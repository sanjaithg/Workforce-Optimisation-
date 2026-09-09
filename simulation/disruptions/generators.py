"""Disruption generation with self-excitation (cascades) and regime switching.

Two properties matter here:

1. Disruptions are not scripted. They arise from point processes whose intensity
   depends on what has already happened, so `failure -> specialist needed ->
   specialist unavailable -> queue -> delay -> more work` emerges from the
   mechanics instead of being hard-coded as a story.

2. Disruptions spawn ordinary Tasks through the ordinary Process engine. There is
   no separate "disruption handler" pathway, which is precisely why cascades
   compose with normal work and compete for the same scarce workers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from simulation.core.config import DomainConfig, ScenarioConfig
from simulation.distributions.library import HawkesProcess, heavy_tailed_severity


@dataclass
class Disruption:
    id: str
    kind: str                  # asset_failure | train_delay | demand_spike
    time: float                # simulation minutes
    location_id: str
    severity_minutes: float
    detail: dict = field(default_factory=dict)
    caused_by: str | None = None


class DisruptionEngine:
    """Poisson baseline + Hawkes excitation, per location.

    lambda_loc(t) = mu_loc + sum over recent events at/near loc of
                    alpha * exp(-beta * (t - t_i))

    A failure therefore makes near-term further trouble at that location more
    likely — the empirical signature of real cascading failure — and neighbouring
    locations receive a damped share of the excitation.
    """

    def __init__(self, domain: DomainConfig, scenario: ScenarioConfig, rng_registry):
        self.domain = domain
        self.scenario = scenario
        self.rng = rng_registry
        cfg = scenario.disruptions

        # Branching ratio: expected direct offspring per event. Must stay < 1 or
        # the disruption process is non-stationary and explodes.
        self.branching_ratio = float(cfg["hawkes_branching_ratio"])
        self.beta = float(cfg["hawkes_beta"]) / 60.0        # config is per hour; sim is per minute
        # Fraction of a location's branching mass that leaks to its neighbours.
        self.spread_fraction = float(cfg.get("hawkes_spread_fraction", 0.25))
        self.severity_median = float(cfg["severity_median"])
        self.severity_sigma = float(cfg["severity_sigma"])
        self.severity_tail_prob = float(cfg["severity_tail_prob"])
        self.severity_tail_alpha = float(cfg["severity_tail_alpha"])

        # Per-hour base rates scaled by station tier: bigger stations have more to
        # go wrong. Converted to per-minute intensities.
        tier_weight = {1: 1.6, 2: 1.0, 3: 0.5}
        self.processes: dict[tuple[str, str], HawkesProcess] = {}
        for loc_id, loc in domain.locations.items():
            w = tier_weight[loc.tier]
            for kind, rate_key in (
                ("asset_failure", "asset_failure_rate"),
                ("train_delay", "train_delay_rate"),
                ("demand_spike", "demand_spike_rate"),
            ):
                mu = float(cfg[rate_key]) * w / 60.0
                self.processes[(loc_id, kind)] = HawkesProcess(
                    mu=mu, branching_ratio=self.branching_ratio, beta=self.beta
                )

        self.regime = "normal"
        self.failure_multiplier = 1.0
        self._counter = 0
        self.history: list[Disruption] = []

    # ---------- regime switching (non-stationarity) ----------

    def step_regime(self, dt_minutes: float) -> str | None:
        """Hidden-Markov style regime jumps: structural change within an episode."""
        drift = self.scenario.drift
        if not drift.get("enabled", False):
            return None
        p_hour = float(drift.get("transition_prob_per_hour", 0.0))
        p = 1.0 - (1.0 - p_hour) ** (dt_minutes / 60.0)
        if self.rng.regime.random() < p:
            regimes = list(drift["regimes"])
            new = regimes[int(self.rng.regime.integers(len(regimes)))]
            if new != self.regime:
                self.regime = new
                self.failure_multiplier = float(drift["regimes"][new]["failure_mult"])
                return new
        return None

    def demand_multiplier(self) -> float:
        drift = self.scenario.drift
        if not drift.get("enabled", False):
            return 1.0
        return float(drift["regimes"][self.regime]["demand_mult"])

    # ---------- event generation ----------

    def sample_window(self, t_start: float, t_end: float) -> list[Disruption]:
        """Draw all disruptions in [t_start, t_end) by thinning each process."""
        out: list[Disruption] = []
        rng = self.rng.disruption_base
        for (loc_id, kind), proc in self.processes.items():
            t = t_start
            while True:
                intensity_cap = (proc.intensity(t) * self.failure_multiplier) + 1e-12
                t = t + float(rng.exponential(1.0 / intensity_cap))
                if t >= t_end:
                    break
                accept = (proc.intensity(t) * self.failure_multiplier) / intensity_cap
                if rng.random() > accept:
                    continue
                proc.record(t)
                self._counter += 1
                severity = heavy_tailed_severity(
                    self.rng.disruption_severity, self.severity_median, self.severity_sigma,
                    self.severity_tail_prob, self.severity_tail_alpha,
                )
                # Provenance: if this event landed while excitation was elevated,
                # it is (probabilistically) a cascade child of an earlier event.
                excited = proc.excitation(t) > proc.mu
                caused_by = None
                if excited and self.history:
                    # Attribute to the most recent event at this location, if it is
                    # recent enough to still be exciting the process.
                    for d_prev in reversed(self.history[-60:]):
                        if d_prev.location_id == loc_id and t - d_prev.time < 180:
                            caused_by = d_prev.id
                            break
                d = Disruption(
                    id=f"D{self._counter:05d}", kind=kind, time=t, location_id=loc_id,
                    severity_minutes=severity, caused_by=caused_by,
                    detail={"regime": self.regime, "excited": bool(excited)},
                )
                out.append(d)
            proc.prune(t_end, lookback=360.0)

        out.sort(key=lambda d: d.time)
        self.history.extend(out)
        if len(self.history) > 4000:
            self.history = self.history[-2000:]
        return out

    def excite_neighbours(self, disruption: Disruption) -> None:
        """Spread a damped share of excitation to connected locations.

        This is what lets trouble propagate along the network rather than staying
        politely inside one station.
        """
        loc = self.domain.locations[disruption.location_id]
        for neighbour, travel in loc.neighbours.items():
            if neighbour not in self.domain.locations:
                continue
            # Damped, weighted excitation. The weight is a fraction of one unit of
            # branching mass, so spreading trouble across the network cannot push
            # the aggregate branching ratio above the stability threshold.
            damping = float(np.exp(-travel / 60.0)) * self.spread_fraction
            if damping <= 0.02:
                continue
            for kind in ("asset_failure", "train_delay"):
                proc = self.processes.get((neighbour, kind))
                if proc is not None:
                    proc.record(disruption.time, weight=damping)
