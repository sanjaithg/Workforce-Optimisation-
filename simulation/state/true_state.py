"""The latent world state: what is true, as opposed to what is observed.

This module exists to make the separation structural rather than a matter of
discipline. Anything recorded here is, by construction, NOT available to the
agent. The observation builder never reads from this object; it reads from
observed instruments (delivered surveys, closed performance periods, realised
counts), which are noisy, delayed and incomplete views of what lives here.

Keeping the split explicit is what lets a POMDP variant be added later without
rewriting the generative model: the state equation is already separate from the
observation equation.

    s_{t+1} = F(s_t, a_t, exogenous_t, process noise)      <- TrueState
    o_t     = H(s_t) + observation noise                   <- observation builder
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrueState:
    """Latent quantities the simulation knows and the agent does not."""

    # Latent service experience by location, as (time, experience) samples.
    experience_samples: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    # Latent demand intensity actually used to generate bookings.
    demand_intensity: dict[str, float] = field(default_factory=dict)
    # True asset health, before any inspection reveals it.
    asset_health: dict[str, float] = field(default_factory=dict)

    def record_experience(self, location_id: str, t: float, experience: float) -> None:
        self.experience_samples.setdefault(location_id, []).append((float(t), float(experience)))

    def record_intensity(self, key: str, value: float) -> None:
        self.demand_intensity[key] = float(value)

    def true_experience_mean(self, since: float = 0.0) -> float:
        """Ground-truth mean experience. For validation and analysis ONLY.

        Comparing this against the observed survey mean is how we demonstrate that
        the survey instrument is biased and lagged, rather than merely asserting it.
        """
        vals = [e for samples in self.experience_samples.values()
                for t, e in samples if t >= since]
        return float(np.mean(vals)) if vals else 0.0

    def summary(self) -> dict:
        return {
            "latent_experience_mean": round(self.true_experience_mean(), 4),
            "latent_experience_samples": sum(len(v) for v in self.experience_samples.values()),
        }
