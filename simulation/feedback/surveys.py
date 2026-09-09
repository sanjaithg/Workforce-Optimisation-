"""Sparse, delayed customer-feedback instruments: CSAT and NPS.

Two properties make this realistic, and both matter for the RL formulation:

SPARSITY  Only a small fraction of service interactions produce a response, and
          responders are biased (unhappy customers answer more often). The agent
          therefore sees a small, skewed sample, not the population mean.

DELAY     Responses arrive well after the interaction. At time t the agent may
          only use responses *delivered* by t. This is enforced structurally in
          `FeedbackRegistry.observed()` rather than left to convention, so
          feedback can never leak information about the present or future.

Consequently NPS/CSAT are NOT dense per-timestep rewards. They are a slow,
noisy, partially observed quality signal — which is exactly what they are in a
real operation, and what makes them interesting for a POMDP-flavoured extension.

Everything here is synthetic and configurable. No IRCTC or Indian Railways
customer-feedback data is used, implied, or required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SurveyConfig:
    """[SYNTH] survey instrument parameters."""

    response_rate: float = 0.06            # fraction of interactions that respond at all
    detractor_response_boost: float = 2.2  # unhappy customers answer more often
    delay_median_minutes: float = 150.0
    delay_sigma: float = 0.7
    csat_noise: float = 0.55               # latent-to-rating noise, in rating points
    nps_noise: float = 1.1
    enabled: bool = True

    @classmethod
    def from_config(cls, cfg: dict | None) -> "SurveyConfig":
        if not cfg:
            return cls()
        return cls(**{k: (float(v) if k != "enabled" else bool(v))
                      for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class SurveyResponse:
    """One returned survey. `delivered_at` gates when the agent may use it."""

    case_id: str
    location_id: str
    process_id: str
    interaction_at: float
    delivered_at: float
    csat: int              # 1..5
    nps: int               # 0..10
    latent_experience: float   # recorded for validation ONLY; never observable


class FeedbackRegistry:
    """Generates and stores survey responses, and enforces the delivery delay."""

    def __init__(self, config: SurveyConfig, rng):
        self.cfg = config
        self.rng = rng
        self.responses: list[SurveyResponse] = []
        self.interactions = 0
        self._sorted_dirty = False

    # ------------------------------------------------------------- generation

    def maybe_survey(self, case_id: str, location_id: str, process_id: str,
                     now: float, latent_experience: float) -> SurveyResponse | None:
        """Sample a survey response for one completed customer-facing interaction."""
        if not self.cfg.enabled:
            return None
        self.interactions += 1

        # Non-response bias: dissatisfied customers are likelier to respond, so the
        # observed mean is pessimistic relative to the true population mean. An
        # agent that treats observed CSAT as unbiased will misjudge service quality.
        dissatisfaction = 1.0 - latent_experience
        p = self.cfg.response_rate * (1.0 + self.cfg.detractor_response_boost * dissatisfaction)
        if self.rng.random() > min(0.95, p):
            return None

        delay = float(self.rng.lognormal(np.log(self.cfg.delay_median_minutes),
                                         self.cfg.delay_sigma))
        response = SurveyResponse(
            case_id=case_id, location_id=location_id, process_id=process_id,
            interaction_at=float(now), delivered_at=float(now + delay),
            csat=self._to_csat(latent_experience), nps=self._to_nps(latent_experience),
            latent_experience=float(latent_experience),
        )
        self.responses.append(response)
        self._sorted_dirty = True
        return response

    def _to_csat(self, e: float) -> int:
        """Latent experience -> 1..5 rating, with rating noise and rounding."""
        raw = 1.0 + 4.0 * e + float(self.rng.normal(0.0, self.cfg.csat_noise))
        return int(np.clip(round(raw), 1, 5))

    def _to_nps(self, e: float) -> int:
        """Latent experience -> 0..10 likelihood-to-recommend.

        Deliberately harsher than CSAT at the same experience level: NPS scores
        cluster high only for genuinely excellent service, which is why NPS and
        CSAT are not redundant instruments.
        """
        raw = 10.0 * (e ** 1.35) + float(self.rng.normal(0.0, self.cfg.nps_noise))
        return int(np.clip(round(raw), 0, 10))

    # ------------------------------------------------------------- observation

    def observed(self, now: float, window_minutes: float = 720.0) -> list[SurveyResponse]:
        """Responses the agent is allowed to see at time `now`.

        Filters on `delivered_at <= now`, NOT on when the interaction happened:
        this is the structural guarantee against feedback leakage. It is also a
        *rolling* window, so old responses age out rather than accumulating — an
        operator watching a live dashboard sees recent sentiment, not an
        ever-growing all-time average that would drown out current conditions.
        """
        lo = now - window_minutes
        return [r for r in self.responses if lo <= r.delivered_at <= now]

    def observed_summary(self, now: float, window_minutes: float = 720.0) -> dict:
        sample = self.observed(now, window_minutes)
        if not sample:
            return {"n_responses": 0, "csat_mean": 0.0, "nps_score": 0.0, "has_data": 0.0}
        csat = float(np.mean([r.csat for r in sample]))
        promoters = sum(1 for r in sample if r.nps >= 9)
        detractors = sum(1 for r in sample if r.nps <= 6)
        nps_score = 100.0 * (promoters - detractors) / len(sample)
        return {"n_responses": len(sample), "csat_mean": csat,
                "nps_score": nps_score, "has_data": 1.0}

    # -------------------------------------------------- validation-only views

    def true_summary(self) -> dict:
        """Ground truth over ALL responses, ignoring delivery delay.

        For validation and analysis only. Never expose this to a policy: it
        contains information no real operator would have at decision time.
        """
        if not self.responses:
            return {"n": 0, "csat_mean": 0.0, "nps_score": 0.0, "latent_mean": 0.0}
        promoters = sum(1 for r in self.responses if r.nps >= 9)
        detractors = sum(1 for r in self.responses if r.nps <= 6)
        return {
            "n": len(self.responses),
            "response_rate": round(len(self.responses) / max(1, self.interactions), 4),
            "csat_mean": round(float(np.mean([r.csat for r in self.responses])), 3),
            "nps_score": round(100.0 * (promoters - detractors) / len(self.responses), 2),
            "latent_mean": round(float(np.mean([r.latent_experience for r in self.responses])), 4),
        }
