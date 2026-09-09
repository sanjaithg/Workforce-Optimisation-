"""Latent service experience — the hidden variable behind customer feedback.

IMPORTANT FRAMING: none of this is derived from IRCTC, Indian Railways, or any
real customer-feedback source. NPS and CSAT are sparse and inconsistently
available in public railway data, so we *generate* them from a latent experience
model. They are synthetic instruments on a synthetic world, configurable in
config/scenarios/*.yaml.

The chain is deliberately one-directional and causal:

    operational reality (wait, crowding, delay, resolution quality)
        -> latent service experience  e in [0, 1]     <- NEVER observed
            -> sparse, delayed survey responses        <- what the agent may see

Keeping `e` latent matters. If the agent could read `e` directly it would have a
noise-free, instantaneous quality signal that no real operator possesses, and any
policy learned against it would be unlearnable in reality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.entities.core import Task


@dataclass
class ExperienceFactors:
    """Weights translating operational conditions into felt experience. [SYNTH]"""

    baseline: float = 0.78
    wait_weight: float = 0.42          # queue time relative to the SLA
    breach_penalty: float = 0.22       # missing the deadline outright
    crowding_weight: float = 0.20      # station pressure at the time of service
    delay_weight: float = 0.18         # exposure to active train delay
    quality_weight: float = 0.16       # served by a proficient, unfatigued worker
    rework_penalty: float = 0.12       # the job had to be redone
    noise_sigma: float = 0.09          # individual variation between passengers

    @classmethod
    def from_config(cls, cfg: dict | None) -> "ExperienceFactors":
        if not cfg:
            return cls()
        return cls(**{k: float(v) for k, v in cfg.items() if k in cls.__dataclass_fields__})


class ExperienceModel:
    """Computes latent experience for a completed customer-facing case."""

    def __init__(self, factors: ExperienceFactors, rng):
        self.f = factors
        self.rng = rng

    def latent_experience(self, task: Task, sla_minutes: float, crowding: float,
                          active_delay_minutes: float, server_proficiency: float,
                          server_fatigue: float, breached: bool) -> float:
        f = self.f
        wait_ratio = min(2.0, task.queue_time / max(sla_minutes, 1.0))
        quality = server_proficiency * (1.0 - 0.5 * server_fatigue)

        e = (
            f.baseline
            - f.wait_weight * wait_ratio
            - f.breach_penalty * float(breached)
            - f.crowding_weight * float(np.tanh(max(0.0, crowding - 0.9)))
            - f.delay_weight * float(np.tanh(active_delay_minutes / 45.0))
            + f.quality_weight * (quality - 0.5)
            - f.rework_penalty * min(1.0, task.rework_count)
        )
        e += float(self.rng.normal(0.0, f.noise_sigma))
        return float(np.clip(e, 0.0, 1.0))
