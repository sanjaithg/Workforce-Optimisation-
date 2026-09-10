"""A trained MaskablePPO checkpoint, wrapped to look like any other Policy.

Deliberately separate from rl/policies/baselines.py: the baselines are always
available (numpy is the only dependency), while this module imports sb3-contrib,
which is an optional dependency (see requirements.txt) only needed once you're
actually training/evaluating a learned policy.
"""

from __future__ import annotations

from pathlib import Path

from rl.policies.baselines import Policy


class LearnedPolicy(Policy):
    """Loads a saved MaskablePPO model and drives it through the Policy interface,
    so it drops into run_sim.py / run_baselines.py / diagnose.py unchanged."""

    name = "learned"

    def __init__(self, model):
        self.model = model

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "LearnedPolicy":
        from sb3_contrib import MaskablePPO

        return cls(MaskablePPO.load(str(path)))

    def act(self, obs, mask, env) -> int:
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)
