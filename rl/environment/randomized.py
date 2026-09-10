"""Episode-reseeding wrapper for training.

`WorkforceEnvA.reset()` falls back to a fixed `base_seed` whenever `reset()` is
called without an explicit `seed=` (see env_a.py). A training loop (SB3's
`model.learn()` included) calls `reset()` with no seed on every new episode, so
without this wrapper every training episode would replay the *exact* same world:
same passenger counts, same disruptions, same absences (same seed -> bit-identical
run, by the engine's own reproducibility guarantee). That defeats the point of
training over many episodes — the policy would just be memorising one fixed
trajectory rather than learning something that generalises.

`RandomizedResetEnv` fixes this by drawing a fresh seed (and, optionally, a fresh
scenario) on every `reset()`, regardless of what the caller passed in. Training
seeds are drawn from a high range so they never collide with the low, fixed seeds
(0, 1, 2, ...) conventionally used for evaluation/comparison runs elsewhere in this
repo (`run_sim.py`, `run_baselines.py`) — that convention, not any code-level lock,
is what keeps train/eval seeds disjoint, so evaluation should keep using plain
`WorkforceEnvA` with small explicit seeds rather than this wrapper.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from simulation.core.config import load_scenario

TRAIN_SEED_FLOOR = 10_000     # eval conventionally uses small seeds (0, 1, 2, ...)
TRAIN_SEED_CEILING = 2**31 - 1


class RandomizedResetEnv(gym.Wrapper):
    """Wraps a WorkforceEnvA so every reset() gets a new seed (and optionally scenario).

    `scenarios`, if given, is a list of scenario names; one is sampled uniformly per
    episode and swapped onto the wrapped env before it builds its engine, so a single
    training run can see the full range of stress conditions instead of just one.
    All scenarios share the same domain (southern_railway), so observation/action
    space shapes are identical across the mix.
    """

    def __init__(self, env: gym.Env, seed: int, scenarios: list[str] | None = None):
        super().__init__(env)
        self._rng = np.random.default_rng(seed)
        self._scenarios = list(scenarios) if scenarios else None
        self._scenario_cfgs = (
            {name: load_scenario(name) for name in self._scenarios} if self._scenarios else None
        )

    def reset(self, *, seed=None, options=None):
        if self._scenario_cfgs:
            name = self._scenarios[int(self._rng.integers(len(self._scenarios)))]
            self.env.unwrapped.scenario_cfg = self._scenario_cfgs[name]
        draw = int(self._rng.integers(TRAIN_SEED_FLOOR, TRAIN_SEED_CEILING))
        return self.env.reset(seed=draw, options=options)

    def __getattr__(self, name):
        # gymnasium's own Wrapper no longer auto-forwards unknown attributes, but
        # sb3-contrib's ActionMasker calls `action_mask_fn(env)` on whatever env it
        # was given -- which is this wrapper -- so WorkforceEnvA-specific members
        # (action_masks(), engine, departments, ...) need to reach the wrapped env.
        return getattr(self.env, name)
