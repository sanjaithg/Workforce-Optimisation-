"""Contract tests for the two RL environments.

Both must satisfy the Gymnasium API, and — more importantly — both must enforce
qualification and skill constraints through the action mask rather than leaving
them to the reward function.
"""

from __future__ import annotations

import numpy as np
import pytest

from rl.environment.env_a import CANDIDATE_SLOTS, STAFFING_ACTIONS
from rl.environment.env_a import WorkforceEnvA
from rl.policies.baselines import build_policy
from simulation.core.config import load_scenario
from simulation.entities.core import EpochType
from simulation.workforce.population import is_eligible

SHORT = {"scenario.duration_days": 1}


def short_env(scenario: str = "normal_weekday", seed: int = 0) -> WorkforceEnvA:
    cfg = load_scenario(scenario).with_overrides(**SHORT)
    return WorkforceEnvA(scenario=cfg, seed=seed, max_steps=250)


def test_gymnasium_api_shapes():
    env = short_env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), "observation outside declared space"
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(term, bool) and isinstance(trunc, bool)


def test_observation_is_finite():
    env = short_env()
    obs, _ = env.reset(seed=0)
    for _ in range(60):
        mask = env.action_masks()
        action = int(np.flatnonzero(mask)[0])
        obs, r, term, trunc, _ = env.step(action)
        assert np.all(np.isfinite(obs)), "non-finite value in observation"
        assert np.isfinite(r)
        if term or trunc:
            break


def test_action_mask_always_offers_a_legal_action():
    env = short_env()
    env.reset(seed=0)
    for _ in range(120):
        mask = env.action_masks()
        assert mask.any(), "no legal action available"
        obs, r, term, trunc, _ = env.step(int(np.flatnonzero(mask)[0]))
        if term or trunc:
            break


def test_mask_only_exposes_eligible_workers():
    """The mask is the safety boundary: it must never offer an ineligible worker."""
    env = short_env()
    env.reset(seed=0)
    checked = 0
    for _ in range(300):
        e = env.engine
        epoch = e.current_epoch
        if epoch is not None and epoch.kind == EpochType.DISPATCH:
            mask = env.action_masks()
            hour = (float(e.now) / 60.0) % 24.0
            for i in range(CANDIDATE_SLOTS):
                if mask[i]:
                    worker = epoch.candidates[i]
                    assert is_eligible(worker, epoch.task, hour, e.domain).eligible
                    assert epoch.task.required_qualifications.issubset(worker.qualifications)
                    checked += 1
        obs, r, term, trunc, _ = env.step(int(np.flatnonzero(env.action_masks())[0]))
        if term or trunc:
            break
    assert checked > 0, "no dispatch decisions were exercised"


def test_staffing_actions_masked_at_dispatch_epochs():
    env = short_env()
    env.reset(seed=0)
    for _ in range(200):
        epoch = env.engine.current_epoch
        mask = env.action_masks()
        if epoch is not None and epoch.kind == EpochType.DISPATCH:
            assert not mask[env.n_dispatch:].any(), "staffing action legal at a dispatch epoch"
        obs, r, term, trunc, _ = env.step(int(np.flatnonzero(mask)[0]))
        if term or trunc:
            break


def test_same_seed_reproduces_episode():
    results = []
    for _ in range(2):
        env = short_env(seed=3)
        policy = build_policy("greedy")
        obs, _ = env.reset(seed=3)
        total = 0.0
        while True:
            obs, r, term, trunc, _ = env.step(policy.act(obs, env.action_masks(), env))
            total += r
            if term or trunc:
                break
        results.append((round(total, 6), env.engine.kpis.tasks_created))
    assert results[0] == results[1], "identical seeds produced different episodes"


def test_different_seeds_produce_different_episodes():
    totals = []
    for seed in (1, 2):
        env = short_env(seed=seed)
        policy = build_policy("greedy")
        obs, _ = env.reset(seed=seed)
        total = 0.0
        while True:
            obs, r, term, trunc, _ = env.step(policy.act(obs, env.action_masks(), env))
            total += r
            if term or trunc:
                break
        totals.append(round(total, 4))
    assert totals[0] != totals[1]


def test_stress_scenario_degrades_service():
    """A harder world must produce worse KPIs, or the scenarios mean nothing."""
    def breach_rate(scenario: str) -> float:
        env = WorkforceEnvA(scenario=load_scenario(scenario).with_overrides(**SHORT),
                            seed=0, max_steps=20000)
        policy = build_policy("greedy")
        obs, _ = env.reset(seed=0)
        while True:
            obs, r, term, trunc, _ = env.step(policy.act(obs, env.action_masks(), env))
            if term or trunc:
                break
        return env.engine.kpis.sla_breach_rate

    assert breach_rate("combined_stress") > breach_rate("normal_weekday")


def test_disruptions_spawn_downstream_work():
    """Cascade check: a disruption must create tasks, not just log a message."""
    env = WorkforceEnvA(scenario=load_scenario("equipment_disruption").with_overrides(**SHORT),
                        seed=1, max_steps=20000)
    policy = build_policy("greedy")
    obs, _ = env.reset(seed=1)
    while True:
        obs, r, term, trunc, _ = env.step(policy.act(obs, env.action_masks(), env))
        if term or trunc:
            break
    e = env.engine
    spawned = [t for t in e.all_tasks.values() if t.spawned_by_event]
    assert e.kpis.disruptions > 0
    assert len(spawned) > 0, "disruptions produced no downstream tasks"


def test_deferral_does_not_stall_the_clock():
    """Repeated deferral must advance time, or the episode could loop forever."""
    env = short_env()
    env.reset(seed=0)
    start = float(env.engine.now)
    for _ in range(40):
        mask = env.action_masks()
        action = CANDIDATE_SLOTS if mask[CANDIDATE_SLOTS] else int(np.flatnonzero(mask)[0])
        obs, r, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    assert float(env.engine.now) > start
