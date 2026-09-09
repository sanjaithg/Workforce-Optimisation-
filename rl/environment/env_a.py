"""Environment A — the Gymnasium interface over the model-driven DES.

The world is generated live: demand, disruptions and their cascades are drawn as
the episode runs, so decisions influence what happens next. `env.step()` is called
once per decision epoch (a task becomes assignable, a worker frees, or a shift
boundary is reached); the simulation clock advances internally between epochs.

The Gymnasium wrapper and the rule-based policies in `rl/policies/baselines.py`
exist because the simulation needs something to make dispatch/staffing decisions
in order to run. No learning is implemented here.
"""

from __future__ import annotations

from datetime import timedelta

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulation.core.config import DomainConfig, ScenarioConfig, load_domain, load_scenario
from simulation.core.engine import SimulationEngine
from simulation.entities.core import EpochType

CANDIDATE_SLOTS = 10
CANDIDATE_FEATURES = 6
FEEDBACK_FEATURES = 5
STAFFING_ACTIONS = ("hold", "activate_reserve", "authorise_overtime", "release_overtime")


class WorkforceEnvA(gym.Env):
    """Gymnasium interface over the live discrete-event simulation."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, scenario: str | ScenarioConfig = "normal_weekday",
                 domain: str | DomainConfig = "southern_railway",
                 seed: int | None = None, collect_trace: bool = False,
                 max_steps: int = 20000):
        super().__init__()
        self.scenario_cfg = scenario if isinstance(scenario, ScenarioConfig) else load_scenario(scenario)
        self.domain_cfg = domain if isinstance(domain, DomainConfig) else load_domain(domain)
        self.base_seed = self.scenario_cfg.seed if seed is None else int(seed)
        self.collect_trace = collect_trace
        self.max_steps = max_steps

        self.departments = list(self.domain_cfg.departments)
        self.n_dispatch = CANDIDATE_SLOTS + 1
        self.n_staffing = len(self.departments) * len(STAFFING_ACTIONS)
        self.action_space = spaces.Discrete(self.n_dispatch + self.n_staffing)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self._observation_dim(),), dtype=np.float32)

        self.engine: SimulationEngine | None = None
        self.steps = 0
        self._last_kpis = None

    def _make_engine(self, seed: int) -> SimulationEngine:
        engine = SimulationEngine(self.domain_cfg, self.scenario_cfg, seed=seed,
                                  collect_trace=self.collect_trace)
        engine.start_world_processes()
        return engine

    # ------------------------------------------------------------- observation

    def _observation_dim(self) -> int:
        # epoch(2) + time(4) + global(5) + per-department(5) + candidates + task(4)
        # + observed feedback block(5)
        return (2 + 4 + 5 + 5 * len(self.departments)
                + CANDIDATE_SLOTS * CANDIDATE_FEATURES + 4 + FEEDBACK_FEATURES)

    def _build_observation(self) -> np.ndarray:
        e = self.engine
        epoch = e.current_epoch
        now = float(e.now)
        hour = (now / 60.0) % 24.0
        active = e.on_duty()
        busy = [w for w in active if w.current_task_id is not None]

        feats: list[float] = []
        feats += [1.0 if epoch and epoch.kind == EpochType.DISPATCH else 0.0,
                  1.0 if epoch and epoch.kind == EpochType.STAFFING else 0.0]
        feats += [np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
                  np.sin(2 * np.pi * (now / 1440.0) / 7.0), np.cos(2 * np.pi * (now / 1440.0) / 7.0)]

        n_pending = len(e.pending_tasks)
        overdue = sum(1 for t in e.pending_tasks if t.deadline < now)
        utilisation = len(busy) / len(active) if active else 0.0
        mean_wait = float(np.mean([now - t.created_at for t in e.pending_tasks])) if e.pending_tasks else 0.0
        pressure = e.observed_pressure(hour)
        feats += [np.tanh(n_pending / 40.0), np.tanh(overdue / 15.0), utilisation,
                  np.tanh(mean_wait / 60.0), np.tanh(pressure)]

        for dept in self.departments:
            d_active = [w for w in active if w.department_id == dept]
            d_busy = [w for w in d_active if w.current_task_id is not None]
            d_queue = [t for t in e.pending_tasks if t.department_id == dept]
            d_overdue = sum(1 for t in d_queue if t.deadline < now)
            feats += [
                np.tanh(len(d_queue) / 20.0),
                np.tanh(d_overdue / 8.0),
                (len(d_busy) / len(d_active)) if d_active else 0.0,
                np.tanh(len(d_active) / 60.0),
                1.0 if e.overtime_authorised.get(dept) else 0.0,
            ]

        candidates = epoch.candidates if (epoch and epoch.kind == EpochType.DISPATCH) else []
        task = epoch.task if epoch else None
        for i in range(CANDIDATE_SLOTS):
            if i < len(candidates) and task is not None:
                w = candidates[i]
                travel = e.domain.travel_minutes(w.location_id or w.home_location_id, task.location_id)
                feats += [1.0, w.proficiency(task.required_skill), np.tanh(travel / 45.0),
                          w.fatigue, np.tanh(w.hourly_cost / 5.0), 1.0 if w.on_overtime else 0.0]
            else:
                feats += [0.0] * CANDIDATE_FEATURES

        if task is not None:
            feats += [task.priority / 3.0, np.tanh((task.deadline - now) / 90.0),
                      np.tanh(task.duration_median_min / 60.0), np.tanh(task.deferrals / 3.0)]
        else:
            feats += [0.0, 0.0, 0.0, 0.0]

        # Observed feedback and outcome signals. These come exclusively from
        # `engine.observed_feedback()`, which filters surveys on delivery time and
        # performance on closed review periods — the agent never sees the latent
        # experience that generated them, nor a survey that has not arrived yet.
        fb = e.observed_feedback()
        feats += [
            fb["has_data"],
            (fb["csat_mean"] - 3.0) / 2.0 if fb["has_data"] else 0.0,
            fb["nps_score"] / 100.0 if fb["has_data"] else 0.0,
            np.tanh(fb["n_responses"] / 20.0),
            np.tanh(fb.get("mean_performance", 0.0)),
        ]

        return np.asarray(feats, dtype=np.float32)

    # ----------------------------------------------------------------- masking

    def action_masks(self) -> np.ndarray:
        """Hard feasibility constraints. sb3-contrib's MaskablePPO calls this by name."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        e = self.engine
        epoch = e.current_epoch if e else None
        if epoch is None:
            mask[CANDIDATE_SLOTS] = True
            return mask

        if epoch.kind == EpochType.DISPATCH:
            for i in range(min(len(epoch.candidates), CANDIDATE_SLOTS)):
                mask[i] = True
            mask[CANDIDATE_SLOTS] = True
        else:
            for d_idx, dept in enumerate(self.departments):
                for a_idx, action in enumerate(STAFFING_ACTIONS):
                    flat = self.n_dispatch + d_idx * len(STAFFING_ACTIONS) + a_idx
                    if action == "activate_reserve":
                        legal = (self.scenario_cfg.workforce.get("reserve_available", True)
                                 and any(w.department_id == dept for w in e.reserve_pool))
                    elif action == "authorise_overtime":
                        legal = (self.scenario_cfg.workforce.get("overtime_allowed", True)
                                 and not e.overtime_authorised.get(dept, False))
                    elif action == "release_overtime":
                        legal = e.overtime_authorised.get(dept, False)
                    else:
                        legal = True
                    mask[flat] = legal
        return mask

    # ------------------------------------------------------------------ reward

    def _reward(self, info: dict, before: dict) -> float:
        w = self.scenario_cfg.reward
        e = self.engine
        now = float(e.now)

        completed = e.kpis.tasks_completed - before["completed"]
        breached = e.kpis.tasks_breached - before["breached"]
        labour = e.kpis.labour_cost - before["labour_cost"]
        overtime = e.kpis.overtime_cost - before["overtime_cost"]
        backlog = sum(1 for t in e.pending_tasks if t.deadline < now)
        wait = float(np.mean([now - t.created_at for t in e.pending_tasks])) if e.pending_tasks else 0.0

        util = []
        for dept in self.departments:
            active = [x for x in e.on_duty() if x.department_id == dept]
            if active:
                util.append(sum(1 for x in active if x.current_task_id) / len(active))
        balance_penalty = float(np.std(util)) if len(util) > 1 else 0.0

        reward = (
            float(w["w_throughput"]) * completed
            - float(w["w_sla"]) * breached
            - float(w["w_wait"]) * (wait / 10.0)
            - float(w["w_cost"]) * labour
            - float(w["w_overtime"]) * overtime
            - float(w["w_backlog"]) * (backlog / 10.0)
            - float(w["w_balance"]) * balance_penalty
        )
        info["reward_components"] = {
            "completed": completed, "breached": breached, "labour": round(labour, 3),
            "overtime": round(overtime, 3), "backlog": backlog, "wait": round(wait, 2),
            "balance_penalty": round(balance_penalty, 3),
        }
        return float(reward)

    # ---------------------------------------------------------------- gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        run_seed = self.base_seed if seed is None else int(seed)
        self.engine = self._make_engine(run_seed)
        self.engine.current_epoch = self.engine.next_epoch()
        self.steps = 0
        return self._build_observation(), {
            "epoch": self.engine.current_epoch.kind.value if self.engine.current_epoch else "none"
        }

    def step(self, action: int):
        e = self.engine
        epoch = e.current_epoch
        action = int(action)
        info: dict = {}
        before = {"completed": e.kpis.tasks_completed, "breached": e.kpis.tasks_breached,
                  "labour_cost": e.kpis.labour_cost, "overtime_cost": e.kpis.overtime_cost}

        if epoch is None:
            return self._build_observation(), 0.0, True, False, {"terminal": "no_epoch"}

        if epoch.kind == EpochType.DISPATCH:
            worker = epoch.candidates[action] if (action < CANDIDATE_SLOTS
                                                  and action < len(epoch.candidates)) else None
            info.update(e.apply_dispatch(epoch.task, worker))
            if e.collect_trace:
                e.trace_decisions.append({
                    "t": round(float(e.now), 1), "kind": "dispatch", "task_id": epoch.task.id,
                    "case_id": epoch.task.case_id, "activity": epoch.task.activity_id,
                    "location": epoch.task.location_id, "department": epoch.task.department_id,
                    "n_candidates": len(epoch.candidates),
                    "chosen": (worker.id if worker else "defer"),
                })
        else:
            idx = action - self.n_dispatch
            if 0 <= idx < self.n_staffing:
                dept = self.departments[idx // len(STAFFING_ACTIONS)]
                staffing_action = STAFFING_ACTIONS[idx % len(STAFFING_ACTIONS)]
            else:
                dept, staffing_action = self.departments[0], "hold"
            result = e.apply_staffing(dept, staffing_action)
            info.update(result)
            if e.collect_trace:
                e.trace_decisions.append({
                    "t": round(float(e.now), 1), "kind": "staffing", "department": dept,
                    "chosen": staffing_action, "changed": result.get("changed", 0),
                })

        e.current_epoch = e.next_epoch()
        self.steps += 1
        reward = self._reward(info, before)
        terminated = e.current_epoch is None or e.done
        truncated = self.steps >= self.max_steps
        obs = self._build_observation()
        if terminated or truncated:
            info["kpis"] = e.kpis.summary()
            self._last_kpis = info["kpis"]
        return obs, reward, bool(terminated), bool(truncated), info

    def render(self):
        e = self.engine
        if e is None:
            return
        stamp = (e.episode_start + timedelta(minutes=float(e.now))).strftime("%d %b %H:%M")
        print(f"[{stamp}] pending={len(e.pending_tasks):3d} completed={e.kpis.tasks_completed:4d} "
              f"breached={e.kpis.tasks_breached:3d}")

    @property
    def kpis(self):
        return self.engine.kpis.summary() if self.engine else self._last_kpis
