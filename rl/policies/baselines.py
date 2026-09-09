"""Baseline policies.

These exist to answer the only question that matters before any RL is attempted:
*is there anything here for a learned policy to improve on?* If a greedy rule
already achieves what PPO achieves, the honest conclusion is that RL adds nothing
to this formulation — and that is a result worth reporting, not hiding.
"""

from __future__ import annotations

import numpy as np

from rl.environment.env_a import CANDIDATE_SLOTS, STAFFING_ACTIONS


class Policy:
    name = "base"

    def act(self, obs, mask, env) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class RandomPolicy(Policy):
    """Uniform over legal actions. The floor any useful policy must clear."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, mask, env) -> int:
        legal = np.flatnonzero(mask)
        return int(self.rng.choice(legal))


class GreedyPolicy(Policy):
    """Always take the best-ranked eligible worker; never defer; never staff up.

    Candidates arrive pre-sorted by (skill, proximity, fatigue, cost), so action 0
    is the greedy choice by construction.
    """

    name = "greedy"

    def act(self, obs, mask, env) -> int:
        if mask[0]:
            return 0
        legal = np.flatnonzero(mask)
        return int(legal[0])


class WFMHeuristicPolicy(Policy):
    """A workforce-management-style rule, in the spirit of what commercial WFM
    tools do: watch the queue, and escalate staffing when service is at risk.

    Dispatch: take the best available worker, but defer low-priority work when the
    backlog is manageable and the candidate is a poor skill match (saving the good
    worker for something that needs them).

    Staffing: if a department's queue and overdue count are high, authorise
    overtime; if it stays high, pull in reserves; release overtime once things calm
    down, since idle overtime is pure cost.
    """

    name = "wfm_heuristic"

    def __init__(self, queue_threshold: int = 6, overdue_threshold: int = 2):
        self.queue_threshold = queue_threshold
        self.overdue_threshold = overdue_threshold

    def act(self, obs, mask, env) -> int:
        e = env.engine
        epoch = e.current_epoch
        if epoch is None:
            return int(np.flatnonzero(mask)[0])

        if epoch.kind.value == "dispatch":
            task = epoch.task
            if not epoch.candidates or not mask[0]:
                return CANDIDATE_SLOTS if mask[CANDIDATE_SLOTS] else int(np.flatnonzero(mask)[0])
            best = epoch.candidates[0]
            slack = task.deadline - float(e.now)
            # Hold back a strong specialist from a low-priority job that has slack,
            # provided the queue is short enough that the delay is affordable.
            if (task.priority >= 3 and slack > 60 and len(e.pending_tasks) < self.queue_threshold
                    and best.proficiency(task.required_skill) > 0.75 and task.deferrals < 2
                    and mask[CANDIDATE_SLOTS]):
                return CANDIDATE_SLOTS
            return 0

        # Staffing epoch: react to the department under most pressure.
        now = float(e.now)
        worst_dept, worst_score, worst_overdue = None, -1.0, 0
        for dept in env.departments:
            queue = [t for t in e.pending_tasks if t.department_id == dept]
            overdue = sum(1 for t in queue if t.deadline < now)
            score = len(queue) + 2.0 * overdue
            if score > worst_score:
                worst_dept, worst_score, worst_overdue = dept, score, overdue

        def flat(dept: str, action: str) -> int:
            return (env.n_dispatch + env.departments.index(dept) * len(STAFFING_ACTIONS)
                    + STAFFING_ACTIONS.index(action))

        if worst_dept is not None:
            if worst_score >= self.queue_threshold * 2 or worst_overdue >= self.overdue_threshold:
                for action in ("authorise_overtime", "activate_reserve"):
                    idx = flat(worst_dept, action)
                    if mask[idx]:
                        return idx
            else:
                # Calm: stand down any overtime that is still running.
                for dept in env.departments:
                    idx = flat(dept, "release_overtime")
                    if mask[idx]:
                        return idx
        legal = np.flatnonzero(mask)
        return int(legal[0])


POLICIES = {
    "random": RandomPolicy,
    "greedy": GreedyPolicy,
    "wfm_heuristic": WFMHeuristicPolicy,
}


def build_policy(name: str, seed: int = 0) -> Policy:
    if name not in POLICIES:
        raise KeyError(f"Unknown policy '{name}'. Available: {', '.join(POLICIES)}")
    cls = POLICIES[name]
    return cls(seed=seed) if cls is RandomPolicy else cls()
