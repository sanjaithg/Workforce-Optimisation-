"""Diagnostics: why is the world behaving the way it is?

Answers the questions that matter when a run looks implausible: is the disruption
process stable, is work reaching eligible staff, and where is capacity going?
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from rl.environment.env_a import WorkforceEnvA
from rl.policies.baselines import build_policy
from simulation.workforce.population import is_eligible


def diagnose(scenario: str = "normal_weekday", policy_name: str = "greedy", seed: int = 0) -> None:
    env = WorkforceEnvA(scenario=scenario, seed=seed)
    policy = build_policy(policy_name, seed=seed)
    obs, _ = env.reset(seed=seed)
    while True:
        mask = env.action_masks()
        obs, reward, term, trunc, info = env.step(policy.act(obs, mask, env))
        if term or trunc:
            break

    e = env.engine
    now = float(e.now)
    roster = e._roster()

    print(f"=== diagnostics: {scenario} / {policy_name} / seed {seed} ===\n")

    print(f"workforce total        : {len(roster)}")
    on_shift = [w for w in roster if e.domain.shifts[w.shift_id].covers((now / 60) % 24)]
    print(f"on shift at episode end: {len(on_shift)}")
    print(f"absent                 : {sum(1 for w in roster if w.absent_today)}")
    by_dept = Counter(w.department_id for w in roster)
    print(f"by department          : {dict(by_dept)}")
    print(f"mean fatigue           : {np.mean([w.fatigue for w in roster]):.3f}")
    print(f"workers who did nothing: {sum(1 for w in roster if w.completed_tasks == 0)}")

    print("\n--- disruptions ---")
    print(f"total                  : {e.kpis.disruptions}  (cascade: {e.kpis.cascade_disruptions})")
    kinds = Counter(ev["kind"] for ev in e.log.events)
    print(f"event kinds            : {dict(kinds)}")

    print("\n--- tasks ---")
    print(f"created                : {e.kpis.tasks_created}")
    print(f"completed              : {e.kpis.tasks_completed}")
    print(f"breached               : {e.kpis.tasks_breached}")
    print(f"still pending at end   : {len(e.pending_tasks)}")
    created_by_process = Counter(t.process_id for t in e.all_tasks.values())
    print(f"by process             : {dict(created_by_process)}")
    created_by_dept = Counter(t.department_id for t in e.all_tasks.values())
    print(f"by department          : {dict(created_by_dept)}")

    print("\n--- where does lateness concentrate? ---")
    breached = [t for t in e.all_tasks.values() if t.status.value == "breached"]
    completed = [t for t in e.all_tasks.values() if t.status.value == "completed"]
    print(f"breached by process    : {dict(Counter(t.process_id for t in breached))}")
    print(f"breached by department : {dict(Counter(t.department_id for t in breached))}")
    print(f"breached by location   : {dict(Counter(t.location_id for t in breached).most_common(5))}")
    if completed:
        print(f"mean queue time (completed): {np.mean([t.queue_time for t in completed]):.1f} min")
    if breached:
        served = [t for t in breached if t.started_at is not None]
        if served:
            print(f"mean queue time (breached) : {np.mean([t.queue_time for t in served]):.1f} min")
        print(f"breached never started     : {sum(1 for t in breached if t.started_at is None)}")

    print("\n--- why are pending tasks stuck? ---")
    reasons: Counter = Counter()
    stuck_by_skill: Counter = Counter()
    hour = (now / 60.0) % 24.0
    for task in e.pending_tasks[:400]:
        blockers = Counter()
        any_eligible = False
        for w in roster:
            res = is_eligible(w, task, hour, e.domain)
            if res.eligible:
                any_eligible = True
                break
            blockers[res.reason.split(":")[0]] += 1
        if not any_eligible:
            reasons[blockers.most_common(1)[0][0] if blockers else "unknown"] += 1
            stuck_by_skill[f"{task.department_id}/{task.required_skill}"] += 1
    print(f"pending with NO eligible worker: {sum(reasons.values())} of {min(len(e.pending_tasks), 400)} sampled")
    print(f"dominant blocking reason       : {dict(reasons)}")
    print(f"stuck task skills              : {dict(stuck_by_skill.most_common(6))}")

    print("\n--- capacity check ---")
    total_worker_minutes = sum(w.busy_minutes for w in roster)
    # Each worker covers one shift per simulated day, so supply is
    # headcount x shift length x days (not divided across shifts).
    mean_shift_len = np.mean([s.duration_hours for s in e.domain.shifts.values()]) * 60
    days = e.horizon / 1440
    supply = len(roster) * mean_shift_len * days
    print(f"worker-minutes worked  : {total_worker_minutes:,.0f}")
    print(f"on-shift supply        : {supply:,.0f}")
    print(f"implied utilisation    : {total_worker_minutes / max(supply, 1):.3f}")
    print(f"reported mean_utilisation (instantaneous, on-duty): {e.kpis.mean_utilisation:.3f}")

    from collections import Counter as _C
    shift_dist = _C(w.shift_id for w in roster)
    print(f"shift distribution     : {dict(shift_dist)}")
    print(f"abandon reasons        : {dict(e.kpis.abandon_reasons)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="normal_weekday")
    ap.add_argument("--policy", default="greedy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    diagnose(args.scenario, args.policy, args.seed)


if __name__ == "__main__":
    main()
