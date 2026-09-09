"""Validation: does this synthetic world behave plausibly?

Two kinds of check, kept deliberately separate:

INVARIANTS  — things that must never happen regardless of parameters. A failure
              here is a bug in the simulator.
PLAUSIBILITY — statistical properties we *expect* of a realistic world (weekly
              seasonality, overdispersed demand, positive-skewed durations,
              stable queues). A failure here means the world is miscalibrated,
              not necessarily broken.

We deliberately do NOT claim agreement with real Southern Railway statistics.
The dataset is synthetic; the honest claim is internal consistency and
distributional plausibility, not fidelity to a real railway.
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from rl.environment.env_a import WorkforceEnvA
from rl.policies.baselines import build_policy
from simulation.core.config import load_domain, load_scenario
from simulation.core.rng import RNGRegistry
from simulation.demand.model import DemandModel


class Check:
    def __init__(self):
        self.results: list[tuple[str, bool, str]] = []

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append((name, bool(ok), detail))

    def report(self) -> bool:
        width = max(len(n) for n, _, _ in self.results) + 2
        passed = 0
        for name, ok, detail in self.results:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name:<{width}} {detail}")
            passed += int(ok)
        print(f"\n  {passed}/{len(self.results)} checks passed")
        return passed == len(self.results)


def run_invariants(scenario: str, seed: int, check: Check) -> None:
    env = WorkforceEnvA(scenario=scenario, seed=seed)
    policy = build_policy("wfm_heuristic")
    obs, _ = env.reset(seed=seed)
    while True:
        obs, r, term, trunc, info = env.step(policy.act(obs, env.action_masks(), env))
        if term or trunc:
            break
    e = env.engine

    # 1. Every task ends in a terminal state or is still pending — none vanish.
    counts = Counter(t.status.value for t in e.all_tasks.values())
    accounted = sum(counts.values())
    check.record("task accounting: none lost", accounted == e.kpis.tasks_created,
                 f"{accounted} tracked vs {e.kpis.tasks_created} created")

    # 2. No worker is ever assigned two tasks at once.
    in_progress = [t for t in e.all_tasks.values() if t.status.value == "in_progress"]
    workers_in_progress = [t.assigned_worker_id for t in in_progress]
    check.record("no double-booked workers",
                 len(workers_in_progress) == len(set(workers_in_progress)),
                 f"{len(in_progress)} tasks in progress at end")

    # 3. Qualification constraints were never violated (the mask must be airtight).
    violations = 0
    for t in e.all_tasks.values():
        if t.assigned_worker_id is None:
            continue
        w = next((x for x in e._roster() if x.id == t.assigned_worker_id), None)
        if w is None:
            continue
        if not t.required_qualifications.issubset(w.qualifications):
            violations += 1
        if w.proficiency(t.required_skill) < t.min_proficiency:
            violations += 1
    check.record("qualification/skill constraints never violated", violations == 0,
                 f"{violations} violations")

    # 4. Timing sanity: nothing completes before it starts or before it was created.
    bad_timing = sum(
        1 for t in e.all_tasks.values()
        if t.started_at is not None and t.completed_at is not None
        and (t.completed_at < t.started_at or t.started_at < t.created_at)
    )
    check.record("task timings monotonic", bad_timing == 0, f"{bad_timing} bad")

    # 5. Queues do not diverge: the backlog at the end is not a runaway multiple
    #    of the mean, which would indicate an unstable (overloaded) world.
    mean_q = e.kpis.mean_queue_length
    final_q = len(e.pending_tasks)
    check.record("queue stability (no runaway backlog)",
                 final_q <= max(25, 6 * mean_q),
                 f"final {final_q} vs mean {mean_q:.1f}")

    # 6. Non-negativity of counts that can never sensibly go negative.
    check.record("no negative station queues",
                 all(v >= 0 for v in e.station_queues.values()), "")

    # 7. The event log is complete enough for process mining.
    df_rows = len(e.log.rows)
    has_cases = len({r.case_id for r in e.log.rows})
    check.record("event log populated (PM4Py-ready)", df_rows > 0 and has_cases > 0,
                 f"{df_rows} rows, {has_cases} cases")

    return e


def run_feedback_invariants(engine, check: Check) -> None:
    """The feedback layer must not hand the agent information it could not have."""
    fb = engine.feedback
    now = float(engine.now)

    # 1. Nothing the agent can see may have been delivered in the future.
    visible = fb.observed(now)
    check.record("no undelivered surveys are observable",
                 all(r.delivered_at <= now for r in visible),
                 f"{len(visible)} responses visible at t={now:.0f}")

    # 2. Feedback must be sparse: a response for every interaction would be an
    #    unrealistically rich signal.
    rate = len(fb.responses) / max(1, fb.interactions)
    check.record("survey sampling is sparse", rate < 0.35,
                 f"response rate {rate:.1%} of {fb.interactions} interactions")

    # 3. Responses must be delayed, not instantaneous.
    if fb.responses:
        delays = [r.delivered_at - r.interaction_at for r in fb.responses]
        check.record("survey responses are delayed", min(delays) > 0,
                     f"median delay {np.median(delays):.0f} min")

    # 4. Non-response bias: responders should be less satisfied than the
    #    population, so observed CSAT is pessimistic rather than unbiased.
    true_pop = engine.true_state.true_experience_mean()
    if fb.responses and true_pop > 0:
        responder_mean = float(np.mean([r.latent_experience for r in fb.responses]))
        check.record("responders skew dissatisfied (non-response bias)",
                     responder_mean < true_pop,
                     f"responders {responder_mean:.3f} vs population {true_pop:.3f}")

    # 5. Performance records must aggregate periods, not mirror single tasks.
    perf = engine.performance
    check.record("performance is periodic, not per-task",
                 len(perf.records) < len(perf.outcomes) if perf.outcomes else True,
                 f"{len(perf.records)} records from {len(perf.outcomes)} task outcomes")

    # 6. Cost decomposition must sum to the reported total.
    s = engine.costs.summary()
    components = (s["cost_labour"] + s["cost_reserve_activation"] + s["cost_sla_penalty"]
                  + s["cost_disruption_response"] + s["cost_idle_capacity"] + s["cost_rework"])
    check.record("cost components sum to total", abs(components - s["cost_total"]) < 1e-6,
                 f"total {s['cost_total']:.1f}")

    # 7. Every scheduled duty window must be well formed.
    entries = engine.roster.entries
    check.record("schedule windows are well formed",
                 all(e.planned_end > e.planned_start for e in entries),
                 f"{len(entries)} schedule entries")


def run_booking_invariants(scenario: str, seed: int, check: Check) -> None:
    """The reservation model must respect capacity by construction."""
    domain = load_domain()
    cfg = load_scenario(scenario)
    rng = RNGRegistry(seed)
    model = DemandModel(domain, cfg, rng)
    day = cfg.start_date

    bad_capacity = bad_negative = bad_conservation = 0
    wl_seen = 0
    for train in domain.trains:
        for b in model.bookings_for(train, day):
            if b.class_id == "UR":
                continue
            if b.confirmed > b.capacity:
                bad_capacity += 1
            if b.rac > b.rac_capacity:
                bad_capacity += 1
            if min(b.confirmed, b.rac, b.waitlist) < 0:
                bad_negative += 1
            # Allocation + cancellations must account for all gross demand.
            if b.confirmed + b.rac + b.waitlist + b.cancellations != b.gross_demand:
                bad_conservation += 1
            wl_seen += int(b.waitlist > 0)

    check.record("confirmed never exceeds capacity", bad_capacity == 0, f"{bad_capacity} violations")
    check.record("no negative CNF/RAC/WL", bad_negative == 0, f"{bad_negative} violations")
    check.record("booking conservation (gross = cnf+rac+wl+cancelled)",
                 bad_conservation == 0, f"{bad_conservation} violations")
    check.record("waitlist emerges under capacity pressure", wl_seen > 0,
                 f"{wl_seen} train-classes with WL")


def run_plausibility(scenario: str, check: Check) -> None:
    """Distributional properties we expect of a realistic demand process."""
    domain = load_domain()
    cfg = load_scenario(scenario)

    # Weekly seasonality: weekend demand should differ from midweek.
    from datetime import timedelta
    daily = []
    for offset in range(28):
        rng = RNGRegistry(1000 + offset)
        model = DemandModel(domain, cfg, rng)
        day = cfg.start_date + timedelta(days=offset)
        total = sum(b.gross_demand for t in domain.trains for b in model.bookings_for(t, day))
        daily.append((day.weekday(), total))

    weekday_means = {}
    for wd in range(7):
        vals = [v for d, v in daily if d == wd]
        if vals:
            weekday_means[wd] = float(np.mean(vals))
    spread = (max(weekday_means.values()) - min(weekday_means.values())) / np.mean(list(weekday_means.values()))
    check.record("weekly seasonality present", spread > 0.05, f"peak-trough spread {spread:.1%}")

    # Overdispersion: demand variance should exceed the mean (Poisson would not).
    rng = RNGRegistry(7)
    model = DemandModel(domain, cfg, rng)
    train = domain.trains[0]
    draws = [model.bookings_for(train, cfg.start_date)[0].gross_demand for _ in range(300)]
    mean, var = float(np.mean(draws)), float(np.var(draws))
    check.record("demand is overdispersed (var > mean)", var > mean,
                 f"mean {mean:.1f}, var {var:.1f}, ratio {var / max(mean, 1e-9):.2f}")

    # Task durations should be positively skewed (lognormal), not symmetric.
    from simulation.distributions.library import lognormal_duration
    r = np.random.default_rng(3)
    durations = [lognormal_duration(r, 30, 0.5) for _ in range(4000)]
    from scipy.stats import skew
    sk = float(skew(durations))
    check.record("durations positively skewed", sk > 0.5, f"skew {sk:.2f}")

    # Hawkes stability: the configured branching ratio must be sub-critical.
    br = float(cfg.disruptions["hawkes_branching_ratio"])
    check.record("disruption process sub-critical (branching < 1)", br < 1.0,
                 f"branching ratio {br}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="normal_weekday")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"\nValidating scenario '{args.scenario}' (seed {args.seed})")
    print("\nINVARIANTS (a failure here is a simulator bug)")
    check = Check()
    engine = run_invariants(args.scenario, args.seed, check)
    run_booking_invariants(args.scenario, args.seed, check)
    run_feedback_invariants(engine, check)
    ok1 = check.report()

    print("\nPLAUSIBILITY (a failure here means miscalibration, not a bug)")
    check2 = Check()
    run_plausibility(args.scenario, check2)
    ok2 = check2.report()

    print("\nOBSERVED BEHAVIOUR (for eyeballing, not asserted)")
    k = engine.kpis.summary()
    for key in ("tasks_created", "completion_rate", "sla_breach_rate", "mean_queue_time_min",
                "mean_utilisation", "disruptions", "cascade_disruptions",
                "passengers_served", "waitlisted_passengers"):
        print(f"  {key:24s}: {k[key]}")

    out = engine.outcome_summary()
    print("\n  -- synthetic feedback layer (observed vs truth) --")
    obs, true = out["feedback_observed"], out["feedback_true"]
    print(f"  {'observed CSAT (agent sees)':30s}: {obs['csat_mean']:.3f} "
          f"from {obs['n_responses']} delivered responses")
    print(f"  {'observed NPS (agent sees)':30s}: {obs['nps_score']:.1f}")
    print(f"  {'all-survey CSAT (truth)':30s}: {true['csat_mean']:.3f} from {true['n']} responses")
    print(f"  {'latent experience (hidden)':30s}: {out['latent']['latent_experience_mean']:.3f} "
          f"over {out['latent']['latent_experience_samples']} interactions")
    print(f"  {'survey response rate':30s}: {true.get('response_rate', 0):.1%}")
    print(f"  {'mean performance index':30s}: {out['performance'].get('mean_performance', 0):.3f}")
    print(f"  {'total operational cost':30s}: {out['costs']['cost_total']:.1f}")
    print("\n  NOTE: these are properties of a synthetic world calibrated to be")
    print("  plausible. They are not claims about real Southern Railway operations.")

    raise SystemExit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
