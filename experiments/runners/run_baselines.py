"""Compare baseline policies across scenarios and seeds, with confidence intervals.

    python -m experiments.runners.run_baselines --scenarios normal_weekday,high_demand --seeds 5

The point of this script is the honesty check: if a greedy rule already does as
well as anything more sophisticated, we need to know that *before* investing in
learned policies.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.runners.run_sim import run_episode
from simulation.core.config import available_scenarios

PRIMARY_METRICS = ("sla_breach_rate", "mean_queue_time_min", "completion_rate",
                   "labour_cost", "mean_utilisation", "total_reward")


def mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of a normal-approximation confidence interval."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float(arr.mean()) if arr.size else 0.0, 0.0
    z = 1.96 if confidence == 0.95 else 2.576
    return float(arr.mean()), float(z * arr.std(ddof=1) / math.sqrt(arr.size))


def run_grid(scenarios: list[str], policies: list[str], seeds: int) -> dict:
    results: dict = {}
    for scenario in scenarios:
        results[scenario] = {}
        for policy in policies:
            runs = []
            for seed in range(seeds):
                r = run_episode(scenario, policy, seed, trace_path=None, verbose=False)
                runs.append(r)
            summary = {}
            for metric in PRIMARY_METRICS:
                vals = [float(run[metric]) for run in runs if metric in run]
                if vals:
                    m, ci = mean_ci(vals)
                    summary[metric] = {"mean": round(m, 4), "ci95": round(ci, 4)}
            results[scenario][policy] = summary
            print(f"  {scenario:22s} {policy:14s} "
                  f"breach={summary['sla_breach_rate']['mean']:.3f}"
                  f"±{summary['sla_breach_rate']['ci95']:.3f}  "
                  f"queue={summary['mean_queue_time_min']['mean']:6.1f}  "
                  f"util={summary['mean_utilisation']['mean']:.3f}  "
                  f"reward={summary['total_reward']['mean']:9.1f}"
                  f"±{summary['total_reward']['ci95']:.1f}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="normal_weekday",
                    help="comma-separated, or 'all'")
    ap.add_argument("--policies", default="random,greedy,wfm_heuristic")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="out/baselines.json")
    args = ap.parse_args()

    scenarios = available_scenarios() if args.scenarios == "all" else args.scenarios.split(",")
    policies = args.policies.split(",")

    print(f"Running {len(scenarios)} scenario(s) x {len(policies)} policies x {args.seeds} seeds\n")
    results = run_grid(scenarios, policies, args.seeds)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
