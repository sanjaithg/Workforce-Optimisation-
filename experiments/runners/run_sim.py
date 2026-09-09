"""Run one episode of Environment A under a chosen policy.

    python -m experiments.runners.run_sim --scenario normal_weekday --policy greedy \
        --seed 0 --trace out/run.json
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

from rl.environment.env_a import WorkforceEnvA
from rl.policies.baselines import build_policy
from simulation.core.config import available_scenarios


def run_episode(scenario: str, policy_name: str, seed: int, trace_path: str | None = None,
                verbose: bool = True, max_steps: int = 20000):
    env = WorkforceEnvA(scenario=scenario, seed=seed,
                        collect_trace=trace_path is not None, max_steps=max_steps)
    policy = build_policy(policy_name, seed=seed)
    obs, _ = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    while True:
        mask = env.action_masks()
        action = policy.act(obs, mask, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    kpis = env.engine.kpis.summary()
    result = {
        "scenario": scenario, "policy": policy_name, "seed": seed,
        "steps": steps, "total_reward": round(total_reward, 2), **kpis,
    }

    if verbose:
        _print_timeline(env)
        print(f"\n=== {scenario} / {policy_name} / seed {seed} ===")
        print(f"decision epochs : {steps}")
        print(f"total reward    : {total_reward:,.1f}")
        for k, v in kpis.items():
            print(f"{k:24s}: {v}")

    if trace_path:
        write_trace(env, trace_path, result)
        if verbose:
            print(f"\ntrace written to {trace_path}  (open viz/index.html and load it)")
    return result


def _print_timeline(env, limit: int = 28) -> None:
    """Show the interleaved event timeline: this is where cascades become visible."""
    e = env.engine
    events = e.log.events
    print("\n--- event timeline (sample) ---")
    interesting = [ev for ev in events if ev["kind"] in
                   ("asset_failure", "train_delay", "demand_spike", "crowd_surge",
                    "wl_pressure", "staffing", "rework", "train_departure")]
    step = max(1, len(interesting) // limit)
    for ev in interesting[::step][:limit]:
        stamp = (e.episode_start + timedelta(minutes=ev["time"])).strftime("%d %b %H:%M")
        cascade = "  <-- cascade" if ev.get("caused_by") else ""
        print(f"{stamp}  {ev['kind']:16s} {ev['message']}{cascade}")


def write_trace(env, path: str, result: dict) -> None:
    e = env.engine
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "scenario": result["scenario"], "policy": result["policy"], "seed": result["seed"],
            "domain": e.domain.name, "start": e.episode_start.isoformat(),
            "horizon_minutes": e.horizon,
            "locations": {lid: {"name": loc.name, "tier": loc.tier}
                          for lid, loc in e.domain.locations.items()},
            "departments": {d: dept.name for d, dept in e.domain.departments.items()},
        },
        "kpis": result,
        "snapshots": e.trace_snapshots,
        "events": e.log.events,
        "decisions": e.trace_decisions,
    }
    Path(path).write_text(json.dumps(payload))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one Environment A episode.")
    ap.add_argument("--scenario", default="normal_weekday", choices=available_scenarios())
    ap.add_argument("--policy", default="greedy", choices=["random", "greedy", "wfm_heuristic"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace", default=None, help="write a visualiser trace JSON here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run_episode(args.scenario, args.policy, args.seed, args.trace, verbose=not args.quiet)


if __name__ == "__main__":
    main()
