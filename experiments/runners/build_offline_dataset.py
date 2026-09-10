"""Build a Minari offline-RL dataset by rolling out an expert (behaviour) policy.

Requires the optional dataset-building dependencies (uncomment in requirements.txt):
    pip install "minari[create,hdf5]"

    python -m experiments.runners.build_offline_dataset \
        --episodes-per-scenario 20 --policy wfm_heuristic \
        --dataset-id workforce/wfm_heuristic-v0

Produces one combined Minari dataset spanning every scenario given (all 8 by
default). They share one domain (southern_railway), so observation/action space
shapes are identical across scenarios and can live in a single dataset. Each
episode uses an explicit seed (0, 1, 2, ... per scenario) so the whole dataset is
regenerable from this command line alone -- unlike training (see
rl/environment/randomized.py), a dataset needs *reproducible*, not merely varied,
episodes.

Every step's `info` is normalised to one fixed flat schema (see
WorkforceStepCallback below) that carries the scenario id, which kind of decision
it was, and the same 7-term reward decomposition env_a.py's live decision-rating
trace uses (rl/environment/env_a.py:_reward()) -- so an offline algorithm (or you,
doing EDA on the dataset) can condition on or inspect *why* a step scored the way
it did, not just the scalar reward.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rl.policies.baselines import build_policy
from simulation.core.config import available_scenarios, load_scenario

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_PATH = REPO_ROOT / "out" / "minari_datasets"
REWARD_KEYS = ("completed", "breached", "labour", "overtime", "backlog", "wait", "balance_penalty")


def _step_callback_class():
    """Built lazily so this module can be imported without minari installed."""
    from minari.data_collector.callbacks import StepDataCallback

    class WorkforceStepCallback(StepDataCallback):
        """Normalises env_a.py's per-branch info dict into one fixed flat schema.

        Dispatch steps merge {deferred, task_id, worker_id}, staffing steps merge
        {department, action, changed}, and a never-actually-reached terminal branch
        merges {terminal}. Minari's episode buffer diffs consecutive steps' info
        dicts with jax.tree_util.tree_map, which requires identical tree structure
        (same keys) on every call -- so the raw, per-branch info dict from env_a.py
        would break dataset creation the first time a dispatch step followed a
        staffing step. This callback is the fix: same keys, same types, every step.
        """

        def __call__(self, env, obs, info, action=None, rew=None, terminated=None,
                     truncated=None):
            step_data = super().__call__(env, obs=obs, info=info, action=action, rew=rew,
                                         terminated=terminated, truncated=truncated)
            rc = info.get("reward_components") or {}
            step_data["info"] = {
                "scenario": str(info.get("scenario", "")),
                "decision_kind": ("staffing" if "department" in info
                                  else "terminal" if "terminal" in info else "dispatch"),
                "deferred": bool(info.get("deferred", False)),
                "worker_id": str(info.get("worker_id") or ""),
                "department": str(info.get("department") or ""),
                "staffing_action": str(info.get("action") or ""),
                "changed": int(info.get("changed", 0)),
                **{f"reward_{k}": float(rc.get(k, 0.0)) for k in REWARD_KEYS},
            }
            return step_data

    return WorkforceStepCallback


def build_dataset(scenarios: list[str], episodes_per_scenario: int, policy_name: str,
                  dataset_id: str, datasets_path: Path, author: str | None,
                  author_email: str | None):
    os.environ.setdefault("MINARI_DATASETS_PATH", str(datasets_path))
    from minari import DataCollector

    from rl.environment.env_a import WorkforceEnvA

    base_env = WorkforceEnvA(scenario=scenarios[0])
    collector = DataCollector(base_env, step_data_callback=_step_callback_class(),
                              record_infos=True)
    policy = build_policy(policy_name)

    for scenario in scenarios:
        base_env.scenario_cfg = load_scenario(scenario)
        for ep in range(episodes_per_scenario):
            seed = ep
            obs, _ = collector.reset(seed=seed)
            policy.reset()
            done = False
            steps = 0
            while not done:
                # gymnasium's Wrapper no longer forwards arbitrary attribute access
                # (no __getattr__), so custom env methods/state are read off
                # base_env directly rather than through the DataCollector wrapper;
                # collector.step()/.reset() still record every transition normally.
                mask = base_env.action_masks()
                action = policy.act(obs, mask, base_env)
                obs, reward, terminated, truncated, info = collector.step(action)
                done = terminated or truncated
                steps += 1
            print(f"  {scenario:22s} episode {ep + 1}/{episodes_per_scenario}  "
                  f"seed={seed}  steps={steps}")

    dataset = collector.create_dataset(
        dataset_id=dataset_id,
        algorithm_name=policy_name,
        author=author,
        author_email=author_email,
        description=(f"Expert ({policy_name}) rollouts across {len(scenarios)} "
                     f"southern_railway scenarios: {', '.join(scenarios)}. "
                     "infos carry scenario id, decision_kind and the 7-term reward "
                     "decomposition from rl/environment/env_a.py:_reward()."),
    )
    return dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a Minari offline-RL dataset.")
    ap.add_argument("--scenarios", default=None,
                    help="comma-separated scenario list (default: all 8)")
    ap.add_argument("--episodes-per-scenario", type=int, default=20)
    ap.add_argument("--policy", default="wfm_heuristic",
                    choices=["random", "greedy", "wfm_heuristic"])
    ap.add_argument("--dataset-id", default="workforce/wfm_heuristic-v0")
    ap.add_argument("--datasets-path", default=str(DEFAULT_DATASETS_PATH),
                    help="where Minari stores the dataset (sets MINARI_DATASETS_PATH)")
    ap.add_argument("--author", default=None)
    ap.add_argument("--author-email", default=None)
    args = ap.parse_args()

    try:
        import minari  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            'build_offline_dataset.py needs the optional dataset-building dependencies. '
            'Install with:\n  pip install "minari[create,hdf5]"\n'
            "(see the commented block in requirements.txt)"
        ) from exc

    scenarios = args.scenarios.split(",") if args.scenarios else available_scenarios()
    print(f"building '{args.dataset_id}' from {len(scenarios)} scenario(s) x "
          f"{args.episodes_per_scenario} episodes, behaviour policy '{args.policy}'")

    dataset = build_dataset(scenarios, args.episodes_per_scenario, args.policy,
                            args.dataset_id, Path(args.datasets_path), args.author,
                            args.author_email)

    print(f"\ndataset '{dataset.id}': {dataset.total_episodes} episodes, "
          f"{dataset.total_steps} steps")
    print(f"stored under {args.datasets_path}")
    print(f"load it back with: minari.load_dataset('{dataset.id}')")


if __name__ == "__main__":
    main()
