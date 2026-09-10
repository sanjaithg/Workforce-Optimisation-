"""Train a simple MaskablePPO policy against Environment A.

This is deliberately minimal: it proves the env/masking/SB3 wiring works end to end
and produces a checkpoint that drops into run_sim.py / run_baselines.py via
`--policy learned`. It does not tune reward weights, pick a training budget that
guarantees a good policy, or claim the result beats the heuristics in
rl/policies/baselines.py -- that comparison, and any tuning it motivates, is the
next step, not this script's job.

Requires the optional RL-training dependencies (uncomment in requirements.txt):
    pip install stable-baselines3 sb3-contrib
Add `--tensorboard` to log training curves (requires `pip install tensorboard`).

Train on one scenario:
    python -m experiments.runners.train_rl --scenario normal_weekday \
        --timesteps 200000 --out out/models/ppo_normal_weekday

Train across a mix of scenarios (recommended -- see rl/environment/randomized.py
for why a single fixed scenario/seed would otherwise just memorise one world):
    python -m experiments.runners.train_rl \
        --scenarios normal_weekday,high_demand,high_absence,combined_stress \
        --timesteps 400000 --out out/models/ppo_mixed

Then evaluate against the heuristics:
    python -m experiments.runners.run_sim --scenario normal_weekday \
        --policy learned --model-path out/models/ppo_mixed --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rl.environment.env_a import WorkforceEnvA
from rl.environment.randomized import RandomizedResetEnv
from simulation.core.config import available_scenarios


def build_training_env(scenario: str, scenarios: list[str] | None, seed: int):
    base = WorkforceEnvA(scenario=scenario, seed=seed)
    return RandomizedResetEnv(base, seed=seed, scenarios=scenarios)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train MaskablePPO on Environment A.")
    ap.add_argument("--scenario", default="normal_weekday", choices=available_scenarios(),
                    help="scenario to train on (ignored if --scenarios is given)")
    ap.add_argument("--scenarios", default=None,
                    help="comma-separated scenario list to train across, e.g. "
                         "normal_weekday,high_demand,combined_stress")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the training reseeding stream and SB3 -- NOT a fixed "
                         "world, every episode still gets a fresh seed (see "
                         "rl/environment/randomized.py)")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--out", default="out/models/ppo", help="checkpoint path (no extension)")
    ap.add_argument("--tensorboard", action="store_true",
                    help="log to out/tb/ (requires `pip install tensorboard`)")
    args = ap.parse_args()

    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
    except ImportError as exc:
        raise SystemExit(
            "train_rl.py needs the optional RL-training dependencies. Install with:\n"
            "  pip install stable-baselines3 sb3-contrib\n"
            "(see the commented block in requirements.txt)"
        ) from exc

    scenarios = args.scenarios.split(",") if args.scenarios else None
    env = build_training_env(args.scenario, scenarios, args.seed)
    env = ActionMasker(env, lambda e: e.action_masks())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = MaskablePPO("MlpPolicy", env, verbose=1, seed=args.seed,
                        tensorboard_log="out/tb/" if args.tensorboard else None)
    model.learn(total_timesteps=args.timesteps)
    model.save(str(out_path))

    print(f"\nsaved checkpoint to {out_path}.zip")
    print("evaluate with:")
    print(f"  python -m experiments.runners.run_sim --scenario {args.scenario} "
          f"--policy learned --model-path {out_path} --seed 0")


if __name__ == "__main__":
    main()
