"""Train PPO on one experiment from the Tail Cave transfer suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from training.env_factory import make_task_env
from training.experiments import DEFAULT_MANIFEST, ExperimentSuite, load_suite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--experiment", "-e", default="A")
    parser.add_argument("--steps", type=int, help="Override the experiment timestep budget")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        help="Run directory (default: artifacts/tail_cave/<experiment>)",
    )
    parser.add_argument(
        "--init-model",
        help="PPO checkpoint to fine-tune; omit for a scratch run",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List experiments without importing SB3",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Reset every referenced state, then exit",
    )
    args = parser.parse_args()

    suite = load_suite(args.manifest)
    if args.list:
        _list_experiments(suite)
        return 0

    experiment = suite.experiment(args.experiment)
    if args.check:
        _check_instances(suite, experiment.train_instances + experiment.eval_instances, args.seed)
        return 0
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")

    _train(suite, experiment.name, args)
    return 0


def _train(suite: ExperimentSuite, experiment_name: str, args) -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
    except ImportError as exc:
        raise RuntimeError("Install training dependencies: pip install -e '.[train]'") from exc
    from training.sb3 import make_vec_env

    experiment = suite.experiment(experiment_name)
    output = Path(args.output or f"artifacts/tail_cave/{experiment.name}")
    output.mkdir(parents=True, exist_ok=True)
    train_env = make_vec_env(suite, experiment.train_instances)
    eval_env = make_vec_env(suite, experiment.eval_instances)
    train_env.seed(args.seed)
    eval_env.seed(args.seed + 10_000)

    ppo_values = {
        key: value
        for key, value in suite.ppo.items()
        if key not in {"frame_stack", "eval_freq", "checkpoint_freq", "eval_episodes"}
    }
    tensorboard_log = str(output / "tensorboard")
    if args.init_model:
        model = PPO.load(
            args.init_model,
            env=train_env,
            device=args.device,
            tensorboard_log=tensorboard_log,
        )
        model.set_random_seed(args.seed)
    else:
        policy = ppo_values.pop("policy", "CnnPolicy")
        model = PPO(
            policy,
            train_env,
            seed=args.seed,
            device=args.device,
            verbose=1,
            tensorboard_log=tensorboard_log,
            **ppo_values,
        )

    n_envs = len(experiment.train_instances)
    checkpoint = CheckpointCallback(
        save_freq=max(int(suite.ppo["checkpoint_freq"]) // n_envs, 1),
        save_path=str(output / "checkpoints"),
        name_prefix="ppo",
    )
    evaluate = EvalCallback(
        eval_env,
        n_eval_episodes=int(suite.ppo["eval_episodes"]),
        eval_freq=max(int(suite.ppo["eval_freq"]) // n_envs, 1),
        best_model_save_path=str(output),
        log_path=str(output / "evaluation"),
        deterministic=True,
    )
    _write_run_metadata(output, suite, experiment.name, args)
    try:
        model.learn(
            total_timesteps=args.steps or experiment.total_timesteps,
            callback=CallbackList([checkpoint, evaluate]),
            progress_bar=False,
        )
        model.save(output / "final_model")
    finally:
        train_env.close()
        eval_env.close()


def _check_instances(suite: ExperimentSuite, names: tuple[str, ...], seed: int) -> None:
    for name in dict.fromkeys(names):
        env = make_task_env(suite.instances[name])
        try:
            _, info = env.reset(seed=seed)
            state = info["game_state"]
            room = state["room"]
            task = info["task"]
            print(
                f"OK {name}: room=({room['is_indoor']},{room['map_id']},0x{room['id']:02X}) "
                f"targets={task['target_slots']} noops={info['reset_noop_frames']}"
            )
        finally:
            env.close()


def _list_experiments(suite: ExperimentSuite) -> None:
    for experiment in suite.experiments.values():
        note = (
            f"; initialize from {experiment.pretrained_from}"
            if experiment.pretrained_from
            else ""
        )
        print(f"{experiment.name}: {experiment.description}{note}")


def _write_run_metadata(
    output: Path,
    suite: ExperimentSuite,
    experiment_name: str,
    args,
) -> None:
    experiment = suite.experiment(experiment_name)
    state_paths = {
        path
        for name in experiment.train_instances + experiment.eval_instances
        for path in suite.instances[name].state_paths
    }
    metadata = {
        "experiment": experiment.name,
        "description": experiment.description,
        "seed": args.seed,
        "timesteps": args.steps or experiment.total_timesteps,
        "init_model": args.init_model,
        "train_instances": experiment.train_instances,
        "eval_instances": experiment.eval_instances,
        "state_sha256": {
            _display_path(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(state_paths)
        },
    }
    (output / "run.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(suite.path, output / "manifest.toml")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
