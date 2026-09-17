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
    parser.add_argument(
        "--experiment",
        "-e",
        help="Experiment in <task>/<variant> form (default: room16_key/fixed)",
    )
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
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cpu, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        help="Parallel training workers (default: one per training instance)",
    )
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

    from training.sb3 import rollout_layout

    try:
        num_envs, n_steps = rollout_layout(
            int(suite.ppo["n_steps"]),
            len(experiment.train_instances),
            args.num_envs,
        )
    except ValueError as exc:
        parser.error(str(exc))

    _train(suite, experiment.name, args, num_envs=num_envs, n_steps=n_steps)
    return 0


def _train(
    suite: ExperimentSuite,
    experiment_name: str,
    args,
    *,
    num_envs: int,
    n_steps: int,
) -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
    except ImportError as exc:
        raise RuntimeError("Install training dependencies: pip install -e '.[train]'") from exc
    from training.sb3 import device_info, make_vec_env, resolve_device

    experiment = suite.experiment(experiment_name)
    device = resolve_device(args.device)
    selected_device_info = device_info(device)
    device_description = selected_device_info["device"]
    if "gpu_name" in selected_device_info:
        device_description += f" ({selected_device_info['gpu_name']})"
    print(f"Training device: {device_description}")

    output = Path(args.output or f"artifacts/tail_cave/{experiment.name}")
    output.mkdir(parents=True, exist_ok=True)
    train_env = make_vec_env(suite, experiment.train_instances, num_envs=num_envs)
    eval_env = make_vec_env(suite, experiment.eval_instances)
    train_env.seed(args.seed)
    eval_env.seed(args.seed + 10_000)

    ppo_values = {
        key: value
        for key, value in suite.ppo.items()
        if key not in {"frame_stack", "eval_freq", "checkpoint_freq", "eval_episodes"}
    }
    ppo_values["n_steps"] = n_steps
    rollout_size = n_steps * num_envs
    print(
        f"Sampling layout: {num_envs} env(s) x {n_steps} steps "
        f"= {rollout_size} transitions/update"
    )
    tensorboard_log = str(output / "tensorboard")
    if args.init_model:
        model = PPO.load(
            args.init_model,
            env=train_env,
            device=device,
            tensorboard_log=tensorboard_log,
            n_steps=n_steps,
        )
        model.set_random_seed(args.seed)
    else:
        policy = ppo_values.pop("policy", "CnnPolicy")
        model = PPO(
            policy,
            train_env,
            seed=args.seed,
            device=device,
            verbose=1,
            tensorboard_log=tensorboard_log,
            **ppo_values,
        )

    checkpoint = CheckpointCallback(
        save_freq=max(int(suite.ppo["checkpoint_freq"]) // num_envs, 1),
        save_path=str(output / "checkpoints"),
        name_prefix="ppo",
    )
    evaluate = EvalCallback(
        eval_env,
        n_eval_episodes=int(suite.ppo["eval_episodes"]),
        eval_freq=max(int(suite.ppo["eval_freq"]) // num_envs, 1),
        best_model_save_path=str(output),
        log_path=str(output / "evaluation"),
        deterministic=True,
    )
    _write_run_metadata(
        output,
        suite,
        experiment.name,
        args,
        selected_device_info,
        num_envs=num_envs,
        n_steps=n_steps,
    )
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
            detail = (
                f"targets={task['target_slots']}"
                if "target_slots" in task
                else f"phase={task['phase']}"
            )
            print(
                f"OK {name}: room=({room['is_indoor']},{room['map_id']},0x{room['id']:02X}) "
                f"{detail} noops={info['reset_noop_frames']}"
            )
        finally:
            env.close()


def _list_experiments(suite: ExperimentSuite) -> None:
    for experiment in suite.experiments.values():
        note = (
            f"; recommended initialization: {experiment.pretrained_from}"
            if experiment.pretrained_from
            else ""
        )
        description = experiment.description.rstrip(".")
        print(f"{experiment.name}: {description}{note}.")


def _write_run_metadata(
    output: Path,
    suite: ExperimentSuite,
    experiment_name: str,
    args,
    selected_device_info: dict[str, object],
    *,
    num_envs: int,
    n_steps: int,
) -> None:
    experiment = suite.experiment(experiment_name)
    state_paths = {
        path
        for name in experiment.train_instances + experiment.eval_instances
        for path in suite.instances[name].state_paths
    }
    task_config_paths = {
        suite.instances[name].task_config_path
        for name in experiment.train_instances + experiment.eval_instances
    }
    metadata = {
        "experiment": experiment.name,
        "task": experiment.task_name,
        "variant": experiment.variant,
        "description": experiment.description,
        "seed": args.seed,
        "timesteps": args.steps or experiment.total_timesteps,
        "init_model": args.init_model,
        "requested_device": args.device,
        **selected_device_info,
        "num_envs": num_envs,
        "n_steps_per_env": n_steps,
        "rollout_size": num_envs * n_steps,
        "train_instances": experiment.train_instances,
        "eval_instances": experiment.eval_instances,
        "state_sha256": {
            _display_path(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(state_paths)
        },
        "task_config_sha256": {
            _display_path(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(task_config_paths)
        },
    }
    (output / "run.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(suite.path, output / "manifest.toml")
    task_snapshot_dir = output / "task_configs"
    task_snapshot_dir.mkdir(exist_ok=True)
    for path in task_config_paths:
        shutil.copy2(path, task_snapshot_dir / path.name)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
