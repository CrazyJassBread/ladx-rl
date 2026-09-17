"""Evaluate a trained PPO policy and report task-level transfer metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

from training.experiments import DEFAULT_MANIFEST, load_suite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--experiment",
        "-e",
        default="room16_key/fixed",
        help="Experiment in <task>/<variant> form (default: room16_key/fixed)",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output", help="Optional JSON result path")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--record-episodes",
        type=int,
        default=1,
        help="Number of episodes to save as GIF per evaluation instance (default: 1)",
    )
    parser.add_argument(
        "--gif-dir",
        help="GIF output directory (default: next to --output or the model)",
    )
    parser.add_argument(
        "--gif-fps",
        type=float,
        help="GIF playback speed (default: Game Boy speed divided by frame_skip)",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.record_episodes < 0:
        parser.error("--record-episodes must be non-negative")
    if args.gif_fps is not None and args.gif_fps <= 0:
        parser.error("--gif-fps must be positive")

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("Install training dependencies: pip install -e '.[train]'") from exc
    from training.sb3 import make_vec_env

    suite = load_suite(args.manifest)
    experiment = suite.experiment(args.experiment)
    gif_dir = _gif_output_dir(args.model, args.output, args.gif_dir)
    results = []
    for index, instance_name in enumerate(experiment.eval_instances):
        env = make_vec_env(suite, [instance_name])
        env.seed(args.seed + index)
        model = PPO.load(args.model, env=env, device=args.device)
        frame_skip = suite.instances[instance_name].frame_skip
        gif_fps = args.gif_fps or 60.0 / frame_skip
        try:
            results.append(
                _evaluate_instance(
                    model,
                    env,
                    instance_name,
                    args.episodes,
                    gif_dir=gif_dir,
                    gif_prefix=f"{experiment.task_name}-{experiment.variant}-{instance_name}",
                    record_episodes=args.record_episodes,
                    frame_duration_ms=_gif_frame_duration_ms(gif_fps),
                )
            )
        finally:
            env.close()

    report = {
        "experiment": experiment.name,
        "task": experiment.task_name,
        "variant": experiment.variant,
        "model": args.model,
        "episodes_per_instance": args.episodes,
        "success_rate": mean(result["success_rate"] for result in results),
        "instances": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    return 0


def _evaluate_instance(
    model,
    env,
    name: str,
    episodes: int,
    *,
    gif_dir: Path | None = None,
    gif_prefix: str | None = None,
    record_episodes: int = 0,
    frame_duration_ms: int = 67,
) -> dict:
    observation = env.reset()
    episode_return = 0.0
    episode_steps = 0
    returns: list[float] = []
    steps: list[int] = []
    successes: list[bool] = []
    damage_taken: list[int] = []
    targets_defeated: list[int] = []
    target_damage_dealt: list[int] = []
    combat_steps: list[int] = []
    keys_collected: list[bool] = []
    items_collected: list[bool] = []
    chests_seen: list[bool] = []
    rupees_collected: list[bool] = []
    dialogs_completed: list[bool] = []
    best_pattern_matches: list[int] = []
    pattern_attempts: list[int] = []
    pattern_mismatches: list[int] = []
    owl_hints_seen: list[bool] = []
    terminal_phases: dict[str, int] = {}
    failures: dict[str, int] = {}
    current_damage = 0
    gif_paths: list[str] = []
    episode_frames = (
        [_current_rgb_frame(observation)] if record_episodes > 0 else None
    )

    while len(returns) < episodes:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, infos = env.step(action)
        if episode_frames is not None:
            frame_observation = (
                infos[0].get("terminal_observation", observation)
                if done[0]
                else observation
            )
            episode_frames.append(_current_rgb_frame(frame_observation))
        episode_return += float(reward[0])
        episode_steps += 1
        current_damage += sum(
            event["data"]["amount"]
            for event in infos[0].get("events", [])
            if event["type"] == "player_damaged"
        )
        if not done[0]:
            continue
        info = infos[0]
        task = info.get("task", {})
        success = bool(info.get("is_success", False))
        failure = task.get("failure")
        if failure is None and info.get("TimeLimit.truncated", False):
            failure = "timeout"
        returns.append(episode_return)
        steps.append(episode_steps)
        successes.append(success)
        damage_taken.append(current_damage)
        targets_defeated.append(
            len(task.get("target_slots", [])) - task.get("targets_remaining", 0)
        )
        target_damage_dealt.append(int(task.get("target_damage_dealt", 0)))
        combat_steps.append(int(task.get("combat_steps", 0)))
        keys_collected.append(bool(task.get("key_collected", False)))
        items_collected.append(
            bool(task.get("item_received", task.get("item_collected", False)))
        )
        chests_seen.append(bool(task.get("chest_seen", False)))
        rupees_collected.append(bool(task.get("rupees_collected", False)))
        dialogs_completed.append(bool(task.get("dialog_completed", False)))
        best_pattern_matches.append(int(task.get("best_pattern_match", 0)))
        pattern_attempts.append(int(task.get("pattern_attempts", 0)))
        pattern_mismatches.append(int(task.get("pattern_mismatches", 0)))
        owl_hints_seen.append(bool(task.get("owl_hint_seen", False)))
        phase = str(task.get("phase", "unknown"))
        terminal_phases[phase] = terminal_phases.get(phase, 0) + 1
        if failure:
            failures[failure] = failures.get(failure, 0) + 1

        if episode_frames is not None:
            if gif_dir is None:
                raise ValueError("gif_dir is required when recording episodes")
            prefix = gif_prefix or name
            gif_path = gif_dir / f"{prefix}-episode-{len(returns):03d}.gif"
            _save_gif(episode_frames, gif_path, frame_duration_ms)
            gif_paths.append(_display_path(gif_path))

        episode_return = 0.0
        episode_steps = 0
        current_damage = 0
        episode_frames = (
            [_current_rgb_frame(observation)]
            if len(returns) < min(record_episodes, episodes)
            else None
        )

    successful_steps = [
        value
        for value, success in zip(steps, successes, strict=True)
        if success
    ]
    return {
        "instance": name,
        "success_rate": sum(successes) / episodes,
        "mean_return": mean(returns),
        "mean_episode_steps": mean(steps),
        "mean_success_steps": mean(successful_steps) if successful_steps else None,
        "mean_damage_taken": mean(damage_taken),
        "mean_targets_defeated": mean(targets_defeated),
        "mean_target_damage_dealt": mean(target_damage_dealt),
        "mean_combat_steps": mean(combat_steps),
        "key_collection_rate": sum(keys_collected) / episodes,
        "item_collection_rate": sum(items_collected) / episodes,
        "chest_seen_rate": sum(chests_seen) / episodes,
        "rupee_collection_rate": sum(rupees_collected) / episodes,
        "dialog_completion_rate": sum(dialogs_completed) / episodes,
        "mean_best_pattern_match": mean(best_pattern_matches),
        "mean_pattern_attempts": mean(pattern_attempts),
        "mean_pattern_mismatches": mean(pattern_mismatches),
        "owl_hint_seen_rate": sum(owl_hints_seen) / episodes,
        "terminal_phases": terminal_phases,
        "failures": failures,
        "gifs": gif_paths,
    }


def _current_rgb_frame(observation) -> np.ndarray:
    """Extract the newest RGB frame from a vectorized, channel-first stack."""

    array = np.asarray(observation)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"Expected one evaluation environment, got shape {array.shape}")
        array = array[0]
    if array.ndim != 3 or array.shape[0] < 3:
        raise ValueError(f"Expected channel-first image observation, got shape {array.shape}")
    return np.moveaxis(array[-3:], 0, -1).astype(np.uint8, copy=True)


def _save_gif(frames: list[np.ndarray], path: Path, duration_ms: int) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install training dependencies: pip install -e '.[train]'") from exc

    if not frames:
        raise ValueError("Cannot save an empty GIF")
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame, mode="RGB") for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def _gif_output_dir(model: str, output: str | None, requested: str | None) -> Path:
    if requested:
        return Path(requested)
    if output:
        return Path(output).parent / "gifs"
    return Path(model).parent / "evaluation_gifs"


def _gif_frame_duration_ms(fps: float) -> int:
    """Convert FPS to GIF's centisecond frame-duration resolution."""

    return max(10, round(1000.0 / fps / 10.0) * 10)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
