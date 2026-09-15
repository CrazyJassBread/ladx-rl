"""Evaluate a trained PPO policy and report task-level transfer metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from training.experiments import DEFAULT_MANIFEST, load_suite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--experiment", "-e", default="A")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output", help="Optional JSON result path")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("Install training dependencies: pip install -e '.[train]'") from exc
    from training.sb3 import make_vec_env

    suite = load_suite(args.manifest)
    experiment = suite.experiment(args.experiment)
    results = []
    for index, instance_name in enumerate(experiment.eval_instances):
        env = make_vec_env(suite, [instance_name])
        env.seed(args.seed + index)
        model = PPO.load(args.model, env=env, device=args.device)
        try:
            results.append(_evaluate_instance(model, env, instance_name, args.episodes))
        finally:
            env.close()

    report = {
        "experiment": experiment.name,
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


def _evaluate_instance(model, env, name: str, episodes: int) -> dict:
    observation = env.reset()
    episode_return = 0.0
    episode_steps = 0
    returns: list[float] = []
    steps: list[int] = []
    successes: list[bool] = []
    damage_taken: list[int] = []
    targets_defeated: list[int] = []
    keys_collected: list[bool] = []
    failures: dict[str, int] = {}
    current_damage = 0

    while len(returns) < episodes:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, infos = env.step(action)
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
        keys_collected.append(bool(task.get("key_collected", False)))
        if failure:
            failures[failure] = failures.get(failure, 0) + 1
        episode_return = 0.0
        episode_steps = 0
        current_damage = 0

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
        "key_collection_rate": sum(keys_collected) / episodes,
        "failures": failures,
    }


if __name__ == "__main__":
    raise SystemExit(main())
