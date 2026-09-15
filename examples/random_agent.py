"""Visually test ZeldaEnv with a random policy in Tail Cave room 1."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from zelda_env import ZeldaEnv

from viewer import GameInfoViewer


DEFAULT_STATE = Path("save_states/azle.gbc.tail_cave.r1.state")
# START and SELECT are omitted by default so the policy does not remain in menus.
GAMEPLAY_ACTIONS = (0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--frame-skip", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--scale", type=int, default=4, help="integer game-frame scale")
    parser.add_argument("--speed", type=float, default=1.0, help="visual playback speed")
    parser.add_argument(
        "--include-menu-actions",
        action="store_true",
        help="also sample START and SELECT",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="disable the visual window (useful for automated smoke tests)",
    )
    args = parser.parse_args(argv)

    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    if args.scale < 1:
        parser.error("--scale must be positive")
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if not args.state.is_file():
        parser.error(f"save state not found: {args.state}")

    rng = random.Random(args.seed)
    actions = tuple(range(17)) if args.include_menu_actions else GAMEPLAY_ACTIONS
    env = ZeldaEnv(
        initial_state_path=args.state,
        frame_skip=args.frame_skip,
    )
    viewer = None if args.headless else GameInfoViewer(
        title="ZeldaEnv random policy — Tail Cave room 1",
        scale=args.scale,
        help_text="Random policy (close window or press Ctrl-C to stop)",
    )

    episodes = 1
    try:
        observation, info = env.reset(seed=args.seed)
        _print_state(0, episodes, info, reward=0.0)
        if viewer is not None and not viewer.update(observation, info, 0.0, 0, episodes):
            return 0

        started_at = time.monotonic()
        simulated_frames = 0

        for step_index in range(1, args.steps + 1):
            action = rng.choice(actions)
            observation, reward, terminated, truncated, info = env.step(action)
            simulated_frames += args.frame_skip
            if viewer is not None:
                if not viewer.update(observation, info, reward, step_index, episodes):
                    break
                target_time = started_at + simulated_frames / (60 * args.speed)
                time.sleep(max(0.0, target_time - time.monotonic()))
            if info["events"] or step_index % args.log_every == 0:
                _print_state(step_index, episodes, info, reward)
            if terminated or truncated:
                episodes += 1
                observation, info = env.reset()
                print(f"reset -> episode {episodes}", flush=True)
    except KeyboardInterrupt:
        print("stopped by user", flush=True)
    finally:
        if viewer is not None:
            viewer.close()
        env.close()
    return 0


def _print_state(step: int, episode: int, info: dict, reward: float) -> None:
    state = info["game_state"]
    room = state["room"]
    player = state["player"]
    events = [event["type"] for event in info["events"]]
    print(
        f"step={step:04d} episode={episode} action={info['action_name']} "
        f"room={room['map_id']:02X}:{room['id']:02X} "
        f"xy=({player['x']:3d},{player['y']:3d}) "
        f"health={player['health']:02X} monsters={len(state['monsters'])} "
        f"reward={reward:+.3f} events={events}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
