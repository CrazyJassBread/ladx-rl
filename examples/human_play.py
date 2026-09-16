"""Play ZeldaEnv manually while inspecting live environment info."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from zelda_env import ZeldaEnv

from viewer import GameInfoViewer


DEFAULT_STATE = Path("save_states/azle.gbc.tail_cave.r3.state")

KEY_TO_BUTTON = {
    "up": "UP",
    "down": "DOWN",
    "left": "LEFT",
    "right": "RIGHT",
    "z": "A",
    "x": "B",
    "return": "START",
    "backspace": "SELECT",
}
ACTION_BY_BUTTONS = {
    frozenset(): 0,
    frozenset({"UP"}): 1,
    frozenset({"DOWN"}): 2,
    frozenset({"LEFT"}): 3,
    frozenset({"RIGHT"}): 4,
    frozenset({"A"}): 5,
    frozenset({"B"}): 6,
    frozenset({"START"}): 7,
    frozenset({"SELECT"}): 8,
    frozenset({"UP", "A"}): 9,
    frozenset({"DOWN", "A"}): 10,
    frozenset({"LEFT", "A"}): 11,
    frozenset({"RIGHT", "A"}): 12,
    frozenset({"UP", "B"}): 13,
    frozenset({"DOWN", "B"}): 14,
    frozenset({"LEFT", "B"}): 15,
    frozenset({"RIGHT", "B"}): 16,
}
DIRECTIONS = frozenset({"UP", "DOWN", "LEFT", "RIGHT"})


class KeyboardController:
    """Translate held keyboard keys to the environment's fixed action IDs."""

    def __init__(self) -> None:
        self.pressed: set[str] = set()
        self.press_order: list[str] = []
        self.exit_requested = False

    def key_down(self, key: str) -> None:
        key = key.lower()
        if key == "escape":
            self.exit_requested = True
            return
        button = KEY_TO_BUTTON.get(key)
        if button is None:
            return
        self.pressed.add(button)
        if button in self.press_order:
            self.press_order.remove(button)
        self.press_order.append(button)

    def key_up(self, key: str) -> None:
        button = KEY_TO_BUTTON.get(key.lower())
        if button is None:
            return
        self.pressed.discard(button)
        if button in self.press_order:
            self.press_order.remove(button)

    def action(self) -> int:
        if "START" in self.pressed:
            return ACTION_BY_BUTTONS[frozenset({"START"})]
        if "SELECT" in self.pressed:
            return ACTION_BY_BUTTONS[frozenset({"SELECT"})]

        direction = next(
            (button for button in reversed(self.press_order) if button in DIRECTIONS),
            None,
        )
        attack = "A" if "A" in self.pressed else "B" if "B" in self.pressed else None
        buttons = frozenset(button for button in (direction, attack) if button is not None)
        return ACTION_BY_BUTTONS[buttons]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args(argv)

    if not args.state.is_file():
        parser.error(f"save state not found: {args.state}")
    if args.frame_skip < 1 or args.scale < 1 or args.speed <= 0:
        parser.error("frame-skip, scale, and speed must be positive")

    env = ZeldaEnv(initial_state_path=args.state, frame_skip=args.frame_skip)
    viewer = GameInfoViewer(
        title="ZeldaEnv human play — Tail Cave room 1",
        scale=args.scale,
        help_text=(
            "Controls: Arrow keys=move | Z=A | X=B | Enter=START | "
            "Backspace=SELECT | Esc=quit"
        ),
    )
    controller = KeyboardController()
    viewer.bind_keys(controller.key_down, controller.key_up)

    episode = 1
    step = 0
    simulated_frames = 0
    started_at = time.monotonic()
    try:
        observation, info = env.reset()
        if not viewer.update(observation, info, 0.0, step, episode):
            return 0

        while not viewer.closed and not controller.exit_requested:
            action = controller.action()
            observation, reward, terminated, truncated, info = env.step(action)
            step += 1
            simulated_frames += args.frame_skip
            if not viewer.update(observation, info, reward, step, episode):
                break

            if terminated or truncated:
                episode += 1
                observation, info = env.reset()

            target_time = started_at + simulated_frames / (60 * args.speed)
            time.sleep(max(0.0, target_time - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
