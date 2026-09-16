"""Minimal Gymnasium environment for Link's Awakening DX."""

from __future__ import annotations

from numbers import Integral
from os import PathLike
from pathlib import Path
from typing import Any, Callable

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    gym = None

from zelda_env.actions import DEFAULT_ACTIONS, ActionSpec, buttons_for_action
from zelda_env.emulator import Emulator, PyBoyEmulator
from zelda_env.events import default_reward, detect_events
from zelda_env.memory import DEFAULT_ROM_PATH, DEFAULT_SYM_PATH, GameMemory, GameState
from zelda_env.utils.symbol_loader import SymbolTable


RewardFn = Callable[[dict[str, Any] | None, dict[str, Any], list[dict[str, Any]]], float]


class ZeldaEnv(gym.Env if gym is not None else object):
    """Pixel-only LADX environment with memory state returned in ``info``."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(
        self,
        *,
        rom_path: str | Path = DEFAULT_ROM_PATH,
        sym_path: str | Path = DEFAULT_SYM_PATH,
        initial_state_path: str | Path | None = None,
        render_mode: str | None = None,
        frame_skip: int = 4,
        max_episode_steps: int | None = None,
        reward_fn: RewardFn = default_reward,
        actions: tuple[ActionSpec, ...] = DEFAULT_ACTIONS,
        emulator: Emulator | None = None,
    ) -> None:
        try:
            import numpy as np
            from gymnasium import spaces
        except ImportError as exc:
            raise RuntimeError("ZeldaEnv requires gymnasium and numpy") from exc
        if render_mode not in {None, "rgb_array", "human"}:
            raise ValueError("render_mode must be None, 'rgb_array', or 'human'")
        if isinstance(frame_skip, bool) or not isinstance(frame_skip, Integral) or frame_skip < 1:
            raise ValueError("frame_skip must be a positive integer")
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive or None")

        self.render_mode = render_mode
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = max_episode_steps
        self.reward_fn = reward_fn
        self.actions = actions
        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = spaces.Box(0, 255, shape=(144, 160, 3), dtype=np.uint8)

        self.emulator = emulator if emulator is not None else PyBoyEmulator(
            rom_path,
            sym_path=sym_path,
            window="SDL2" if render_mode == "human" else "null",
        )
        self.symbols = SymbolTable.from_file(sym_path)
        self.memory = GameMemory(self.emulator, self.symbols)
        self._initial_state = Path(initial_state_path).read_bytes() if initial_state_path else None
        self._boot_state = self.emulator.save_state()
        self._previous_state: dict[str, Any] | None = None
        self._visited_rooms: set[tuple[int, int, int]] = set()
        self.elapsed_steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.elapsed_steps = 0
        self.emulator.release_all()

        state_data: bytes | bytearray | memoryview | str | PathLike[str] | None = self._initial_state
        noop_frames = 0
        if options:
            state_data = options.get("state", options.get("state_path", state_data))
            noop_frames = options.get("noop_frames", 0)
        if isinstance(noop_frames, bool) or not isinstance(noop_frames, Integral) or noop_frames < 0:
            raise ValueError("noop_frames must be a non-negative integer")
        self.emulator.load_state(self._state_bytes(state_data if state_data is not None else self._boot_state))
        if noop_frames:
            self.emulator.advance(int(noop_frames))

        game_state = self.get_game_state()
        self._previous_state = game_state
        self._visited_rooms = {_room_key(game_state)}
        info = self._info(game_state, [], action=None, num_frames=0)
        info["reset_noop_frames"] = int(noop_frames)
        return self.get_frame(), info

    def step(self, action: int, num_frames: int | None = None):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        frames = self.frame_skip if num_frames is None else num_frames
        if isinstance(frames, bool) or not isinstance(frames, Integral) or frames < 1:
            raise ValueError("num_frames must be a positive integer")
        frames = int(frames)
        action = int(action)

        self.emulator.press(buttons_for_action(action, self.actions))
        try:
            self.emulator.advance(frames)
        finally:
            self.emulator.release_all()

        self.elapsed_steps += 1
        game_state = self.get_game_state()
        events = detect_events(self._previous_state, game_state, self._visited_rooms)
        reward = float(self.reward_fn(self._previous_state, game_state, events))
        terminated = any(event["type"] == "player_died" for event in events)
        truncated = (
            self.max_episode_steps is not None
            and self.elapsed_steps >= self.max_episode_steps
            and not terminated
        )
        self._previous_state = game_state
        return (
            self.get_frame(),
            reward,
            terminated,
            truncated,
            self._info(game_state, events, action=action, num_frames=frames),
        )

    def get_frame(self):
        """Return the current ``(144, 160, 3)`` RGB uint8 frame."""

        return self.emulator.get_frame()

    def read_state(self) -> GameState:
        """Read the current semantic state as a typed snapshot."""

        return self.memory.read_state()

    def get_game_state(self) -> dict[str, Any]:
        """Read the legacy dictionary-shaped semantic state."""

        return self.memory.game_state()

    def save_state(self, path: str | PathLike[str] | None = None) -> bytes:
        data = self.emulator.save_state()
        if path is not None:
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
        return data

    def load_state(self, state: bytes | bytearray | memoryview | str | PathLike[str]) -> None:
        self.emulator.release_all()
        self.emulator.load_state(self._state_bytes(state))
        self._previous_state = self.get_game_state()
        self._visited_rooms.add(_room_key(self._previous_state))

    def read_memory(self, address: int | str, length: int = 1) -> int | bytes:
        """Read by numeric address or by a name from ``azle.sym``."""

        return self.memory.read(address, length)

    def render(self):
        return self.get_frame()

    def close(self) -> None:
        self.emulator.close()

    def _info(self, game_state, events, *, action, num_frames):
        action_spec = self.actions[action] if action is not None else None
        return {
            "game_state": game_state,
            "events": events,
            "action": action,
            "action_name": action_spec.name if action_spec else None,
            "num_frames": num_frames,
            "elapsed_steps": self.elapsed_steps,
        }

    @staticmethod
    def _state_bytes(state: bytes | bytearray | memoryview | str | PathLike[str]) -> bytes:
        if isinstance(state, (str, PathLike)):
            return Path(state).read_bytes()
        if isinstance(state, (bytes, bytearray, memoryview)):
            return bytes(state)
        raise TypeError("state must be bytes-like or a filesystem path")


def _room_key(state: dict[str, Any]) -> tuple[int, int, int]:
    room = state["room"]
    return room["is_indoor"], room["map_id"], room["id"]
