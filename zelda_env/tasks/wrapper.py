"""Gymnasium wrapper for task rewards and reset-state sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import gymnasium as gym
import numpy as np

from zelda_env.actions import DEFAULT_ACTIONS
from zelda_env.tasks.base import Task


DEFAULT_TRAINING_ACTIONS = tuple(
    action.name for action in DEFAULT_ACTIONS if action.name not in {"START", "SELECT"}
)


class TaskEnv(gym.Wrapper):
    """Keep observations pixel-only while deriving task signals from ``info``."""

    def __init__(
        self,
        env: gym.Env,
        task: Task,
        state_paths: Iterable[str | Path],
        *,
        noop_frames: tuple[int, int] = (0, 0),
        action_names: Iterable[str] = DEFAULT_TRAINING_ACTIONS,
        expected_room: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__(env)
        self.task = task
        self.state_paths = tuple(Path(path) for path in state_paths)
        if not self.state_paths:
            raise ValueError("state_paths cannot be empty")
        low, high = noop_frames
        if low < 0 or high < low:
            raise ValueError("noop_frames must be an inclusive non-negative range")
        self.noop_frames = int(low), int(high)
        self.expected_room = expected_room

        base_actions = {
            action.name: index
            for index, action in enumerate(env.unwrapped.actions)
        }
        try:
            self._action_map = tuple(base_actions[name] for name in action_names)
        except KeyError as exc:
            raise ValueError(f"Unknown action name: {exc.args[0]}") from exc
        if not self._action_map:
            raise ValueError("action_names cannot be empty")
        self.action_space = gym.spaces.Discrete(len(self._action_map))
        self._rng = np.random.default_rng()
        self._previous_state = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        state_path = self.state_paths[int(self._rng.integers(len(self.state_paths)))]
        low, high = self.noop_frames
        jitter = int(self._rng.integers(low, high + 1))
        reset_options = dict(options or {})
        reset_options.update({"state_path": state_path, "noop_frames": jitter})
        observation, info = self.env.reset(seed=seed, options=reset_options)
        self._previous_state = info["game_state"]
        actual_room = _room_key(self._previous_state)
        if self.expected_room is not None and actual_room != self.expected_room:
            raise ValueError(
                f"State {state_path} opened room {actual_room}, expected {self.expected_room}"
            )
        task_info = self.task.reset(self._previous_state)
        info = dict(info)
        info.update(
            {
                "task": task_info,
                "is_success": False,
                "initial_state_path": str(state_path),
            }
        )
        return observation, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid task action: {action}")
        mapped_action = self._action_map[int(action)]
        observation, _, base_terminated, truncated, info = self.env.step(mapped_action)
        current = info["game_state"]
        result = self.task.step(self._previous_state, current, info["events"])
        self._previous_state = current
        info = dict(info)
        info["task"] = result.info
        info["is_success"] = result.info["success"]
        return observation, result.reward, result.terminated or base_terminated, truncated, info


def _room_key(state: dict) -> tuple[int, int, int]:
    room = state["room"]
    return room["is_indoor"], room["map_id"], room["id"]
