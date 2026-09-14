"""Gymnasium environment for the preserved LADX disassembly build."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    gym = None

from zelda_env.actions import DEFAULT_ACTIONS, ActionSpec, buttons_for_action
from zelda_env.backends.pyboy_backend import PyBoyBackend
from zelda_env.config import PROJECT_ROOT, LadxPaths
from zelda_env.games.ladx.event_detector import LadxEventDetector
from zelda_env.games.ladx.state_extractor import LadxStateExtractor
from zelda_env.games.ladx.symbols import SymbolTable
from zelda_env.memory.delta import diff_states
from zelda_env.rewards import EventReward
from zelda_env.tasks import TaskSpec

RewardFn = Callable[[dict[str, Any] | None, dict[str, Any], int], tuple[float, dict[str, Any]]]


class ZeldaEnv(gym.Env if gym is not None else object):
    """Pixel observations with semantic state, deltas, and events in ``info``."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(
        self, *, game: str = "ladx", backend: str = "pyboy",
        rom_path: str | Path | None = None, sym_path: str | Path | None = None,
        initial_state_path: str | Path | None = None, render_mode: str | None = None,
        frame_skip: int = 4, max_episode_steps: int | None = None,
        reward_fn: RewardFn | None = None, actions: tuple[ActionSpec, ...] = DEFAULT_ACTIONS,
        project_root: str | Path = PROJECT_ROOT, repo_root: str | Path | None = None,
        state_mode: str = "reward", include_legacy_aliases: bool = False,
        include_state_delta: bool = False, require_initial_state: bool = False,
        task: TaskSpec | None = None,
    ) -> None:
        try:
            import numpy as np
            from gymnasium import spaces
        except ImportError as exc:
            raise RuntimeError("ZeldaEnv requires gymnasium and numpy") from exc
        if game != "ladx" or backend != "pyboy":
            raise ValueError(f"Unsupported game/backend: {game}/{backend}")
        root = Path(repo_root) if repo_root is not None else Path(project_root)
        paths = LadxPaths.default(root)
        rom = Path(rom_path) if rom_path is not None else paths.rom_path
        sym = Path(sym_path) if sym_path is not None else paths.sym_path
        if require_initial_state and initial_state_path is None:
            raise ValueError("This configuration requires initial_state_path")

        self._np, self.game, self.render_mode = np, game, render_mode
        self.frame_skip, self.max_episode_steps = frame_skip, max_episode_steps
        self.reward_fn, self.actions = reward_fn or EventReward(), actions
        self.task, self.include_state_delta = task or TaskSpec(max_steps=max_episode_steps), include_state_delta
        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = spaces.Box(0, 255, shape=(144, 160, 3), dtype=np.uint8)
        self.elapsed_steps = 0
        self._previous_info: dict[str, Any] | None = None
        self._previous_state: dict[str, Any] | None = None
        self._initial_state_data = Path(initial_state_path).read_bytes() if initial_state_path else None
        self.backend = PyBoyBackend(rom, sym_path=sym, window="SDL2" if render_mode == "human" else "null")
        self.extractor = LadxStateExtractor(
            SymbolTable.from_sym_file(sym), repo_root=paths.disassembly_root,
            state_mode=state_mode, include_legacy_aliases=include_legacy_aliases,
        )
        self.event_detector = LadxEventDetector()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if gym is not None:
            super().reset(seed=seed)
        self.elapsed_steps = 0
        self.backend.reset()
        state_data = options.get("state") if options and "state" in options else self._initial_state_data
        if state_data is not None:
            self.backend.load_state(state_data)
        state = self.extractor.extract(self.backend)
        self.event_detector.reset(state)
        reset_reward = getattr(self.reward_fn, "reset", None)
        if callable(reset_reward):
            reset_reward()
        info = self._make_info(state, events=[], action=None)
        self._previous_state, self._previous_info = state, info
        return self.backend.screen_rgb(), info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        self.elapsed_steps += 1
        self.backend.press(buttons_for_action(int(action), self.actions))
        self.backend.advance(self.frame_skip)
        self.backend.release_all()
        state = self.extractor.extract(self.backend)
        previous = self._previous_state or state
        delta = diff_states(previous, state)
        frame = int(state.get("meta", {}).get("frame", self.elapsed_steps))
        event_records = self.event_detector.detect(previous, state, delta, frame=frame)
        events = [event.as_dict() for event in event_records]
        info = self._make_info(state, events=events, action=int(action))
        if self.include_state_delta:
            info["state_delta"] = delta.as_dict()
        reward, terms = self.reward_fn(self._previous_info, info, int(action))
        info["reward_terms"] = terms
        success, failure, reason = self.task.evaluate(event_records)
        terminated = success or failure
        limit = self.task.max_steps if self.task.max_steps is not None else self.max_episode_steps
        truncated = limit is not None and self.elapsed_steps >= limit and not terminated
        info["task"] = {"id": self.task.id, "success": success, "failure": failure,
                        "termination_reason": reason if terminated else ("time_limit" if truncated else None)}
        self._previous_state, self._previous_info = state, info
        return self.backend.screen_rgb(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self.backend.screen_rgb()

    def close(self) -> None:
        self.backend.close()

    def _make_info(self, state, *, events, action):
        return {"state": state, "events": events, "reward_terms": {},
                "transition": {"action": action, "elapsed_steps": self.elapsed_steps},
                "task": {"id": self.task.id, "success": False, "failure": False,
                         "termination_reason": None}}
