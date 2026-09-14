"""Central environment factory used by training and evaluation."""

from pathlib import Path
from typing import Any

from zelda_env import ZeldaEnv
from zelda_env.tasks import TaskSpec


def make_env(config: dict[str, Any]) -> ZeldaEnv:
    task_config = config.get("task", {})
    task = TaskSpec(
        id=task_config.get("id", "free_play"),
        max_steps=task_config.get("max_steps"),
        success_events=frozenset(task_config.get("success_events", [])),
        failure_events=frozenset(task_config.get("failure_events", ["player_died"])),
    )
    return ZeldaEnv(
        initial_state_path=Path(config["initial_state_path"]),
        frame_skip=config.get("frame_skip", 4),
        state_mode=config.get("state_mode", "reward"),
        task=task,
    )
