"""Reusable mechanic-level task definitions layered on top of :class:`ZeldaEnv`."""

from zelda_env.tasks.chest_task import (
    DefeatAndCollectItemTask,
    DefeatAndCollectRupeesTask,
)
from zelda_env.tasks.entity_task import DefeatEntitiesTask
from zelda_env.tasks.exit_task import DefeatAndExitTask
from zelda_env.tasks.key_task import KillAndCollectTask
from zelda_env.tasks.rolling_bones_task import RollingBonesTask
from zelda_env.tasks.switch_task import PressSwitchOpenChestTask
from zelda_env.tasks.wrapper import TaskEnv

__all__ = [
    "DefeatAndCollectItemTask",
    "DefeatAndCollectRupeesTask",
    "DefeatAndExitTask",
    "DefeatEntitiesTask",
    "KillAndCollectTask",
    "PressSwitchOpenChestTask",
    "RollingBonesTask",
    "TaskEnv",
]
