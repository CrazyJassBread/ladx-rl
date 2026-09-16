"""Reusable task definitions layered on top of :class:`ZeldaEnv`."""

from zelda_env.tasks.entity_task import (
    DefeatAndCollectItemTask,
    DefeatEntitiesTask,
    KillAndCollectTask,
)
from zelda_env.tasks.switch_task import PressSwitchOpenChestTask
from zelda_env.tasks.wrapper import TaskEnv

__all__ = [
    "DefeatAndCollectItemTask",
    "DefeatEntitiesTask",
    "KillAndCollectTask",
    "PressSwitchOpenChestTask",
    "TaskEnv",
]
