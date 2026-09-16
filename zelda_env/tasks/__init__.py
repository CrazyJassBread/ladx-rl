"""Reusable task definitions layered on top of :class:`ZeldaEnv`."""

from zelda_env.tasks.entity_task import (
    DefeatAndCollectItemTask,
    DefeatEntitiesTask,
    KillAndCollectTask,
)
from zelda_env.tasks.wrapper import TaskEnv

__all__ = [
    "DefeatAndCollectItemTask",
    "DefeatEntitiesTask",
    "KillAndCollectTask",
    "TaskEnv",
]
