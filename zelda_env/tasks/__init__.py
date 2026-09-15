"""Reusable task definitions layered on top of :class:`ZeldaEnv`."""

from zelda_env.tasks.entity_task import DefeatEntitiesTask, KillAndCollectTask
from zelda_env.tasks.wrapper import TaskEnv

__all__ = ["DefeatEntitiesTask", "KillAndCollectTask", "TaskEnv"]
