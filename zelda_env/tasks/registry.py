"""Construct task objects from small declarative configurations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zelda_env.tasks.base import Task
from zelda_env.tasks.entity_task import DefeatEntitiesTask, KillAndCollectTask


def make_task(config: Mapping[str, Any], *, task_id: str) -> Task:
    values = dict(config)
    kind = values.pop("kind")
    values["task_id"] = task_id
    if kind == "defeat_entities":
        return DefeatEntitiesTask(**values)
    if kind == "kill_and_collect":
        return KillAndCollectTask(**values)
    raise ValueError(f"Unknown task kind: {kind}")
