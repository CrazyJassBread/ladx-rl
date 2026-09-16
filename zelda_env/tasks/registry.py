"""Construct task objects from small declarative configurations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zelda_env.tasks.base import Task
from zelda_env.tasks.chest_task import (
    DefeatAndCollectItemTask,
    DefeatAndCollectRupeesTask,
)
from zelda_env.tasks.entity_task import DefeatEntitiesTask
from zelda_env.tasks.exit_task import DefeatAndExitTask
from zelda_env.tasks.key_task import KillAndCollectTask
from zelda_env.tasks.switch_task import PressSwitchOpenChestTask


def make_task(config: Mapping[str, Any], *, task_id: str) -> Task:
    values = dict(config)
    kind = values.pop("kind")
    values["task_id"] = task_id
    if kind == "defeat_entities":
        return DefeatEntitiesTask(**values)
    if kind == "defeat_and_exit":
        return DefeatAndExitTask(**values)
    if kind == "kill_and_collect":
        return KillAndCollectTask(**values)
    if kind == "defeat_and_collect_item":
        return DefeatAndCollectItemTask(**values)
    if kind == "defeat_and_collect_rupees":
        return DefeatAndCollectRupeesTask(**values)
    if kind == "press_switch_open_chest":
        return PressSwitchOpenChestTask(**values)
    raise ValueError(f"Unknown task kind: {kind}")
