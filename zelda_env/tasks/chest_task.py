"""Chest objectives unlocked by defeating enemies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from zelda_env.tasks.base import EventList, GameState, TaskStep
from zelda_env.tasks.entity_task import DefeatEntitiesTask


class DefeatAndOpenChestTask(DefeatEntitiesTask, ABC):
    """Shared defeat, chest interaction, reward, and dialog lifecycle."""

    reward_signal: str
    completion_signal: str | None = None

    def __init__(
        self,
        target_types: Iterable[int],
        *,
        chest_type: int = 0x07,
        **kwargs: Any,
    ) -> None:
        super().__init__(target_types, **kwargs)
        self.chest_type = int(chest_type)
        self._chest_seen = False
        self._dialog_completed = False

    def reset(self, state: GameState) -> dict[str, Any]:
        self._reset_reward(state)
        self._chest_seen = False
        self._dialog_completed = False
        return super().reset(state)

    def step(
        self,
        previous: GameState,
        current: GameState,
        events: EventList,
    ) -> TaskStep:
        result = super().step(previous, current, events)
        if result.info["failure"] is not None:
            return result

        chest_visible = self._chest_visible(current)
        first_chest = chest_visible and not self._chest_seen
        self._chest_seen |= chest_visible

        collected = self._reward_collected(current)
        was_collected = self._reward_collected(previous)
        completed_now = self._chest_seen and collected and not chest_visible
        first_completion = completed_now and not self._dialog_completed
        self._dialog_completed |= completed_now

        signals = dict(result.info["reward_signals"])
        if first_chest:
            signals["chest_revealed"] = 1.0
        if collected and not was_collected:
            signals[self.reward_signal] = 1.0
        if first_completion and self.completion_signal is not None:
            signals[self.completion_signal] = 1.0

        success = self._cleared and self._dialog_completed
        return self._result(current, signals, success=success, failure=None)

    def _success_condition(self, state: GameState) -> bool:
        return False

    def _phase(self, state: GameState) -> str:
        if self._cleared and self._dialog_completed:
            return "complete"
        if self._chest_seen and not self._dialog_completed:
            return self._interaction_phase(state)
        return "open_chest" if self._cleared else "defeat_targets"

    def _extra_info(self, state: GameState) -> dict[str, Any]:
        return {
            "chest_type": self.chest_type,
            "chest_seen": self._chest_seen,
            **self._reward_info(state),
        }

    def _chest_visible(self, state: GameState) -> bool:
        return any(entity["type"] == self.chest_type for entity in state["entities"])

    @abstractmethod
    def _reset_reward(self, state: GameState) -> None:
        """Capture and validate the reset-time reward state."""

    @abstractmethod
    def _reward_collected(self, state: GameState) -> bool:
        """Return whether the configured chest reward has been received."""

    @abstractmethod
    def _interaction_phase(self, state: GameState) -> str:
        """Describe the active chest-item interaction phase."""

    @abstractmethod
    def _reward_info(self, state: GameState) -> dict[str, Any]:
        """Return reward-specific task diagnostics."""


class DefeatAndCollectItemTask(DefeatAndOpenChestTask):
    """Defeat reset-time targets and finish receiving a chest inventory item."""

    reward_signal = "item_collected"

    def __init__(
        self,
        target_types: Iterable[int],
        *,
        item_field: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(target_types, **kwargs)
        self.item_field = item_field
        self._initial_item = 0

    def _reset_reward(self, state: GameState) -> None:
        try:
            self._initial_item = int(state["inventory"][self.item_field])
        except KeyError as exc:
            raise ValueError(f"Unknown inventory item field: {self.item_field}") from exc
        if self._initial_item:
            raise ValueError(
                f"Task {self.task_id!r} requires {self.item_field!r} to be absent at reset"
            )

    def _reward_collected(self, state: GameState) -> bool:
        return int(state["inventory"][self.item_field]) > self._initial_item

    def _interaction_phase(self, state: GameState) -> str:
        return "receive_item"

    def _reward_info(self, state: GameState) -> dict[str, Any]:
        current_item = int(state["inventory"][self.item_field])
        return {
            "item_field": self.item_field,
            "initial_item_value": self._initial_item,
            "current_item_value": current_item,
            "item_collected": current_item > self._initial_item,
            "item_received": self._dialog_completed,
        }


class DefeatAndCollectRupeesTask(DefeatAndOpenChestTask):
    """Defeat reset-time targets and finish receiving Rupees from a chest."""

    reward_signal = "rupees_collected"
    completion_signal = "dialog_completed"

    def __init__(
        self,
        target_types: Iterable[int],
        *,
        rupee_amount: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(target_types, **kwargs)
        self.rupee_amount = int(rupee_amount)
        if self.rupee_amount < 1:
            raise ValueError("rupee_amount must be positive")
        self._initial_rupees = 0

    def _reset_reward(self, state: GameState) -> None:
        self._initial_rupees = int(state["progress"]["rupees"])

    def _reward_collected(self, state: GameState) -> bool:
        return int(state["progress"]["rupees"]) >= self._initial_rupees + self.rupee_amount

    def _interaction_phase(self, state: GameState) -> str:
        return "finish_dialog" if self._reward_collected(state) else "receive_rupees"

    def _reward_info(self, state: GameState) -> dict[str, Any]:
        current_rupees = int(state["progress"]["rupees"])
        gained = max(current_rupees - self._initial_rupees, 0)
        return {
            "initial_rupees": self._initial_rupees,
            "current_rupees": current_rupees,
            "rupees_gained": gained,
            "required_rupees": self.rupee_amount,
            "rupees_collected": gained >= self.rupee_amount,
            "dialog_completed": self._dialog_completed,
        }
