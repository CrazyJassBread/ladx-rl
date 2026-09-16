"""Entity-removal skills used by the first Tail Cave experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from zelda_env.reward_signals import RewardComposer, transition_signals
from zelda_env.tasks.base import EventList, GameState, Task, TaskStep


class DefeatEntitiesTask(Task):
    """Succeed when the target entities present at reset have disappeared."""

    def __init__(
        self,
        target_types: Iterable[int],
        *,
        reward: Mapping[str, float],
        task_id: str = "defeat_entities",
    ) -> None:
        self.task_id = task_id
        self.target_types = frozenset(int(value) for value in target_types)
        if not self.target_types:
            raise ValueError("target_types cannot be empty")
        self.rewards = RewardComposer(reward)
        self._room: tuple[int, int, int] | None = None
        self._targets: dict[int, int] = {}
        self._removed: set[int] = set()
        self._cleared = False

    def reset(self, state: GameState) -> dict[str, Any]:
        self._room = _room_key(state)
        self._targets = {
            entity["slot"]: entity["type"]
            for entity in state["entities"]
            if entity["type"] in self.target_types
        }
        if not self._targets:
            expected = ", ".join(f"0x{value:02X}" for value in sorted(self.target_types))
            raise ValueError(f"Task {self.task_id!r} found no target entities of type {expected}")
        self._removed = set()
        self._cleared = False
        info = self._info(state, success=False, failure=None)
        info["reward_weights"] = dict(self.rewards.weights)
        return info

    def step(
        self,
        previous: GameState,
        current: GameState,
        events: EventList,
    ) -> TaskStep:
        signals = transition_signals(current, events)

        failure = None
        if current["player"]["health"] == 0:
            failure = "player_died"
        elif _room_key(current) != self._room:
            failure = "left_task_room"
            signals["premature_room_exit"] = 1.0
        if failure is not None:
            return self._result(current, signals, success=False, failure=failure)

        active = {entity["slot"]: entity["type"] for entity in current["entities"]}
        removed_now = {
            slot
            for slot, entity_type in self._targets.items()
            if slot not in self._removed and active.get(slot) != entity_type
        }
        self._removed.update(removed_now)
        if removed_now:
            signals["target_defeated"] = float(len(removed_now))

        all_cleared = len(self._removed) == len(self._targets)
        if all_cleared and not self._cleared:
            signals["all_targets_cleared"] = 1.0
            self._cleared = True

        success = all_cleared and self._success_condition(current)
        return self._result(current, signals, success=success, failure=None)

    def _success_condition(self, state: GameState) -> bool:
        return True

    def _phase(self, state: GameState) -> str:
        return "complete" if self._cleared else "defeat_targets"

    def _extra_info(self, state: GameState) -> dict[str, Any]:
        return {}

    def _result(
        self,
        state: GameState,
        signals: dict[str, float],
        *,
        success: bool,
        failure: str | None,
    ) -> TaskStep:
        reward, terms = self.rewards.compose(signals)
        info = self._info(state, success=success, failure=failure)
        info["reward_signals"] = signals
        info["reward_terms"] = terms
        return TaskStep(
            reward=reward,
            terminated=success or failure is not None,
            info=info,
        )

    def _info(
        self,
        state: GameState,
        *,
        success: bool,
        failure: str | None,
    ) -> dict[str, Any]:
        remaining = sorted(set(self._targets) - self._removed)
        return {
            "task_id": self.task_id,
            "phase": self._phase(state),
            "target_types": sorted(self.target_types),
            "target_slots": sorted(self._targets),
            "targets_remaining": len(remaining),
            "remaining_slots": remaining,
            "success": success,
            "failure": failure,
            **self._extra_info(state),
        }


class KillAndCollectTask(DefeatEntitiesTask):
    """Defeat reset-time targets, then collect the resulting small key."""

    def __init__(
        self,
        target_types: Iterable[int],
        *,
        drop_type: int = 0x30,
        **kwargs: Any,
    ) -> None:
        super().__init__(target_types, **kwargs)
        self.drop_type = int(drop_type)
        self._initial_keys = 0
        self._drop_seen = False

    def reset(self, state: GameState) -> dict[str, Any]:
        self._initial_keys = state["progress"]["small_keys"]
        self._drop_seen = False
        return super().reset(state)

    def step(
        self,
        previous: GameState,
        current: GameState,
        events: EventList,
    ) -> TaskStep:
        drop_visible = any(entity["type"] == self.drop_type for entity in current["entities"])
        first_drop = drop_visible and not self._drop_seen
        self._drop_seen |= drop_visible

        result = super().step(previous, current, events)
        if result.info["failure"] is not None:
            return result
        collected = current["progress"]["small_keys"] > self._initial_keys
        was_collected = previous["progress"]["small_keys"] > self._initial_keys
        success = self._cleared and collected
        signals = dict(result.info["reward_signals"])
        if first_drop:
            signals["key_drop_seen"] = 1.0
        if collected and not was_collected:
            signals["key_collected"] = 1.0

        return self._result(current, signals, success=success, failure=None)

    def _success_condition(self, state: GameState) -> bool:
        return state["progress"]["small_keys"] > self._initial_keys

    def _phase(self, state: GameState) -> str:
        if state["progress"]["small_keys"] > self._initial_keys:
            return "complete"
        if self._cleared:
            return "collect_key"
        return "defeat_targets"

    def _extra_info(self, state: GameState) -> dict[str, Any]:
        current_keys = state["progress"]["small_keys"]
        return {
            "drop_type": self.drop_type,
            "drop_seen": self._drop_seen,
            "initial_small_keys": self._initial_keys,
            "current_small_keys": current_keys,
            "key_collected": current_keys > self._initial_keys,
        }


class DefeatAndCollectItemTask(DefeatEntitiesTask):
    """Defeat reset-time targets, interact with a chest, and acquire its item."""

    def __init__(
        self,
        target_types: Iterable[int],
        *,
        item_field: str,
        chest_type: int = 0x07,
        **kwargs: Any,
    ) -> None:
        super().__init__(target_types, **kwargs)
        self.item_field = item_field
        self.chest_type = int(chest_type)
        self._initial_item = 0
        self._chest_seen = False

    def reset(self, state: GameState) -> dict[str, Any]:
        try:
            self._initial_item = int(state["inventory"][self.item_field])
        except KeyError as exc:
            raise ValueError(f"Unknown inventory item field: {self.item_field}") from exc
        if self._initial_item:
            raise ValueError(
                f"Task {self.task_id!r} requires {self.item_field!r} to be absent at reset"
            )
        self._chest_seen = False
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
        collected = self._item_collected(current)
        was_collected = self._item_collected(previous)
        success = self._cleared and self._item_received(current)
        signals = dict(result.info["reward_signals"])
        if first_chest:
            signals["chest_revealed"] = 1.0
        if collected and not was_collected:
            signals["item_collected"] = 1.0

        return self._result(current, signals, success=success, failure=None)

    def _success_condition(self, state: GameState) -> bool:
        return self._item_received(state)

    def _phase(self, state: GameState) -> str:
        if self._cleared and self._item_received(state):
            return "complete"
        if self._chest_seen and not self._item_received(state):
            return "receive_item"
        return "open_chest" if self._cleared else "defeat_targets"

    def _extra_info(self, state: GameState) -> dict[str, Any]:
        current_item = int(state["inventory"][self.item_field])
        return {
            "chest_type": self.chest_type,
            "chest_seen": self._chest_seen,
            "item_field": self.item_field,
            "initial_item_value": self._initial_item,
            "current_item_value": current_item,
            "item_collected": current_item > self._initial_item,
            "item_received": self._item_received(state),
        }

    def _item_collected(self, state: GameState) -> bool:
        return int(state["inventory"][self.item_field]) > self._initial_item

    def _item_received(self, state: GameState) -> bool:
        return self._item_collected(state) and self._chest_seen and not self._chest_visible(state)

    def _chest_visible(self, state: GameState) -> bool:
        return any(entity["type"] == self.chest_type for entity in state["entities"])


def _room_key(state: GameState) -> tuple[int, int, int]:
    room = state["room"]
    return room["is_indoor"], room["map_id"], room["id"]
