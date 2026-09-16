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
        expected_target_count: int | None = None,
    ) -> None:
        self.task_id = task_id
        self.target_types = frozenset(int(value) for value in target_types)
        if not self.target_types:
            raise ValueError("target_types cannot be empty")
        if expected_target_count is not None and expected_target_count < 1:
            raise ValueError("expected_target_count must be positive")
        self.expected_target_count = expected_target_count
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
        if (
            self.expected_target_count is not None
            and len(self._targets) != self.expected_target_count
        ):
            raise ValueError(
                f"Task {self.task_id!r} found {len(self._targets)} targets, "
                f"expected {self.expected_target_count}"
            )
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

        if current["player"]["health"] == 0:
            return self._result(current, signals, success=False, failure="player_died")
        if _room_key(current) != self._room:
            success = self._room_exit_success(current)
            if success:
                signals["destination_reached"] = 1.0
            else:
                signals["premature_room_exit"] = 1.0
            return self._result(
                current,
                signals,
                success=success,
                failure=None if success else "left_task_room",
            )

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

    def _room_exit_success(self, state: GameState) -> bool:
        return False

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


def _room_key(state: GameState) -> tuple[int, int, int]:
    room = state["room"]
    return room["is_indoor"], room["map_id"], room["id"]
