"""Entity-removal skills used by the first Tail Cave experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from zelda_env.reward_signals import RewardComposer, transition_signals
from zelda_env.tasks.base import EventList, GameState, Task, TaskStep
from zelda_env.tasks.navigation import manhattan_distance


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
        self._target_damage_dealt = 0
        self._combat_steps = 0
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
        self._target_damage_dealt = 0
        self._combat_steps = 0
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

        if not self._cleared:
            self._combat_steps += 1
            signals["combat_step"] = 1.0
            self._add_target_approach_signal(previous, current, signals)

        damage_by_slot: dict[int, int] = {}
        for event in events:
            if event["type"] not in {"entity_damaged", "monster_damaged"}:
                continue
            slot = int(event["data"]["slot"])
            if (
                slot not in self._targets
                or int(event["data"]["entity_type"]) != self._targets[slot]
            ):
                continue
            # The generic event detector emits both views for ordinary hits;
            # max-per-slot avoids double counting while retaining the final hit
            # that may only appear as monster_damaged when the entity vanishes.
            damage_by_slot[slot] = max(
                damage_by_slot.get(slot, 0), int(event["data"]["amount"])
            )
        damage = sum(damage_by_slot.values())
        if damage:
            self._target_damage_dealt += damage
            signals["target_damaged"] = float(damage)

        self._add_task_signals(previous, current, events, signals)

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

    def _add_task_signals(
        self,
        previous: GameState,
        current: GameState,
        events: EventList,
        signals: dict[str, float],
    ) -> None:
        """Hook for mechanic-specific transition signals."""

    def _add_target_approach_signal(
        self,
        previous: GameState,
        current: GameState,
        signals: dict[str, float],
    ) -> None:
        active_targets = [
            entity
            for entity in current["entities"]
            if entity["slot"] not in self._removed
            and self._targets.get(entity["slot"]) == entity["type"]
        ]
        if not active_targets:
            return
        target = min(
            active_targets,
            key=lambda entity: manhattan_distance(
                previous["player"], (int(entity["x"]), int(entity["y"]))
            ),
        )
        target_position = int(target["x"]), int(target["y"])
        progress = manhattan_distance(
            previous["player"], target_position
        ) - manhattan_distance(current["player"], target_position)
        if progress:
            signals["target_approach"] = float(progress)

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
            "target_damage_dealt": self._target_damage_dealt,
            "combat_steps": self._combat_steps,
            "success": success,
            "failure": failure,
            **self._extra_info(state),
        }


def _room_key(state: GameState) -> tuple[int, int, int]:
    room = state["room"]
    return room["is_indoor"], room["map_id"], room["id"]
