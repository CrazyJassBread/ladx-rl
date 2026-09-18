"""Rolling Bones miniboss combat with explicit rolling-bar evasion signals."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from zelda_env.tasks.base import EventList, GameState
from zelda_env.tasks.entity_task import DefeatEntitiesTask


class RollingBonesTask(DefeatEntitiesTask):
    """Defeat Rolling Bones while rewarding bounded airborne bar crossings."""

    def __init__(
        self,
        target_types: Iterable[int],
        *,
        bar_type: int = 0x82,
        expected_bar_count: int = 1,
        bar_crossing_tolerance: int = 12,
        max_rewarded_bar_dodges: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(target_types, **kwargs)
        self.bar_type = int(bar_type)
        self.expected_bar_count = int(expected_bar_count)
        self.bar_crossing_tolerance = int(bar_crossing_tolerance)
        self.max_rewarded_bar_dodges = int(max_rewarded_bar_dodges)
        if self.expected_bar_count < 1:
            raise ValueError("expected_bar_count must be positive")
        if self.bar_crossing_tolerance < 0:
            raise ValueError("bar_crossing_tolerance cannot be negative")
        if self.max_rewarded_bar_dodges < 0:
            raise ValueError("max_rewarded_bar_dodges cannot be negative")
        self._bar_slots: set[int] = set()
        self._bar_dodge_credited = False
        self._bar_dodges = 0
        self._rewarded_bar_dodges = 0
        self._jumps_started = 0
        self._airborne_steps = 0

    def reset(self, state: GameState) -> dict[str, Any]:
        self._bar_slots = {
            int(entity["slot"])
            for entity in state["entities"]
            if int(entity["type"]) == self.bar_type
        }
        if len(self._bar_slots) != self.expected_bar_count:
            raise ValueError(
                f"Task {self.task_id!r} found {len(self._bar_slots)} rolling bars, "
                f"expected {self.expected_bar_count}"
            )
        self._bar_dodge_credited = False
        self._bar_dodges = 0
        self._rewarded_bar_dodges = 0
        self._jumps_started = 0
        self._airborne_steps = 0
        return super().reset(state)

    def _add_task_signals(
        self,
        previous: GameState,
        current: GameState,
        events: EventList,
        signals: dict[str, float],
    ) -> None:
        previous_z = int(previous["player"]["z"])
        current_z = int(current["player"]["z"])
        if previous_z == 0 and current_z > 0:
            self._jumps_started += 1
        if current_z > 0:
            self._airborne_steps += 1

        previous_bar = self._bar(previous)
        current_bar = self._bar(current)
        if previous_bar is None or current_bar is None:
            return

        # State 1 is the active horizontal roll. Returning to state 0 or 2
        # rearms the per-pass credit before the next launch.
        if int(current_bar["state"]) != 1:
            self._bar_dodge_credited = False
            return
        if self._bar_dodge_credited:
            return

        previous_delta = int(previous_bar["x"]) - int(previous["player"]["x"])
        current_delta = int(current_bar["x"]) - int(current["player"]["x"])
        crossed = (
            previous_delta == 0
            or current_delta == 0
            or (previous_delta < 0 < current_delta)
            or (current_delta < 0 < previous_delta)
            or min(abs(previous_delta), abs(current_delta))
            <= self.bar_crossing_tolerance
        )
        airborne = previous_z > 0 or current_z > 0
        damaged = any(event["type"] == "player_damaged" for event in events)
        if not crossed or not airborne or damaged:
            return

        self._bar_dodge_credited = True
        self._bar_dodges += 1
        if self._rewarded_bar_dodges < self.max_rewarded_bar_dodges:
            self._rewarded_bar_dodges += 1
            signals["bar_dodged"] = 1.0

    def _phase(self, state: GameState) -> str:
        return "complete" if self._cleared else "defeat_rolling_bones"

    def _extra_info(self, state: GameState) -> dict[str, Any]:
        bar = self._bar(state)
        return {
            "bar_type": self.bar_type,
            "bar_slots": sorted(self._bar_slots),
            "bar_state": int(bar["state"]) if bar is not None else None,
            "bar_x": int(bar["x"]) if bar is not None else None,
            "bar_dodges": self._bar_dodges,
            "rewarded_bar_dodges": self._rewarded_bar_dodges,
            "max_rewarded_bar_dodges": self.max_rewarded_bar_dodges,
            "jumps_started": self._jumps_started,
            "airborne_steps": self._airborne_steps,
        }

    def _bar(self, state: GameState) -> dict[str, Any] | None:
        return next(
            (
                entity
                for entity in state["entities"]
                if int(entity["slot"]) in self._bar_slots
                and int(entity["type"]) == self.bar_type
            ),
            None,
        )
