"""Tasks built around dungeon floor switches and revealed chests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from zelda_env.reward_signals import RewardComposer, transition_signals
from zelda_env.tasks.base import EventList, GameState, Task, TaskStep


PIT_GROUND_STATUS = 0x07
FALLING_DOWN_MOTION_STATE = 0x06


class PressSwitchOpenChestTask(Task):
    """Press a floor switch, reveal its chest, and collect the small key."""

    def __init__(
        self,
        *,
        reward: Mapping[str, float],
        chest_type: int = 0x07,
        switch_waypoints: Iterable[Iterable[int]] = (),
        post_hazard_waypoints: Iterable[Iterable[int]] = (),
        chest_waypoints: Iterable[Iterable[int]] = (),
        blocking_hazard_types: Iterable[int] = (),
        waypoint_tolerance: int = 8,
        success_stage: str = "chest",
        task_id: str = "press_switch_open_chest",
    ) -> None:
        self.task_id = task_id
        self.chest_type = int(chest_type)
        self.rewards = RewardComposer(reward)
        self.switch_waypoints = _parse_waypoints(switch_waypoints, "switch_waypoints")
        self.post_hazard_waypoints = _parse_waypoints(
            post_hazard_waypoints, "post_hazard_waypoints"
        )
        self.chest_waypoints = _parse_waypoints(chest_waypoints, "chest_waypoints")
        self.blocking_hazard_types = frozenset(int(value) for value in blocking_hazard_types)
        self.waypoint_tolerance = int(waypoint_tolerance)
        if self.waypoint_tolerance < 0:
            raise ValueError("waypoint_tolerance cannot be negative")
        if success_stage not in {"switch", "chest"}:
            raise ValueError("success_stage must be 'switch' or 'chest'")
        self.success_stage = success_stage
        self._room: tuple[int, int, int] | None = None
        self._initial_keys = 0
        self._switch_pressed = False
        self._chest_revealed = False
        self._chest_seen = False
        self._pit_contact = False
        self._hazards: dict[int, int] = {}
        self._removed_hazards: set[int] = set()
        self._route_indices = {"switch": 0, "post_hazard": 0, "chest": 0}

    def reset(self, state: GameState) -> dict[str, Any]:
        self._room = _room_key(state)
        self._initial_keys = int(state["progress"]["small_keys"])
        if int(state["event_flags"]["switch_button_pressed"]):
            raise ValueError(
                f"Task {self.task_id!r} requires the floor switch to be unpressed at reset"
            )
        if self._chest_visible(state):
            raise ValueError(
                f"Task {self.task_id!r} requires the revealed chest to be absent at reset"
            )
        self._switch_pressed = False
        self._chest_revealed = False
        self._chest_seen = False
        self._pit_contact = self._touching_pit(state)
        self._hazards = {
            entity["slot"]: entity["type"]
            for entity in state["entities"]
            if entity["type"] in self.blocking_hazard_types
        }
        if self.blocking_hazard_types and not self._hazards:
            expected = ", ".join(
                f"0x{value:02X}" for value in sorted(self.blocking_hazard_types)
            )
            raise ValueError(
                f"Task {self.task_id!r} found no blocking hazards of type {expected}"
            )
        self._removed_hazards = set()
        self._route_indices = {"switch": 0, "post_hazard": 0, "chest": 0}
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
        elif int(current["player"]["motion_state"]) == FALLING_DOWN_MOTION_STATE:
            failure = "fell_in_pit"
            signals["fell_in_pit"] = 1.0
        if failure is not None:
            return self._result(current, signals, success=False, failure=failure)

        touching_pit = self._touching_pit(current)
        if touching_pit and not self._pit_contact:
            signals["pit_contact"] = 1.0
        self._pit_contact = touching_pit

        self._update_hazards(current, signals)
        self._add_route_signals(previous, current, signals)

        if not self._switch_pressed:
            previous_hold = int(previous["event_flags"]["switch_button_hold_frames"])
            current_hold = int(current["event_flags"]["switch_button_hold_frames"])
            hold_progress = current_hold - previous_hold
            if hold_progress:
                signals["switch_hold_progress"] = float(hold_progress)

        switch_now = int(current["event_flags"]["switch_button_pressed"])
        if switch_now and not self._switch_pressed:
            self._switch_pressed = True
            self._route_indices["chest"] = 0
            signals["switch_pressed"] = 1.0

        reveal_effect = int(current["event_flags"]["room_event_executed"])
        if self._switch_pressed and reveal_effect and not self._chest_revealed:
            self._chest_revealed = True
            signals["chest_revealed"] = 1.0

        chest_visible = self._chest_visible(current)
        if chest_visible and not self._chest_seen:
            self._chest_seen = True

        key_collected = int(current["progress"]["small_keys"]) > self._initial_keys
        was_collected = int(previous["progress"]["small_keys"]) > self._initial_keys
        if key_collected and not was_collected:
            signals["chest_opened"] = 1.0

        success = self._switch_pressed and (
            self.success_stage == "switch" or (self._chest_seen and key_collected)
        )
        return self._result(current, signals, success=success, failure=None)

    def _add_route_signals(
        self,
        previous: GameState,
        current: GameState,
        signals: dict[str, float],
    ) -> None:
        route = self._active_route()
        if route is None:
            return
        name, waypoints = route
        index = self._route_indices[name]
        if index >= len(waypoints):
            return
        target = waypoints[index]
        previous_distance = _manhattan(previous["player"], target)
        current_distance = _manhattan(current["player"], target)
        progress = previous_distance - current_distance
        if progress:
            signals["route_progress"] = float(progress)
        if current_distance <= self.waypoint_tolerance:
            self._route_indices[name] += 1
            signals["waypoint_reached"] = 1.0

    def _update_hazards(
        self,
        current: GameState,
        signals: dict[str, float],
    ) -> None:
        active = {entity["slot"]: entity["type"] for entity in current["entities"]}
        removed_now = {
            slot
            for slot, entity_type in self._hazards.items()
            if slot not in self._removed_hazards and active.get(slot) != entity_type
        }
        self._removed_hazards.update(removed_now)
        if removed_now:
            signals["blocking_hazard_removed"] = float(len(removed_now))

    def _active_route(self) -> tuple[str, tuple[tuple[int, int], ...]] | None:
        if self._switch_pressed:
            return "chest", self.chest_waypoints
        if self._route_indices["switch"] < len(self.switch_waypoints):
            return "switch", self.switch_waypoints
        if len(self._removed_hazards) < len(self._hazards):
            return None
        return "post_hazard", self.post_hazard_waypoints

    def _touching_pit(self, state: GameState) -> bool:
        player = state["player"]
        return (
            int(player["ground_status"]) == PIT_GROUND_STATUS
            or int(player["pit_slipping_counter"]) > 0
        )

    def _chest_visible(self, state: GameState) -> bool:
        return any(entity["type"] == self.chest_type for entity in state["entities"])

    def _phase(self, state: GameState) -> str:
        if self.success_stage == "switch" and self._switch_pressed:
            return "complete"
        if (
            self._switch_pressed
            and self._chest_seen
            and int(state["progress"]["small_keys"]) > self._initial_keys
        ):
            return "complete"
        if self._chest_seen:
            return "open_chest"
        if self._chest_revealed:
            return "open_chest"
        if self._switch_pressed:
            return "wait_for_chest"
        if len(self._removed_hazards) < len(self._hazards) and (
            self._route_indices["switch"] >= len(self.switch_waypoints)
        ):
            return "clear_blocking_hazard"
        return "press_switch"

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
        return TaskStep(reward, success or failure is not None, info)

    def _info(
        self,
        state: GameState,
        *,
        success: bool,
        failure: str | None,
    ) -> dict[str, Any]:
        current_keys = int(state["progress"]["small_keys"])
        player = state["player"]
        route = self._active_route()
        route_name = route[0] if route is not None else None
        route_target = None
        waypoint_index = 0
        if route is not None:
            route_name, waypoints = route
            waypoint_index = self._route_indices[route_name]
            if waypoint_index < len(waypoints):
                route_target = list(waypoints[waypoint_index])
        remaining_hazards = sorted(set(self._hazards) - self._removed_hazards)
        return {
            "task_id": self.task_id,
            "phase": self._phase(state),
            "success": success,
            "failure": failure,
            "switch_pressed": self._switch_pressed,
            "switch_button_value": int(state["event_flags"]["switch_button_pressed"]),
            "switch_button_hold_frames": int(
                state["event_flags"]["switch_button_hold_frames"]
            ),
            "success_stage": self.success_stage,
            "route_name": route_name,
            "waypoint_index": waypoint_index,
            "route_target": route_target,
            "blocking_hazard_types": sorted(self.blocking_hazard_types),
            "blocking_hazard_slots": sorted(self._hazards),
            "blocking_hazards_remaining": len(remaining_hazards),
            "remaining_blocking_hazard_slots": remaining_hazards,
            "chest_type": self.chest_type,
            "chest_seen": self._chest_seen,
            "chest_revealed": self._chest_revealed,
            "initial_small_keys": self._initial_keys,
            "current_small_keys": current_keys,
            "chest_opened": current_keys > self._initial_keys,
            "touching_pit": self._touching_pit(state),
            "motion_state": int(player["motion_state"]),
            "pit_slipping_counter": int(player["pit_slipping_counter"]),
        }


def _room_key(state: GameState) -> tuple[int, int, int]:
    room = state["room"]
    return room["is_indoor"], room["map_id"], room["id"]


def _parse_waypoints(
    values: Iterable[Iterable[int]],
    name: str,
) -> tuple[tuple[int, int], ...]:
    waypoints = tuple(tuple(int(coordinate) for coordinate in value) for value in values)
    if any(len(waypoint) != 2 for waypoint in waypoints):
        raise ValueError(f"{name} entries must be [x, y] pairs")
    return waypoints


def _manhattan(player: Mapping[str, Any], target: tuple[int, int]) -> int:
    return abs(int(player["x"]) - target[0]) + abs(int(player["y"]) - target[1])
