"""Translate LADX semantic state changes into reward-facing game events."""

from __future__ import annotations

from typing import Any

from zelda_env.events.types import GameEvent
from zelda_env.memory.delta import StateDelta


class LadxEventDetector:
    """Stateful detector for stable, interpretable LADX events."""

    def __init__(self) -> None:
        self._visited_rooms: set[tuple[Any, Any, Any]] = set()
        self._entity_generations: dict[str, int] = {}

    def reset(self, state: dict[str, Any]) -> None:
        self._visited_rooms = {_location_key(state)}
        self._entity_generations.clear()

    def detect(
        self,
        previous: dict[str, Any],
        current: dict[str, Any],
        delta: StateDelta,
        *,
        frame: int,
    ) -> tuple[GameEvent, ...]:
        events: list[GameEvent] = []
        self._detect_location(previous, current, frame, events)
        self._detect_player(previous, current, frame, events)
        self._detect_progress(previous, current, frame, events)
        self._detect_inventory(previous, current, frame, events)
        self._detect_entities(previous, current, frame, events)
        self._detect_room_flags(delta, frame, events)
        return tuple(events)

    def _detect_location(self, previous, current, frame, events) -> None:
        old_key = _location_key(previous)
        new_key = _location_key(current)
        if old_key == new_key:
            return
        events.append(_event("room_changed", frame, {"previous": old_key, "current": new_key}, ("map.location",)))
        if new_key not in self._visited_rooms:
            self._visited_rooms.add(new_key)
            events.append(_event("new_room_visited", frame, {"location": new_key}, ("map.location",)))

    def _detect_player(self, previous, current, frame, events) -> None:
        old_health = _get(previous, "sprites.player.health.current")
        new_health = _get(current, "sprites.player.health.current")
        if not isinstance(old_health, int) or not isinstance(new_health, int):
            return
        path = ("sprites.player.health.current",)
        if new_health < old_health:
            events.append(_event("player_damaged", frame, {"amount": old_health - new_health, "previous": old_health, "current": new_health}, path))
        elif new_health > old_health:
            events.append(_event("player_healed", frame, {"amount": new_health - old_health, "previous": old_health, "current": new_health}, path))
        if old_health > 0 and new_health == 0:
            events.append(_event("player_died", frame, {}, path))

    def _detect_progress(self, previous, current, frame, events) -> None:
        for field, acquired, consumed in (
            ("small_keys", "small_key_acquired", "small_key_used"),
            ("rupees", "rupees_gained", "rupees_spent"),
            ("heart_pieces", "heart_piece_acquired", None),
        ):
            path = f"progress.{field}"
            old = _get(previous, path)
            new = _get(current, path)
            if not isinstance(old, int) or not isinstance(new, int) or old == new:
                continue
            event_type = acquired if new > old else consumed
            if event_type:
                events.append(_event(event_type, frame, {"amount": abs(new - old), "previous": old, "current": new}, (path,)))

        old_instruments = _get(previous, "progress.instruments") or []
        new_instruments = _get(current, "progress.instruments") or []
        for index, (old, new) in enumerate(zip(old_instruments, new_instruments)):
            if not old and new:
                events.append(_event("instrument_acquired", frame, {"index": index + 1}, (f"progress.instruments.{index}",)))

    def _detect_inventory(self, previous, current, frame, events) -> None:
        old_items = _get(previous, "sprites.player.inventory.items") or []
        new_items = _get(current, "sprites.player.inventory.items") or []
        for slot, (old, new) in enumerate(zip(old_items, new_items)):
            if old == new:
                continue
            event_type = "item_acquired" if new else "item_removed"
            events.append(_event(event_type, frame, {"slot": slot, "previous": old, "current": new}, (f"sprites.player.inventory.items.{slot}",)))

    def _detect_entities(self, previous, current, frame, events) -> None:
        old_slots = _get(previous, "sprites.slots") or {}
        new_slots = _get(current, "sprites.slots") or {}
        for slot_id in sorted(set(old_slots) | set(new_slots)):
            old = old_slots.get(slot_id, {})
            new = new_slots.get(slot_id, {})
            old_enabled = bool(old.get("enabled"))
            new_enabled = bool(new.get("enabled"))
            same_entity = old.get("type") == new.get("type") and old.get("load_order") == new.get("load_order")
            if new_enabled and (not old_enabled or not same_entity):
                generation = self._entity_generations.get(slot_id, 0) + 1
                self._entity_generations[slot_id] = generation
                events.append(_event("entity_spawned", frame, _entity_payload(slot_id, new, generation), (f"sprites.slots.{slot_id}",)))
                continue
            if old_enabled and not new_enabled:
                payload = _entity_payload(slot_id, old, self._entity_generations.get(slot_id, 0))
                events.append(_event("entity_despawned", frame, payload, (f"sprites.slots.{slot_id}",), confidence=0.8))
                if old.get("status") in {1, 2, 3} or old.get("health") == 0:
                    events.append(_event("entity_defeated", frame, payload, (f"sprites.slots.{slot_id}",), confidence=0.8))
                continue
            if old_enabled and new_enabled and same_entity:
                old_health, new_health = old.get("health"), new.get("health")
                if isinstance(old_health, int) and isinstance(new_health, int) and new_health < old_health:
                    payload = _entity_payload(slot_id, new, self._entity_generations.get(slot_id, 0))
                    payload.update({"amount": old_health - new_health, "previous_health": old_health, "current_health": new_health})
                    events.append(_event("entity_damaged", frame, payload, (f"sprites.slots.{slot_id}.health",)))

    def _detect_room_flags(self, delta: StateDelta, frame: int, events: list[GameEvent]) -> None:
        path = "map.location.room_status"
        change = delta.get(path)
        if change is not None:
            events.append(_event("room_status_changed", frame, change.as_dict(), (path,), confidence=0.7))


def _event(event_type, frame, data, paths, confidence=1.0) -> GameEvent:
    return GameEvent(event_type, frame, data, tuple(paths), confidence)


def _location_key(state: dict[str, Any]) -> tuple[Any, Any, Any]:
    location = _get(state, "map.location") or {}
    return location.get("is_indoor"), location.get("map_id"), location.get("room")


def _get(target: dict[str, Any], path: str) -> Any:
    current: Any = target
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _entity_payload(slot_id: str, entity: dict[str, Any], generation: int) -> dict[str, Any]:
    return {
        "slot": slot_id,
        "generation": generation,
        "type": entity.get("type"),
        "type_name": entity.get("type_name"),
        "category": entity.get("category"),
        "room": entity.get("room"),
    }
