"""Events and the small default reward derived from state transitions."""

from __future__ import annotations

from typing import Any


EVENT_REWARDS = {
    "new_room_visited": 0.05,
    "player_damaged": -0.02,
    "player_died": -1.0,
    "item_acquired": 0.2,
    "key_acquired": 0.2,
    "instrument_acquired": 2.0,
    "monster_defeated": 0.05,
}


def detect_events(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    visited_rooms: set[tuple[int, int, int]],
) -> list[dict[str, Any]]:
    """Compare two memory-derived states and return JSON-safe events."""

    if previous is None:
        return []
    frame = current["frame"]
    events: list[dict[str, Any]] = []

    old_room, new_room = _room_key(previous), _room_key(current)
    if old_room != new_room:
        events.append(_event("room_changed", frame, previous=old_room, current=new_room))
        if new_room not in visited_rooms:
            visited_rooms.add(new_room)
            events.append(_event("new_room_visited", frame, room=new_room))

    old_health = previous["player"]["health"]
    new_health = current["player"]["health"]
    if new_health < old_health:
        events.append(_event("player_damaged", frame, amount=old_health - new_health))
    elif new_health > old_health:
        events.append(_event("player_healed", frame, amount=new_health - old_health))
    if old_health > 0 and new_health == 0:
        events.append(_event("player_died", frame))

    _detect_inventory(previous, current, frame, events)
    _detect_flags(previous, current, frame, events)
    if old_room == new_room:
        _detect_entities(previous, current, frame, events)
        _detect_monsters(previous, current, frame, events)
    return events


def default_reward(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    events: list[dict[str, Any]],
) -> float:
    """A deliberately small baseline reward; replace it for real tasks."""

    reward = -0.001
    for event in events:
        value = EVENT_REWARDS.get(event["type"], 0.0)
        if event["type"] == "player_damaged":
            value *= event["data"]["amount"]
        reward += value
    return reward


def _detect_inventory(previous, current, frame, events) -> None:
    old, new = previous["inventory"], current["inventory"]
    for slot, (before, after) in enumerate(zip(old["items"], new["items"], strict=True)):
        if before != after:
            event_type = "item_acquired" if after else "item_removed"
            events.append(_event(event_type, frame, slot=slot, previous=before, current=after))
    for key in ("tail_key", "angler_key", "face_key", "bird_key"):
        if not old[key] and new[key]:
            events.append(_event("key_acquired", frame, key=key))
    for index, (before, after) in enumerate(zip(old["instruments"], new["instruments"], strict=True), 1):
        if not before and after:
            events.append(_event("instrument_acquired", frame, instrument=index))
    old_keys = previous["progress"]["small_keys"]
    new_keys = current["progress"]["small_keys"]
    if new_keys > old_keys:
        events.append(_event("small_key_acquired", frame, amount=new_keys - old_keys))


def _detect_flags(previous, current, frame, events) -> None:
    for name, before in previous["event_flags"].items():
        after = current["event_flags"][name]
        if before != after:
            events.append(_event("event_flag_changed", frame, flag=name, previous=before, current=after))


def _detect_entities(previous, current, frame, events) -> None:
    old = {entity["slot"]: entity for entity in previous["entities"]}
    new = {entity["slot"]: entity for entity in current["entities"]}
    for slot in sorted(old.keys() | new.keys()):
        before, after = old.get(slot), new.get(slot)
        if before is None:
            events.append(_event("entity_spawned", frame, entity=after))
        elif after is None:
            events.append(_event("entity_removed", frame, entity=before))
        elif before["type"] != after["type"]:
            events.append(_event("entity_removed", frame, entity=before))
            events.append(_event("entity_spawned", frame, entity=after))
        elif after["health"] < before["health"]:
            events.append(_event(
                "entity_damaged",
                frame,
                slot=slot,
                entity_type=before["type"],
                amount=before["health"] - after["health"],
            ))


def _detect_monsters(previous, current, frame, events) -> None:
    old = {entity["slot"]: entity for entity in previous["monsters"]}
    new = {entity["slot"]: entity for entity in current["monsters"]}
    for slot, before in old.items():
        after = new.get(slot)
        if after is not None and after["type"] != before["type"]:
            continue
        new_health = after["health"] if after else 0
        if new_health < before["health"]:
            events.append(_event(
                "monster_damaged",
                frame,
                slot=slot,
                entity_type=before["type"],
                amount=before["health"] - new_health,
            ))
        if before["health"] > 0 and new_health == 0:
            events.append(_event("monster_defeated", frame, slot=slot, entity_type=before["type"]))


def _room_key(state: dict[str, Any]) -> tuple[int, int, int]:
    room = state["room"]
    return room["is_indoor"], room["map_id"], room["id"]


def _event(event_type: str, frame: int, **data: Any) -> dict[str, Any]:
    return {"type": event_type, "frame": frame, "data": data}
