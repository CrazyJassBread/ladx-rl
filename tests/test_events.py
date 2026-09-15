from zelda_env.events import default_reward, detect_events


def _state(*, room=1, health=16, item=0, monster_health=2):
    monster = {"slot": 0, "type": 9, "health": monster_health}
    return {
        "frame": 4,
        "room": {"is_indoor": 0, "map_id": 0, "id": room},
        "player": {"health": health},
        "inventory": {
            "items": [item] + [0] * 11,
            "instruments": [0] * 8,
            "tail_key": 0,
            "angler_key": 0,
            "face_key": 0,
            "bird_key": 0,
        },
        "event_flags": {"room_status": 0},
        "progress": {"small_keys": 0},
        "entities": [monster] if monster_health else [],
        "monsters": [monster] if monster_health else [],
    }


def test_events_are_plain_state_transition_rules():
    old = _state()
    new = _state(room=2, health=12, item=1)
    visited = {(0, 0, 1)}

    events = detect_events(old, new, visited)
    event_types = {event["type"] for event in events}

    assert event_types >= {"room_changed", "new_room_visited", "player_damaged", "item_acquired"}
    assert default_reward(old, new, events) != 0


def test_monster_damage_and_defeat_are_detected_in_the_same_room():
    visited = {(0, 0, 1)}
    damaged = detect_events(_state(monster_health=2), _state(monster_health=1), visited)
    defeated = detect_events(_state(monster_health=1), _state(monster_health=0), visited)

    assert any(event["type"] == "monster_damaged" for event in damaged)
    assert any(event["type"] == "monster_defeated" for event in defeated)
    assert any(event["type"] == "entity_damaged" for event in damaged)
    assert any(event["type"] == "entity_removed" for event in defeated)


def test_small_key_increase_is_detected():
    old = _state()
    new = _state()
    new["progress"]["small_keys"] = 1

    events = detect_events(old, new, {(0, 0, 1)})

    assert any(event["type"] == "small_key_acquired" for event in events)
