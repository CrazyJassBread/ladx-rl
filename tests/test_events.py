from zelda_env.games.ladx.event_detector import LadxEventDetector
from zelda_env.memory.delta import diff_states


def _state(room=1, health=16, keys=0):
    return {
        "meta": {"frame": 0},
        "map": {"location": {"is_indoor": 0, "map_id": 0, "room": room}},
        "sprites": {"player": {"health": {"current": health}, "inventory": {"items": [0] * 12}}, "slots": {}},
        "progress": {"small_keys": keys, "rupees": 0, "heart_pieces": 0, "instruments": [0] * 8},
    }


def test_detector_maps_state_changes_to_named_events():
    old, new = _state(), _state(room=2, health=12, keys=1)
    detector = LadxEventDetector()
    detector.reset(old)
    events = detector.detect(old, new, diff_states(old, new), frame=4)
    assert {event.type for event in events} >= {"room_changed", "new_room_visited", "player_damaged", "small_key_acquired"}
