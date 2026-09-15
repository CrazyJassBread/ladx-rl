from copy import deepcopy

from zelda_env.tasks.entity_task import DefeatEntitiesTask, KillAndCollectTask


def _state(*, entities=None, keys=0, health=24, room=0x16):
    return {
        "room": {"is_indoor": 1, "map_id": 0, "id": room},
        "player": {"health": health},
        "progress": {"small_keys": keys},
        "entities": deepcopy(entities or []),
    }


def _hardhats():
    return [
        {"slot": 1, "type": 0x20, "health": 4},
        {"slot": 2, "type": 0x20, "health": 4},
    ]


def test_entity_disappearance_counts_as_defeat_even_without_zero_health():
    task = DefeatEntitiesTask([0x20])
    initial = _state(entities=_hardhats())
    task.reset(initial)

    one_left = _state(entities=[_hardhats()[1]])
    first = task.step(initial, one_left, [])
    finished = task.step(one_left, _state(), [])

    assert first.info["targets_remaining"] == 1
    assert not first.terminated
    assert finished.info["success"]
    assert finished.terminated


def test_key_task_waits_for_collection_after_targets_are_removed():
    task = KillAndCollectTask([0x20], drop_type=0x30)
    initial = _state(entities=_hardhats())
    task.reset(initial)

    key_visible = _state(entities=[{"slot": 1, "type": 0x30, "health": 0}])
    dropped = task.step(initial, key_visible, [])
    collected = task.step(key_visible, _state(keys=1), [])

    assert dropped.info["phase"] == "collect_key"
    assert dropped.info["drop_seen"]
    assert not dropped.terminated
    assert collected.info["key_collected"]
    assert collected.info["success"]
    assert collected.terminated


def test_leaving_the_start_room_is_a_failure():
    task = DefeatEntitiesTask([0x20])
    initial = _state(entities=_hardhats())
    task.reset(initial)

    result = task.step(initial, _state(entities=_hardhats(), room=0x15), [])

    assert result.info["failure"] == "left_task_room"
    assert result.info["targets_remaining"] == 2
    assert result.reward < 0
    assert result.terminated
