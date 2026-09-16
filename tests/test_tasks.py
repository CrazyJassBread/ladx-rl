from copy import deepcopy

from zelda_env.tasks.entity_task import (
    DefeatAndCollectItemTask,
    DefeatEntitiesTask,
    KillAndCollectTask,
)


def _state(*, entities=None, keys=0, compass=0, health=24, room=0x16):
    return {
        "room": {"is_indoor": 1, "map_id": 0, "id": room},
        "player": {"health": health},
        "progress": {"small_keys": keys},
        "inventory": {"dungeon_compass": compass},
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


def test_compass_task_waits_for_all_targets_and_chest_item():
    zols = [
        {"slot": slot, "type": 0x9B, "health": 1}
        for slot in range(4)
    ]
    task = DefeatAndCollectItemTask(
        [0x9B],
        item_field="dungeon_compass",
    )
    initial = _state(entities=zols, room=0x15)
    reset_info = task.reset(initial)

    assert reset_info["targets_remaining"] == 4
    assert reset_info["phase"] == "defeat_targets"

    chest_ready = _state(room=0x15)
    cleared = task.step(initial, chest_ready, [])

    assert not cleared.terminated
    assert cleared.info["phase"] == "open_chest"
    assert not cleared.info["chest_seen"]
    assert not cleared.info["item_collected"]

    chest = {"slot": 4, "type": 0x07, "health": 0}
    receiving_state = _state(entities=[chest], compass=1, room=0x15)
    receiving = task.step(chest_ready, receiving_state, [])

    assert not receiving.terminated
    assert receiving.info["phase"] == "receive_item"
    assert receiving.info["item_collected"]
    assert not receiving.info["item_received"]

    collected_state = _state(compass=1, room=0x15)
    collected = task.step(receiving_state, collected_state, [])

    assert collected.terminated
    assert collected.info["success"]
    assert collected.info["phase"] == "complete"
    assert collected.info["item_collected"]
    assert collected.info["item_received"]


def test_compass_alone_does_not_succeed_before_targets_are_defeated():
    zols = [
        {"slot": slot, "type": 0x9B, "health": 1}
        for slot in range(4)
    ]
    task = DefeatAndCollectItemTask(
        [0x9B],
        item_field="dungeon_compass",
    )
    initial = _state(entities=zols, room=0x15)
    task.reset(initial)

    result = task.step(initial, _state(entities=zols, compass=1, room=0x15), [])

    assert not result.terminated
    assert not result.info["success"]
    assert result.info["targets_remaining"] == 4


def test_compass_task_allows_chest_interaction_before_final_target():
    zols = [
        {"slot": slot, "type": 0x9B, "health": 1}
        for slot in range(4)
    ]
    chest = {"slot": 4, "type": 0x07, "health": 0}
    task = DefeatAndCollectItemTask(
        [0x9B],
        item_field="dungeon_compass",
    )
    initial = _state(entities=zols, room=0x15)
    task.reset(initial)

    receiving_state = _state(entities=[*zols, chest], compass=1, room=0x15)
    receiving = task.step(initial, receiving_state, [])
    item_received_state = _state(entities=zols, compass=1, room=0x15)
    received = task.step(receiving_state, item_received_state, [])
    completed = task.step(item_received_state, _state(compass=1, room=0x15), [])

    assert not receiving.terminated
    assert receiving.info["phase"] == "receive_item"
    assert not received.terminated
    assert received.info["phase"] == "defeat_targets"
    assert completed.terminated
    assert completed.info["success"]
