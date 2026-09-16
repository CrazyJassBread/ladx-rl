from copy import deepcopy

import pytest

from zelda_env.tasks.chest_task import (
    DefeatAndCollectItemTask,
    DefeatAndCollectRupeesTask,
)
from zelda_env.tasks.entity_task import DefeatEntitiesTask
from zelda_env.tasks.exit_task import DefeatAndExitTask
from zelda_env.tasks.key_task import KillAndCollectTask
from zelda_env.tasks.switch_task import PressSwitchOpenChestTask


BASE_REWARD = {
    "step": -0.001,
    "damage_taken": -0.02,
    "player_died": -2.0,
    "premature_room_exit": -1.0,
    "target_defeated": 1.0,
    "all_targets_cleared": 0.5,
}
KEY_REWARD = {
    **BASE_REWARD,
    "key_drop_seen": 0.2,
    "key_collected": 5.0,
}
ITEM_REWARD = {
    **BASE_REWARD,
    "chest_revealed": 0.2,
    "item_collected": 5.0,
}
SWITCH_CHEST_REWARD = {
    "step": -0.001,
    "damage_taken": -0.02,
    "player_died": -2.0,
    "premature_room_exit": -1.0,
    "pit_contact": -0.25,
    "fell_in_pit": -2.0,
    "switch_pressed": 1.0,
    "chest_revealed": 0.5,
    "chest_opened": 5.0,
}


def _state(
    *,
    entities=None,
    keys=0,
    compass=0,
    health=24,
    room=0x16,
    switch=0,
    motion=0,
    ground=0,
    pit_counter=0,
    switch_hold=0,
    x=82,
    y=127,
    room_event_executed=0,
    rupees=0,
):
    return {
        "room": {"is_indoor": 1, "map_id": 0, "id": room},
        "player": {
            "health": health,
            "motion_state": motion,
            "ground_status": ground,
            "pit_slipping_counter": pit_counter,
            "x": x,
            "y": y,
        },
        "progress": {"small_keys": keys, "rupees": rupees},
        "inventory": {"dungeon_compass": compass},
        "event_flags": {
            "switch_button_pressed": switch,
            "switch_button_hold_frames": switch_hold,
            "room_event_executed": room_event_executed,
        },
        "entities": deepcopy(entities or []),
    }


def _hardhats():
    return [
        {"slot": 1, "type": 0x20, "health": 4},
        {"slot": 2, "type": 0x20, "health": 4},
    ]


def test_entity_disappearance_counts_as_defeat_even_without_zero_health():
    task = DefeatEntitiesTask([0x20], reward=BASE_REWARD)
    initial = _state(entities=_hardhats())
    task.reset(initial)

    one_left = _state(entities=[_hardhats()[1]])
    first = task.step(initial, one_left, [])
    finished = task.step(one_left, _state(), [])

    assert first.info["targets_remaining"] == 1
    assert first.info["reward_signals"]["target_defeated"] == 1
    assert first.info["reward_terms"]["target_defeated"] == 1.0
    assert not first.terminated
    assert finished.info["success"]
    assert finished.terminated


def test_key_task_waits_for_collection_after_targets_are_removed():
    task = KillAndCollectTask([0x20], drop_type=0x30, reward=KEY_REWARD)
    initial = _state(entities=_hardhats())
    task.reset(initial)

    key_visible = _state(entities=[{"slot": 1, "type": 0x30, "health": 0}])
    dropped = task.step(initial, key_visible, [])
    collected = task.step(key_visible, _state(keys=1), [])

    assert dropped.info["phase"] == "collect_key"
    assert dropped.info["drop_seen"]
    assert dropped.info["reward_terms"]["key_drop_seen"] == 0.2
    assert not dropped.terminated
    assert collected.info["key_collected"]
    assert collected.info["reward_terms"]["key_collected"] == 5.0
    assert collected.info["success"]
    assert collected.terminated


def test_leaving_the_start_room_is_a_failure():
    task = DefeatEntitiesTask([0x20], reward=BASE_REWARD)
    initial = _state(entities=_hardhats())
    task.reset(initial)

    result = task.step(initial, _state(entities=_hardhats(), room=0x15), [])

    assert result.info["failure"] == "left_task_room"
    assert result.info["targets_remaining"] == 2
    assert result.reward < 0
    assert result.info["reward_terms"]["premature_room_exit"] == -1.0
    assert result.terminated


def test_defeat_and_exit_requires_all_targets_then_destination_room():
    keese = [
        {"slot": slot, "type": 0x19, "health": 1}
        for slot in range(4)
    ]
    reward = {**BASE_REWARD, "destination_reached": 5.0}
    task = DefeatAndExitTask(
        [0x19],
        target_room=[1, 0, 0x0D],
        expected_target_count=4,
        reward=reward,
    )
    initial = _state(entities=keese, room=0x12)
    reset_info = task.reset(initial)

    assert reset_info["targets_remaining"] == 4
    assert reset_info["target_room"] == [1, 0, 0x0D]

    cleared_state = _state(room=0x12)
    cleared = task.step(initial, cleared_state, [])
    assert not cleared.terminated
    assert cleared.info["phase"] == "exit_room"
    assert cleared.info["reward_terms"]["target_defeated"] == 4.0

    destination = _state(room=0x0D)
    completed = task.step(cleared_state, destination, [])
    assert completed.terminated
    assert completed.info["success"]
    assert completed.info["phase"] == "complete"
    assert completed.info["destination_reached"]
    assert completed.info["reward_terms"]["destination_reached"] == 5.0


def test_defeat_and_exit_rejects_early_or_wrong_room_exit():
    keese = [
        {"slot": slot, "type": 0x19, "health": 1}
        for slot in range(4)
    ]
    reward = {**BASE_REWARD, "destination_reached": 5.0}
    task = DefeatAndExitTask(
        [0x19],
        target_room=[1, 0, 0x0D],
        expected_target_count=4,
        reward=reward,
    )
    initial = _state(entities=keese, room=0x12)
    task.reset(initial)

    early = task.step(initial, _state(entities=keese, room=0x0D), [])
    assert early.terminated
    assert not early.info["success"]
    assert not early.info["destination_reached"]
    assert early.info["failure"] == "left_task_room"

    task.reset(initial)
    cleared_state = _state(room=0x12)
    task.step(initial, cleared_state, [])
    wrong_room = task.step(cleared_state, _state(room=0x13), [])
    assert wrong_room.terminated
    assert not wrong_room.info["success"]
    assert wrong_room.info["failure"] == "left_task_room"


def test_defeat_and_exit_validates_four_reset_time_targets():
    task = DefeatAndExitTask(
        [0x19],
        target_room=[1, 0, 0x0D],
        expected_target_count=4,
        reward={**BASE_REWARD, "destination_reached": 5.0},
    )
    three_keese = [
        {"slot": slot, "type": 0x19, "health": 1}
        for slot in range(3)
    ]

    with pytest.raises(ValueError, match="found 3 targets, expected 4"):
        task.reset(_state(entities=three_keese, room=0x12))


def test_compass_task_waits_for_all_targets_and_chest_item():
    zols = [
        {"slot": slot, "type": 0x9B, "health": 1}
        for slot in range(4)
    ]
    task = DefeatAndCollectItemTask(
        [0x9B],
        item_field="dungeon_compass",
        reward=ITEM_REWARD,
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
    assert receiving.info["reward_terms"]["chest_revealed"] == 0.2
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
        reward=ITEM_REWARD,
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
        reward=ITEM_REWARD,
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


def test_rupee_chest_task_waits_for_twenty_rupees_and_dialog_completion():
    moldorm = {"slot": 0, "type": 0x29, "health": 2}
    reward = {
        **BASE_REWARD,
        "chest_revealed": 0.2,
        "rupees_collected": 4.0,
        "dialog_completed": 1.0,
    }
    task = DefeatAndCollectRupeesTask(
        [0x29],
        expected_target_count=1,
        rupee_amount=20,
        reward=reward,
    )
    initial = _state(entities=[moldorm], room=0x0D, rupees=41)
    reset_info = task.reset(initial)
    assert reset_info["targets_remaining"] == 1
    assert reset_info["required_rupees"] == 20

    cleared_state = _state(room=0x0D, rupees=41)
    cleared = task.step(initial, cleared_state, [])
    assert not cleared.terminated
    assert cleared.info["phase"] == "open_chest"

    chest = {"slot": 0, "type": 0x07, "health": 0}
    dialog_state = _state(entities=[chest], room=0x0D, rupees=61)
    opened = task.step(cleared_state, dialog_state, [])
    assert not opened.terminated
    assert opened.info["phase"] == "finish_dialog"
    assert opened.info["rupees_gained"] == 20
    assert opened.info["reward_terms"]["chest_revealed"] == 0.2
    assert opened.info["reward_terms"]["rupees_collected"] == 4.0
    assert not opened.info["dialog_completed"]

    finished_state = _state(room=0x0D, rupees=61)
    finished = task.step(dialog_state, finished_state, [])
    assert finished.terminated
    assert finished.info["success"]
    assert finished.info["phase"] == "complete"
    assert finished.info["dialog_completed"]
    assert finished.info["reward_terms"]["dialog_completed"] == 1.0


def test_rupee_chest_task_rejects_partial_reward_and_unseen_chest():
    moldorm = {"slot": 0, "type": 0x29, "health": 2}
    reward = {**BASE_REWARD, "rupees_collected": 4.0, "dialog_completed": 1.0}
    initial = _state(entities=[moldorm], room=0x0D, rupees=41)
    cleared_state = _state(room=0x0D, rupees=41)

    partial_task = DefeatAndCollectRupeesTask(
        [0x29], rupee_amount=20, reward=reward
    )
    partial_task.reset(initial)
    partial_task.step(initial, cleared_state, [])
    chest = {"slot": 0, "type": 0x07, "health": 0}
    partial_state = _state(entities=[chest], room=0x0D, rupees=60)
    partial = partial_task.step(cleared_state, partial_state, [])
    assert not partial.terminated
    assert not partial.info["rupees_collected"]

    unseen_task = DefeatAndCollectRupeesTask(
        [0x29], rupee_amount=20, reward=reward
    )
    unseen_task.reset(initial)
    unseen_task.step(initial, cleared_state, [])
    unseen = unseen_task.step(cleared_state, _state(room=0x0D, rupees=61), [])
    assert not unseen.terminated
    assert unseen.info["rupees_collected"]
    assert not unseen.info["dialog_completed"]


def test_switch_chest_task_requires_switch_reveal_and_key_collection():
    task = PressSwitchOpenChestTask(reward=SWITCH_CHEST_REWARD)
    initial = _state(room=0x13, keys=1)
    reset_info = task.reset(initial)

    assert reset_info["phase"] == "press_switch"

    switched_state = _state(room=0x13, keys=1, switch=0x60)
    switched = task.step(initial, switched_state, [])
    assert not switched.terminated
    assert switched.info["phase"] == "wait_for_chest"
    assert switched.info["reward_terms"]["switch_pressed"] == 1.0

    revealed_state = _state(
        room=0x13,
        keys=1,
        switch=0x60,
        room_event_executed=1,
    )
    revealed = task.step(switched_state, revealed_state, [])
    assert not revealed.terminated
    assert revealed.info["phase"] == "open_chest"
    assert revealed.info["reward_terms"]["chest_revealed"] == 0.5

    chest = {"slot": 4, "type": 0x07, "health": 0}
    opened_state = _state(
        room=0x13,
        keys=2,
        switch=0x60,
        room_event_executed=1,
        entities=[chest],
    )
    opened = task.step(revealed_state, opened_state, [])
    assert opened.terminated
    assert opened.info["success"]
    assert opened.info["phase"] == "complete"
    assert opened.info["reward_terms"]["chest_opened"] == 5.0


def test_switch_chest_task_penalizes_pit_contact_and_fails_on_fall():
    task = PressSwitchOpenChestTask(reward=SWITCH_CHEST_REWARD)
    initial = _state(room=0x13, keys=1)
    task.reset(initial)

    slipping_state = _state(room=0x13, keys=1, ground=7, pit_counter=4)
    slipping = task.step(initial, slipping_state, [])
    assert not slipping.terminated
    assert slipping.info["reward_terms"]["pit_contact"] == -0.25

    falling_state = _state(room=0x13, keys=1, motion=6, ground=7, pit_counter=15)
    falling = task.step(slipping_state, falling_state, [])
    assert falling.terminated
    assert not falling.info["success"]
    assert falling.info["failure"] == "fell_in_pit"
    assert falling.info["reward_terms"]["fell_in_pit"] == -2.0


def test_switch_chest_task_rejects_an_already_pressed_switch():
    task = PressSwitchOpenChestTask(reward=SWITCH_CHEST_REWARD)

    with pytest.raises(ValueError, match="floor switch"):
        task.reset(_state(room=0x13, keys=1, switch=0x60))


def test_switch_curriculum_emits_route_and_hold_progress():
    reward = {
        **SWITCH_CHEST_REWARD,
        "route_progress": 0.01,
        "waypoint_reached": 0.25,
        "switch_hold_progress": 0.02,
    }
    task = PressSwitchOpenChestTask(
        reward=reward,
        switch_waypoints=[[82, 107], [36, 107]],
        waypoint_tolerance=8,
        success_stage="switch",
    )
    initial = _state(room=0x13, keys=1, x=82, y=127)
    task.reset(initial)

    waypoint = _state(room=0x13, keys=1, x=82, y=107)
    progressed = task.step(initial, waypoint, [])
    assert progressed.info["waypoint_index"] == 1
    assert progressed.info["route_target"] == [36, 107]
    assert progressed.info["reward_signals"]["route_progress"] == 20
    assert progressed.info["reward_terms"]["waypoint_reached"] == 0.25

    holding = _state(room=0x13, keys=1, x=84, y=59, switch_hold=4)
    held = task.step(waypoint, holding, [])
    assert held.info["reward_signals"]["switch_hold_progress"] == 4
    assert held.info["reward_terms"]["switch_hold_progress"] == 0.08

    switched = _state(
        room=0x13,
        keys=1,
        x=84,
        y=59,
        switch=0x60,
        switch_hold=24,
    )
    completed = task.step(holding, switched, [])
    assert completed.terminated
    assert completed.info["success"]
    assert completed.info["phase"] == "complete"


def test_switch_task_pauses_route_until_blocking_hazard_is_removed():
    reward = {**SWITCH_CHEST_REWARD, "blocking_hazard_removed": 1.0}
    hardhat = {"slot": 0, "type": 0x20, "health": 4}
    task = PressSwitchOpenChestTask(
        reward=reward,
        blocking_hazard_types=[0x20],
        post_hazard_waypoints=[[84, 59]],
    )
    initial = _state(room=0x13, keys=1, entities=[hardhat], x=84, y=27)
    task.reset(initial)

    waiting = task.step(initial, initial, [])
    assert waiting.info["phase"] == "clear_blocking_hazard"
    assert waiting.info["route_target"] is None

    removed_state = _state(room=0x13, keys=1, x=84, y=27)
    removed = task.step(initial, removed_state, [])
    assert removed.info["blocking_hazards_remaining"] == 0
    assert removed.info["route_target"] == [84, 59]
    assert removed.info["reward_terms"]["blocking_hazard_removed"] == 1.0
