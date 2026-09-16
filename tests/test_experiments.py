from training.experiments import load_suite


def test_tail_cave_suite_defines_all_transfer_experiments():
    suite = load_suite()

    assert set(suite.experiments) == {
        "room16_key/fixed",
        "room16_key/reset_jitter",
        "hardhat_transfer/room16_to_room09",
        "kill_all_transfer/room16_room12_to_room03",
        "room09_hardhat/reset_jitter",
        "room15_compass/reset_jitter",
        "room13_switch_chest/reset_jitter",
        "room13_press_switch/curriculum",
        "room13_switch_chest/curriculum_finetune",
    }
    transfer = suite.experiment("HARDHAT_TRANSFER/ROOM16_TO_ROOM09")
    assert transfer.train_instances == ("room16_hardhats",)
    assert transfer.eval_instances == ("room09_hardhat_eval",)
    fine_tune = suite.experiment("room09_hardhat/reset_jitter")
    assert fine_tune.pretrained_from == "hardhat_transfer/room16_to_room09"
    compass = suite.experiment("room15_compass/reset_jitter")
    assert compass.task_name == "room15_compass"
    assert compass.variant == "reset_jitter"
    assert compass.train_instances == ("room15_compass_train",)
    assert compass.eval_instances == ("room15_compass_eval",)
    switch_chest = suite.experiment("room13_switch_chest/reset_jitter")
    assert switch_chest.train_instances == ("room13_switch_chest_train",)
    assert switch_chest.eval_instances == ("room13_switch_chest_eval",)
    curriculum = suite.experiment("room13_switch_chest/curriculum_finetune")
    assert curriculum.pretrained_from == "room13_press_switch/curriculum"


def test_pixel_policy_actions_exclude_menu_buttons():
    suite = load_suite()
    actions = suite.instances["room16_key_fixed"].action_names

    assert "START" not in actions
    assert "SELECT" not in actions
    assert {"UP", "DOWN", "LEFT", "RIGHT", "A", "B"} <= set(actions)


def test_instances_load_reward_weights_from_separate_task_toml():
    suite = load_suite()
    compass = suite.instances["room15_compass_train"]

    assert compass.task_config_path.name == "room15_compass.toml"
    assert compass.task["kind"] == "defeat_and_collect_item"
    assert compass.task["reward"]["target_defeated"] == 1.0
    assert compass.task["reward"]["item_collected"] == 5.0

    switch_chest = suite.instances["room13_switch_chest_train"]
    assert switch_chest.expected_room == (1, 0, 0x13)
    assert switch_chest.max_episode_steps == 1200
    assert switch_chest.task["kind"] == "press_switch_open_chest"
    assert switch_chest.task["reward"]["fell_in_pit"] == -3.0
    assert switch_chest.task["switch_waypoints"][0] == [82, 107]
    assert switch_chest.task["blocking_hazard_types"] == [0x20]
    assert switch_chest.task["chest_waypoints"][-1] == [139, 58]
    assert "LEFT_A" in switch_chest.action_names
    assert "LEFT_B" not in switch_chest.action_names

    press_switch = suite.instances["room13_press_switch_train"]
    assert press_switch.task["success_stage"] == "switch"
    assert press_switch.max_episode_steps == 800
    assert press_switch.action_names == switch_chest.action_names
