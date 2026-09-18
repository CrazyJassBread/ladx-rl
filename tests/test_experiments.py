from training.experiments import load_suite


def test_tail_cave_suite_defines_all_transfer_experiments():
    suite = load_suite()

    assert set(suite.experiments) == {
        "room16_key/fixed",
        "room16_key/reset_jitter",
        "hardhat_transfer/room16_to_room09",
        "kill_all_transfer/room16_room12_to_room03",
        "room12_keese_exit/reset_jitter",
        "room0d_moldorm_rupees/reset_jitter",
        "room0d_moldorm_rupees/chest_route",
        "room0d_moldorm_rupees/chest_route_finetune",
        "room0d_moldorm_rupees/safe_route",
        "room0d_moldorm_rupees/safe_route_finetune",
        "room0a_three_of_a_kind/pattern_curriculum",
        "room0a_three_of_a_kind/full_scratch",
        "room0a_three_of_a_kind/curriculum_finetune",
        "room09_hardhat/reset_jitter",
        "room15_compass/reset_jitter",
        "room13_switch_chest/reset_jitter",
        "room13_press_switch/curriculum",
        "room13_switch_chest/curriculum_finetune",
        "room19_goomba_ladder_exit/reset_jitter",
        "room11_rolling_bones/reset_jitter",
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
    keese_exit = suite.experiment("room12_keese_exit/reset_jitter")
    assert keese_exit.train_instances == ("room12_keese_exit_train",)
    assert keese_exit.eval_instances == ("room12_keese_exit_eval",)
    goomba_exit = suite.experiment("room19_goomba_ladder_exit/reset_jitter")
    assert goomba_exit.train_instances == (
        "room19_goomba_ladder_exit_train",
    )
    assert goomba_exit.eval_instances == (
        "room19_goomba_ladder_exit_eval",
    )
    rolling_bones = suite.experiment("room11_rolling_bones/reset_jitter")
    assert rolling_bones.train_instances == ("room11_rolling_bones_train",)
    assert rolling_bones.eval_instances == ("room11_rolling_bones_eval",)
    moldorm = suite.experiment("room0d_moldorm_rupees/reset_jitter")
    assert moldorm.train_instances == ("room0d_moldorm_rupees_train",)
    assert moldorm.eval_instances == ("room0d_moldorm_rupees_eval",)
    moldorm_route = suite.experiment("room0d_moldorm_rupees/chest_route")
    assert moldorm_route.train_instances == ("room0d_moldorm_rupees_route_train",)
    assert moldorm_route.eval_instances == ("room0d_moldorm_rupees_route_eval",)
    moldorm_finetune = suite.experiment("room0d_moldorm_rupees/chest_route_finetune")
    assert moldorm_finetune.pretrained_from == "room0d_moldorm_rupees/reset_jitter"
    moldorm_safe = suite.experiment("room0d_moldorm_rupees/safe_route")
    assert moldorm_safe.train_instances == ("room0d_moldorm_rupees_safe_train",)
    assert moldorm_safe.eval_instances == ("room0d_moldorm_rupees_safe_eval",)
    moldorm_safe_finetune = suite.experiment(
        "room0d_moldorm_rupees/safe_route_finetune"
    )
    assert moldorm_safe_finetune.pretrained_from == (
        "room0d_moldorm_rupees/reset_jitter"
    )
    pattern = suite.experiment("room0a_three_of_a_kind/pattern_curriculum")
    assert pattern.train_instances == ("room0a_three_of_a_kind_pattern_train",)
    assert pattern.ppo == {"n_steps": 2048, "batch_size": 256}
    full = suite.experiment("room0a_three_of_a_kind/full_scratch")
    assert full.eval_instances == ("room0a_three_of_a_kind_stone_beak_eval",)
    assert full.ppo == {"n_steps": 2048, "batch_size": 256}
    pattern_finetune = suite.experiment(
        "room0a_three_of_a_kind/curriculum_finetune"
    )
    assert pattern_finetune.pretrained_from == (
        "room0a_three_of_a_kind/pattern_curriculum"
    )
    assert pattern_finetune.ppo == {"n_steps": 2048, "batch_size": 256}


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

    keese_exit = suite.instances["room12_keese_exit_train"]
    assert keese_exit.expected_room == (1, 0, 0x12)
    assert keese_exit.max_episode_steps == 1200
    assert keese_exit.task["kind"] == "defeat_and_exit"
    assert keese_exit.task["expected_target_count"] == 4
    assert keese_exit.task["target_room"] == [1, 0, 0x0D]
    assert keese_exit.task["reward"]["destination_reached"] == 5.0

    goomba_exit = suite.instances["room19_goomba_ladder_exit_train"]
    assert goomba_exit.expected_room == (1, 0, 0x19)
    assert goomba_exit.noop_frames == (0, 15)
    assert goomba_exit.max_episode_steps == 1500
    assert goomba_exit.task_config_path.name == (
        "room19_goomba_ladder_exit.toml"
    )
    assert goomba_exit.task["kind"] == "defeat_and_exit"
    assert goomba_exit.task["target_types"] == [0x9F]
    assert goomba_exit.task["expected_target_count"] == 2
    assert goomba_exit.task["target_room"] == [1, 0, 0x03]
    assert goomba_exit.task["reward"]["destination_reached"] == 5.0

    rolling_bones = suite.instances["room11_rolling_bones_train"]
    assert rolling_bones.expected_room == (1, 0, 0x11)
    assert rolling_bones.noop_frames == (0, 31)
    assert rolling_bones.frame_skip == 2
    assert rolling_bones.frame_stack == 8
    assert rolling_bones.max_episode_steps == 1800
    assert rolling_bones.task_config_path.name == "room11_rolling_bones.toml"
    assert rolling_bones.task["kind"] == "rolling_bones"
    assert rolling_bones.task["target_types"] == [0x81]
    assert rolling_bones.task["bar_type"] == 0x82
    assert rolling_bones.task["max_rewarded_bar_dodges"] == 3
    assert rolling_bones.task["reward"]["bar_dodged"] == 0.25

    moldorm = suite.instances["room0d_moldorm_rupees_train"]
    assert moldorm.expected_room == (1, 0, 0x0D)
    assert moldorm.max_episode_steps == 1200
    assert moldorm.task["kind"] == "defeat_and_collect_rupees"
    assert moldorm.task["target_types"] == [0x29]
    assert moldorm.task["expected_target_count"] == 1
    assert moldorm.task["rupee_amount"] == 20

    moldorm_route = suite.instances["room0d_moldorm_rupees_route_train"]
    assert moldorm_route.task_config_path.name == "room0d_moldorm_rupees_route.toml"
    assert moldorm_route.task["chest_waypoints"] == [[136, 48]]
    assert moldorm_route.task["waypoint_tolerance"] == 16
    assert moldorm_route.task["reward"]["route_progress"] == 0.01
    assert moldorm_route.task["reward"]["waypoint_reached"] == 0.5

    moldorm_safe = suite.instances["room0d_moldorm_rupees_safe_train"]
    assert moldorm_safe.noop_frames == (0, 60)
    assert moldorm_safe.task_config_path.name == (
        "room0d_moldorm_rupees_safe_route.toml"
    )
    assert moldorm_safe.task["chest_waypoints"] == [
        [136, 112],
        [136, 72],
        [136, 48],
    ]
    assert moldorm_safe.task["route_progress_mode"] == "remaining_path"
    assert moldorm_safe.task["fail_on_fall"] is True
    assert "waypoint_reached" not in moldorm_safe.task["reward"]
    assert moldorm_safe.task["reward"]["target_damaged"] == 0.25

    pattern = suite.instances["room0a_three_of_a_kind_pattern_train"]
    assert pattern.expected_room == (1, 0, 0x0A)
    assert pattern.noop_frames == (0, 63)
    assert pattern.task["kind"] == "match_pattern_and_collect_item"
    assert pattern.task["target_types"] == [0x90]
    assert pattern.task["valid_patterns"] == [0, 1, 2, 3]
    assert pattern.task["success_stage"] == "pattern"
    assert pattern.frame_skip == 2
    assert pattern.frame_stack == 8
    assert pattern.max_episode_steps == 1000
    assert pattern.task_config_path.name == (
        "room0a_three_of_a_kind_pattern_guided.toml"
    )
    assert pattern.task["anchor_shaping"] is True
    assert pattern.task["terminate_on_prefix_mismatch"] is True
    assert pattern.task["reward"]["pattern_consistent_freeze"] == 0.5
    assert pattern.task["reward"]["pattern_prefix_mismatch"] == -0.1

    pattern_eval = suite.instances["room0a_three_of_a_kind_pattern_eval"]
    assert pattern_eval.frame_skip == 2
    assert pattern_eval.frame_stack == 8
    assert "anchor_shaping" not in pattern_eval.task
    assert pattern_eval.task["reward"]["pattern_mismatch"] == -1.0

    full_pattern = suite.instances["room0a_three_of_a_kind_stone_beak_eval"]
    assert full_pattern.noop_frames == (64, 127)
    assert full_pattern.frame_skip == 2
    assert full_pattern.frame_stack == 8
    assert full_pattern.task["success_stage"] == "item"
    assert full_pattern.task["item_field"] == "dungeon_stone_beak"
    assert full_pattern.task["hint_dialog_id"] == 0x280
    assert full_pattern.task["chest_waypoints"] == [[136, 48]]
    assert full_pattern.task["route_progress_mode"] == "remaining_path"
