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


def test_pixel_policy_actions_exclude_menu_buttons():
    suite = load_suite()
    actions = suite.instances["room16_key_fixed"].action_names

    assert "START" not in actions
    assert "SELECT" not in actions
    assert {"UP", "DOWN", "LEFT", "RIGHT", "A", "B"} <= set(actions)
