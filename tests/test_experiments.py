from training.experiments import load_suite


def test_tail_cave_suite_defines_the_five_transfer_experiments():
    suite = load_suite()

    assert set(suite.experiments) == {"A", "B", "C", "D", "E"}
    assert suite.experiment("c").train_instances == ("room16_hardhats",)
    assert suite.experiment("C").eval_instances == ("room09_hardhat_eval",)
    assert suite.experiment("E").pretrained_from == "C"


def test_pixel_policy_actions_exclude_menu_buttons():
    suite = load_suite()
    actions = suite.instances["room16_key_fixed"].action_names

    assert "START" not in actions
    assert "SELECT" not in actions
    assert {"UP", "DOWN", "LEFT", "RIGHT", "A", "B"} <= set(actions)
