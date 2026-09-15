from zelda_env.actions import buttons_for_action


def test_fixed_action_map():
    assert buttons_for_action(0) == frozenset()
    assert buttons_for_action(9) == frozenset({"UP", "A"})
