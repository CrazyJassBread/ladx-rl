import pytest

from zelda_env.reward_signals import RewardComposer, transition_signals


def test_transition_signals_are_unweighted_facts():
    current = {"player": {"health": 0}}
    events = [{"type": "player_damaged", "data": {"amount": 3}}]

    signals = transition_signals(current, events)

    assert signals == {
        "step": 1.0,
        "damage_taken": 3.0,
        "player_died": 1.0,
    }


def test_reward_composer_uses_only_selected_toml_weights():
    composer = RewardComposer({"step": -0.001, "target_defeated": 1.5})

    reward, terms = composer.compose(
        {"step": 1.0, "damage_taken": 2.0, "target_defeated": 2.0}
    )

    assert reward == pytest.approx(2.999)
    assert terms == {"step": -0.001, "target_defeated": 3.0}


def test_reward_composer_rejects_unknown_configured_signal():
    with pytest.raises(ValueError, match="Unknown reward signal"):
        RewardComposer({"made_up_signal": 1.0})
