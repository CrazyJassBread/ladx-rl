from pathlib import Path

import pytest

from zelda_env import ZeldaEnv


@pytest.mark.rom
def test_real_rom_state_and_deterministic_replay():
    state_path = Path("save_states/azle.gbc.start.state")
    env = ZeldaEnv(initial_state_path=state_path, max_episode_steps=20)
    try:
        env.reset(seed=7)
        initial_state = env.save_state()
        first = _semantic_rollout(env)

        env.load_state(initial_state)
        second = _semantic_rollout(env)
    finally:
        env.close()

    assert first == [
        (1, 16, 0xA3, 110, 80, 24),
        (1, 16, 0xA3, 110, 80, 24),
        (1, 16, 0xA3, 110, 77, 24),
        (1, 16, 0xA3, 110, 73, 24),
        (1, 16, 0xA3, 110, 72, 24),
        (1, 16, 0xA3, 110, 72, 24),
    ]
    assert first == second


def _semantic_rollout(env):
    trace = [_selected_semantics(env.read_state())]
    for action in (0, 1, 1, 5, 0):
        env.step(action)
        trace.append(_selected_semantics(env.read_state()))
    return trace


def _selected_semantics(state):
    room = state.room
    player = state.player
    return (
        room["is_indoor"],
        room["map_id"],
        room["id"],
        player["x"],
        player["y"],
        player["health"],
    )
