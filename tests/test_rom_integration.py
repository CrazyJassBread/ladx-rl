from pathlib import Path

import pytest

from zelda_env import ZeldaEnv


@pytest.mark.rom
def test_real_rom_state_and_deterministic_replay():
    state_path = Path("save_states/azle.gbc.start.state")
    env = ZeldaEnv(initial_state_path=state_path, max_episode_steps=20)
    try:
        first = _rollout(env)
        second = _rollout(env)
    finally:
        env.close()
    assert first == second


def _rollout(env):
    observation, info = env.reset(seed=7)
    trace = [(observation.tobytes(), info["game_state"]["room"])]
    for action in (0, 1, 1, 5, 0):
        observation, reward, terminated, truncated, info = env.step(action)
        trace.append((observation.tobytes(), info["game_state"]["room"], reward))
    return trace
