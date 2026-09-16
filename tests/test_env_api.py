from __future__ import annotations

import numpy as np
import pytest

from zelda_env import GameState, ZeldaEnv
from zelda_env.memory import DEFAULT_SYM_PATH
from zelda_env.utils.symbol_loader import SymbolTable


class FakeBackend:
    platform = "gbc"
    buttons = ("UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT")

    def __init__(self):
        self.memory = bytearray(0x10000)
        self.frame = np.zeros((144, 160, 3), dtype=np.uint8)
        self.pressed = set()
        self.advanced = 0

    def close(self):
        pass

    def press(self, buttons):
        self.pressed = set(buttons)

    def release_all(self):
        self.pressed.clear()

    def advance(self, frames):
        self.advanced += frames
        self.frame[:, :, 0] = self.advanced % 256
        self.memory[0xFF98] = self.advanced % 256  # hLinkPositionX

    def read_u8(self, address):
        return self.memory[address]

    def read_u16(self, address, *, endian="little"):
        return int.from_bytes(self.read_bytes(address, 2), endian)

    def read_bytes(self, address, length):
        return bytes(self.memory[address : address + length])

    def save_state(self):
        return bytes(self.memory) + self.frame.tobytes() + self.advanced.to_bytes(8, "little")

    def load_state(self, data):
        memory_end = 0x10000
        frame_end = memory_end + self.frame.nbytes
        self.memory[:] = data[:memory_end]
        self.frame[:] = np.frombuffer(data[memory_end:frame_end], dtype=np.uint8).reshape(self.frame.shape)
        self.advanced = int.from_bytes(data[frame_end : frame_end + 8], "little")

    def get_frame(self):
        return self.frame.copy()


def _env(**kwargs):
    return ZeldaEnv(emulator=FakeBackend(), **kwargs)


def test_pixel_observation_matches_space():
    env = _env()
    try:
        observation, info = env.reset(seed=3)
        assert observation.shape == (144, 160, 3)
        assert env.observation_space.contains(observation)
        assert info["game_state"]["player"]["x"] == 0
    finally:
        env.close()


def test_read_state_exposes_typed_state_without_changing_info_schema():
    env = _env()
    try:
        _, info = env.reset()
        state = env.read_state()

        assert isinstance(state, GameState)
        assert state.player["x"] == 0
        assert isinstance(info["game_state"], dict)
    finally:
        env.close()


def test_variable_frame_advance():
    env = _env(frame_skip=4)
    try:
        observation, _ = env.reset()
        assert env.observation_space.contains(observation)

        env.step(0)
        assert env.emulator.advanced == 4

        observation, _, _, _, info = env.step(9, num_frames=7)
        assert env.emulator.advanced == 11
        assert env.observation_space.contains(observation)
        assert info["action"] == 9
        assert info["action_name"] == "UP_A"
        assert info["num_frames"] == 7
        assert info["elapsed_steps"] == 2
    finally:
        env.close()


def test_reset_save_load_frame_and_memory_helpers(tmp_path):
    env = _env()
    symbols = SymbolTable.from_file(DEFAULT_SYM_PATH)
    link_x = symbols.resolve("hLinkPositionX")
    try:
        initial_frame, _ = env.reset()
        env.step(0, 5)
        assert env.read_memory("hLinkPositionX") == 5
        assert env.read_memory(link_x, 2) == bytes((5, 0))
        assert np.array_equal(env.get_frame(), env.render())

        state_path = tmp_path / "states" / "checkpoint.state"
        state_data = env.save_state(state_path)
        env.step(0, 3)
        env.load_state(state_path)
        assert env.read_memory("hLinkPositionX") == 5
        assert env.save_state() == state_data

        reset_frame, reset_info = env.reset()
        assert np.array_equal(reset_frame, initial_frame)
        assert reset_info["elapsed_steps"] == 0
    finally:
        env.close()


def test_invalid_num_frames_is_rejected():
    env = _env()
    try:
        env.reset()
        with pytest.raises(ValueError, match="positive integer"):
            env.step(0, 0)
    finally:
        env.close()


def test_reset_can_advance_deterministic_noop_frames():
    env = _env()
    try:
        _, info = env.reset(seed=4, options={"noop_frames": 7})
        assert env.emulator.advanced == 7
        assert info["game_state"]["player"]["x"] == 7
        assert info["reset_noop_frames"] == 7
        assert info["elapsed_steps"] == 0
    finally:
        env.close()
