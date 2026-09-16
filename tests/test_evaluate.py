from pathlib import Path

import numpy as np

from scripts.evaluate import _current_rgb_frame, _evaluate_instance, _gif_frame_duration_ms


def _observation(value: int) -> np.ndarray:
    observation = np.zeros((1, 12, 2, 3), dtype=np.uint8)
    observation[:, -3:] = value
    return observation


class FakeModel:
    def predict(self, observation, deterministic):
        assert deterministic is True
        return np.array([0]), None


class FakeEnv:
    def __init__(self):
        self.steps = 0

    def reset(self):
        return _observation(1)

    def step(self, action):
        self.steps += 1
        if self.steps == 1:
            return _observation(2), np.array([0.25]), np.array([False]), [{}]
        terminal = _observation(3)[0]
        info = {
            "terminal_observation": terminal,
            "is_success": True,
            "task": {
                "target_slots": [0, 1],
                "targets_remaining": 0,
                "key_collected": True,
            },
        }
        return _observation(9), np.array([1.0]), np.array([True]), [info]


def test_current_rgb_frame_uses_newest_channels_and_copies():
    observation = _observation(7)

    frame = _current_rgb_frame(observation)
    observation[:, -3:] = 0

    assert frame.shape == (2, 3, 3)
    assert np.all(frame == 7)


def test_gif_duration_matches_centisecond_format():
    assert _gif_frame_duration_ms(15.0) == 70


def test_evaluate_records_terminal_frame_instead_of_auto_reset(monkeypatch, tmp_path):
    saved = {}

    def fake_save(frames, path: Path, duration_ms: int):
        saved["values"] = [int(frame[0, 0, 0]) for frame in frames]
        saved["path"] = path
        saved["duration_ms"] = duration_ms

    monkeypatch.setattr("scripts.evaluate._save_gif", fake_save)

    result = _evaluate_instance(
        FakeModel(),
        FakeEnv(),
        "room",
        1,
        gif_dir=tmp_path,
        gif_prefix="A-room",
        record_episodes=1,
        frame_duration_ms=67,
    )

    assert saved["values"] == [1, 2, 3]
    assert saved["path"] == tmp_path / "A-room-episode-001.gif"
    assert saved["duration_ms"] == 67
    assert result["success_rate"] == 1.0
    assert result["gifs"] == [str(saved["path"].resolve())]
