"""Stable-Baselines3 helpers kept separate from the core environment."""

from __future__ import annotations

from collections.abc import Iterable

from training.env_factory import make_task_env
from training.experiments import ExperimentSuite


def make_vec_env(suite: ExperimentSuite, instance_names: Iterable[str]):
    """Build channel-first, four-frame-stacked task environments."""

    try:
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
    except ImportError as exc:
        raise RuntimeError("Install training dependencies: pip install -e '.[train]'") from exc

    configs = [suite.instances[name] for name in instance_names]
    if not configs:
        raise ValueError("At least one task instance is required")

    def factory(config):
        return lambda: Monitor(make_task_env(config), info_keywords=("is_success",))

    env = DummyVecEnv([factory(config) for config in configs])
    env = VecTransposeImage(env)
    frame_stack = int(suite.ppo.get("frame_stack", 1))
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack, channels_order="first")
    return env
