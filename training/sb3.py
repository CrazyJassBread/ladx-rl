"""Stable-Baselines3 helpers kept separate from the core environment."""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial

from training.env_factory import make_task_env
from training.experiments import ExperimentSuite


def resolve_device(requested: str):
    """Resolve an SB3 device, failing clearly when requested CUDA is unavailable."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install training dependencies: pip install -e '.[train]'") from exc

    requested = requested.strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid training device: {requested!r}") from exc

    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA GPU. "
            "Check the NVIDIA driver and install a CUDA-enabled PyTorch build, "
            "or use --device cpu."
        )

    index = device.index if device.index is not None else torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {index} was requested, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are visible."
        )
    return torch.device(f"cuda:{index}")


def device_info(device) -> dict[str, object]:
    """Return reproducibility metadata for the selected PyTorch device."""

    import torch

    info: dict[str, object] = {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        info["gpu_name"] = torch.cuda.get_device_name(index)
    return info


def rollout_layout(
    base_n_steps: int,
    instance_count: int,
    requested_num_envs: int | None,
) -> tuple[int, int]:
    """Return worker count and per-worker steps without changing rollout size."""

    if base_n_steps < 1 or instance_count < 1:
        raise ValueError("n_steps and instance count must be positive")
    num_envs = instance_count if requested_num_envs is None else requested_num_envs
    if num_envs < instance_count or num_envs % instance_count != 0:
        raise ValueError(
            f"--num-envs must be a positive multiple of the {instance_count} "
            "training instance(s)"
        )

    rollout_size = base_n_steps * instance_count
    if rollout_size % num_envs != 0:
        raise ValueError(
            f"--num-envs={num_envs} cannot preserve the baseline rollout size "
            f"of {rollout_size}; choose a divisor of {rollout_size}"
        )
    return num_envs, rollout_size // num_envs


def make_vec_env(
    suite: ExperimentSuite,
    instance_names: Iterable[str],
    *,
    num_envs: int | None = None,
):
    """Build channel-first, four-frame-stacked task environments."""

    try:
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            SubprocVecEnv,
            VecFrameStack,
            VecTransposeImage,
        )
    except ImportError as exc:
        raise RuntimeError("Install training dependencies: pip install -e '.[train]'") from exc

    names = tuple(instance_names)
    if not names:
        raise ValueError("At least one task instance is required")
    worker_count = len(names) if num_envs is None else num_envs
    if worker_count < len(names) or worker_count % len(names) != 0:
        raise ValueError("num_envs must be a positive multiple of the instance count")

    worker_names = tuple(names[index % len(names)] for index in range(worker_count))
    factories = [partial(_make_monitored_env, suite.instances[name]) for name in worker_names]

    env = SubprocVecEnv(factories) if worker_count > 1 else DummyVecEnv(factories)
    env = VecTransposeImage(env)
    frame_stacks = {suite.instances[name].frame_stack for name in names}
    if len(frame_stacks) != 1:
        raise ValueError("All instances in one vector environment need one frame_stack")
    frame_stack = frame_stacks.pop()
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack, channels_order="first")
    return env


def _make_monitored_env(config):
    """Top-level process target so spawn/forkserver workers can import it."""

    from stable_baselines3.common.monitor import Monitor

    return Monitor(make_task_env(config), info_keywords=("is_success",))
