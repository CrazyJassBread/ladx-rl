import pytest

torch = pytest.importorskip("torch")

from training.sb3 import device_info, resolve_device, rollout_layout


def test_auto_device_falls_back_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    device = resolve_device("auto")

    assert str(device) == "cpu"


def test_explicit_cuda_fails_instead_of_silently_using_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")


def test_cpu_device_metadata_records_runtime_versions(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    info = device_info(torch.device("cpu"))

    assert info["device"] == "cpu"
    assert info["torch_version"] == torch.__version__
    assert info["cuda_available"] is False
    assert "cuda_version" in info
    assert "gpu_name" not in info


def test_parallel_layout_preserves_single_instance_rollout_size():
    assert rollout_layout(512, 1, 8) == (8, 64)


def test_parallel_layout_balances_multiple_instances():
    assert rollout_layout(512, 2, 8) == (8, 128)


@pytest.mark.parametrize("num_envs", [1, 3, 6])
def test_parallel_layout_rejects_unbalanced_or_inexact_worker_counts(num_envs):
    with pytest.raises(ValueError, match="multiple|preserve"):
        rollout_layout(512, 2, num_envs)
