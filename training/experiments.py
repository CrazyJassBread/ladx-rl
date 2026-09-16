"""Load and validate the Tail Cave transfer experiment manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "experiments" / "tail_cave_transfer.toml"


@dataclass(frozen=True)
class InstanceConfig:
    name: str
    state_paths: tuple[Path, ...]
    expected_room: tuple[int, int, int]
    noop_frames: tuple[int, int]
    frame_skip: int
    max_episode_steps: int
    action_names: tuple[str, ...]
    task: dict[str, Any]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    train_instances: tuple[str, ...]
    eval_instances: tuple[str, ...]
    total_timesteps: int
    pretrained_from: str | None = None

    @property
    def task_name(self) -> str:
        return self.name.split("/", 1)[0]

    @property
    def variant(self) -> str:
        return self.name.split("/", 1)[1]


@dataclass(frozen=True)
class ExperimentSuite:
    path: Path
    instances: dict[str, InstanceConfig]
    experiments: dict[str, ExperimentConfig]
    ppo: dict[str, Any]

    def experiment(self, name: str) -> ExperimentConfig:
        normalized = name.strip().casefold()
        for experiment_name, experiment in self.experiments.items():
            if experiment_name.casefold() == normalized:
                return experiment
        choices = ", ".join(sorted(self.experiments))
        raise ValueError(f"Unknown experiment {name!r}; choose one of: {choices}")


def load_suite(path: str | Path = DEFAULT_MANIFEST) -> ExperimentSuite:
    manifest_path = Path(path).resolve()
    data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    defaults = data["defaults"]

    instances = {
        name: _parse_instance(name, values, defaults)
        for name, values in data["instances"].items()
    }
    experiments = {
        name: ExperimentConfig(
            name=name,
            description=values["description"],
            train_instances=tuple(values["train"]),
            eval_instances=tuple(values["eval"]),
            total_timesteps=int(values.get("total_timesteps", defaults["total_timesteps"])),
            pretrained_from=values.get("pretrained_from"),
        )
        for name, values in data["experiments"].items()
    }
    for experiment in experiments.values():
        if experiment.name.count("/") != 1 or not all(experiment.name.split("/")):
            raise ValueError(
                f"Experiment {experiment.name!r} must use '<task>/<variant>' naming"
            )
        for instance_name in experiment.train_instances + experiment.eval_instances:
            if instance_name not in instances:
                raise ValueError(
                    f"Experiment {experiment.name} references unknown instance {instance_name}"
                )
        if experiment.pretrained_from and experiment.pretrained_from not in experiments:
            raise ValueError(
                f"Experiment {experiment.name} references unknown pretrained experiment "
                f"{experiment.pretrained_from}"
            )
    return ExperimentSuite(manifest_path, instances, experiments, dict(data["ppo"]))


def _parse_instance(
    name: str,
    values: dict[str, Any],
    defaults: dict[str, Any],
) -> InstanceConfig:
    paths = tuple(_project_path(value) for value in values["states"])
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing state for {name}: {missing[0]}")
    room = tuple(int(value) for value in values["expected_room"])
    noops = tuple(int(value) for value in values.get("noop_frames", [0, 0]))
    if len(room) != 3 or len(noops) != 2:
        raise ValueError(f"Invalid room or noop range for instance {name}")
    return InstanceConfig(
        name=name,
        state_paths=paths,
        expected_room=room,
        noop_frames=noops,
        frame_skip=int(values.get("frame_skip", defaults["frame_skip"])),
        max_episode_steps=int(values.get("max_episode_steps", defaults["max_episode_steps"])),
        action_names=tuple(values.get("action_names", defaults["action_names"])),
        task=dict(values["task"]),
    )


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
