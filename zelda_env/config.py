"""Paths and runtime configuration for the Zelda research environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LadxPaths:
    """Explicit locations of the preserved disassembly and local artifacts."""

    disassembly_root: Path
    rom_path: Path
    sym_path: Path

    @classmethod
    def default(cls, project_root: str | Path = PROJECT_ROOT) -> "LadxPaths":
        root = Path(project_root).resolve()
        disassembly = root / "ladx-disassembly"
        return cls(
            disassembly_root=disassembly,
            rom_path=disassembly / "azle.gbc",
            sym_path=disassembly / "azle.sym",
        )


@dataclass(frozen=True)
class ZeldaEnvConfig:
    """Serializable environment settings shared by scripts and experiments."""

    frame_skip: int = 4
    max_episode_steps: int | None = None
    state_mode: str = "reward"
    include_legacy_aliases: bool = False
    include_state_delta: bool = False
    require_initial_state: bool = False
