"""Small interface shared by Zelda task implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


GameState = dict[str, Any]
EventList = list[dict[str, Any]]


@dataclass(frozen=True)
class TaskStep:
    reward: float
    terminated: bool
    info: dict[str, Any]


class Task(ABC):
    """Stateful reward and success logic for one episode."""

    @abstractmethod
    def reset(self, state: GameState) -> dict[str, Any]:
        """Initialize episode-local task state and return diagnostics."""

    @abstractmethod
    def step(self, previous: GameState, current: GameState, events: EventList) -> TaskStep:
        """Evaluate one environment transition."""
