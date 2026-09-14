"""Rewards derived from named game events, not emulator addresses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_EVENT_WEIGHTS = {
    "new_room_visited": 0.05,
    "player_damaged": -0.02,
    "player_died": -1.0,
    "small_key_acquired": 0.2,
    "heart_piece_acquired": 0.5,
    "instrument_acquired": 2.0,
    "item_acquired": 0.2,
    "entity_defeated": 0.05,
}


@dataclass
class EventReward:
    """Configurable baseline reward over ``info['events']``."""

    step_penalty: float = -0.001
    event_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_EVENT_WEIGHTS))

    def reset(self) -> None:
        pass

    def __call__(self, prev_info, info, action):
        reward = self.step_penalty
        terms: dict[str, float] = {"step": self.step_penalty}
        for event in info.get("events", []):
            event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
            weight = self.event_weights.get(str(event_type), 0.0)
            if not weight:
                continue
            amount = 1
            if isinstance(event, dict):
                data = event.get("data", {})
                if event_type == "player_damaged":
                    amount = data.get("amount", 1)
            value = weight * amount
            terms[str(event_type)] = terms.get(str(event_type), 0.0) + value
            reward += value
        return reward, terms


_DEFAULT_REWARD = EventReward()


def default_progress_reward(
    prev_info: dict[str, Any] | None,
    info: dict[str, Any],
    action: int,
    *,
    step_penalty: float = -0.001,
    new_room_reward: float = 0.05,
    damage_penalty_per_health_unit: float = -0.02,
    death_penalty: float = -1.0,
) -> tuple[float, dict[str, Any]]:
    """Backward-compatible function entry point for the event reward."""
    weights = dict(DEFAULT_EVENT_WEIGHTS)
    weights.update(
        new_room_visited=new_room_reward,
        player_damaged=damage_penalty_per_health_unit,
        player_died=death_penalty,
    )
    return EventReward(step_penalty=step_penalty, event_weights=weights)(prev_info, info, action)
