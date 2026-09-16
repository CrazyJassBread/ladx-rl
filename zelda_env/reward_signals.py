"""Canonical, unweighted reward signals and their weighted composition."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from types import MappingProxyType
from typing import Any


# This is the public vocabulary accepted by task TOML files. Signals describe
# facts; task configs decide whether and how strongly each fact is rewarded.
REWARD_SIGNAL_NAMES = frozenset(
    {
        "step",
        "damage_taken",
        "player_died",
        "premature_room_exit",
        "target_defeated",
        "all_targets_cleared",
        "destination_reached",
        "key_drop_seen",
        "key_collected",
        "switch_pressed",
        "switch_hold_progress",
        "route_progress",
        "waypoint_reached",
        "blocking_hazard_removed",
        "chest_revealed",
        "chest_opened",
        "item_collected",
        "rupees_collected",
        "dialog_completed",
        "pit_contact",
        "fell_in_pit",
    }
)


def transition_signals(
    current: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, float]:
    """Extract task-independent signals from one game transition."""

    signals = {"step": 1.0}
    damage = sum(
        event["data"]["amount"]
        for event in events
        if event["type"] == "player_damaged"
    )
    if damage:
        signals["damage_taken"] = float(damage)
    if current["player"]["health"] == 0:
        signals["player_died"] = 1.0
    return signals


class RewardComposer:
    """Validate TOML weights and combine them with emitted signal values."""

    def __init__(self, weights: Mapping[str, Real]) -> None:
        if not weights:
            raise ValueError("reward configuration cannot be empty")
        unknown = sorted(set(weights) - REWARD_SIGNAL_NAMES)
        if unknown:
            raise ValueError(f"Unknown reward signal: {unknown[0]}")

        normalized: dict[str, float] = {}
        for name, weight in weights.items():
            if isinstance(weight, bool) or not isinstance(weight, Real):
                raise TypeError(f"Reward weight {name!r} must be numeric")
            normalized[name] = float(weight)
        self.weights = MappingProxyType(normalized)

    def compose(
        self,
        signals: Mapping[str, Real],
    ) -> tuple[float, dict[str, float]]:
        unknown = sorted(set(signals) - REWARD_SIGNAL_NAMES)
        if unknown:
            raise ValueError(f"Unknown emitted reward signal: {unknown[0]}")

        terms: dict[str, float] = {}
        for name, weight in self.weights.items():
            value = signals.get(name, 0.0)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Reward signal {name!r} must be numeric")
            if value:
                terms[name] = weight * float(value)
        return sum(terms.values()), terms
