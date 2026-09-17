"""Shared player-state checks for room hazards."""

from __future__ import annotations

from zelda_env.tasks.base import GameState


PIT_GROUND_STATUS = 0x07
FALLING_DOWN_MOTION_STATE = 0x06


def touching_pit(state: GameState) -> bool:
    player = state["player"]
    return (
        int(player["ground_status"]) == PIT_GROUND_STATUS
        or int(player["pit_slipping_counter"]) > 0
    )
