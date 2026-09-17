"""Small helpers shared by tasks with privileged route supervision."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def parse_waypoints(
    values: Iterable[Iterable[int]],
    name: str,
) -> tuple[tuple[int, int], ...]:
    waypoints = tuple(tuple(int(coordinate) for coordinate in value) for value in values)
    if any(len(waypoint) != 2 for waypoint in waypoints):
        raise ValueError(f"{name} entries must be [x, y] pairs")
    return waypoints


def manhattan_distance(player: Mapping[str, Any], target: tuple[int, int]) -> int:
    return abs(int(player["x"]) - target[0]) + abs(int(player["y"]) - target[1])


def remaining_path_distance(
    player: Mapping[str, Any],
    waypoints: tuple[tuple[int, int], ...],
    index: int,
) -> int:
    """Estimate remaining route length without rewarding each annotation."""

    if index >= len(waypoints):
        return 0
    distance = manhattan_distance(player, waypoints[index])
    for start, end in zip(
        waypoints[index:], waypoints[index + 1 :], strict=False
    ):
        distance += abs(start[0] - end[0]) + abs(start[1] - end[1])
    return distance
