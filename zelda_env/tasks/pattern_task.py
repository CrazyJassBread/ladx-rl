"""Timing puzzles where several enemies must be frozen on one pattern."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

from zelda_env.tasks.base import EventList, GameState, TaskStep
from zelda_env.tasks.chest_task import DefeatAndCollectItemTask
from zelda_env.tasks.navigation import manhattan_distance


class MatchPatternAndCollectItemTask(DefeatAndCollectItemTask):
    """Freeze all reset-time targets on one valid pattern, then open a chest.

    The Tail Cave Three-of-a-Kind enemy stores its displayed pattern in the
    entity direction byte. A hit changes its entity state to ``2`` and freezes
    that pattern. All four direction values are valid patterns; values 0 and 1
    only select guaranteed Heart and Rupee drops after a successful match.
    """

    def __init__(
        self,
        target_types: Iterable[int],
        *,
        valid_patterns: Iterable[int] = (0, 1, 2, 3),
        frozen_state: int = 2,
        success_stage: str = "item",
        hint_dialog_id: int | None = None,
        reward: Mapping[str, Real],
        **kwargs: Any,
    ) -> None:
        self.valid_patterns = frozenset(int(value) for value in valid_patterns)
        if not self.valid_patterns:
            raise ValueError("valid_patterns cannot be empty")
        self.frozen_state = int(frozen_state)
        if success_stage not in {"pattern", "item"}:
            raise ValueError("success_stage must be 'pattern' or 'item'")
        if hint_dialog_id is not None and not 0 <= int(hint_dialog_id) <= 0xFFFF:
            raise ValueError("hint_dialog_id must be a 16-bit dialog id")
        self.success_stage = success_stage
        self.hint_dialog_id = (
            None if hint_dialog_id is None else int(hint_dialog_id)
        )
        self._best_pattern_match = 0
        self._pattern_attempts = 0
        self._pattern_mismatches = 0
        self._pattern_matches = 0
        self._attempt_resolved = False
        self._owl_hint_seen = False
        super().__init__(target_types, reward=reward, **kwargs)

    def reset(self, state: GameState) -> dict[str, Any]:
        target_entities = [
            entity
            for entity in state["entities"]
            if int(entity["type"]) in self.target_types
        ]
        for field in ("state", "direction", "transition_countdown"):
            if any(field not in entity for entity in target_entities):
                raise ValueError(
                    f"Task {self.task_id!r} requires entity field {field!r}"
                )
        self._best_pattern_match = self._matching_frozen_count(target_entities)
        self._pattern_attempts = 0
        self._pattern_mismatches = 0
        self._pattern_matches = 0
        self._attempt_resolved = False
        self._owl_hint_seen = self._is_hint_dialog(state)
        return super().reset(state)

    def step(
        self,
        previous: GameState,
        current: GameState,
        events: EventList,
    ) -> TaskStep:
        result = super().step(previous, current, events)
        if (
            self.success_stage == "pattern"
            and self._cleared
            and result.info["failure"] is None
        ):
            return self._result(
                current,
                dict(result.info["reward_signals"]),
                success=True,
                failure=None,
            )
        return result

    def _add_task_signals(
        self,
        previous: GameState,
        current: GameState,
        events: EventList,
        signals: dict[str, float],
    ) -> None:
        if not self._owl_hint_seen and self._is_hint_dialog(current):
            self._owl_hint_seen = True
            signals["owl_hint_seen"] = 1.0

        previous_targets = self._target_entities(previous)
        current_targets = self._target_entities(current)
        newly_frozen = [
            slot
            for slot, entity in current_targets.items()
            if int(entity["state"]) == self.frozen_state
            and int(previous_targets.get(slot, {}).get("state", -1))
            != self.frozen_state
        ]
        if newly_frozen:
            signals["target_frozen"] = float(len(newly_frozen))

        current_match = self._matching_frozen_count(current_targets.values())
        if current_match > self._best_pattern_match:
            signals["pattern_match_progress"] = float(
                current_match - self._best_pattern_match
            )
            self._best_pattern_match = current_match

        all_frozen = (
            len(current_targets) == len(self._targets)
            and all(
                int(entity["state"]) == self.frozen_state
                for entity in current_targets.values()
            )
        )
        if not all_frozen:
            self._attempt_resolved = False
            return
        if self._attempt_resolved:
            return

        self._attempt_resolved = True
        self._pattern_attempts += 1
        patterns = {int(entity["direction"]) for entity in current_targets.values()}
        if len(patterns) == 1 and next(iter(patterns)) in self.valid_patterns:
            self._pattern_matches += 1
            signals["pattern_matched"] = 1.0
        else:
            self._pattern_mismatches += 1
            signals["pattern_mismatch"] = 1.0

    def _add_target_approach_signal(
        self,
        previous: GameState,
        current: GameState,
        signals: dict[str, float],
    ) -> None:
        # Once a target is frozen, pursuing it is no longer useful. Direct the
        # potential toward the nearest still-cycling target instead.
        candidates = [
            entity
            for entity in current["entities"]
            if self._targets.get(int(entity["slot"])) == int(entity["type"])
            and int(entity["state"]) != self.frozen_state
        ]
        if not candidates:
            return
        target = min(
            candidates,
            key=lambda entity: manhattan_distance(
                previous["player"], (int(entity["x"]), int(entity["y"]))
            ),
        )
        target_position = int(target["x"]), int(target["y"])
        progress = manhattan_distance(
            previous["player"], target_position
        ) - manhattan_distance(current["player"], target_position)
        if progress:
            signals["target_approach"] = float(progress)

    def _phase(self, state: GameState) -> str:
        if self.success_stage == "pattern" and self._cleared:
            return "complete"
        if not self._cleared and self._frozen_targets(state):
            return "match_pattern"
        return super()._phase(state)

    def _extra_info(self, state: GameState) -> dict[str, Any]:
        frozen = self._frozen_targets(state)
        return {
            **super()._extra_info(state),
            "success_stage": self.success_stage,
            "valid_patterns": sorted(self.valid_patterns),
            "frozen_targets": len(frozen),
            "frozen_patterns": [
                {
                    "slot": int(entity["slot"]),
                    "pattern": int(entity["direction"]),
                    "countdown": int(entity["transition_countdown"]),
                }
                for entity in frozen
            ],
            "current_pattern_match": self._matching_frozen_count(frozen),
            "best_pattern_match": self._best_pattern_match,
            "pattern_attempts": self._pattern_attempts,
            "pattern_mismatches": self._pattern_mismatches,
            "pattern_matches": self._pattern_matches,
            "owl_hint_seen": self._owl_hint_seen,
            "hint_dialog_id": self.hint_dialog_id,
        }

    def _target_entities(self, state: GameState) -> dict[int, dict[str, Any]]:
        return {
            int(entity["slot"]): entity
            for entity in state["entities"]
            if self._targets.get(int(entity["slot"])) == int(entity["type"])
        }

    def _frozen_targets(self, state: GameState) -> list[dict[str, Any]]:
        return [
            entity
            for entity in self._target_entities(state).values()
            if int(entity["state"]) == self.frozen_state
        ]

    def _matching_frozen_count(
        self, entities: Iterable[dict[str, Any]]
    ) -> int:
        counts = Counter(
            int(entity["direction"])
            for entity in entities
            if int(entity["state"]) == self.frozen_state
            and int(entity["direction"]) in self.valid_patterns
        )
        return max(counts.values(), default=0)

    def _is_hint_dialog(self, state: GameState) -> bool:
        if self.hint_dialog_id is None:
            return False
        dialog = state.get("dialog", {})
        if not int(dialog.get("state", 0)):
            return False
        dialog_id = (int(dialog.get("index_hi", 0)) << 8) | int(
            dialog.get("index", 0)
        )
        return dialog_id == self.hint_dialog_id
