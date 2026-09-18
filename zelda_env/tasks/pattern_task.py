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
        anchor_shaping: bool = False,
        terminate_on_prefix_mismatch: bool = False,
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
        if not isinstance(anchor_shaping, bool):
            raise TypeError("anchor_shaping must be a boolean")
        if not isinstance(terminate_on_prefix_mismatch, bool):
            raise TypeError("terminate_on_prefix_mismatch must be a boolean")
        if terminate_on_prefix_mismatch and not anchor_shaping:
            raise ValueError(
                "terminate_on_prefix_mismatch requires anchor_shaping"
            )
        if hint_dialog_id is not None and not 0 <= int(hint_dialog_id) <= 0xFFFF:
            raise ValueError("hint_dialog_id must be a 16-bit dialog id")
        self.success_stage = success_stage
        self.anchor_shaping = anchor_shaping
        self.terminate_on_prefix_mismatch = terminate_on_prefix_mismatch
        self.hint_dialog_id = (
            None if hint_dialog_id is None else int(hint_dialog_id)
        )
        self._best_pattern_match = 0
        self._pattern_attempts = 0
        self._pattern_mismatches = 0
        self._pattern_matches = 0
        self._attempt_resolved = False
        self._prefix_mismatch_seen = False
        self._pattern_anchor_sets = 0
        self._pattern_consistent_freezes = 0
        self._pattern_prefix_mismatches = 0
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
        initial_frozen = [
            entity
            for entity in target_entities
            if int(entity["state"]) == self.frozen_state
        ]
        self._prefix_mismatch_seen = len(
            {int(entity["direction"]) for entity in initial_frozen}
        ) > 1
        self._pattern_anchor_sets = 0
        self._pattern_consistent_freezes = 0
        self._pattern_prefix_mismatches = 0
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
            self.terminate_on_prefix_mismatch
            and "pattern_prefix_mismatch" in result.info["reward_signals"]
            and result.info["failure"] is None
        ):
            return self._result(
                current,
                dict(result.info["reward_signals"]),
                success=False,
                failure="pattern_prefix_mismatch",
            )
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

        current_frozen = [
            entity
            for entity in current_targets.values()
            if int(entity["state"]) == self.frozen_state
        ]
        if not current_frozen:
            self._prefix_mismatch_seen = False

        # Always expose anchor diagnostics, including in the unshaped
        # evaluation task. Only the guided curriculum assigns them weights or
        # terminates on a prefix mismatch.
        self._add_anchor_signals(
            previous_targets,
            current_targets,
            newly_frozen,
            signals,
        )
        if self.anchor_shaping:
            self._best_pattern_match = max(
                self._best_pattern_match,
                self._matching_frozen_count(current_targets.values()),
            )
        else:
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
            "anchor_shaping": self.anchor_shaping,
            "terminate_on_prefix_mismatch": self.terminate_on_prefix_mismatch,
            "pattern_anchor_sets": self._pattern_anchor_sets,
            "pattern_consistent_freezes": self._pattern_consistent_freezes,
            "pattern_prefix_mismatches": self._pattern_prefix_mismatches,
            "owl_hint_seen": self._owl_hint_seen,
            "hint_dialog_id": self.hint_dialog_id,
        }

    def _add_anchor_signals(
        self,
        previous_targets: Mapping[int, dict[str, Any]],
        current_targets: Mapping[int, dict[str, Any]],
        newly_frozen: Iterable[int],
        signals: dict[str, float],
    ) -> None:
        """Reward only freezes that preserve the first frozen pattern."""

        prefix = [
            int(entity["direction"])
            for entity in previous_targets.values()
            if int(entity["state"]) == self.frozen_state
            and int(entity["direction"]) in self.valid_patterns
        ]
        for slot in sorted(newly_frozen):
            pattern = int(current_targets[slot]["direction"])
            if not prefix:
                if pattern in self.valid_patterns:
                    signals["pattern_anchor_set"] = (
                        signals.get("pattern_anchor_set", 0.0) + 1.0
                    )
                    self._pattern_anchor_sets += 1
                prefix.append(pattern)
                continue

            prefix_is_consistent = len(set(prefix)) == 1
            if (
                prefix_is_consistent
                and pattern in self.valid_patterns
                and pattern == prefix[0]
            ):
                signals["pattern_consistent_freeze"] = (
                    signals.get("pattern_consistent_freeze", 0.0) + 1.0
                )
                self._pattern_consistent_freezes += 1
            elif prefix_is_consistent and not self._prefix_mismatch_seen:
                signals["pattern_prefix_mismatch"] = 1.0
                self._pattern_prefix_mismatches += 1
                self._prefix_mismatch_seen = True
            prefix.append(pattern)

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
