"""Raw memory snapshots for discovering state and event addresses."""

from __future__ import annotations

from dataclasses import dataclass

from zelda_env.emulator import Emulator
from zelda_env.utils.symbol_loader import SymbolTable


@dataclass(frozen=True)
class MemorySnapshot:
    start: int
    data: bytes

    @classmethod
    def capture(
        cls,
        emulator: Emulator,
        *,
        start: int = 0xC000,
        length: int = 0x4000,
    ) -> "MemorySnapshot":
        return cls(start, emulator.read_bytes(start, length))

    def diff(
        self,
        other: "MemorySnapshot",
        symbols: SymbolTable | None = None,
    ) -> list[dict[str, object]]:
        """Return changed addresses, optionally annotated with symbol names."""

        if self.start != other.start or len(self.data) != len(other.data):
            raise ValueError("Snapshots must cover the same memory range")
        changes = []
        for offset, (before, after) in enumerate(zip(self.data, other.data, strict=True)):
            if before == after:
                continue
            address = self.start + offset
            changes.append({
                "address": address,
                "before": before,
                "after": after,
                "symbols": symbols.names_at(address) if symbols else [],
            })
        return changes
