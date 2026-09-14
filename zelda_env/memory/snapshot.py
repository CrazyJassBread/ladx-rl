"""Raw memory snapshots used by reverse-engineering tools."""

from __future__ import annotations

from dataclasses import dataclass

from zelda_env.backends.base import EmulatorBackend


@dataclass(frozen=True)
class MemorySnapshot:
    start: int
    data: bytes

    @classmethod
    def capture(cls, backend: EmulatorBackend, start: int = 0xC000, length: int = 0x4000) -> "MemorySnapshot":
        return cls(start=start, data=backend.read_bytes(start, length))

    def changed_addresses(self, other: "MemorySnapshot") -> list[tuple[int, int, int]]:
        if self.start != other.start or len(self.data) != len(other.data):
            raise ValueError("Snapshots must cover the same address range")
        return [
            (self.start + offset, before, after)
            for offset, (before, after) in enumerate(zip(self.data, other.data, strict=True))
            if before != after
        ]
