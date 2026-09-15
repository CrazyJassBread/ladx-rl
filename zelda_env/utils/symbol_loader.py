"""Read RGBDS symbol files produced by the LADX build."""

from __future__ import annotations

import re
from pathlib import Path


_SYMBOL_LINE = re.compile(r"^\s*[0-9A-Fa-f]{2}:([0-9A-Fa-f]{4})\s+(.+?)\s*$")


class SymbolTable:
    def __init__(self, addresses: dict[str, int]) -> None:
        self.addresses = addresses
        self._names_by_address: dict[int, list[str]] = {}
        for name, address in addresses.items():
            self._names_by_address.setdefault(address, []).append(name)

    @classmethod
    def from_file(cls, path: str | Path) -> "SymbolTable":
        addresses: dict[str, int] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            match = _SYMBOL_LINE.match(line)
            if match:
                addresses[match.group(2)] = int(match.group(1), 16)
        return cls(addresses)

    def resolve(self, name: str) -> int:
        try:
            return self.addresses[name]
        except KeyError as exc:
            raise KeyError(f"Unknown LADX symbol: {name}") from exc

    def get(self, name: str) -> int | None:
        return self.addresses.get(name)

    def names_at(self, address: int) -> list[str]:
        return self._names_by_address.get(address, []).copy()
