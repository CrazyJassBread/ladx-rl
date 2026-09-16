"""LADX memory address map and semantic state extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zelda_env.emulator import Emulator
from zelda_env.utils.symbol_loader import SymbolTable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROM_PATH = PROJECT_ROOT / "ladx-disassembly" / "azle.gbc"
DEFAULT_SYM_PATH = PROJECT_ROOT / "ladx-disassembly" / "azle.sym"

INVENTORY_SIZE = 12
ENTITY_SLOTS = 16

# The only maintained game-specific mapping: semantic field -> RGBDS symbol.
ROOM_ADDRESSES = {
    "id": "hMapRoom",
    "map_id": "hMapId",
    "indoor_room": "wIndoorRoom",
    "is_indoor": "wIsIndoor",
    "is_side_scrolling": "hIsSideScrolling",
}
PLAYER_ADDRESSES = {
    "x": "hLinkPositionX",
    "y": "hLinkPositionY",
    "z": "hLinkPositionZ",
    "direction": "hLinkDirection",
    "health": "wHealth",
    "max_hearts": "wMaxHearts",
}
INVENTORY_ADDRESSES = {
    "dungeon_compass": "wHasDungeonCompass",
    "flippers": "wHasFlippers",
    "medicine": "wHasMedicine",
    "seashells": "wSeashellsCount",
    "tail_key": "wHasTailKey",
    "angler_key": "wHasAnglerKey",
    "face_key": "wHasFaceKey",
    "bird_key": "wHasBirdKey",
    "golden_leaves_or_slime_key": "wGoldenLeavesCount",
    "bracelet_level": "wPowerBraceletLevel",
    "shield_level": "wShieldLevel",
    "sword_level": "wSwordLevel",
    "arrows": "wArrowCount",
    "bombs": "wBombCount",
    "magic_powder": "wMagicPowderCount",
}
EVENT_FLAG_ADDRESSES = {
    "room_status": "hRoomStatus",
    "door_event": "wDoorEvent",
    "room_event": "wRoomEvent",
    "room_event_executed": "wRoomEventEffectExecuted",
    "shutter_event_executed": "wShutterDoorEventExecuted",
    "stole_from_shop": "wHasStolenFromShop",
    "tarin": "wTarinFlag",
    "richard_spoken": "wRichardSpokenFlag",
    "bow_wow": "wIsBowWowFollowingLink",
}
ENTITY_TABLES = {
    "status": "wEntitiesStatusTable",
    "type": "wEntitiesTypeTable",
    "x": "wEntitiesPosXTable",
    "y": "wEntitiesPosYTable",
    "z": "wEntitiesPosZTable",
    "health": "wEntitiesHealthTable",
    "direction": "wEntitiesDirectionTable",
    "room_id": "wEntitiesRoomTable",
}


@dataclass(frozen=True)
class GameState:
    """Top-level frozen envelope for one decoded semantic state snapshot.

    The nested dictionaries and lists intentionally retain the existing state
    schema and are not deeply immutable.
    """

    frame: int
    room: dict[str, int]
    player: dict[str, int]
    inventory: dict[str, Any]
    progress: dict[str, int]
    event_flags: dict[str, int]
    entities: list[dict[str, int]]
    monsters: list[dict[str, int]]


class GameMemory:
    """Read LADX state using the declarative symbol tables above."""

    def __init__(self, emulator: Emulator, symbols: SymbolTable) -> None:
        self.emulator = emulator
        self.symbols = symbols

    def read(self, address: int | str, length: int = 1) -> int | bytes:
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise ValueError("length must be a positive integer")
        if isinstance(address, str):
            address = self.symbols.resolve(address)
        if isinstance(address, bool) or not isinstance(address, int):
            raise TypeError("address must be an integer or symbol name")
        if not 0 <= address <= 0xFFFF or address + length > 0x10000:
            raise ValueError("memory range must be within 0x0000..0xFFFF")
        return self.emulator.read_u8(address) if length == 1 else self.emulator.read_bytes(address, length)

    def read_state(self) -> GameState:
        """Return a typed snapshot of the currently decoded semantic state."""

        return GameState(**self._decode_state())

    def game_state(self) -> dict[str, Any]:
        """Return the legacy dictionary-shaped semantic state."""

        return self._decode_state()

    def _decode_state(self) -> dict[str, Any]:
        inventory = self._read_fields(INVENTORY_ADDRESSES)
        inventory["items"] = list(self._read_symbol_bytes("wInventoryItems", INVENTORY_SIZE))
        inventory["instruments"] = [self._read_symbol(f"wHasInstrument{index}") for index in range(1, 9)]

        entities = self._read_entities()
        return {
            "frame": self._read_symbol("hFrameCounter"),
            "room": self._read_fields(ROOM_ADDRESSES),
            "player": self._read_fields(PLAYER_ADDRESSES),
            "inventory": inventory,
            "progress": {
                "rupees": self._read_symbol("wRupeeCountHigh") * 100 + self._read_symbol("wRupeeCountLow"),
                "heart_pieces": self._read_symbol("wHeartPiecesCount"),
                "small_keys": self._read_symbol("wSmallKeysCount"),
            },
            "event_flags": self._read_fields(EVENT_FLAG_ADDRESSES),
            "entities": entities,
            # A useful best-effort view. `entities` remains authoritative because
            # some NPCs and special objects also have a non-zero health byte.
            "monsters": [entity for entity in entities if entity["health"] > 0],
        }

    def _read_fields(self, fields: dict[str, str]) -> dict[str, int]:
        return {name: self._read_symbol(symbol) for name, symbol in fields.items()}

    def _read_entities(self) -> list[dict[str, int]]:
        bases = {field: self.symbols.resolve(symbol) for field, symbol in ENTITY_TABLES.items()}
        entities = []
        for slot in range(ENTITY_SLOTS):
            status = self.emulator.read_u8(bases["status"] + slot)
            if status == 0:
                continue
            entity = {"slot": slot}
            entity.update({field: self.emulator.read_u8(base + slot) for field, base in bases.items()})
            entities.append(entity)
        return entities

    def _read_symbol(self, symbol: str) -> int:
        return self.emulator.read_u8(self.symbols.resolve(symbol))

    def _read_symbol_bytes(self, symbol: str, length: int) -> bytes:
        return self.emulator.read_bytes(self.symbols.resolve(symbol), length)
