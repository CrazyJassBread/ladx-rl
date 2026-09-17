import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from zelda_env.memory import GameMemory
from zelda_env.utils.symbol_loader import SymbolTable


class FakeEmulator:
    def __init__(self):
        self.data = bytearray(0x10000)

    def read_u8(self, address):
        return self.data[address]

    def read_bytes(self, address, length):
        return bytes(self.data[address : address + length])


def test_memory_table_builds_simple_game_state():
    root = Path(__file__).parents[1]
    symbols = SymbolTable.from_file(root / "ladx-disassembly/azle.sym")
    emulator = FakeEmulator()
    emulator.data[symbols.resolve("hMapRoom")] = 0x92
    emulator.data[symbols.resolve("hLinkPositionX")] = 42
    emulator.data[symbols.resolve("hLinkPositionY")] = 99
    emulator.data[symbols.resolve("wHealth")] = 0x18
    emulator.data[symbols.resolve("wLinkMotionState")] = 6
    emulator.data[symbols.resolve("wLinkGroundStatus")] = 7
    emulator.data[symbols.resolve("wPitSlippingCounter")] = 12
    emulator.data[symbols.resolve("wInventoryItems")] = 1
    emulator.data[symbols.resolve("wHasTailKey")] = 1
    emulator.data[symbols.resolve("wHasDungeonCompass")] = 1
    emulator.data[symbols.resolve("wHasDungeonStoneSlab")] = 1
    emulator.data[symbols.resolve("wDialogState")] = 4
    emulator.data[symbols.resolve("wDialogIndex")] = 0x80
    emulator.data[symbols.resolve("wDialogIndexHi")] = 2
    emulator.data[symbols.resolve("wRoomEvent")] = 0x42
    emulator.data[symbols.resolve("wSwitchButtonPressed")] = 0x60
    emulator.data[symbols.resolve("wC1CA")] = 17
    emulator.data[symbols.resolve("wEntitiesStatusTable")] = 5
    emulator.data[symbols.resolve("wEntitiesTypeTable")] = 9
    emulator.data[symbols.resolve("wEntitiesPosXTable")] = 80
    emulator.data[symbols.resolve("wEntitiesPosYTable")] = 64
    emulator.data[symbols.resolve("wEntitiesHealthTable")] = 2
    emulator.data[symbols.resolve("wEntitiesStateTable")] = 2
    emulator.data[symbols.resolve("wEntitiesTransitionCountdownTable")] = 37
    emulator.data[symbols.resolve("wEntitiesIgnoreHitsCountdownTable")] = 5

    state = GameMemory(emulator, symbols).game_state()

    assert state["room"]["id"] == 0x92
    assert state["player"] == {
        "x": 42,
        "y": 99,
        "z": 0,
        "direction": 0,
        "health": 0x18,
        "max_hearts": 0,
        "motion_state": 6,
        "ground_status": 7,
        "pit_slipping_counter": 12,
    }
    assert state["inventory"]["items"][0] == 1
    assert state["inventory"]["tail_key"] == 1
    assert state["inventory"]["dungeon_compass"] == 1
    assert state["inventory"]["dungeon_stone_beak"] == 1
    assert state["dialog"]["state"] == 4
    assert state["dialog"]["index"] == 0x80
    assert state["dialog"]["index_hi"] == 2
    assert state["event_flags"]["room_event"] == 0x42
    assert state["event_flags"]["switch_button_pressed"] == 0x60
    assert state["event_flags"]["switch_button_hold_frames"] == 17
    assert state["entities"][0]["type"] == 9
    assert state["monsters"][0] == state["entities"][0]
    assert state["monsters"][0]["x"] == 80
    assert state["monsters"][0]["y"] == 64
    assert state["monsters"][0]["health"] == 2
    assert state["monsters"][0]["state"] == 2
    assert state["monsters"][0]["transition_countdown"] == 37
    assert state["monsters"][0]["ignore_hits_countdown"] == 5


def test_read_state_returns_a_frozen_semantic_envelope_without_changing_legacy_dict():
    root = Path(__file__).parents[1]
    symbols = SymbolTable.from_file(root / "ladx-disassembly/azle.sym")
    emulator = FakeEmulator()
    emulator.data[symbols.resolve("hMapRoom")] = 0x92
    emulator.data[symbols.resolve("hLinkPositionX")] = 42
    memory = GameMemory(emulator, symbols)

    state = memory.read_state()

    assert state.room["id"] == 0x92
    assert state.player["x"] == 42
    with pytest.raises(FrozenInstanceError):
        state.frame = 10

    emulator.data[symbols.resolve("hLinkPositionX")] = 7
    assert state.player["x"] == 42

    legacy = memory.game_state()
    assert isinstance(legacy, dict)
    assert legacy["player"]["x"] == 7
    json.dumps(legacy)
