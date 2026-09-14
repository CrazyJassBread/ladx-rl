from pathlib import Path
import json

from zelda_env.games.ladx.state_extractor import LadxStateExtractor
from zelda_env.games.ladx.symbols import default_ladx_symbol_table


class FakeBackend:
    platform = "gbc"
    buttons = ("UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT")

    def __init__(self):
        self.memory = bytearray(0x10000)

    def reset(self): ...

    def close(self): ...

    def press(self, buttons): ...

    def release_all(self): ...

    def advance(self, frames): ...

    def read_u8(self, address):
        return self.memory[address]

    def read_u16(self, address, *, endian="little"):
        return int.from_bytes(self.read_bytes(address, 2), endian)

    def read_bytes(self, address, length):
        return bytes(self.memory[address : address + length])

    def save_state(self):
        return bytes(self.memory)

    def load_state(self, data):
        self.memory[:] = data

    def screen_rgb(self):
        return None


def test_ladx_state_extractor_returns_generic_json_safe_schema():
    root = Path(__file__).parents[1]
    symbols = default_ladx_symbol_table(root)
    backend = FakeBackend()
    backend.memory[symbols.resolve("hLinkPositionX")] = 42
    backend.memory[symbols.resolve("hLinkPositionY")] = 99
    backend.memory[symbols.resolve("hMapRoom")] = 0x92
    backend.memory[symbols.resolve("hMapId")] = 0x00
    backend.memory[symbols.resolve("wHealth")] = 0x18
    backend.memory[symbols.resolve("wMaxHearts")] = 3
    backend.memory[symbols.resolve("wInventoryItems")] = 1
    backend.memory[symbols.resolve("wInventoryItems.AButtonSlot")] = 4
    backend.memory[symbols.resolve("wEntitiesStatusTable")] = 5
    backend.memory[symbols.resolve("wEntitiesTypeTable")] = 9
    backend.memory[symbols.resolve("wEntitiesPosXTable")] = 80
    backend.memory[symbols.resolve("wEntitiesPosYTable")] = 64

    state = LadxStateExtractor(symbols, repo_root=root).extract(backend)
    json.dumps(state)

    assert state["meta"]["game"] == "ladx"
    assert state["meta"]["schema_version"] == 3
    assert state["map"]["location"]["room"] == 0x92
    assert state["sprites"]["player"]["x"] == 42
    assert state["sprites"]["player"]["inventory"]["b_button_item"] == 1
    assert state["sprites"]["slots"]["slot_00"]["type"] == 9
    assert state["sprites"]["slots"]["slot_00"]["category"] == "enemy"
    assert state["sprites"]["active"][0]["type_name"] == "ENTITY_OCTOROK"
    assert state["sprites"]["player"]["health"]["current"] == 0x18
    assert "world" not in state
    assert "raw" not in state


def test_full_mode_and_legacy_aliases_are_opt_in():
    root = Path(__file__).parents[1]
    symbols = default_ladx_symbol_table(root)
    state = LadxStateExtractor(
        symbols, repo_root=root, state_mode="full", include_legacy_aliases=True
    ).extract(FakeBackend())
    assert state["world"] is state["map"]["location"]
    assert len(state["entities"]) == 0x10
    assert "entity_tables" in state["raw"]
