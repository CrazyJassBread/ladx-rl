from zelda_env.emulator import Emulator


class ContractBackend:
    def close(self): ...

    def press(self, buttons): ...

    def release_all(self): ...

    def advance(self, frames): ...

    def read_u8(self, address):
        return 0

    def read_bytes(self, address, length):
        return bytes(length)

    def save_state(self):
        return b""

    def load_state(self, data): ...

    def get_frame(self):
        return None


def test_emulator_protocol_accepts_contract_emulator():
    assert isinstance(ContractBackend(), Emulator)
