from zelda_env.emulator import Emulator, PyBoyEmulator


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


def test_pyboy_advance_batches_frames_and_renders_only_the_last_frame():
    class TickRecorder:
        def __init__(self):
            self.calls = []

        def tick(self, **kwargs):
            self.calls.append(kwargs)

    emulator = object.__new__(PyBoyEmulator)
    emulator.pyboy = TickRecorder()
    emulator._needs_render_warmup = False

    emulator.advance(4)

    assert emulator.pyboy.calls == [{"count": 4, "render": True, "sound": False}]


def test_first_advance_after_load_warms_the_renderer_before_batching():
    class PyBoyRecorder:
        def __init__(self):
            self.calls = []

        def load_state(self, stream):
            assert stream.read() == b"state"

        def tick(self, **kwargs):
            self.calls.append(kwargs)

    emulator = object.__new__(PyBoyEmulator)
    emulator.pyboy = PyBoyRecorder()
    emulator._needs_render_warmup = False

    emulator.load_state(b"state")
    emulator.advance(2)
    emulator.advance(4)

    assert emulator.pyboy.calls == [
        {"count": 1, "render": True, "sound": False},
        {"count": 1, "render": True, "sound": False},
        {"count": 4, "render": True, "sound": False},
    ]
