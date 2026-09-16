"""Small PyBoy wrapper used by the LADX environment."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Emulator(Protocol):
    """The emulator operations needed by :class:`ZeldaEnv`."""

    def press(self, buttons: frozenset[str]) -> None: ...
    def release_all(self) -> None: ...
    def advance(self, frames: int) -> None: ...
    def read_u8(self, address: int) -> int: ...
    def read_bytes(self, address: int, length: int) -> bytes: ...
    def save_state(self) -> bytes: ...
    def load_state(self, data: bytes) -> None: ...
    def get_frame(self): ...
    def close(self) -> None: ...


class PyBoyEmulator:
    """Expose screen, controls, state and memory through a compact API."""

    _BUTTONS = {
        "UP": "up",
        "DOWN": "down",
        "LEFT": "left",
        "RIGHT": "right",
        "A": "a",
        "B": "b",
        "START": "start",
        "SELECT": "select",
    }

    def __init__(
        self,
        rom_path: str | Path,
        *,
        sym_path: str | Path | None = None,
        window: str = "null",
    ) -> None:
        try:
            from pyboy import PyBoy
        except ImportError as exc:
            raise RuntimeError("PyBoy is required; install the project dependencies first") from exc

        rom = Path(rom_path)
        if not rom.is_file():
            raise FileNotFoundError(f"LADX ROM not found: {rom}. Run `make rom` first.")
        if sym_path is not None and not Path(sym_path).is_file():
            raise FileNotFoundError(f"LADX symbol file not found: {sym_path}. Run `make rom` first.")

        kwargs = {"window": window, "cgb": True}
        if sym_path is not None:
            kwargs["symbols"] = str(sym_path)
        self.pyboy = PyBoy(str(rom), **kwargs)
        self._pressed: set[str] = set()
        self._needs_render_warmup = False

    def press(self, buttons: frozenset[str]) -> None:
        self.release_all()
        for button in buttons:
            try:
                mapped = self._BUTTONS[button]
            except KeyError as exc:
                raise ValueError(f"Unsupported Game Boy button: {button}") from exc
            self.pyboy.button_press(mapped)
            self._pressed.add(button)

    def release_all(self) -> None:
        for button in tuple(self._pressed):
            self.pyboy.button_release(self._BUTTONS[button])
        self._pressed.clear()

    def advance(self, frames: int) -> None:
        if self._needs_render_warmup:
            # PyBoy save states do not restore the renderer's partial-frame
            # cache. Render every skipped frame once after loading so the first
            # pixel observation is deterministic, then use the fast path.
            for _ in range(frames):
                self.pyboy.tick(count=1, render=True, sound=False)
            self._needs_render_warmup = False
            return
        # PyBoy renders only the final frame when count > 1. This preserves the
        # observed frame while avoiding repeated rendering during frame skip.
        self.pyboy.tick(count=frames, render=True, sound=False)

    def read_u8(self, address: int) -> int:
        return int(self.pyboy.memory[address]) & 0xFF

    def read_bytes(self, address: int, length: int) -> bytes:
        return bytes(self.read_u8(address + offset) for offset in range(length))

    def save_state(self) -> bytes:
        buffer = BytesIO()
        self.pyboy.save_state(buffer)
        return buffer.getvalue()

    def load_state(self, data: bytes) -> None:
        self.pyboy.load_state(BytesIO(data))
        self._needs_render_warmup = True

    def get_frame(self):
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required for screen capture") from exc
        frame = np.asarray(self.pyboy.screen.ndarray)
        return frame[:, :, :3].copy()

    def close(self) -> None:
        try:
            self.pyboy.stop(save=False)
        except TypeError:  # PyBoy < 2.6
            self.pyboy.stop()
