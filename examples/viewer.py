"""Tk viewer shared by the random and human environment tests."""

from __future__ import annotations

import pprint
import tkinter as tk
from tkinter import scrolledtext
from typing import Callable


class GameInfoViewer:
    """Show the scaled game frame on the left and live info on the right."""

    def __init__(self, *, title: str, scale: int, help_text: str = "") -> None:
        self.scale = scale
        self.help_text = help_text
        self.closed = False
        self.root = tk.Tk()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.frame_label = tk.Label(self.root, bg="black")
        self.frame_label.pack(side="left", fill="both", expand=False)
        self.info_text = scrolledtext.ScrolledText(
            self.root,
            width=72,
            height=36,
            font=("TkFixedFont", 11),
            wrap="none",
        )
        self.info_text.pack(side="right", fill="both", expand=True)
        self.info_text.configure(state="disabled")
        self._photo = None

    def bind_keys(
        self,
        on_press: Callable[[str], None],
        on_release: Callable[[str], None],
    ) -> None:
        self.root.bind_all("<KeyPress>", lambda event: on_press(event.keysym))
        self.root.bind_all("<KeyRelease>", lambda event: on_release(event.keysym))
        self.root.focus_force()

    def update(self, frame, info: dict, reward: float, step: int, episode: int) -> bool:
        if self.closed:
            return False
        try:
            height, width, channels = frame.shape
            if channels != 3:
                raise ValueError(f"expected an RGB frame, got shape {frame.shape}")
            ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + frame.tobytes()
            photo = tk.PhotoImage(data=ppm, format="PPM")
            self._photo = photo.zoom(self.scale, self.scale)
            self.frame_label.configure(image=self._photo)

            self.info_text.configure(state="normal")
            self.info_text.delete("1.0", "end")
            self.info_text.insert("1.0", self._format_info(info, reward, step, episode))
            self.info_text.configure(state="disabled")
            self.root.update_idletasks()
            self.root.update()
            return not self.closed
        except tk.TclError:
            self.closed = True
            return False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _format_info(self, info: dict, reward: float, step: int, episode: int) -> str:
        state = info["game_state"]
        display = {
            "step": step,
            "episode": episode,
            "action": info["action_name"],
            "reward": reward,
            "num_frames": info["num_frames"],
            "room": state["room"],
            "player": state["player"],
            "inventory": state["inventory"],
            "progress": state["progress"],
            "event_flags": state["event_flags"],
            "monster_count": len(state["monsters"]),
            "entities": state["entities"],
            "events": info["events"],
        }
        body = pprint.pformat(display, sort_dicts=False, width=100)
        return f"{self.help_text}\n\n{body}" if self.help_text else body
