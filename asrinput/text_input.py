from __future__ import annotations

import time

import pyperclip
from pynput.keyboard import Controller, Key


class TextInputSimulator:
    """Simulate text input to the active field.

    By default, it types directly without touching the clipboard. The clipboard
    based paste workflow can be enabled for environments where pasting is more
    reliable than simulated keystrokes.
    """

    def __init__(self, keyboard_controller: Controller | None = None, use_clipboard: bool = False):
        self.keyboard = keyboard_controller or Controller()
        self.use_clipboard = use_clipboard

    def set_use_clipboard(self, enabled: bool) -> None:
        """Enable/disable clipboard-based pasting."""

        self.use_clipboard = enabled

    def type_text(self, text: str) -> None:
        if not text.strip():
            return

        if not self.use_clipboard:
            # Type directly to avoid mutating the clipboard.
            self.keyboard.type(text)
            return

        try:
            old_clipboard = pyperclip.paste()
        except Exception:
            old_clipboard = ""

        try:
            pyperclip.copy(text)
            time.sleep(0.05)  # allow clipboard to update

            self.keyboard.press(Key.ctrl)
            self.keyboard.press("v")
            self.keyboard.release("v")
            self.keyboard.release(Key.ctrl)

            time.sleep(0.1)  # allow paste to finish
        finally:
            try:
                pyperclip.copy(old_clipboard)
            except Exception:
                pass
