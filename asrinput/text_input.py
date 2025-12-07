from __future__ import annotations

import time

import pyperclip
from pynput.keyboard import Controller, Key


class TextInputSimulator:
    """Simulate text input to the active field via clipboard paste."""

    def __init__(self, keyboard_controller: Controller | None = None):
        self.keyboard = keyboard_controller or Controller()

    def type_text(self, text: str) -> None:
        if not text.strip():
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
