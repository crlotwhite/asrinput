from __future__ import annotations

import threading
from typing import Iterable

import numpy as np
import sounddevice as sd

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_BLOCK_SECONDS = 0.1


def _list_input_devices() -> Iterable[tuple[int, str]]:
    """Yield (index, name) for all available input devices."""
    for index, device in enumerate(sd.query_devices()):
        if device.get("max_input_channels", 0) > 0:
            yield index, device["name"]


class AudioCapture:
    """Capture audio from the microphone using sounddevice."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        device: int | None = None,
        block_seconds: float = DEFAULT_BLOCK_SECONDS,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device  # None = default input
        self.block_seconds = block_seconds

        self.stream: sd.InputStream | None = None
        self.is_recording = False
        self._buffer: list[np.ndarray] = []
        self._lock = threading.Lock()

    # ---------- Device helpers ----------
    @staticmethod
    def get_input_devices() -> list[tuple[int, str]]:
        """Return list of available input devices."""
        return list(_list_input_devices())

    def set_device(self, device: int | None) -> None:
        """Set the input device and reset the current stream if needed."""
        if self.stream:
            self.stream.close()
            self.stream = None
        self.device = device

    # ---------- Stream lifecycle ----------
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        if self.is_recording:
            with self._lock:
                self._buffer.append(indata.copy())

    def _ensure_stream(self) -> None:
        """Create the input stream if it does not exist."""
        if self.stream is None:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * self.block_seconds),
                device=self.device,
            )

    def start(self) -> None:
        """Start capturing audio."""
        self._ensure_stream()
        self.is_recording = True
        self._buffer.clear()
        if self.stream:
            self.stream.start()

    def stop(self) -> np.ndarray | None:
        """Stop capturing and return the recorded audio as a 1-D numpy array."""
        self.is_recording = False
        if self.stream:
            self.stream.stop()

        with self._lock:
            if not self._buffer:
                return None
            audio_data = np.concatenate(self._buffer, axis=0)
            self._buffer.clear()
        return audio_data.flatten()

    def close(self) -> None:
        """Completely close the stream."""
        if self.stream:
            self.stream.close()
            self.stream = None
