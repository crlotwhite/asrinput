from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .audio import DEFAULT_SAMPLE_RATE

LanguageMap = dict[str | None, str]
LoadCallback = Callable[[bool, str | None], None]


class SpeechRecognizer:
    """Speech recognition wrapper around Faster-Whisper."""

    LANGUAGES: LanguageMap = {
        None: "자동 감지 (Auto)",
        "ko": "한국어 (Korean)",
        "en": "English",
        "ja": "日本語 (Japanese)",
        "zh": "中文 (Chinese)",
    }

    def __init__(self, model_size: str = "small", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self.language: str | None = None  # None = auto
        self.model = None
        self._loading = False
        self._loaded = threading.Event()

    # ---------- Lifecycle ----------
    def load_model(self, callback: LoadCallback | None = None) -> None:
        """Load the model in a background thread."""

        def _load():
            self._loading = True
            try:
                from faster_whisper import WhisperModel

                compute_type = "int8" if self.device == "cpu" else "float16"
                self.model = WhisperModel(
                    self.model_size, device=self.device, compute_type=compute_type
                )
                self._loaded.set()
                if callback:
                    callback(True, None)
            except Exception as exc:  # noqa: BLE001 - surface model errors
                if callback:
                    callback(False, str(exc))
            finally:
                self._loading = False

        threading.Thread(target=_load, daemon=True).start()

    def is_ready(self) -> bool:
        return self._loaded.is_set()

    def is_loading(self) -> bool:
        return self._loading

    # ---------- Transcription ----------
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> str:
        """Transcribe audio and return text."""
        if not self.is_ready():
            return ""

        if not self._has_sufficient_audio(audio_data, sample_rate):
            return ""

        temp_path = self._write_temp_wav(audio_data, sample_rate)
        try:
            segments, _info = self.model.transcribe(
                temp_path,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                language=self.language,
            )
            return self._join_segments(segments)
        except Exception as exc:  # noqa: BLE001 - surface transcription errors
            print(f"Transcription error: {exc}")
            return ""
        finally:
            self._safe_delete(temp_path)

    @staticmethod
    def _has_sufficient_audio(audio_data: np.ndarray, sample_rate: int) -> bool:
        return audio_data is not None and len(audio_data) >= sample_rate * 0.5

    @staticmethod
    def _join_segments(segments: Iterable) -> str:
        texts = (segment.text.strip() for segment in segments)
        return " ".join(filter(None, texts)).strip()

    @staticmethod
    def _write_temp_wav(audio_data: np.ndarray, sample_rate: int) -> str:
        import wave

        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # close the low-level handle; wave will reopen

        with wave.open(temp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())

        return temp_path

    @staticmethod
    def _safe_delete(path: str) -> None:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass
