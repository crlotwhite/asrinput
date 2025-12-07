from __future__ import annotations

from collections.abc import Callable

import numpy as np
import sounddevice as sd
import customtkinter as ctk

from .audio import AudioCapture

DEFAULT_DEVICE_LABEL = "기본 장치 (Default)"


class SettingsWindow(ctk.CTkToplevel):
    """Settings window for choosing and testing the input device."""

    def __init__(
        self,
        parent,
        audio_capture: AudioCapture,
        use_clipboard: bool,
        on_clipboard_change: Callable[[bool], None],
    ):
        super().__init__(parent)
        self.audio_capture = audio_capture
        self.test_stream: sd.InputStream | None = None
        self.is_testing = False
        self.devices: list[tuple[int, str]] = []
        self.on_clipboard_change = on_clipboard_change
        self.use_clipboard_var = ctk.BooleanVar(value=use_clipboard)

        # window setup
        self.title("설정")
        self.geometry("460x480")
        self.minsize(420, 380)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.device_var = ctk.StringVar(value=DEFAULT_DEVICE_LABEL)
        self._build_ui()
        self._refresh_devices()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI ----------
    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)

        device_frame = ctk.CTkFrame(self)
        device_frame.grid(row=0, column=0, padx=16, pady=16, sticky="ew")
        device_frame.grid_columnconfigure(1, weight=1)

        title_label = ctk.CTkLabel(
            device_frame,
            text="음성 입력 장치",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        title_label.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 12), sticky="w")

        device_label = ctk.CTkLabel(device_frame, text="장치 선택:")
        device_label.grid(row=1, column=0, padx=10, pady=6, sticky="w")

        self.device_menu = ctk.CTkOptionMenu(
            device_frame,
            variable=self.device_var,
            values=[DEFAULT_DEVICE_LABEL],
            width=260,
        )
        self.device_menu.grid(row=1, column=1, padx=10, pady=6, sticky="ew")

        refresh_btn = ctk.CTkButton(
            device_frame,
            text="새로고침",
            width=80,
            command=self._refresh_devices,
        )
        refresh_btn.grid(row=1, column=2, padx=(0, 10), pady=6)

        # --- Test section ---
        test_frame = ctk.CTkFrame(self)
        test_frame.grid(row=1, column=0, padx=16, pady=12, sticky="ew")
        test_frame.grid_columnconfigure(0, weight=1)

        test_title = ctk.CTkLabel(
            test_frame,
            text="마이크 테스트",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        test_title.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 8), sticky="w")

        self.volume_bar = ctk.CTkProgressBar(test_frame, width=300)
        self.volume_bar.grid(row=1, column=0, padx=10, pady=8, sticky="ew")
        self.volume_bar.set(0)

        self.test_btn = ctk.CTkButton(test_frame, text="테스트 시작", command=self._toggle_test)
        self.test_btn.grid(row=1, column=1, padx=10, pady=8)

        self.test_status = ctk.CTkLabel(
            test_frame,
            text="테스트 버튼을 눌러 마이크를 확인하세요",
            font=ctk.CTkFont(size=12),
        )
        self.test_status.grid(row=2, column=0, columnspan=2, padx=10, pady=(2, 10))

        # --- Input / automation section ---
        input_frame = ctk.CTkFrame(self)
        input_frame.grid(row=2, column=0, padx=16, pady=12, sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)

        input_title = ctk.CTkLabel(
            input_frame,
            text="입력 설정",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        input_title.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="w")

        self.clipboard_switch = ctk.CTkSwitch(
            input_frame,
            text="클립보드 자동 복사/붙여넣기",
            variable=self.use_clipboard_var,
        )
        self.clipboard_switch.grid(row=1, column=0, padx=10, pady=(0, 6), sticky="w")

        clipboard_desc = ctk.CTkLabel(
            input_frame,
            text="꺼짐(기본): 클립보드를 건드리지 않고 직접 타이핑합니다.",
            font=ctk.CTkFont(size=11),
        )
        clipboard_desc.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="w")

        # --- Buttons ---
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=3, column=0, padx=16, pady=16, sticky="e")

        apply_btn = ctk.CTkButton(btn_frame, text="적용", command=self._apply_settings, width=80)
        apply_btn.grid(row=0, column=0, padx=6)

        cancel_btn = ctk.CTkButton(btn_frame, text="취소", fg_color="gray", command=self._on_close, width=80)
        cancel_btn.grid(row=0, column=1, padx=6)

    # ---------- Device handling ----------
    def _refresh_devices(self) -> None:
        self.devices = AudioCapture.get_input_devices()
        values = [DEFAULT_DEVICE_LABEL] + [name for _, name in self.devices]
        self.device_menu.configure(values=values)

        # keep previously selected device if present
        if self.audio_capture.device is not None:
            for idx, name in self.devices:
                if idx == self.audio_capture.device:
                    self.device_var.set(name)
                    break
        elif self.device_var.get() not in values:
            self.device_var.set(DEFAULT_DEVICE_LABEL)

    def _selected_device_index(self) -> int | None:
        selected = self.device_var.get()
        if selected == DEFAULT_DEVICE_LABEL:
            return None
        for idx, name in self.devices:
            if name == selected:
                return idx
        return None

    # ---------- Test controls ----------
    def _toggle_test(self) -> None:
        if self.is_testing:
            self._stop_test()
        else:
            self._start_test()

    def _start_test(self) -> None:
        self.is_testing = True
        self.test_btn.configure(text="테스트 중지")
        self.test_status.configure(text="마이크에 말해보세요...")

        device_idx = self._selected_device_index()

        def audio_callback(indata, frames, time_info, status):
            if self.is_testing:
                volume = np.sqrt(np.mean(indata**2))
                level = min(1.0, volume * 10)
                self.after(0, lambda: self.volume_bar.set(level))

        try:
            self.test_stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype=np.float32,
                callback=audio_callback,
                device=device_idx,
            )
            self.test_stream.start()
        except Exception as exc:  # noqa: BLE001 - surface to UI
            self.test_status.configure(text=f"오류: {str(exc)[:30]}")
            self._stop_test()

    def _stop_test(self) -> None:
        self.is_testing = False
        self.test_btn.configure(text="테스트 시작")
        self.test_status.configure(text="테스트 버튼을 눌러 마이크를 확인하세요")
        self.volume_bar.set(0)

        if self.test_stream:
            self.test_stream.stop()
            self.test_stream.close()
            self.test_stream = None

    # ---------- Apply / close ----------
    def _apply_settings(self) -> None:
        device_idx = self._selected_device_index()
        self.audio_capture.set_device(device_idx)
        self.on_clipboard_change(self.use_clipboard_var.get())
        self._on_close()

    def _on_close(self) -> None:
        self._stop_test()
        self.grab_release()
        self.destroy()
