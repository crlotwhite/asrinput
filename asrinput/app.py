from __future__ import annotations

import threading
import time

import customtkinter as ctk
from pynput import keyboard

from .audio import AudioCapture
from .recognizer import SpeechRecognizer
from .settings_window import SettingsWindow
from .text_input import TextInputSimulator

PTT_KEY = keyboard.Key.f2


class ASRApp(ctk.CTk):
    """Main application window for ASRInput."""

    def __init__(self):
        super().__init__()

        # window setup
        self.title("ASRInput - 음성 인식 입력기")
        self.geometry("520x420")
        self.minsize(420, 320)

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # components
        self.audio_capture = AudioCapture()
        self.recognizer = SpeechRecognizer(model_size="small", device="cpu")
        self.text_simulator = TextInputSimulator()

        # state
        self.is_toggle_active = False
        self.is_ptt_active = False
        self.recognition_thread: threading.Thread | None = None
        self.should_stop = threading.Event()

        # UI
        self._build_ui()
        self._setup_ptt_listener()
        self._start_model_loading()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI ----------
    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        top_frame = ctk.CTkFrame(self)
        top_frame.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="ew")
        top_frame.grid_columnconfigure(1, weight=1)

        self.status_label = ctk.CTkLabel(
            top_frame,
            text="모델 로딩 중...",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.status_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.language_var = ctk.StringVar(value="자동 감지 (Auto)")
        self.language_menu = ctk.CTkOptionMenu(
            top_frame,
            variable=self.language_var,
            values=list(SpeechRecognizer.LANGUAGES.values()),
            command=self._on_language_changed,
            width=150,
        )
        self.language_menu.grid(row=0, column=1, padx=8, pady=10)

        self.toggle_var = ctk.BooleanVar(value=False)
        self.toggle_switch = ctk.CTkSwitch(
            top_frame,
            text="연속 인식",
            variable=self.toggle_var,
            command=self._on_toggle_changed,
            state="disabled",
        )
        self.toggle_switch.grid(row=0, column=2, padx=10, pady=10, sticky="e")

        self.text_display = ctk.CTkTextbox(self, font=ctk.CTkFont(size=13), wrap="word")
        self.text_display.grid(row=1, column=0, padx=10, pady=6, sticky="nsew")
        self.text_display.insert("1.0", "인식된 텍스트가 여기에 표시됩니다.\n")
        self.text_display.configure(state="disabled")

        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.grid(row=2, column=0, padx=10, pady=(6, 10), sticky="ew")
        bottom_frame.grid_columnconfigure(0, weight=1)

        self.ptt_label = ctk.CTkLabel(
            bottom_frame,
            text="F2 키를 누르고 있는 동안 음성 인식 (Push-to-Talk)",
            font=ctk.CTkFont(size=12),
        )
        self.ptt_label.grid(row=0, column=0, padx=10, pady=6, sticky="w")

        self.auto_input_var = ctk.BooleanVar(value=True)
        self.auto_input_check = ctk.CTkCheckBox(
            bottom_frame, text="자동 입력", variable=self.auto_input_var
        )
        self.auto_input_check.grid(row=0, column=1, padx=8, pady=6, sticky="e")

        self.always_on_top_var = ctk.BooleanVar(value=False)
        self.always_on_top_check = ctk.CTkCheckBox(
            bottom_frame,
            text="항상 위",
            variable=self.always_on_top_var,
            command=self._on_always_on_top_changed,
        )
        self.always_on_top_check.grid(row=0, column=2, padx=8, pady=6, sticky="e")

        self.settings_btn = ctk.CTkButton(
            bottom_frame,
            text="설정",
            width=80,
            command=self._open_settings,
        )
        self.settings_btn.grid(row=0, column=3, padx=10, pady=6, sticky="e")

    # ---------- Model loading ----------
    def _start_model_loading(self) -> None:
        def on_model_loaded(success: bool, error: str | None):
            if success:
                self.after(0, lambda: self._update_status("준비 완료", "green"))
                self.after(0, lambda: self.toggle_switch.configure(state="normal"))
            else:
                message = f"모델 로딩 실패: {error}"
                self.after(0, lambda: self._update_status(message, "red"))

        self.recognizer.load_model(callback=on_model_loaded)

    def _update_status(self, text: str, color: str | None = None) -> None:
        self.status_label.configure(text=text)
        if color:
            self.status_label.configure(text_color=color)

    # ---------- Keyboard listener ----------
    def _setup_ptt_listener(self) -> None:
        def on_press(key):
            if key == PTT_KEY and not self.is_ptt_active:
                if self.recognizer.is_ready() and not self.is_toggle_active:
                    self.is_ptt_active = True
                    self.after(0, self._start_ptt_recording)

        def on_release(key):
            if key == PTT_KEY and self.is_ptt_active:
                self.is_ptt_active = False
                self.after(0, self._stop_ptt_recording)

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()
        self.ptt_listener = listener

    # ---------- UI callbacks ----------
    def _on_language_changed(self, selection: str) -> None:
        for code, name in SpeechRecognizer.LANGUAGES.items():
            if name == selection:
                self.recognizer.language = code
                break

    def _on_always_on_top_changed(self) -> None:
        self.attributes("-topmost", self.always_on_top_var.get())

    def _open_settings(self) -> None:
        SettingsWindow(self, self.audio_capture)

    def _on_toggle_changed(self) -> None:
        if self.toggle_var.get():
            self._start_continuous_recognition()
        else:
            self._stop_continuous_recognition()

    # ---------- Continuous recognition ----------
    def _start_continuous_recognition(self) -> None:
        if not self.recognizer.is_ready():
            self.toggle_var.set(False)
            return

        self.is_toggle_active = True
        self.should_stop.clear()
        self._update_status("연속 인식 중...", "red")

        self.recognition_thread = threading.Thread(
            target=self._continuous_recognition_loop, daemon=True
        )
        self.recognition_thread.start()

    def _stop_continuous_recognition(self) -> None:
        self.is_toggle_active = False
        self.should_stop.set()
        self.audio_capture.stop()
        self._update_status("준비 완료", "green")

    def _continuous_recognition_loop(self) -> None:
        self.audio_capture.start()

        while not self.should_stop.is_set():
            time.sleep(2.0)  # capture roughly 2 seconds
            if self.should_stop.is_set():
                break

            audio_data = self.audio_capture.stop()
            if not self.should_stop.is_set():
                self.audio_capture.start()

            if audio_data is not None and len(audio_data) > 0:
                text = self.recognizer.transcribe(audio_data)
                if text:
                    self.after(0, lambda t=text: self._on_text_recognized(t))

    # ---------- Push-to-talk ----------
    def _start_ptt_recording(self) -> None:
        if not self.recognizer.is_ready():
            return
        self._update_status("녹음 중... (F2 유지)", "red")
        self.audio_capture.start()

    def _stop_ptt_recording(self) -> None:
        self._update_status("인식 중...", "orange")
        audio_data = self.audio_capture.stop()

        if audio_data is None or len(audio_data) == 0:
            self._update_status("준비 완료", "green")
            return

        threading.Thread(
            target=self._transcribe_and_handle, args=(audio_data,), daemon=True
        ).start()

    def _transcribe_and_handle(self, audio_data) -> None:
        text = self.recognizer.transcribe(audio_data)
        if text:
            self.after(0, lambda: self._on_text_recognized(text))
        self.after(0, lambda: self._update_status("준비 완료", "green"))

    # ---------- Text handling ----------
    def _append_text(self, text: str) -> None:
        self.text_display.configure(state="normal")
        self.text_display.insert("end", f"{text}\n")
        self.text_display.see("end")
        self.text_display.configure(state="disabled")

    def _on_text_recognized(self, text: str) -> None:
        self._append_text(f"[인식] {text}")
        if self.auto_input_var.get():
            self.text_simulator.type_text(text)

    # ---------- Cleanup ----------
    def _on_close(self) -> None:
        self.should_stop.set()
        self.is_toggle_active = False
        self.is_ptt_active = False

        if hasattr(self, "ptt_listener"):
            self.ptt_listener.stop()

        self.audio_capture.close()
        self.destroy()


def create_app() -> ASRApp:
    return ASRApp()


def run_app() -> None:
    create_app().mainloop()


__all__ = ["ASRApp", "create_app", "run_app"]
