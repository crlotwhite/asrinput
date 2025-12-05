"""
ASRInput - Real-time Speech Recognition Tool
Supports Korean and English with Faster-Whisper
"""

import threading
import queue
import time
import tempfile
import os
from pathlib import Path

import numpy as np
import sounddevice as sd
import pyperclip
import customtkinter as ctk
from pynput import keyboard
from pynput.keyboard import Key, Controller


class TextInputSimulator:
    """Simulates text input to active input fields using clipboard."""
    
    def __init__(self):
        self.keyboard = Controller()
    
    def type_text(self, text: str) -> None:
        """Type text to the currently focused input field using clipboard paste."""
        if not text.strip():
            return
        
        # Save current clipboard content
        try:
            old_clipboard = pyperclip.paste()
        except Exception:
            old_clipboard = ""
        
        try:
            # Copy text to clipboard and paste
            pyperclip.copy(text)
            time.sleep(0.05)  # Small delay for clipboard
            
            # Simulate Ctrl+V
            self.keyboard.press(Key.ctrl)
            self.keyboard.press('v')
            self.keyboard.release('v')
            self.keyboard.release(Key.ctrl)
            
            time.sleep(0.1)  # Wait for paste to complete
        finally:
            # Restore old clipboard content (optional)
            try:
                pyperclip.copy(old_clipboard)
            except Exception:
                pass


class AudioCapture:
    """Captures audio from microphone using sounddevice."""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue: queue.Queue = queue.Queue()
        self.is_recording = False
        self.stream = None
        self._buffer: list[np.ndarray] = []
        self._lock = threading.Lock()
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f"Audio status: {status}")
        if self.is_recording:
            with self._lock:
                self._buffer.append(indata.copy())
    
    def start(self) -> None:
        """Start audio capture."""
        if self.stream is None:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
            )
        self.is_recording = True
        self._buffer.clear()
        self.stream.start()
    
    def stop(self) -> np.ndarray | None:
        """Stop audio capture and return recorded audio."""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
        
        with self._lock:
            if self._buffer:
                audio_data = np.concatenate(self._buffer, axis=0)
                self._buffer.clear()
                return audio_data.flatten()
        return None
    
    def close(self) -> None:
        """Close the audio stream."""
        if self.stream:
            self.stream.close()
            self.stream = None


class SpeechRecognizer:
    """Speech recognition using Faster-Whisper."""
    
    # Supported languages: code -> display name
    LANGUAGES = {
        None: "자동 감지 (Auto)",
        "ko": "한국어 (Korean)",
        "en": "English",
        "ja": "日本語 (Japanese)",
        "zh": "中文 (Chinese)",
    }
    
    def __init__(self, model_size: str = "small", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self.model = None
        self.language = None  # None = auto-detect
        self._loading = False
        self._loaded = threading.Event()
    
    def load_model(self, callback=None) -> None:
        """Load the Whisper model in background."""
        def _load():
            self._loading = True
            try:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type="int8" if self.device == "cpu" else "float16"
                )
                self._loaded.set()
                if callback:
                    callback(True, None)
            except Exception as e:
                if callback:
                    callback(False, str(e))
            finally:
                self._loading = False
        
        thread = threading.Thread(target=_load, daemon=True)
        thread.start()
    
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self._loaded.is_set()
    
    def is_loading(self) -> bool:
        """Check if model is currently loading."""
        return self._loading
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio data to text."""
        if not self.is_ready():
            return ""
        
        if len(audio_data) < sample_rate * 0.5:  # Less than 0.5 seconds
            return ""
        
        # Save audio to temporary file (faster-whisper requires file input)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            import wave
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                # Convert float32 to int16
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
            
            # Transcribe
            segments, info = self.model.transcribe(
                temp_path,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),
                language=self.language,  # None = auto-detect
            )
            
            # Combine all segments
            text = " ".join(segment.text.strip() for segment in segments)
            return text.strip()
        
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except Exception:
                pass


class ASRApp(ctk.CTk):
    """Main application window for ASR Input."""
    
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.title("ASRInput - 음성 인식 입력기")
        self.geometry("500x400")
        self.minsize(400, 300)
        
        # Set appearance
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        
        # Components
        self.audio_capture = AudioCapture()
        self.recognizer = SpeechRecognizer(model_size="small", device="cpu")
        self.text_simulator = TextInputSimulator()
        
        # State
        self.is_toggle_active = False
        self.is_ptt_active = False
        self.recognition_thread = None
        self.should_stop = threading.Event()
        
        # Setup UI
        self._setup_ui()
        
        # Setup keyboard listener for PTT
        self._setup_ptt_listener()
        
        # Start model loading
        self._start_model_loading()
        
        # Cleanup on close
        self.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _setup_ui(self):
        """Setup the user interface."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Top frame - Status and controls
        top_frame = ctk.CTkFrame(self)
        top_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        top_frame.grid_columnconfigure(1, weight=1)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            top_frame,
            text="⏳ 모델 로딩 중...",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.status_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        # Language selection dropdown
        self.language_var = ctk.StringVar(value="자동 감지 (Auto)")
        self.language_menu = ctk.CTkOptionMenu(
            top_frame,
            variable=self.language_var,
            values=list(SpeechRecognizer.LANGUAGES.values()),
            command=self._on_language_changed,
            width=140
        )
        self.language_menu.grid(row=0, column=1, padx=10, pady=10)
        
        # Toggle switch for continuous recognition
        self.toggle_var = ctk.BooleanVar(value=False)
        self.toggle_switch = ctk.CTkSwitch(
            top_frame,
            text="연속 인식",
            variable=self.toggle_var,
            command=self._on_toggle_changed,
            state="disabled"
        )
        self.toggle_switch.grid(row=0, column=2, padx=10, pady=10, sticky="e")
        
        # Text display area
        self.text_display = ctk.CTkTextbox(
            self,
            font=ctk.CTkFont(size=13),
            wrap="word"
        )
        self.text_display.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.text_display.insert("1.0", "인식된 텍스트가 여기에 표시됩니다.\n")
        self.text_display.configure(state="disabled")
        
        # Bottom frame - Info and controls
        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="ew")
        bottom_frame.grid_columnconfigure(0, weight=1)
        
        # PTT info label
        self.ptt_label = ctk.CTkLabel(
            bottom_frame,
            text="🎤 F2 키를 누르고 있는 동안 음성 인식 (Push-to-Talk)",
            font=ctk.CTkFont(size=12)
        )
        self.ptt_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Auto-input checkbox
        self.auto_input_var = ctk.BooleanVar(value=True)
        self.auto_input_check = ctk.CTkCheckBox(
            bottom_frame,
            text="인식 후 자동 입력",
            variable=self.auto_input_var
        )
        self.auto_input_check.grid(row=0, column=1, padx=10, pady=5, sticky="e")
    
    def _setup_ptt_listener(self):
        """Setup Push-to-Talk keyboard listener."""
        def on_press(key):
            if key == keyboard.Key.f2 and not self.is_ptt_active:
                if self.recognizer.is_ready() and not self.is_toggle_active:
                    self.is_ptt_active = True
                    self.after(0, self._start_ptt_recording)
        
        def on_release(key):
            if key == keyboard.Key.f2 and self.is_ptt_active:
                self.is_ptt_active = False
                self.after(0, self._stop_ptt_recording)
        
        self.ptt_listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )
        self.ptt_listener.daemon = True
        self.ptt_listener.start()
    
    def _start_model_loading(self):
        """Start loading the Whisper model."""
        def on_model_loaded(success, error):
            if success:
                self.after(0, lambda: self._update_status("✅ 준비 완료", "green"))
                self.after(0, lambda: self.toggle_switch.configure(state="normal"))
            else:
                self.after(0, lambda: self._update_status(f"❌ 모델 로딩 실패: {error}", "red"))
        
        self.recognizer.load_model(callback=on_model_loaded)
    
    def _update_status(self, text: str, color: str = None):
        """Update status label."""
        self.status_label.configure(text=text)
        if color:
            self.status_label.configure(text_color=color)
    
    def _append_text(self, text: str):
        """Append text to the display."""
        self.text_display.configure(state="normal")
        self.text_display.insert("end", f"{text}\n")
        self.text_display.see("end")
        self.text_display.configure(state="disabled")
    
    def _on_language_changed(self, selection: str):
        """Handle language selection change."""
        # Find language code from display name
        for code, name in SpeechRecognizer.LANGUAGES.items():
            if name == selection:
                self.recognizer.language = code
                break
    
    def _on_toggle_changed(self):
        """Handle toggle switch state change."""
        if self.toggle_var.get():
            self._start_continuous_recognition()
        else:
            self._stop_continuous_recognition()
    
    def _start_continuous_recognition(self):
        """Start continuous recognition mode."""
        if not self.recognizer.is_ready():
            self.toggle_var.set(False)
            return
        
        self.is_toggle_active = True
        self.should_stop.clear()
        self._update_status("🔴 연속 인식 중...", "red")
        
        # Start recognition thread
        self.recognition_thread = threading.Thread(
            target=self._continuous_recognition_loop,
            daemon=True
        )
        self.recognition_thread.start()
    
    def _stop_continuous_recognition(self):
        """Stop continuous recognition mode."""
        self.is_toggle_active = False
        self.should_stop.set()
        self.audio_capture.stop()
        self._update_status("✅ 준비 완료", "green")
    
    def _continuous_recognition_loop(self):
        """Background loop for continuous recognition."""
        self.audio_capture.start()
        
        while not self.should_stop.is_set():
            time.sleep(2.0)  # Record for 2 seconds
            
            if self.should_stop.is_set():
                break
            
            # Get audio and restart recording
            audio_data = self.audio_capture.stop()
            if not self.should_stop.is_set():
                self.audio_capture.start()
            
            if audio_data is not None and len(audio_data) > 0:
                # Transcribe in background
                text = self.recognizer.transcribe(audio_data)
                if text:
                    self.after(0, lambda t=text: self._on_text_recognized(t))
    
    def _start_ptt_recording(self):
        """Start PTT recording."""
        if not self.recognizer.is_ready():
            return
        
        self._update_status("🔴 녹음 중... (F2 유지)", "red")
        self.audio_capture.start()
    
    def _stop_ptt_recording(self):
        """Stop PTT recording and transcribe."""
        self._update_status("⏳ 인식 중...", "orange")
        audio_data = self.audio_capture.stop()
        
        if audio_data is not None and len(audio_data) > 0:
            # Transcribe in background thread
            def transcribe():
                text = self.recognizer.transcribe(audio_data)
                if text:
                    self.after(0, lambda: self._on_text_recognized(text))
                self.after(0, lambda: self._update_status("✅ 준비 완료", "green"))
            
            thread = threading.Thread(target=transcribe, daemon=True)
            thread.start()
        else:
            self._update_status("✅ 준비 완료", "green")
    
    def _on_text_recognized(self, text: str):
        """Handle recognized text."""
        self._append_text(f"[인식] {text}")
        
        # Auto-input to active field
        if self.auto_input_var.get():
            self.text_simulator.type_text(text)
    
    def _on_close(self):
        """Cleanup on window close."""
        self.should_stop.set()
        self.is_toggle_active = False
        self.is_ptt_active = False
        
        if hasattr(self, 'ptt_listener'):
            self.ptt_listener.stop()
        
        self.audio_capture.close()
        self.destroy()


def main():
    app = ASRApp()
    app.mainloop()


if __name__ == "__main__":
    main()
