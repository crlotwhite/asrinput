ASRInput - real-time speech recognition (Korean/English) using Faster-Whisper with a CustomTkinter UI.

## Requirements
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (provides `uv` and `uvx`)
- Audio input device and basic desktop UI support

## Run this tool via `uvx`
```sh
uvx --from git+https://github.com/crlotwhite/asrinput asrinput
```

## Make targets
- `make run` (default): sync deps if needed, then run the app
- `make run-uvx`: run via `uvx` without touching a local venv
- `make deps`: sync dependencies
- `make clean`: remove caches/old build artifacts

## Notes
- On first run, Faster-Whisper downloads the selected model (~hundreds of MB).
- The previous PyInstaller-based exe build has been removed in favor of uv/uvx usage.
