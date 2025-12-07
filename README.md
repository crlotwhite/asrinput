ASRInput - real-time speech recognition (Korean/English) using Faster-Whisper with a CustomTkinter UI.

## Requirements
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (provides `uv` and `uvx`)
- Audio input device and basic desktop UI support

## Run locally (managed venv)
```sh
uv sync
uv run python main.py
```

## Run ad-hoc with uvx (no local venv)
```sh
uvx --from . python main.py
```
`uvx` will resolve dependencies in an ephemeral environment and launch the app.

## Make targets
- `make run` (default): sync deps if needed, then run the app
- `make run-uvx`: run via `uvx` without touching a local venv
- `make deps`: sync dependencies
- `make clean`: remove caches/old build artifacts

## Notes
- On first run, Faster-Whisper downloads the selected model (~hundreds of MB).
- The previous PyInstaller-based exe build has been removed in favor of uv/uvx usage.
