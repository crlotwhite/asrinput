"""ASRInput entrypoint.

Launches the CustomTkinter GUI for Korean/English speech recognition powered by
Faster-Whisper. The core logic lives in the `asrinput` package.
"""

from asrinput.app import run_app


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
