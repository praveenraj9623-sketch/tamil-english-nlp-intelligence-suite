from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.preprocessing import PROJECT_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg"):
        return

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "Whisper needs FFmpeg. Install it with `winget install Gyan.FFmpeg` "
            "or install the Python fallback with `pip install imageio-ffmpeg`."
        ) from exc

    ffmpeg_source = Path(imageio_ffmpeg.get_ffmpeg_exe())
    ffmpeg_dir = PROJECT_ROOT / ".bin"
    ffmpeg_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_target = ffmpeg_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not ffmpeg_target.exists():
        shutil.copy2(ffmpeg_source, ffmpeg_target)

    os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{os.environ.get('PATH', '')}"


@lru_cache(maxsize=2)
def _load_whisper_model(model_name: str):
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError("Install Whisper with `pip install openai-whisper`.") from exc

    return whisper.load_model(model_name)


def transcribe_audio(file_path: str | Path, model_name: str = "base") -> dict[str, Any]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    _ensure_ffmpeg_available()
    model = _load_whisper_model(model_name)
    result = model.transcribe(str(file_path), task="transcribe")
    return {
        "text": str(result.get("text", "")).strip(),
        "language_detected": result.get("language", "unknown"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio with OpenAI Whisper.")
    parser.add_argument("file_path")
    parser.add_argument("--model-name", default="base")
    args = parser.parse_args()
    print(json.dumps(transcribe_audio(args.file_path, model_name=args.model_name), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
