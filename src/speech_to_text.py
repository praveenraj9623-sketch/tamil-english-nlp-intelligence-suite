from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.preprocessing import PROJECT_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".aac", ".ogg", ".flac"}
SUPPORTED_TRANSCRIPTION_LANGUAGES = {"ta": "Tamil / Tanglish", "en": "English"}


def _ensure_ffmpeg_available() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

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
    ffmpeg_path = shutil.which("ffmpeg") or str(ffmpeg_target)
    if not Path(ffmpeg_path).exists() and not shutil.which(ffmpeg_path):
        raise RuntimeError(
            "Whisper needs FFmpeg. Install it with `winget install Gyan.FFmpeg` "
            "or install the Python fallback with `pip install imageio-ffmpeg`."
        )
    return ffmpeg_path


def audio_file_diagnostics(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    return {
        "extension": path.suffix.lower() or "(none)",
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def normalize_audio_for_whisper(input_path: str | Path) -> Path:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    diagnostics = audio_file_diagnostics(input_path)
    extension = str(diagnostics["extension"])
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(
            f"Unsupported audio file type `{extension}` ({diagnostics['size_bytes']} bytes). "
            f"Use one of: {supported}."
        )

    ffmpeg_path = _ensure_ffmpeg_available()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".whisper.wav") as normalized_file:
        normalized_path = Path(normalized_file.name)

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        "-acodec",
        "pcm_s16le",
        str(normalized_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        normalized_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not decode this audio file. Please try WAV/MP3/M4A recorded from a phone, "
            f"or check FFmpeg installation. Original extension: `{extension}`, "
            f"size: {diagnostics['size_bytes']} bytes."
        )
    return normalized_path


@lru_cache(maxsize=2)
def _load_whisper_model(model_name: str):
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError("Install Whisper with `pip install openai-whisper`.") from exc

    return whisper.load_model(model_name)


def _normalize_language_hint(language: str | None) -> str | None:
    if language is None:
        return None
    language = language.strip().lower()
    if language in {"", "auto", "detect"}:
        return None
    if language not in SUPPORTED_TRANSCRIPTION_LANGUAGES:
        supported = ", ".join(["auto", *sorted(SUPPORTED_TRANSCRIPTION_LANGUAGES)])
        raise ValueError(f"Unsupported transcription language `{language}`. Use one of: {supported}.")
    return language


def _initial_prompt_for_language(language: str | None) -> str | None:
    if language == "ta":
        return (
            "Tamil or Tamil-English Tanglish speech. Transcribe the speech faithfully. "
            "Preserve Tamil names, English support words, and code-mixed phrases."
        )
    if language == "en":
        return "English speech. Transcribe clearly and preserve customer-support wording."
    return None


def transcribe_audio(
    file_path: str | Path,
    model_name: str = "base",
    language: str | None = None,
) -> dict[str, Any]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    diagnostics = audio_file_diagnostics(file_path)
    language_hint = _normalize_language_hint(language)
    normalized_path = normalize_audio_for_whisper(file_path)
    model = _load_whisper_model(model_name)
    try:
        transcribe_kwargs: dict[str, Any] = {
            "task": "transcribe",
            "fp16": False,
            "condition_on_previous_text": False,
        }
        if language_hint:
            transcribe_kwargs["language"] = language_hint
            transcribe_kwargs["initial_prompt"] = _initial_prompt_for_language(language_hint)

        result = model.transcribe(str(normalized_path), **transcribe_kwargs)
        return {
            "text": str(result.get("text", "")).strip(),
            "language_detected": result.get("language", "unknown"),
            "language_requested": language_hint or "auto",
            "model_name": model_name,
            "normalized_audio": "wav_16khz_mono_pcm_s16le",
            "original_extension": diagnostics["extension"],
            "original_size_bytes": diagnostics["size_bytes"],
        }
    finally:
        normalized_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio with OpenAI Whisper.")
    parser.add_argument("file_path")
    parser.add_argument("--model-name", default="base")
    parser.add_argument("--language", default=None, help="Use ta, en, or auto.")
    args = parser.parse_args()
    print(
        json.dumps(
            transcribe_audio(args.file_path, model_name=args.model_name, language=args.language),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
