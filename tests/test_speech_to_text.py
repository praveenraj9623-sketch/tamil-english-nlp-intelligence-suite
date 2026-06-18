import pytest

from src.speech_to_text import (
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_TRANSCRIPTION_LANGUAGES,
    normalize_audio_for_whisper,
    transcribe_audio,
)


def test_phone_audio_extensions_are_supported() -> None:
    assert {".m4a", ".mp4", ".aac"} <= SUPPORTED_AUDIO_EXTENSIONS
    assert {"ta", "en"} <= set(SUPPORTED_TRANSCRIPTION_LANGUAGES)


def test_unsupported_audio_extension_fails_before_whisper(tmp_path) -> None:
    bad_audio = tmp_path / "sample.txt"
    bad_audio.write_text("not audio", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported audio file type"):
        normalize_audio_for_whisper(bad_audio)


def test_invalid_language_hint_fails_before_whisper_load(tmp_path) -> None:
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"not real wav")

    with pytest.raises(ValueError, match="Unsupported transcription language"):
        transcribe_audio(wav_path, language="fr")
