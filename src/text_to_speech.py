from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from src.preprocessing import PROJECT_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def text_to_speech(text: str, output_dir: str | Path | None = None) -> str:
    if not text or not text.strip():
        raise ValueError("Text-to-speech input cannot be empty.")

    output_path = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "reports" / "generated_audio"
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    audio_path = output_path / f"ta_tts_{timestamp}_{uuid4().hex[:8]}.mp3"
    try:
        from gtts import gTTS
    except ImportError as exc:
        raise RuntimeError("Install Tamil text-to-speech support with `pip install gTTS`.") from exc

    try:
        gTTS(text=text.strip(), lang="ta", slow=False).save(str(audio_path))
    except Exception as exc:
        raise RuntimeError(
            "gTTS could not create audio. Check your internet connection and Tamil text input."
        ) from exc
    return str(audio_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Tamil speech with gTTS.")
    parser.add_argument("text")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    print(json.dumps({"audio_path": text_to_speech(args.text, args.output_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
