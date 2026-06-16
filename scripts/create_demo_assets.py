from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "demo"
DEMO_TEXT_PATH = DEMO_DIR / "sample_tamil_text.txt"
DEMO_AUDIO_PATH = DEMO_DIR / "sample_tamil_gtts.mp3"
DEMO_TEXT = (
    "\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd. "
    "\u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd "
    "\u0b95\u0bcb\u0bb0\u0bbf\u0b95\u0bcd\u0b95\u0bc8 "
    "\u0baa\u0ba4\u0bbf\u0bb5\u0bc1 "
    "\u0b9a\u0bc6\u0baf\u0bcd\u0baf\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f\u0ba4\u0bc1. "
    "\u0ba8\u0bbe\u0b99\u0bcd\u0b95\u0bb3\u0bcd "
    "\u0bb5\u0bbf\u0bb0\u0bc8\u0bb5\u0bbf\u0bb2\u0bcd "
    "\u0b89\u0ba4\u0bb5\u0bc1\u0bb5\u0bcb\u0bae\u0bcd."
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_TEXT_PATH.write_text(DEMO_TEXT + "\n", encoding="utf-8")

    try:
        from gtts import gTTS
    except ImportError as exc:
        raise RuntimeError("Install gTTS first: pip install gTTS") from exc

    gTTS(text=DEMO_TEXT, lang="ta", slow=False).save(str(DEMO_AUDIO_PATH))
    print(f"Wrote {DEMO_TEXT_PATH}")
    print(f"Wrote {DEMO_AUDIO_PATH}")


if __name__ == "__main__":
    main()

