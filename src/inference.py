from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.preprocessing import PROJECT_ROOT, clean_text


DEFAULT_CHAMPION_PATH = PROJECT_ROOT / "models" / "champion_model.joblib"


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@lru_cache(maxsize=4)
def load_model(model_path: str | Path = DEFAULT_CHAMPION_PATH) -> dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run baseline and transformer training first."
        )
    return joblib.load(model_path)


def predict_batch_proba(
    bundle: dict[str, Any],
    texts: list[str],
) -> tuple[list[str], np.ndarray, list[str]]:
    cleaned_texts = [clean_text(text) for text in texts]
    model_type = bundle.get("model_type")

    if model_type == "baseline":
        pipeline = bundle["pipeline"]
        probabilities = pipeline.predict_proba(cleaned_texts)
        classes = list(pipeline.classes_)
    elif model_type == "transformer":
        model_name = bundle["sentence_transformer_name"]
        encoder = _load_sentence_transformer(model_name)
        embeddings = encoder.encode(
            cleaned_texts,
            batch_size=int(bundle.get("batch_size", 32)),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        classifier = bundle["classifier"]
        probabilities = classifier.predict_proba(embeddings)
        classes = list(classifier.classes_)
    else:
        raise ValueError(f"Unsupported model_type in model bundle: {model_type}")

    return classes, np.asarray(probabilities), cleaned_texts


def model_family(bundle: dict[str, Any]) -> str:
    return "transformer" if bundle.get("model_type") == "transformer" else "baseline"


def predict_sentiment(
    text: str,
    model_path: str | Path = DEFAULT_CHAMPION_PATH,
    top_n: int = 3,
) -> dict[str, Any]:
    bundle = load_model(model_path)
    classes, probabilities, cleaned_texts = predict_batch_proba(bundle, [text])
    row = probabilities[0]
    best_idx = int(np.argmax(row))
    ranked = sorted(
        (
            {"label": label, "probability": float(row[class_idx])}
            for class_idx, label in enumerate(classes)
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )
    return {
        "sentiment": classes[best_idx],
        "confidence": float(row[best_idx]),
        "model": model_family(bundle),
        "model_name": bundle.get("model_name", bundle.get("model_type", "unknown")),
        "cleaned_text": cleaned_texts[0],
        "top_probabilities": ranked[: max(int(top_n), 0)],
    }
