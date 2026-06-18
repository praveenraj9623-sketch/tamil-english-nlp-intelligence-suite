from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.inference import DEFAULT_CHAMPION_PATH, load_model, model_family, predict_batch_proba
from src.preprocessing import PROJECT_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else PROJECT_ROOT / path_obj


def _plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8.5, 6.8), facecolor="#0b1220")
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, cmap="Oranges", values_format="d", xticks_rotation=35, colorbar=False)
    ax.set_title("Champion Model Confusion Matrix", color="#e8edf5", pad=16)
    ax.set_xlabel("Predicted label", color="#cbd5e1", labelpad=12)
    ax.set_ylabel("True label", color="#cbd5e1", labelpad=12)
    ax.set_facecolor("#0b1220")
    ax.tick_params(axis="x", colors="#cbd5e1", labelsize=8)
    ax.tick_params(axis="y", colors="#cbd5e1", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    for text in display.text_.ravel():
        text.set_fontsize(8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28, left=0.20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def generate_error_analysis(
    model_path: str | Path | None = None,
    test_path: str | Path | None = None,
    reports_dir: str | Path | None = None,
    top_n: int = 20,
) -> dict[str, Any]:
    model_path = _path(model_path, DEFAULT_CHAMPION_PATH)
    test_path = _path(test_path, PROJECT_ROOT / "data" / "processed" / "test.csv")
    reports_path = _path(reports_dir, PROJECT_ROOT / "reports")
    reports_path.mkdir(parents=True, exist_ok=True)

    if not test_path.exists():
        raise FileNotFoundError(f"Missing test split at {test_path}. Run preprocessing first.")

    test_df = pd.read_csv(test_path)
    bundle = load_model(model_path)
    classes, probabilities, cleaned_texts = predict_batch_proba(bundle, test_df["text"].astype(str).tolist())
    pred_indices = np.argmax(probabilities, axis=1)
    predicted_labels = [classes[index] for index in pred_indices]
    confidences = probabilities[np.arange(len(pred_indices)), pred_indices]
    true_labels = test_df["label"].astype(str).tolist()
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    total_examples = int(len(test_df))
    total_correct = int(np.trace(cm))
    total_wrong = int(total_examples - total_correct)
    derived_accuracy = float(total_correct / total_examples) if total_examples else 0.0

    _plot_confusion_matrix(true_labels, predicted_labels, classes, reports_path / "confusion_matrix_champion.png")

    wrong_examples: list[dict[str, Any]] = []
    original_texts = test_df.get("original_text", test_df["text"]).astype(str).tolist()
    for row_idx, (true_label, predicted_label, confidence) in enumerate(
        zip(true_labels, predicted_labels, confidences, strict=False)
    ):
        if true_label == predicted_label:
            continue
        probability_map = {
            label: float(probabilities[row_idx, class_idx]) for class_idx, label in enumerate(classes)
        }
        wrong_examples.append(
            {
                "row_index": int(row_idx),
                "original_text": original_texts[row_idx],
                "cleaned_text": cleaned_texts[row_idx],
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": float(confidence),
                "top_probabilities": dict(
                    sorted(probability_map.items(), key=lambda item: item[1], reverse=True)[:3]
                ),
            }
        )

    hardest = sorted(wrong_examples, key=lambda item: item["confidence"], reverse=True)[:top_n]
    analysis = {
        "model_path": str(model_path),
        "model_family": model_family(bundle),
        "model_name": bundle.get("model_name"),
        "labels": list(classes),
        "total_examples": total_examples,
        "total_correct": total_correct,
        "total_wrong": total_wrong,
        "derived_accuracy": derived_accuracy,
        "confusion_matrix": cm.tolist(),
        "top_n": int(top_n),
        "confusion_matrix_path": str(reports_path / "confusion_matrix_champion.png"),
        "hardest_misclassifications": hardest,
    }

    json_path = reports_path / "error_analysis.json"
    csv_path = reports_path / "error_analysis_top20.csv"
    json_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(hardest).to_csv(csv_path, index=False, encoding="utf-8")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate champion model error analysis.")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    generate_error_analysis(
        model_path=args.model_path,
        test_path=args.test_path,
        reports_dir=args.reports_dir,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
