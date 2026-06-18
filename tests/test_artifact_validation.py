import json

import pandas as pd

from scripts.validate_artifacts import validate_artifacts


def test_validate_artifacts_accepts_consistent_baseline_only_state(tmp_path) -> None:
    reports_dir = tmp_path / "reports"
    processed_dir = tmp_path / "data" / "processed"
    reports_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "text": ["a", "b", "c", "d"],
            "label": ["Positive", "Negative", "Positive", "Negative"],
        }
    ).to_csv(processed_dir / "test.csv", index=False)

    metrics = {
        "model_name": "tfidf_baseline",
        "model_family": "baseline",
        "accuracy": 0.75,
        "macro_f1": 0.7,
        "weighted_f1": 0.74,
        "labels": ["Positive", "Negative"],
        "confusion_matrix": [[1, 1], [0, 2]],
    }
    comparison = {
        "mode": "baseline_only",
        "baseline_only": True,
        "champion_model": "tfidf_baseline",
        "models": {"tfidf_baseline": metrics},
    }
    error_analysis = {
        "model_name": "tfidf_baseline",
        "total_examples": 4,
        "total_wrong": 1,
        "derived_accuracy": 0.75,
    }

    (reports_dir / "model_comparison.json").write_text(json.dumps(comparison), encoding="utf-8")
    (reports_dir / "error_analysis.json").write_text(json.dumps(error_analysis), encoding="utf-8")

    result = validate_artifacts(tmp_path, verbose=False, load_model_bundle=False)

    assert result["ok"] is True
    assert result["mode"] == "baseline_only"
    assert result["total_wrong"] == 1
    assert result["derived_accuracy"] == 0.75
