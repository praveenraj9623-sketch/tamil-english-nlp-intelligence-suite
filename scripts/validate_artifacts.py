from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover - exercised only in stripped environments
    joblib = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTION_METRIC_ORDER = ["macro_f1", "weighted_f1", "accuracy"]
SELECTION_METRIC_ORDER_LABEL = "Champion selected by macro F1, then weighted F1, then accuracy."


def load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def confusion_matrix_stats(metrics: dict[str, Any]) -> dict[str, int | float | None]:
    matrix = metrics.get("confusion_matrix")
    if not isinstance(matrix, list) or not matrix:
        return {"total": None, "correct": None, "wrong": None, "derived_accuracy": None}

    total = 0
    correct = 0
    for row_index, row in enumerate(matrix):
        if not isinstance(row, list):
            return {"total": None, "correct": None, "wrong": None, "derived_accuracy": None}
        row_values = [int(value) for value in row]
        total += sum(row_values)
        if row_index < len(row_values):
            correct += row_values[row_index]

    wrong = total - correct
    derived_accuracy = correct / total if total else None
    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "derived_accuracy": derived_accuracy,
    }


def select_display_metrics(reports_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    comparison = load_json(reports_dir / "model_comparison.json")
    if isinstance(comparison, dict):
        champion_name = comparison.get("champion_model")
        models = comparison.get("models", {})
        if isinstance(champion_name, str) and isinstance(models, dict) and champion_name in models:
            return models[champion_name], comparison

    champion_metrics = load_json(reports_dir / "champion_metrics.json")
    if isinstance(champion_metrics, dict):
        return champion_metrics, {
            "mode": "champion_metrics_only",
            "champion_model": champion_metrics.get("model_name"),
            "models": {champion_metrics.get("model_name", "champion"): champion_metrics},
        }

    baseline_metrics = load_json(reports_dir / "baseline_metrics.json")
    if isinstance(baseline_metrics, dict):
        return baseline_metrics, {
            "mode": "baseline_only",
            "baseline_only": True,
            "champion_model": baseline_metrics.get("model_name", "tfidf_baseline"),
            "models": {baseline_metrics.get("model_name", "tfidf_baseline"): baseline_metrics},
        }

    return None, {}


def _almost_equal(left: float | None, right: float | None, tolerance: float = 1e-8) -> bool:
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) <= tolerance


def validate_artifacts(
    project_root: str | Path = PROJECT_ROOT,
    *,
    verbose: bool = True,
    load_model_bundle: bool = True,
) -> dict[str, Any]:
    project_root = Path(project_root)
    reports_dir = project_root / "reports"
    models_dir = project_root / "models"
    test_path = project_root / "data" / "processed" / "test.csv"
    warnings: list[str] = []

    champion_metrics, comparison = select_display_metrics(reports_dir)
    if champion_metrics is None:
        warnings.append("No champion, baseline, or comparison metrics found in reports/.")
        result = {
            "ok": False,
            "warnings": warnings,
            "rerun_commands": rerun_commands(),
        }
        if verbose:
            print_validation_result(result)
        return result

    champion_name = str(comparison.get("champion_model") or champion_metrics.get("model_name") or "unknown")
    champion_stats = confusion_matrix_stats(champion_metrics)
    error_analysis = load_json(reports_dir / "error_analysis.json")

    test_rows = None
    if test_path.exists():
        test_rows = int(len(pd.read_csv(test_path)))
    else:
        warnings.append(f"Missing test split: {test_path}")

    if test_rows is not None and champion_stats["total"] is not None and test_rows != champion_stats["total"]:
        warnings.append(
            f"Champion confusion matrix total ({champion_stats['total']}) does not match test rows ({test_rows})."
        )

    if isinstance(error_analysis, dict):
        error_total = error_analysis.get("total_examples")
        error_wrong = error_analysis.get("total_wrong")
        if test_rows is not None and error_total != test_rows:
            warnings.append(f"error_analysis total_examples ({error_total}) does not match test rows ({test_rows}).")
        if champion_stats["wrong"] is not None and error_wrong != champion_stats["wrong"]:
            warnings.append(
                "error_analysis total_wrong "
                f"({error_wrong}) does not match champion confusion matrix wrong count ({champion_stats['wrong']})."
            )
        reported_accuracy = champion_metrics.get("accuracy")
        if not _almost_equal(champion_stats["derived_accuracy"], reported_accuracy):
            warnings.append(
                "Champion metric accuracy "
                f"({reported_accuracy}) does not match confusion-matrix derived accuracy "
                f"({champion_stats['derived_accuracy']})."
            )
        error_model_name = error_analysis.get("model_name")
        if error_model_name and error_model_name != champion_metrics.get("model_name"):
            warnings.append(
                f"error_analysis model_name ({error_model_name}) does not match champion metrics "
                f"({champion_metrics.get('model_name')})."
            )
    else:
        warnings.append("Missing reports/error_analysis.json. Run python -m src.error_analysis.")

    champion_path = models_dir / "champion_model.joblib"
    bundle_model_name = None
    if load_model_bundle:
        if not champion_path.exists():
            warnings.append(f"Missing champion model artifact: {champion_path}")
        elif joblib is None:
            warnings.append("joblib is not installed, so champion_model.joblib could not be inspected.")
        else:
            try:
                bundle = joblib.load(champion_path)
                bundle_model_name = bundle.get("model_name")
                if bundle_model_name != champion_name:
                    warnings.append(
                        f"champion_model.joblib model_name ({bundle_model_name}) does not match "
                        f"displayed champion ({champion_name})."
                    )
            except Exception as exc:  # pragma: no cover - defensive for corrupted local artifacts
                warnings.append(f"Could not inspect champion_model.joblib: {exc}")

    baseline_only = bool(comparison.get("baseline_only")) or comparison.get("mode") == "baseline_only"
    transformer_metrics_exists = (reports_dir / "transformer_metrics.json").exists()
    if baseline_only and transformer_metrics_exists:
        warnings.append(
            "model_comparison.json says baseline-only, but transformer_metrics.json exists. "
            "Rerun python -m src.transformer_model to refresh comparison."
        )

    result = {
        "ok": not warnings,
        "warnings": warnings,
        "mode": comparison.get("mode", "comparison" if not baseline_only else "baseline_only"),
        "baseline_only": baseline_only,
        "champion_model": champion_name,
        "bundle_model_name": bundle_model_name,
        "model_family": champion_metrics.get("model_family"),
        "accuracy": champion_metrics.get("accuracy"),
        "macro_f1": champion_metrics.get("macro_f1"),
        "weighted_f1": champion_metrics.get("weighted_f1"),
        "test_rows": test_rows,
        "confusion_matrix_total": champion_stats["total"],
        "total_correct": champion_stats["correct"],
        "total_wrong": champion_stats["wrong"],
        "derived_accuracy": champion_stats["derived_accuracy"],
        "selection_metric_order": SELECTION_METRIC_ORDER,
        "selection_metric_order_label": SELECTION_METRIC_ORDER_LABEL,
        "rerun_commands": rerun_commands(),
    }
    if verbose:
        print_validation_result(result)
    return result


def rerun_commands() -> list[str]:
    return [
        "python -m src.preprocessing --input data/raw/Tamil_sentiments.csv --label-col 0 --text-col 1",
        "python -m src.baseline_model",
        "python -m src.transformer_model",
        "python -m src.error_analysis",
        "python scripts/validate_artifacts.py",
    ]


def print_validation_result(result: dict[str, Any]) -> None:
    if result["ok"]:
        print("Artifact validation: OK")
    else:
        print("Artifact validation: WARNINGS")
        for warning in result["warnings"]:
            print(f"- {warning}")
        print("\nRecommended rebuild commands:")
        for command in result["rerun_commands"]:
            print(f"  {command}")

    summary_keys = [
        "mode",
        "champion_model",
        "model_family",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "test_rows",
        "total_wrong",
        "derived_accuracy",
    ]
    print("\nSummary:")
    for key in summary_keys:
        if key in result:
            print(f"- {key}: {result[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate model/report artifact consistency.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--skip-model-bundle", action="store_true")
    args = parser.parse_args()
    result = validate_artifacts(
        args.project_root,
        verbose=True,
        load_model_bundle=not args.skip_model_bundle,
    )
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
