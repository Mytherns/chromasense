from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from chromasense.features import FEATURE_COLUMNS, assert_feature_schema
from chromasense.style_features import STYLE_MODEL_FEATURE_COLUMNS, style_model_matrix


def train_style_classifier(
    features_path: Path,
    model_output: Path,
    metrics_output: Path,
    min_per_class: int = 150,
    seed: int = 42,
    test_size: float = 0.2,
) -> dict:
    rows = _read_feature_rows(features_path)
    labels = sorted({row["style"] for row in rows})
    class_counts = Counter(row["style"] for row in rows)
    too_small = {label: count for label, count in class_counts.items() if count < min_per_class}
    if too_small:
        raise ValueError(
            "not saving model; classes below minimum usable images: "
            + ", ".join(f"{label}={count}" for label, count in sorted(too_small.items()))
            + f" (minimum={min_per_class})"
        )
    if len(labels) < 2:
        raise ValueError("need at least two style classes to train classifier")

    x = style_model_matrix(rows)
    y = np.array([row["style"] for row in rows])
    image_paths = np.array([row["image_path"] for row in rows])

    train_idx, val_idx = train_test_split(
        np.arange(len(rows)),
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    overlap = set(image_paths[train_idx]).intersection(set(image_paths[val_idx]))
    if overlap:
        raise ValueError("same image path appears in train and validation split")

    model_name, model, candidate_results = _fit_best_model(x[train_idx], y[train_idx], x[val_idx], y[val_idx], seed)
    predictions = model.predict(x[val_idx])
    validation_accuracy = float(accuracy_score(y[val_idx], predictions))
    matrix = confusion_matrix(y[val_idx], predictions, labels=labels)

    metrics = {
        "validation_accuracy": round(validation_accuracy, 6),
        "random_baseline": round(1 / len(labels), 6),
        "selected_model": model_name,
        "candidate_results": candidate_results,
        "model_feature_columns": list(STYLE_MODEL_FEATURE_COLUMNS),
        "labels": labels,
        "per_class_counts": dict(sorted(class_counts.items())),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y[val_idx],
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "train_size": int(len(train_idx)),
        "validation_size": int(len(val_idx)),
        "split_strategy": "stratified_80_20_by_path",
        "leakage_limitation": "Artist grouping not used; document possible artist/style leakage if filenames lack artist metadata.",
        "style_features_source": "kmeans",
        "qualitative_check": _qualitative_check_note(Path("test_images/demo_generalization")),
    }

    examples = [
        {
            "style": row["style"],
            "image_path": row["image_path"],
            "thumbnail_ref": None,
        }
        for row in rows
    ]
    feature_stats = _feature_stats(x)
    bundle = {
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "model_feature_columns": list(STYLE_MODEL_FEATURE_COLUMNS),
        "labels": labels,
        "metrics": metrics,
        "seed": seed,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_examples": examples,
        "training_feature_matrix": x.tolist(),
        "training_feature_stats": feature_stats,
    }
    assert_feature_schema(bundle["feature_columns"])

    model_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_output)
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _fit_best_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    seed: int,
) -> tuple[str, object, list[dict[str, float | str]]]:
    candidates = [
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=700,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(
                n_estimators=160,
                learning_rate=0.05,
                max_depth=3,
                random_state=seed,
            ),
        ),
    ]

    ranked = []
    for order, (name, model) in enumerate(candidates):
        model.fit(train_x, train_y)
        predictions = model.predict(val_x)
        accuracy = float(accuracy_score(val_y, predictions))
        ranked.append((accuracy, -order, name, model))

    ranked.sort(reverse=True)
    candidate_results = [
        {"model": name, "validation_accuracy": round(float(accuracy), 6)}
        for accuracy, _order, name, _model in ranked
    ]
    best_accuracy, _best_order, best_name, best_model = ranked[0]
    best_model.fit(train_x, train_y)
    return best_name, best_model, candidate_results


def _read_feature_rows(features_path: Path) -> list[dict[str, str]]:
    with features_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        actual_columns = list(reader.fieldnames or [])
        feature_columns = [column for column in actual_columns if column in FEATURE_COLUMNS]
        assert_feature_schema(feature_columns)
        rows = list(reader)
    if not rows:
        raise ValueError(f"training feature CSV has no rows: {features_path}")
    required = {"style", "image_path", *FEATURE_COLUMNS}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError("training feature CSV missing columns: " + ", ".join(sorted(missing)))
    return rows


def _feature_stats(matrix: np.ndarray) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for index, column in enumerate(STYLE_MODEL_FEATURE_COLUMNS):
        values = matrix[:, index]
        stats[column] = {
            "mean": round(float(values.mean()), 6),
            "std": round(float(values.std()), 6),
            "min": round(float(values.min()), 6),
            "max": round(float(values.max()), 6),
        }
    return stats


def _qualitative_check_note(path: Path) -> dict[str, str | int]:
    image_count = 0
    if path.exists():
        image_count = sum(1 for item in path.rglob("*") if item.is_file())
    return {
        "path": str(path),
        "image_count": image_count,
        "note": "Use for qualitative demo discussion only; no formal accuracy claim.",
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RandomForest art-style classifier from extracted features.")
    parser.add_argument("--features", type=Path, default=Path("data/training_features_style.csv"))
    parser.add_argument("--model-output", type=Path, default=Path("models/style_classifier.joblib"))
    parser.add_argument("--metrics-output", type=Path, default=Path("models/style_classifier_metrics.json"))
    parser.add_argument("--min-per-class", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = train_style_classifier(
        features_path=args.features,
        model_output=args.model_output,
        metrics_output=args.metrics_output,
        min_per_class=args.min_per_class,
        seed=args.seed,
        test_size=args.test_size,
    )
    print(
        "validation_accuracy={accuracy:.3f} random_baseline={baseline:.3f} labels={labels}".format(
            accuracy=metrics["validation_accuracy"],
            baseline=metrics["random_baseline"],
            labels=",".join(metrics["labels"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
