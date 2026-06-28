from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from chromasense.features import FEATURE_COLUMNS
from chromasense.style_classifier import StyleClassifier, load_style_classifier


def test_style_classifier_returns_scores_explanation_and_nearest_examples(tmp_path):
    model_path = tmp_path / "style_classifier.joblib"
    _write_model_bundle(model_path)

    classifier = load_style_classifier(model_path)
    result = classifier.predict(_feature_mapping([0.81, 0.72, 210.0, 0.51, 0.42, 0.23]), nearest_count=2)

    assert result["status"] == "ok"
    assert result["prediction"] in {"baroque", "pop_art"}
    assert set(result["all_scores"]) == {"baroque", "pop_art"}
    assert sum(result["all_scores"].values()) == pytest.approx(1.0, abs=1e-6)
    assert result["confidence"] == max(result["all_scores"].values())
    assert "training mean" in result["explanation"]
    assert "do not prove" in result["explanation"]
    assert result["style_features_source"] == "kmeans"
    assert len(result["nearest_training_examples"]) == 2
    assert result["nearest_training_examples"][0]["distance"] >= 0
    assert result["nearest_training_examples"][0]["thumbnail_ref"] is None
    assert result["nearest_training_examples"][0]["thumbnail_available"] is False


def test_style_classifier_rejects_bundle_feature_schema_mismatch(tmp_path):
    model_path = tmp_path / "style_classifier.joblib"
    bundle = _model_bundle()
    bundle["feature_columns"] = list(reversed(FEATURE_COLUMNS))
    joblib.dump(bundle, model_path)

    with pytest.raises(ValueError, match="feature schema mismatch"):
        StyleClassifier.load(model_path)


def test_style_classifier_rejects_missing_or_non_finite_features(tmp_path):
    model_path = tmp_path / "style_classifier.joblib"
    _write_model_bundle(model_path)
    classifier = StyleClassifier.load(model_path)

    with pytest.raises(ValueError, match="missing columns"):
        classifier.predict({"avg_saturation": 0.5})

    invalid = _feature_mapping([0.5, 0.5, np.nan, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="non-finite"):
        classifier.predict(invalid)


def _write_model_bundle(path: Path) -> None:
    joblib.dump(_model_bundle(), path)


def _model_bundle() -> dict:
    matrix = np.array(
        [
            [0.20, 0.30, 25.0, 0.20, 0.10, 0.90],
            [0.25, 0.35, 35.0, 0.25, 0.15, 0.80],
            [0.75, 0.70, 200.0, 0.50, 0.40, 0.20],
            [0.85, 0.75, 220.0, 0.55, 0.45, 0.10],
        ],
        dtype=np.float64,
    )
    labels = np.array(["baroque", "baroque", "pop_art", "pop_art"])
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(matrix, labels)
    stats = {}
    for index, column in enumerate(FEATURE_COLUMNS):
        values = matrix[:, index]
        stats[column] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "labels": [str(label) for label in model.classes_],
        "metrics": {},
        "seed": 42,
        "trained_at": "2026-06-24T00:00:00+00:00",
        "training_examples": [
            {"style": str(label), "image_path": f"example-{index}.jpg", "thumbnail_ref": None}
            for index, label in enumerate(labels)
        ],
        "training_feature_matrix": matrix.tolist(),
        "training_feature_stats": stats,
    }


def _feature_mapping(values: list[float]) -> dict[str, float]:
    return dict(zip(FEATURE_COLUMNS, values))
