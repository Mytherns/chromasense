from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

from chromasense.features import FEATURE_COLUMNS, assert_feature_schema
from chromasense.style_features import (
    STYLE_MODEL_FEATURE_COLUMNS,
    style_model_vector,
    validate_style_model_feature_schema,
)


DEFAULT_MODEL_PATH = Path("models/style_classifier.joblib")

_FEATURE_NAMES = {
    "avg_saturation": "average saturation",
    "avg_lightness": "average lightness",
    "avg_hue": "average hue",
    "hue_sin": "hue position",
    "hue_cos": "hue temperature axis",
    "contrast_spread": "lightness contrast",
    "hue_variance": "hue variation",
    "warm_ratio": "warm-color ratio",
    "saturation_lightness": "saturation and brightness blend",
    "saturation_contrast": "saturation and contrast blend",
    "lightness_contrast": "brightness and contrast blend",
    "hue_variance_warmth": "hue variation and warmth blend",
}


class StyleClassifier:
    def __init__(self, bundle: Mapping, model_path: Path | None = None) -> None:
        self.model_path = model_path
        self._bundle = dict(bundle)
        self._validate_bundle()

        self.model = self._bundle["model"]
        self.feature_columns = tuple(self._bundle["feature_columns"])
        self.model_feature_columns = tuple(self._bundle.get("model_feature_columns", self.feature_columns))
        self.labels = tuple(str(label) for label in self._bundle["labels"])
        self.training_examples = list(self._bundle["training_examples"])
        self.training_feature_stats = dict(self._bundle["training_feature_stats"])
        self._training_matrix = np.asarray(self._bundle["training_feature_matrix"], dtype=np.float64)

        self._means = np.array(
            [float(self.training_feature_stats[column]["mean"]) for column in self.model_feature_columns],
            dtype=np.float64,
        )
        self._scales = np.array(
            [float(self.training_feature_stats[column]["std"]) for column in self.model_feature_columns],
            dtype=np.float64,
        )
        self._scales[~np.isfinite(self._scales) | (self._scales <= 0)] = 1.0

        standardized_matrix = (self._training_matrix - self._means) / self._scales
        self._neighbors = NearestNeighbors(metric="euclidean")
        self._neighbors.fit(standardized_matrix)

    @classmethod
    def load(cls, model_path: Path | str = DEFAULT_MODEL_PATH) -> "StyleClassifier":
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"style model not found: {path}")
        bundle = joblib.load(path)
        if not isinstance(bundle, Mapping):
            raise ValueError("style model bundle must be a mapping")
        return cls(bundle, model_path=path)

    def predict(self, features: Mapping[str, float], nearest_count: int = 3) -> dict:
        if nearest_count < 1:
            raise ValueError("nearest_count must be at least 1")

        vector = self._feature_vector(features)
        probabilities = np.asarray(self.model.predict_proba(vector.reshape(1, -1))[0], dtype=np.float64)
        model_classes = [str(label) for label in self.model.classes_]
        scores = {label: round(float(score), 6) for label, score in zip(model_classes, probabilities)}
        prediction_index = int(np.argmax(probabilities))
        prediction = model_classes[prediction_index]

        return {
            "status": "ok",
            "prediction": prediction,
            "confidence": round(float(probabilities[prediction_index]), 6),
            "all_scores": {label: scores[label] for label in self.labels},
            "explanation": self._explain(prediction, vector),
            "nearest_training_examples": self._nearest_examples(vector, nearest_count),
            "style_features_source": "kmeans",
        }

    def _validate_bundle(self) -> None:
        required = {
            "model",
            "feature_columns",
            "labels",
            "metrics",
            "seed",
            "trained_at",
            "training_examples",
            "training_feature_matrix",
            "training_feature_stats",
        }
        missing = required.difference(self._bundle)
        if missing:
            raise ValueError("style model bundle missing keys: " + ", ".join(sorted(missing)))

        assert_feature_schema(self._bundle["feature_columns"])
        model_feature_columns = tuple(self._bundle.get("model_feature_columns", self._bundle["feature_columns"]))
        if "model_feature_columns" in self._bundle:
            validate_style_model_feature_schema(model_feature_columns)
        model = self._bundle["model"]
        if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
            raise ValueError("style model must support predict_proba and classes_")
        if not hasattr(model, "feature_importances_"):
            raise ValueError("style model must expose feature_importances_")

        labels = [str(label) for label in self._bundle["labels"]]
        model_classes = [str(label) for label in model.classes_]
        if labels != model_classes:
            raise ValueError(f"style label mismatch: bundle={labels}, model={model_classes}")

        matrix = np.asarray(self._bundle["training_feature_matrix"], dtype=np.float64)
        expected_shape = (len(self._bundle["training_examples"]), len(model_feature_columns))
        if matrix.ndim != 2 or matrix.shape != expected_shape or matrix.shape[0] == 0:
            raise ValueError(
                f"invalid training feature matrix shape: expected {expected_shape}, got {matrix.shape}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("training feature matrix contains non-finite values")

        importances = np.asarray(model.feature_importances_, dtype=np.float64)
        if importances.shape != (len(model_feature_columns),):
            raise ValueError("style model feature importance count does not match feature schema")

        stats = self._bundle["training_feature_stats"]
        for column in model_feature_columns:
            if column not in stats or "mean" not in stats[column] or "std" not in stats[column]:
                raise ValueError(f"training feature stats missing mean/std for {column}")

    def _feature_vector(self, features: Mapping[str, float]) -> np.ndarray:
        missing = [column for column in self.feature_columns if column not in features]
        if missing:
            raise ValueError("style features missing columns: " + ", ".join(missing))
        if self.model_feature_columns == tuple(FEATURE_COLUMNS):
            vector = np.array([float(features[column]) for column in self.feature_columns], dtype=np.float64)
        else:
            vector = np.array(style_model_vector(features), dtype=np.float64)
        if not np.all(np.isfinite(vector)):
            raise ValueError("style features contain non-finite values")
        return vector

    def _explain(self, prediction: str, vector: np.ndarray) -> str:
        importances = np.asarray(self.model.feature_importances_, dtype=np.float64)
        deviations = (vector - self._means) / self._scales
        influence = importances * np.abs(deviations)
        top_indices = np.argsort(-influence, kind="stable")[:3]

        factors = [
            f"{_FEATURE_NAMES[self.model_feature_columns[index]]} {_deviation_phrase(float(deviations[index]))}"
            for index in top_indices
        ]
        readable_prediction = prediction.replace("_", " ")
        return (
            f"Model leaned toward {readable_prediction}. Strongest signals were "
            f"{', '.join(factors)}. These signals influence the prediction; they do not prove the artwork's style."
        )

    def _nearest_examples(self, vector: np.ndarray, nearest_count: int) -> list[dict]:
        count = min(nearest_count, len(self.training_examples))
        standardized_vector = (vector - self._means) / self._scales
        distances, indices = self._neighbors.kneighbors(
            standardized_vector.reshape(1, -1),
            n_neighbors=count,
        )

        results = []
        for distance, index in zip(distances[0], indices[0]):
            example = self.training_examples[int(index)]
            thumbnail_ref = example.get("thumbnail_ref")
            thumbnail_path = Path(thumbnail_ref) if thumbnail_ref else None
            thumbnail_available = bool(thumbnail_path and thumbnail_path.is_file())
            results.append(
                {
                    "style": str(example["style"]),
                    "distance": round(float(distance), 6),
                    "image_path": example.get("image_path"),
                    "thumbnail_ref": str(thumbnail_path) if thumbnail_available else None,
                    "thumbnail_available": thumbnail_available,
                }
            )
        return results


def load_style_classifier(model_path: Path | str = DEFAULT_MODEL_PATH) -> StyleClassifier:
    return StyleClassifier.load(model_path)


def _deviation_phrase(z_score: float) -> str:
    if not math.isfinite(z_score) or abs(z_score) < 0.25:
        return "near the training mean"
    if z_score >= 1.0:
        return "well above the training mean"
    if z_score > 0:
        return "above the training mean"
    if z_score <= -1.0:
        return "well below the training mean"
    return "below the training mean"
