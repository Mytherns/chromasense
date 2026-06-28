from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from chromasense.features import FEATURE_COLUMNS, assert_feature_schema

STYLE_MODEL_FEATURE_COLUMNS = (
    "avg_saturation",
    "avg_lightness",
    "hue_sin",
    "hue_cos",
    "contrast_spread",
    "hue_variance",
    "warm_ratio",
    "saturation_lightness",
    "saturation_contrast",
    "lightness_contrast",
    "hue_variance_warmth",
)


def style_model_vector(features: Mapping[str, float]) -> list[float]:
    base = _validated_base_features(features)
    hue_radians = math.radians(base["avg_hue"] % 360.0)
    return [
        base["avg_saturation"],
        base["avg_lightness"],
        math.sin(hue_radians),
        math.cos(hue_radians),
        base["contrast_spread"],
        base["hue_variance"],
        base["warm_ratio"],
        base["avg_saturation"] * base["avg_lightness"],
        base["avg_saturation"] * base["contrast_spread"],
        base["avg_lightness"] * base["contrast_spread"],
        base["hue_variance"] * base["warm_ratio"],
    ]


def style_model_matrix(rows: Sequence[Mapping[str, float | str]]) -> np.ndarray:
    return np.array([style_model_vector(row) for row in rows], dtype=np.float64)


def validate_style_model_feature_schema(feature_columns: Sequence[str]) -> None:
    expected = list(STYLE_MODEL_FEATURE_COLUMNS)
    actual = list(feature_columns)
    if actual != expected:
        raise ValueError(f"style model feature schema mismatch: expected {expected}, got {actual}")


def _validated_base_features(features: Mapping[str, float | str]) -> dict[str, float]:
    assert_feature_schema(FEATURE_COLUMNS)
    missing = [column for column in FEATURE_COLUMNS if column not in features]
    if missing:
        raise ValueError("style features missing columns: " + ", ".join(missing))

    values = {column: float(features[column]) for column in FEATURE_COLUMNS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("style features contain non-finite values")

    unit_interval = (
        "avg_saturation",
        "avg_lightness",
        "contrast_spread",
        "hue_variance",
        "warm_ratio",
    )
    invalid = [column for column in unit_interval if not 0.0 <= values[column] <= 1.0]
    if invalid:
        raise ValueError("style features outside 0..1: " + ", ".join(invalid))

    values["avg_hue"] %= 360.0
    return values
