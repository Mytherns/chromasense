from __future__ import annotations

import colorsys
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

FEATURE_COLUMNS = (
    "avg_saturation",
    "avg_lightness",
    "avg_hue",
    "contrast_spread",
    "hue_variance",
    "warm_ratio",
)


def palette_to_features(colors: Sequence[Mapping]) -> dict[str, float]:
    # Convert palette color dicts into stable classifier feature schema.
    if not colors:
        raise ValueError("colors must not be empty")

    weights = _normalized_weights(colors)
    saturations = np.array([_color_metric(color, "saturation") for color in colors], dtype=np.float64)
    lightness = np.array([_color_metric(color, "lightness") for color in colors], dtype=np.float64)
    hues = np.array([_color_metric(color, "hue") for color in colors], dtype=np.float64)

    avg_hue, hue_variance = _weighted_circular_hue(hues, weights)
    warm_mask = np.array([_is_warm_hue(hue) for hue in hues], dtype=np.float64)

    return {
        "avg_saturation": round(float(np.average(saturations, weights=weights)), 6),
        "avg_lightness": round(float(np.average(lightness, weights=weights)), 6),
        "avg_hue": round(avg_hue, 6),
        "contrast_spread": round(float(lightness.max() - lightness.min()), 6),
        "hue_variance": round(hue_variance, 6),
        "warm_ratio": round(float(np.average(warm_mask, weights=weights)), 6),
    }


def feature_vector(colors: Sequence[Mapping]) -> list[float]:
    features = palette_to_features(colors)
    return [features[column] for column in FEATURE_COLUMNS]


def assert_feature_schema(feature_columns: Sequence[str]) -> None:
    expected = list(FEATURE_COLUMNS)
    actual = list(feature_columns)
    if actual != expected:
        raise ValueError(f"feature schema mismatch: expected {expected}, got {actual}")


def _normalized_weights(colors: Sequence[Mapping]) -> np.ndarray:
    raw = np.array([float(color.get("percentage", 0.0)) for color in colors], dtype=np.float64)
    if np.any(raw < 0):
        raise ValueError("color percentages must be non-negative")
    total = float(raw.sum())
    if total <= 0:
        return np.full(len(colors), 1 / len(colors), dtype=np.float64)
    return raw / total


def _color_metric(color: Mapping, key: str) -> float:
    if key in color:
        return float(color[key])
    rgb = color.get("rgb")
    if rgb is None:
        raise ValueError(f"color missing {key} and rgb fallback")
    hue, lightness, saturation = _rgb_to_hsl(rgb)
    return {"hue": hue, "lightness": lightness, "saturation": saturation}[key]


def _weighted_circular_hue(hues: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    radians = np.deg2rad(hues)
    sin_mean = float(np.average(np.sin(radians), weights=weights))
    cos_mean = float(np.average(np.cos(radians), weights=weights))
    mean_angle = math.atan2(sin_mean, cos_mean)
    if mean_angle < 0:
        mean_angle += 2 * math.pi
    resultant_length = min(1.0, math.hypot(sin_mean, cos_mean))
    circular_variance = 1.0 - resultant_length
    return math.degrees(mean_angle), circular_variance


def _is_warm_hue(hue: float) -> bool:
    normalized = hue % 360.0
    return normalized <= 75.0 or normalized >= 330.0


def _rgb_to_hsl(rgb: Iterable[float]) -> tuple[float, float, float]:
    r, g, b = [float(value) / 255.0 for value in rgb]
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    return hue * 360.0, lightness, saturation
