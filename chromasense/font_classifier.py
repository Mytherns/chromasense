from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from chromasense.features import FEATURE_COLUMNS


@dataclass(frozen=True)
class _FontProfile:
    fonts: tuple[str, ...]
    targets: Mapping[str, float]


_FEATURE_WEIGHTS = {
    "avg_saturation": 1.4,
    "avg_lightness": 1.2,
    "avg_hue": 0.4,
    "contrast_spread": 1.5,
    "hue_variance": 1.0,
    "warm_ratio": 1.3,
}

_FONT_PROFILES = {
    "Serif-Elegant": _FontProfile(
        fonts=("Playfair Display", "Cormorant Garamond", "Lora"),
        targets={
            "avg_saturation": 0.30,
            "avg_lightness": 0.42,
            "avg_hue": 35.0,
            "contrast_spread": 0.58,
            "hue_variance": 0.20,
            "warm_ratio": 0.65,
        },
    ),
    "Geometric-Sans": _FontProfile(
        fonts=("Poppins", "Montserrat", "Inter"),
        targets={
            "avg_saturation": 0.52,
            "avg_lightness": 0.56,
            "avg_hue": 200.0,
            "contrast_spread": 0.45,
            "hue_variance": 0.22,
            "warm_ratio": 0.40,
        },
    ),
    "Display-Bold": _FontProfile(
        fonts=("Bebas Neue", "Anton", "Archivo Black"),
        targets={
            "avg_saturation": 0.88,
            "avg_lightness": 0.45,
            "avg_hue": 15.0,
            "contrast_spread": 0.88,
            "hue_variance": 0.34,
            "warm_ratio": 0.72,
        },
    ),
    "Handwritten-Organic": _FontProfile(
        fonts=("Pacifico", "Caveat", "Amatic SC"),
        targets={
            "avg_saturation": 0.62,
            "avg_lightness": 0.62,
            "avg_hue": 90.0,
            "contrast_spread": 0.34,
            "hue_variance": 0.68,
            "warm_ratio": 0.76,
        },
    ),
    "Monospace-Technical": _FontProfile(
        fonts=("Space Mono", "JetBrains Mono", "IBM Plex Mono"),
        targets={
            "avg_saturation": 0.18,
            "avg_lightness": 0.34,
            "avg_hue": 215.0,
            "contrast_spread": 0.76,
            "hue_variance": 0.12,
            "warm_ratio": 0.18,
        },
    ),
    "Soft-Rounded": _FontProfile(
        fonts=("Quicksand", "Comfortaa", "Nunito"),
        targets={
            "avg_saturation": 0.38,
            "avg_lightness": 0.80,
            "avg_hue": 305.0,
            "contrast_spread": 0.18,
            "hue_variance": 0.28,
            "warm_ratio": 0.46,
        },
    ),
}


class FontHeuristic:
    """Recommend a font category from palette features without claiming trained ML."""

    def predict(self, features: Mapping[str, float]) -> dict:
        vector = _validated_features(features)
        scores = {
            category: _profile_score(vector, profile.targets)
            for category, profile in _FONT_PROFILES.items()
        }
        ranked = sorted(scores, key=lambda category: (-scores[category], category))
        category = ranked[0]
        winning_score = scores[category]
        margin = winning_score - scores[ranked[1]]

        # Confidence combines absolute profile fit and separation from runner-up.
        # It is a bounded heuristic score, not an ML probability.
        confidence = max(0.0, min(1.0, 0.7 * winning_score + 0.3 * margin))
        fonts = _FONT_PROFILES[category].fonts

        return {
            "category": category,
            "confidence": round(confidence, 6),
            "recommended_fonts": list(fonts),
            "font_used": fonts[0],
            "preview_image_base64": None,
            "status": "heuristic",
        }


def recommend_font(features: Mapping[str, float]) -> dict:
    return FontHeuristic().predict(features)


def _validated_features(features: Mapping[str, float]) -> dict[str, float]:
    missing = [column for column in FEATURE_COLUMNS if column not in features]
    if missing:
        raise ValueError("font features missing columns: " + ", ".join(missing))

    values = {column: float(features[column]) for column in FEATURE_COLUMNS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("font features contain non-finite values")

    unit_interval = (
        "avg_saturation",
        "avg_lightness",
        "contrast_spread",
        "hue_variance",
        "warm_ratio",
    )
    invalid = [column for column in unit_interval if not 0.0 <= values[column] <= 1.0]
    if invalid:
        raise ValueError("font features outside 0..1: " + ", ".join(invalid))

    values["avg_hue"] %= 360.0
    return values


def _profile_score(features: Mapping[str, float], targets: Mapping[str, float]) -> float:
    weighted_distance = 0.0
    total_weight = 0.0
    for column in FEATURE_COLUMNS:
        weight = _FEATURE_WEIGHTS[column]
        if column == "avg_hue":
            distance = _circular_hue_distance(features[column], targets[column]) / 180.0
        else:
            distance = abs(features[column] - targets[column])
        weighted_distance += weight * distance
        total_weight += weight
    return max(0.0, 1.0 - weighted_distance / total_weight)


def _circular_hue_distance(left: float, right: float) -> float:
    difference = abs((left % 360.0) - (right % 360.0))
    return min(difference, 360.0 - difference)
