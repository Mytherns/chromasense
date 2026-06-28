from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Mapping, Sequence

from chromasense.features import FEATURE_COLUMNS


DEFAULT_LEXICON_PATH = Path("data/mood_lexicon.json")

_TAG_THRESHOLDS = {
    "avg_saturation": 0.5,
    "avg_lightness": 0.5,
}


def generate_mood(
    art_style: str | None,
    features: Mapping[str, float],
    lexicon_path: Path | str = DEFAULT_LEXICON_PATH,
) -> dict:
    """Generate deterministic descriptive mood metadata from style and palette features."""
    vector = _validated_feature_vector(features)
    lexicon = _load_lexicon(Path(lexicon_path))
    style_key = _normalize_style(art_style)
    resolved_style = style_key if style_key in lexicon else "default"
    if resolved_style not in lexicon:
        raise ValueError("mood lexicon missing default style")

    temperature = "warm" if vector["warm_ratio"] > 0.5 else "cool"
    candidates = lexicon[resolved_style].get(temperature)
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"mood lexicon missing entries for {resolved_style}.{temperature}")

    candidate = candidates[_stable_index(vector, len(candidates))]
    if not isinstance(candidate, Mapping) or not candidate.get("name") or not candidate.get("tagline"):
        raise ValueError(f"invalid mood entry for {resolved_style}.{temperature}")

    return {
        "name": str(candidate["name"]),
        "tagline": str(candidate["tagline"]),
        "tags": _palette_tags(vector),
        "temperature": temperature,
        "status": "ok",
    }


def _load_lexicon(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"mood lexicon not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        lexicon = json.load(handle)
    if not isinstance(lexicon, dict):
        raise ValueError("mood lexicon must be an object")
    return lexicon


def _validated_feature_vector(features: Mapping[str, float]) -> dict[str, float]:
    missing = [column for column in FEATURE_COLUMNS if column not in features]
    if missing:
        raise ValueError("mood features missing columns: " + ", ".join(missing))

    vector = {column: float(features[column]) for column in FEATURE_COLUMNS}
    if not all(math.isfinite(value) for value in vector.values()):
        raise ValueError("mood features contain non-finite values")

    bounded = (
        "avg_saturation",
        "avg_lightness",
        "contrast_spread",
        "hue_variance",
        "warm_ratio",
    )
    invalid = [column for column in bounded if not 0.0 <= vector[column] <= 1.0]
    if invalid:
        raise ValueError("mood features outside 0..1: " + ", ".join(invalid))

    vector["avg_hue"] %= 360.0
    return vector


def _stable_index(features: Mapping[str, float], candidate_count: int) -> int:
    rounded_vector: Sequence[float] = [round(features[column], 4) for column in FEATURE_COLUMNS]
    payload = json.dumps(rounded_vector, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("ascii")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % candidate_count


def _palette_tags(features: Mapping[str, float]) -> list[str]:
    return [
        "warm" if features["warm_ratio"] > 0.5 else "cool",
        "vivid" if features["avg_saturation"] >= _TAG_THRESHOLDS["avg_saturation"] else "muted",
        "bright" if features["avg_lightness"] >= _TAG_THRESHOLDS["avg_lightness"] else "deep",
    ]


def _normalize_style(art_style: str | None) -> str:
    if not art_style:
        return "default"
    normalized = re.sub(r"[^a-z0-9]+", "_", str(art_style).strip().lower())
    return normalized.strip("_") or "default"
