from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.color import deltaE_ciede2000, rgb2lab

DEFAULT_REFERENCE_PALETTES_PATH = Path("data/reference_palettes.json")

_HEX_PATTERN = re.compile(r"^#?[0-9a-fA-F]{6}$")
_CARDINALITY_PENALTY = 8.0


def find_similar_palettes(
    palette: Sequence[Mapping | str],
    reference_path: Path | str = DEFAULT_REFERENCE_PALETTES_PATH,
    limit: int = 3,
) -> dict:
    """Find nearest named reference palettes using CIEDE2000 color distance."""
    if limit < 1:
        raise ValueError("limit must be >= 1")

    query_hexes = _palette_hexes(palette)
    references = load_reference_palettes(Path(reference_path))
    scored = []
    for reference in references:
        distance = palette_distance(query_hexes, reference["hexes"])
        scored.append(
            {
                "name": reference["name"],
                "hexes": reference["hexes"],
                "source": reference["source"],
                "source_url": reference.get("source_url"),
                "distance": round(distance, 4),
                "similarity": round(_distance_to_similarity(distance), 6),
            }
        )

    matches = sorted(scored, key=lambda item: (item["distance"], item["name"]))[:limit]
    return {
        "matches": matches,
        "metric": "mean_ciede2000_delta_e_with_hungarian_matching",
        "reference_count": len(references),
        "status": "ok",
    }


def load_reference_palettes(path: Path = DEFAULT_REFERENCE_PALETTES_PATH) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"reference palettes not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_palettes = payload.get("palettes") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_palettes, list) or not raw_palettes:
        raise ValueError("reference palettes must be a non-empty list")

    return [_validated_reference_palette(item, index) for index, item in enumerate(raw_palettes)]


def palette_distance(left: Sequence[Mapping | str], right: Sequence[Mapping | str]) -> float:
    left_hexes = _palette_hexes(left)
    right_hexes = _palette_hexes(right)
    distance = _distance_matrix(_hexes_to_lab(left_hexes), _hexes_to_lab(right_hexes))
    left_indices, right_indices = linear_sum_assignment(distance)
    matched = [float(distance[left_index, right_index]) for left_index, right_index in zip(left_indices, right_indices)]
    mean_distance = float(np.mean(matched)) if matched else 0.0
    size_penalty = abs(len(left_hexes) - len(right_hexes)) * _CARDINALITY_PENALTY
    return round(mean_distance + size_penalty, 6)


def _validated_reference_palette(item: object, index: int) -> dict:
    if not isinstance(item, Mapping):
        raise ValueError(f"reference palette #{index} must be an object")

    name = str(item.get("name", "")).strip()
    source = str(item.get("source", "")).strip()
    hexes = item.get("hexes")
    if not name:
        raise ValueError(f"reference palette #{index} missing name")
    if not source:
        raise ValueError(f"reference palette {name!r} missing source")
    if not isinstance(hexes, list) or not hexes:
        raise ValueError(f"reference palette {name!r} must include hexes")

    validated = {
        "name": name,
        "hexes": [_normalize_hex(hex_color) for hex_color in hexes],
        "source": source,
    }
    source_url = item.get("source_url")
    if source_url:
        validated["source_url"] = str(source_url)
    return validated


def _palette_hexes(palette: Sequence[Mapping | str]) -> list[str]:
    if not palette:
        raise ValueError("palette must not be empty")

    hexes = []
    for item in palette:
        if isinstance(item, str):
            hexes.append(_normalize_hex(item))
            continue
        if not isinstance(item, Mapping) or "hex" not in item:
            raise ValueError("palette entries must be hex strings or mappings with hex")
        hexes.append(_normalize_hex(str(item["hex"])))
    return hexes


def _normalize_hex(hex_color: str) -> str:
    value = hex_color.strip()
    if not _HEX_PATTERN.match(value):
        raise ValueError(f"invalid hex color: {hex_color!r}")
    if not value.startswith("#"):
        value = "#" + value
    return value.upper()


def _hexes_to_lab(hexes: Sequence[str]) -> np.ndarray:
    rgb = np.array([_hex_to_rgb(hex_color) for hex_color in hexes], dtype=np.float64) / 255.0
    return rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3)


def _hex_to_rgb(hex_color: str) -> list[int]:
    value = _normalize_hex(hex_color)[1:]
    return [int(value[index : index + 2], 16) for index in (0, 2, 4)]


def _distance_matrix(left_lab: np.ndarray, right_lab: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(left_lab), len(right_lab)), dtype=np.float64)
    for left_index, left_color in enumerate(left_lab):
        left_repeated = np.repeat(left_color.reshape(1, 3), len(right_lab), axis=0)
        matrix[left_index] = deltaE_ciede2000(left_repeated, right_lab)
    return matrix


def _distance_to_similarity(distance: float) -> float:
    if not math.isfinite(distance):
        return 0.0
    return max(0.0, min(1.0, 1.0 - (distance / 100.0)))
