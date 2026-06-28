import json

import pytest

from chromasense.palette_match import find_similar_palettes, load_reference_palettes, palette_distance


def test_find_similar_palettes_returns_exact_reference_first():
    result = find_similar_palettes(
        ["#390099", "#FFBD00", "#00BBF9", "#FF0054", "#9E0059"],
        limit=2,
    )

    assert result["status"] == "ok"
    assert result["reference_count"] == 12
    assert result["metric"] == "mean_ciede2000_delta_e_with_hungarian_matching"
    assert len(result["matches"]) == 2
    assert result["matches"][0]["name"] == "Pop Poster"
    assert result["matches"][0]["distance"] == 0.0
    assert result["matches"][0]["similarity"] == 1.0
    assert result["matches"][0]["source"] == "manual_curated"
    assert result["matches"][0]["source_url"].startswith("https://coolors.co/")


def test_find_similar_palettes_accepts_extracted_color_dicts_and_sorts():
    palette = [
        {"hex": "#264653", "percentage": 0.35},
        {"hex": "#2A9D8F", "percentage": 0.25},
        {"hex": "#E9C46A", "percentage": 0.2},
        {"hex": "#F4A261", "percentage": 0.12},
        {"hex": "#E76F51", "percentage": 0.08},
    ]

    matches = find_similar_palettes(palette, limit=3)["matches"]

    assert [match["distance"] for match in matches] == sorted(match["distance"] for match in matches)
    assert matches[0]["name"] == "Sunset Product"


def test_palette_distance_is_order_independent_with_hungarian_matching():
    left = ["#111111", "#EEEEEE", "#AA0000"]
    right = ["#AA0000", "#111111", "#EEEEEE"]

    assert palette_distance(left, right) == 0.0


def test_palette_distance_penalizes_missing_colors():
    full = ["#111111", "#EEEEEE", "#AA0000"]
    partial = ["#111111", "#EEEEEE"]

    assert palette_distance(full, partial) > 0.0


def test_find_similar_palettes_rejects_invalid_input():
    with pytest.raises(ValueError, match="limit"):
        find_similar_palettes(["#FFFFFF"], limit=0)

    with pytest.raises(ValueError, match="invalid hex"):
        find_similar_palettes(["not-a-color"])


def test_load_reference_palettes_validates_required_metadata(tmp_path):
    path = tmp_path / "reference_palettes.json"
    path.write_text(
        json.dumps({"palettes": [{"name": "Missing Source", "hexes": ["#FFFFFF"]}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing source"):
        load_reference_palettes(path)
