import json

import pytest

from chromasense.generate import generate_mood


def _features(**overrides):
    values = {
        "avg_saturation": 0.7,
        "avg_lightness": 0.6,
        "avg_hue": 30.0,
        "contrast_spread": 0.5,
        "hue_variance": 0.2,
        "warm_ratio": 0.8,
    }
    values.update(overrides)
    return values


def test_generate_mood_is_deterministic_and_uses_style_bucket():
    first = generate_mood("Pop Art", _features())
    second = generate_mood("pop-art", _features())

    assert first == second
    assert first["name"] in {"Electric Heat", "Poster Pulse"}
    assert first["temperature"] == "warm"
    assert first["tags"] == ["warm", "vivid", "bright"]
    assert first["status"] == "ok"


def test_generate_mood_uses_cool_palette_tags():
    result = generate_mood(
        "impressionism",
        _features(avg_saturation=0.2, avg_lightness=0.3, warm_ratio=0.5),
    )

    assert result["temperature"] == "cool"
    assert result["tags"] == ["cool", "muted", "deep"]


def test_generate_mood_falls_back_for_unknown_style():
    result = generate_mood("unknown future style", _features())

    assert result["name"] in {"Warm Resonance", "Sunward Tone"}


def test_generate_mood_rejects_incomplete_features():
    with pytest.raises(ValueError, match="mood features missing columns"):
        generate_mood("baroque", {"warm_ratio": 0.8})


def test_generate_mood_rejects_missing_temperature_bucket(tmp_path):
    lexicon_path = tmp_path / "mood.json"
    lexicon_path.write_text(
        json.dumps({"default": {"warm": [{"name": "Warm", "tagline": "Warm."}]}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"default\.cool"):
        generate_mood("missing", _features(warm_ratio=0.2), lexicon_path=lexicon_path)
