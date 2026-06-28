import pytest

from chromasense.font_classifier import FontHeuristic, recommend_font


def test_display_bold_profile_returns_required_response_shape():
    result = recommend_font(
        {
            "avg_saturation": 0.88,
            "avg_lightness": 0.45,
            "avg_hue": 15.0,
            "contrast_spread": 0.88,
            "hue_variance": 0.34,
            "warm_ratio": 0.72,
        }
    )

    assert result["category"] == "Display-Bold"
    assert result["confidence"] > 0.7
    assert result["recommended_fonts"] == ["Bebas Neue", "Anton", "Archivo Black"]
    assert result["font_used"] == "Bebas Neue"
    assert result["preview_image_base64"] is None
    assert result["status"] == "heuristic"


def test_soft_rounded_profile_is_deterministic():
    features = {
        "avg_saturation": 0.38,
        "avg_lightness": 0.80,
        "avg_hue": 305.0,
        "contrast_spread": 0.18,
        "hue_variance": 0.28,
        "warm_ratio": 0.46,
    }
    heuristic = FontHeuristic()

    first = heuristic.predict(features)
    second = heuristic.predict(features)

    assert first == second
    assert first["category"] == "Soft-Rounded"
    assert 0.0 <= first["confidence"] <= 1.0


def test_hue_is_normalized_before_scoring():
    base = {
        "avg_saturation": 0.30,
        "avg_lightness": 0.42,
        "avg_hue": 35.0,
        "contrast_spread": 0.58,
        "hue_variance": 0.20,
        "warm_ratio": 0.65,
    }
    wrapped = dict(base, avg_hue=395.0)

    assert recommend_font(base) == recommend_font(wrapped)


def test_missing_or_invalid_features_are_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        recommend_font({"avg_saturation": 0.5})

    with pytest.raises(ValueError, match="outside 0..1"):
        recommend_font(
            {
                "avg_saturation": 1.2,
                "avg_lightness": 0.5,
                "avg_hue": 0.0,
                "contrast_spread": 0.5,
                "hue_variance": 0.5,
                "warm_ratio": 0.5,
            }
        )
