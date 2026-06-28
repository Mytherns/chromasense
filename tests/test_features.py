import pytest

from chromasense.features import FEATURE_COLUMNS, assert_feature_schema, feature_vector, palette_to_features


def test_palette_to_features_uses_stable_schema_order():
    colors = [
        {"rgb": [255, 0, 0], "percentage": 0.7},
        {"rgb": [0, 0, 255], "percentage": 0.3},
    ]

    features = palette_to_features(colors)

    assert list(features) == list(FEATURE_COLUMNS)
    assert feature_vector(colors) == [features[column] for column in FEATURE_COLUMNS]
    assert features["warm_ratio"] == 0.7
    assert 0.0 <= features["avg_saturation"] <= 1.0
    assert 0.0 <= features["avg_lightness"] <= 1.0


def test_assert_feature_schema_rejects_drift():
    with pytest.raises(ValueError, match="feature schema mismatch"):
        assert_feature_schema(["avg_saturation"])

