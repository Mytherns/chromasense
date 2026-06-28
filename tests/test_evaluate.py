from pathlib import Path

from evaluate import choose_primary_algorithm, evaluate_manifest, hex_to_rgb, match_palettes


def test_hex_to_rgb_accepts_hash_and_plain_values():
    assert hex_to_rgb("#0010FF") == [0, 16, 255]
    assert hex_to_rgb("D7322D") == [215, 50, 45]


def test_match_palettes_uses_hungarian_matching_order_independent():
    matches = match_palettes(
        truth_hexes=["#FF0000", "#0000FF"],
        predicted_hexes=["#0000F8", "#F80000"],
    )

    assert len(matches) == 2
    assert {match.truth_hex for match in matches} == {"#FF0000", "#0000FF"}
    assert all(match.delta_e < 3 for match in matches)


def test_choose_primary_algorithm_prefers_kmeans_on_exact_tie():
    results = {
        "kmeans": {"matched_delta_e_lt_threshold_rate": 1.0, "mean_delta_e": 0.0},
        "gmm": {"matched_delta_e_lt_threshold_rate": 1.0, "mean_delta_e": 0.0},
    }

    assert choose_primary_algorithm(results) == "kmeans"


def test_evaluate_manifest_reports_both_algorithms():
    report = evaluate_manifest(
        ground_truth_path=Path("data/ground_truth_palettes.json"),
        images_dir=Path("test_images/color_eval"),
        n_colors=5,
        seed=42,
        delta_e_threshold=6.0,
    )

    assert report["primary_algorithm"] in {"kmeans", "gmm"}
    assert report["style_features_source"] == "kmeans"
    assert set(report["results"]) == {"kmeans", "gmm"}
    assert report["results"]["kmeans"]["image_count"] == 2
