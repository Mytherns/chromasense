import csv
from pathlib import Path

import joblib
import pytest
from PIL import Image

from chromasense.features import FEATURE_COLUMNS
from scripts.extract_features_style import extract_style_features
from scripts.extract_features_style_hf import extract_hf_style_features, list_style_labels
from scripts.prepare_subset import collect_subset, normalize_style_name
from scripts.train_style_classifier import train_style_classifier


def test_normalize_style_name_maps_aliases():
    assert normalize_style_name("Pop Art") == "pop_art"
    assert normalize_style_name("ukiyo-e") == "ukiyo_e"
    assert normalize_style_name("Art Nouveau!") == "art_nouveau"


def test_collect_subset_selects_target_without_copying_raw_images(tmp_path):
    source = tmp_path / "wikiart"
    style_dir = source / "Pop Art"
    style_dir.mkdir(parents=True)
    for index in range(3):
        Image.new("RGB", (8, 8), (index * 60, 10, 20)).save(style_dir / f"img_{index}.jpg")

    rows, missing = collect_subset(source, styles=["pop_art"], target_per_class=2, seed=1)

    assert missing == []
    assert len(rows) == 2
    assert all(row["style"] == "pop_art" for row in rows)
    assert not (tmp_path / "data").exists()


def test_extract_style_features_skips_corrupt_images(tmp_path):
    valid = tmp_path / "valid.jpg"
    corrupt = tmp_path / "corrupt.jpg"
    Image.new("RGB", (12, 12), "#ff0000").save(valid)
    corrupt.write_text("not an image", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    _write_csv(
        manifest,
        ["style", "image_path", "filename", "artist", "source_folder"],
        [
            {"style": "pop_art", "image_path": str(valid), "filename": valid.name, "artist": "", "source_folder": ""},
            {
                "style": "pop_art",
                "image_path": str(corrupt),
                "filename": corrupt.name,
                "artist": "",
                "source_folder": "",
            },
        ],
    )

    usable, skipped = extract_style_features(
        manifest_path=manifest,
        output_path=tmp_path / "features.csv",
        errors_output=tmp_path / "errors.csv",
        n_colors=3,
        seed=42,
    )

    assert usable == 1
    assert skipped == 1
    assert "corrupt.jpg" in (tmp_path / "errors.csv").read_text(encoding="utf-8")


def test_train_style_classifier_blocks_save_when_class_count_too_small(tmp_path):
    features = tmp_path / "features.csv"
    _write_feature_csv(features, rows_per_class=2)

    with pytest.raises(ValueError, match="not saving model"):
        train_style_classifier(
            features_path=features,
            model_output=tmp_path / "model.joblib",
            metrics_output=tmp_path / "metrics.json",
            min_per_class=3,
            seed=42,
        )

    assert not (tmp_path / "model.joblib").exists()


def test_train_style_classifier_saves_required_bundle_keys(tmp_path):
    features = tmp_path / "features.csv"
    model_output = tmp_path / "model.joblib"
    metrics_output = tmp_path / "metrics.json"
    _write_feature_csv(features, rows_per_class=6)

    metrics = train_style_classifier(
        features_path=features,
        model_output=model_output,
        metrics_output=metrics_output,
        min_per_class=2,
        seed=42,
    )

    bundle = joblib.load(model_output)
    assert set(
        [
            "model",
            "feature_columns",
            "labels",
            "metrics",
            "seed",
            "trained_at",
            "training_examples",
            "training_feature_matrix",
            "training_feature_stats",
        ]
    ).issubset(bundle)
    assert bundle["feature_columns"] == list(FEATURE_COLUMNS)
    assert metrics["random_baseline"] == 0.5
    assert metrics_output.exists()


def test_hf_stream_feature_extraction_uses_images_without_saving_raw_files(tmp_path):
    dataset = _FakeHfDataset(
        [
            {"style": 0, "artist": "a", "image": Image.new("RGB", (12, 12), "#C93030")},
            {"style": 1, "artist": "b", "image": Image.new("RGB", (12, 12), "#3030C9")},
            {"style": 0, "artist": "c", "image": Image.new("RGB", (12, 12), "#D25020")},
            {"style": 1, "artist": "d", "image": Image.new("RGB", (12, 12), "#2040D2")},
        ],
        style_names=["Baroque", "Impressionism"],
    )

    counts, skipped = extract_hf_style_features(
        dataset=dataset,
        output_path=tmp_path / "features.csv",
        errors_output=tmp_path / "errors.csv",
        styles=["baroque", "impressionism"],
        target_per_class=2,
        n_colors=3,
        seed=42,
    )

    assert counts["baroque"] == 2
    assert counts["impressionism"] == 2
    assert skipped == 0
    assert not list(tmp_path.glob("*.jpg"))


def test_hf_list_style_labels_normalizes_class_names():
    dataset = _FakeHfDataset([], style_names=["Art_Nouveau", "Post Impressionism"])

    assert list_style_labels(dataset) == ["art_nouveau", "post_impressionism"]


def _write_feature_csv(path: Path, rows_per_class: int) -> None:
    rows = []
    for label_index, label in enumerate(["baroque", "pop_art"]):
        for sample_index in range(rows_per_class):
            base = label_index * 0.4 + sample_index * 0.01
            row = {
                "style": label,
                "image_path": str(path.parent / f"{label}_{sample_index}.jpg"),
                "filename": f"{label}_{sample_index}.jpg",
                "artist": "",
                "style_features_source": "kmeans",
                "avg_saturation": 0.3 + base,
                "avg_lightness": 0.4 + base,
                "avg_hue": 40 + label_index * 120 + sample_index,
                "contrast_spread": 0.2 + sample_index * 0.01,
                "hue_variance": 0.1 + label_index * 0.1,
                "warm_ratio": 0.8 if label == "pop_art" else 0.2,
            }
            rows.append(row)
    _write_csv(path, ["style", "image_path", "filename", "artist", "style_features_source", *FEATURE_COLUMNS], rows)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class _FakeClassLabel:
    def __init__(self, names):
        self.names = names


class _FakeHfDataset:
    def __init__(self, rows, style_names):
        self._rows = rows
        self.features = {"style": _FakeClassLabel(style_names)}

    def __iter__(self):
        return iter(self._rows)

    def take(self, limit):
        return iter(self._rows[:limit])
