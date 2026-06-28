from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chromasense.color_extract import DEFAULT_K, extract_colors
from chromasense.features import FEATURE_COLUMNS, palette_to_features
from scripts.prepare_subset import normalize_style_name

DEFAULT_DATASET = "huggan/wikiart"
DEFAULT_SPLIT = "train"
DEFAULT_STYLES = (
    "impressionism",
    "baroque",
    "art_nouveau",
    "expressionism",
)


def load_hf_stream(dataset_name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("missing dependency: install with `py -m pip install datasets`") from exc
    return load_dataset(dataset_name, split=split, streaming=True)


def list_style_labels(dataset) -> list[str]:
    feature = getattr(dataset, "features", {}).get("style")
    names = getattr(feature, "names", None)
    if names:
        return [normalize_style_name(name) for name in names]

    labels: list[str] = []
    seen = set()
    for example in dataset.take(500):
        label = _style_label(example, None)
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def extract_hf_style_features(
    dataset,
    output_path: Path,
    errors_output: Path,
    styles: Sequence[str] = DEFAULT_STYLES,
    target_per_class: int = 150,
    n_colors: int = DEFAULT_K,
    max_sample_pixels: int = 50_000,
    seed: int = 42,
    progress_every: int = 25,
) -> tuple[Counter, int]:
    wanted = {normalize_style_name(style) for style in styles}
    if len(wanted) < 2:
        raise ValueError("need at least two styles")
    if target_per_class < 1:
        raise ValueError("target_per_class must be >= 1")

    label_names = _style_names(dataset)
    counts: Counter = Counter()
    skipped = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_output.parent.mkdir(parents=True, exist_ok=True)
    feature_fieldnames = [
        "style",
        "image_path",
        "filename",
        "artist",
        "style_features_source",
        *FEATURE_COLUMNS,
    ]
    error_fieldnames = ["image_path", "style", "reason"]

    with output_path.open("w", newline="", encoding="utf-8") as feature_handle, errors_output.open(
        "w", newline="", encoding="utf-8"
    ) as error_handle:
        feature_writer = csv.DictWriter(feature_handle, fieldnames=feature_fieldnames)
        error_writer = csv.DictWriter(error_handle, fieldnames=error_fieldnames)
        feature_writer.writeheader()
        error_writer.writeheader()

        for index, example in enumerate(dataset):
            style = _style_label(example, label_names)
            if style not in wanted or counts[style] >= target_per_class:
                if _targets_met(counts, wanted, target_per_class):
                    break
                continue

            image_ref = _image_ref(example, index)
            try:
                image = _example_image(example)
                palette = extract_colors(
                    image,
                    k=n_colors,
                    algorithm="kmeans",
                    seed=seed,
                    max_sample_pixels=max_sample_pixels,
                )
                features = palette_to_features(palette)
            except Exception as exc:
                error_writer.writerow(
                    {
                        "image_path": image_ref,
                        "style": style,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                error_handle.flush()
                skipped += 1
                continue

            feature_writer.writerow(
                {
                    "style": style,
                    "image_path": image_ref,
                    "filename": image_ref,
                    "artist": _artist_label(example, dataset),
                    "style_features_source": "kmeans",
                    **features,
                }
            )
            feature_handle.flush()
            counts[style] += 1
            total = sum(counts.values())
            if progress_every > 0 and total % progress_every == 0:
                print(
                    "progress usable_images={total} styles={styles}".format(
                        total=total,
                        styles=",".join(f"{style}:{count}" for style, count in sorted(counts.items())),
                    ),
                    flush=True,
                )

            if _targets_met(counts, wanted, target_per_class):
                break

    missing = {style: counts[style] for style in wanted if counts[style] < target_per_class}
    if missing:
        raise ValueError(
            "stream ended before target_per_class: "
            + ", ".join(f"{style}={count}/{target_per_class}" for style, count in sorted(missing.items()))
        )

    return counts, skipped


def _style_names(dataset) -> list[str] | None:
    feature = getattr(dataset, "features", {}).get("style")
    names = getattr(feature, "names", None)
    return list(names) if names else None


def _style_label(example: Mapping, label_names: Sequence[str] | None) -> str:
    raw = example.get("style")
    if isinstance(raw, int) and label_names is not None:
        return normalize_style_name(label_names[raw])
    return normalize_style_name(str(raw))


def _artist_label(example: Mapping, dataset) -> str:
    raw = example.get("artist", "")
    feature = getattr(dataset, "features", {}).get("artist")
    names = getattr(feature, "names", None)
    if isinstance(raw, int) and names:
        return str(names[raw])
    return str(raw or "")


def _example_image(example: Mapping) -> Image.Image:
    raw = example.get("image")
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, dict):
        if raw.get("bytes") is not None:
            return Image.open(BytesIO(raw["bytes"])).convert("RGB")
        if raw.get("path"):
            return Image.open(raw["path"]).convert("RGB")
    raise ValueError("example missing decodable image")


def _image_ref(example: Mapping, index: int) -> str:
    image = example.get("image")
    if isinstance(image, dict) and image.get("path"):
        return str(image["path"])
    artist = str(example.get("artist", "unknown"))
    return f"hf://{DEFAULT_DATASET}/{index}/{artist}"


def _targets_met(counts: Counter, wanted: set[str], target_per_class: int) -> bool:
    return all(counts[style] >= target_per_class for style in wanted)


def _write_features(rows: list[dict[str, str | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "style",
        "image_path",
        "filename",
        "artist",
        "style_features_source",
        *FEATURE_COLUMNS,
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_errors(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "style", "reason"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream WikiArt from Hugging Face and save KMeans style features only.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--styles", nargs="*", default=list(DEFAULT_STYLES))
    parser.add_argument("--target-per-class", type=int, default=150)
    parser.add_argument("--output", type=Path, default=Path("data/training_features_style.csv"))
    parser.add_argument("--errors-output", type=Path, default=Path("data/feature_extraction_errors.csv"))
    parser.add_argument("--n-colors", type=int, default=DEFAULT_K)
    parser.add_argument("--max-sample-pixels", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--list-labels", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = load_hf_stream(args.dataset, args.split)
    if args.list_labels:
        for label in list_style_labels(dataset):
            print(label)
        return 0

    counts, skipped = extract_hf_style_features(
        dataset=dataset,
        output_path=args.output,
        errors_output=args.errors_output,
        styles=args.styles,
        target_per_class=args.target_per_class,
        n_colors=args.n_colors,
        max_sample_pixels=args.max_sample_pixels,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    print(
        "usable_images={total} skipped_images={skipped} styles={styles} output={output}".format(
            total=sum(counts.values()),
            skipped=skipped,
            styles=",".join(f"{style}:{count}" for style, count in sorted(counts.items())),
            output=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
