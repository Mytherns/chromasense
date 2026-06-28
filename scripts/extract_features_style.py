from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chromasense.color_extract import DEFAULT_K, extract_colors
from chromasense.features import FEATURE_COLUMNS, palette_to_features


def extract_style_features(
    manifest_path: Path,
    output_path: Path,
    errors_output: Path,
    n_colors: int = DEFAULT_K,
    seed: int = 42,
    allow_empty: bool = False,
) -> tuple[int, int]:
    rows = _read_manifest(manifest_path)
    feature_rows: list[dict[str, str | float]] = []
    errors: list[dict[str, str]] = []

    for row in rows:
        image_path = Path(row["image_path"])
        try:
            palette = extract_colors(image_path, k=n_colors, algorithm="kmeans", seed=seed)
            features = palette_to_features(palette)
        except Exception as exc:
            errors.append(
                {
                    "image_path": str(image_path),
                    "style": row["style"],
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        feature_rows.append(
            {
                "style": row["style"],
                "image_path": str(image_path),
                "filename": row.get("filename", image_path.name),
                "artist": row.get("artist", ""),
                "style_features_source": "kmeans",
                **features,
            }
        )

    if not feature_rows and not allow_empty:
        raise ValueError("no usable images found during style feature extraction")

    _write_features(feature_rows, output_path)
    _write_errors(errors, errors_output)
    return len(feature_rows), len(errors)


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"manifest has no rows: {manifest_path}")
    for index, row in enumerate(rows, start=2):
        if not row.get("style") or not row.get("image_path"):
            raise ValueError(f"manifest row {index} needs style and image_path")
    return rows


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
    parser = argparse.ArgumentParser(description="Extract KMeans palette features for WikiArt style training.")
    parser.add_argument("--manifest", type=Path, default=Path("data/wikiart_subset_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/training_features_style.csv"))
    parser.add_argument("--errors-output", type=Path, default=Path("data/feature_extraction_errors.csv"))
    parser.add_argument("--n-colors", type=int, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    usable_count, error_count = extract_style_features(
        manifest_path=args.manifest,
        output_path=args.output,
        errors_output=args.errors_output,
        n_colors=args.n_colors,
        seed=args.seed,
        allow_empty=args.allow_empty,
    )
    print(f"usable_images={usable_count} skipped_images={error_count} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
