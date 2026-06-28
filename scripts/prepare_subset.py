from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Iterable

DEFAULT_STYLES = (
    "impressionism",
    "pop_art",
    "fauvism",
    "baroque",
    "art_nouveau",
    "ukiyo_e",
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
STYLE_ALIASES = {
    "popart": "pop_art",
    "pop_art": "pop_art",
    "artnouveau": "art_nouveau",
    "art_nouveau_modern": "art_nouveau",
    "ukiyoe": "ukiyo_e",
    "ukiyo_e": "ukiyo_e",
    "ukiyo": "ukiyo_e",
}


def normalize_style_name(value: str) -> str:
    lowered = value.lower().strip()
    token = re.sub(r"[\s\-]+", "_", lowered)
    token = re.sub(r"[^a-z0-9_]+", "", token)
    token = re.sub(r"_+", "_", token).strip("_")
    compact = token.replace("_", "")
    return STYLE_ALIASES.get(token, STYLE_ALIASES.get(compact, token))


def collect_subset(
    source_root: Path,
    styles: Iterable[str] = DEFAULT_STYLES,
    target_per_class: int = 300,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[str]]:
    if target_per_class < 1:
        raise ValueError("target_per_class must be >= 1")

    rng = random.Random(seed)
    wanted = [normalize_style_name(style) for style in styles]
    folders_by_style = _style_folders(source_root)
    rows: list[dict[str, str]] = []
    missing: list[str] = []

    for style in wanted:
        folders = folders_by_style.get(style, [])
        if not folders:
            missing.append(style)
            continue

        images = []
        for folder in folders:
            images.extend(_image_files(folder))
        images = sorted(set(images), key=lambda path: str(path).lower())
        rng.shuffle(images)

        for image_path in images[:target_per_class]:
            rows.append(
                {
                    "style": style,
                    "image_path": str(image_path),
                    "filename": image_path.name,
                    "artist": "",
                    "source_folder": str(image_path.parent),
                }
            )

    rows.sort(key=lambda row: (row["style"], row["filename"], row["image_path"]))
    return rows, missing


def write_manifest(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["style", "image_path", "filename", "artist", "source_folder"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _style_folders(source_root: Path) -> dict[str, list[Path]]:
    folders: dict[str, list[Path]] = {}
    if not source_root.exists():
        return folders

    for child in source_root.iterdir():
        if child.is_dir():
            folders.setdefault(normalize_style_name(child.name), []).append(child)
    return folders


def _image_files(folder: Path) -> list[Path]:
    return [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create WikiArt style training manifest without copying raw images.")
    parser.add_argument("--source-root", type=Path, default=Path("data/raw/wikiart"))
    parser.add_argument("--output", type=Path, default=Path("data/wikiart_subset_manifest.csv"))
    parser.add_argument("--styles", nargs="*", default=list(DEFAULT_STYLES))
    parser.add_argument("--target-per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    rows, missing = collect_subset(
        source_root=args.source_root,
        styles=args.styles,
        target_per_class=args.target_per_class,
        seed=args.seed,
    )
    write_manifest(rows, args.output)
    print(f"selected_images={len(rows)} output={args.output}")
    if missing:
        print("missing_styles=" + ",".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
