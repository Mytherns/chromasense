from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.color import deltaE_ciede2000, rgb2lab

from chromasense.color_extract import DEFAULT_K, DEFAULT_SEED, Algorithm, extract_colors

ALGORITHMS: tuple[Algorithm, ...] = ("kmeans", "gmm")
DEFAULT_DELTA_E_THRESHOLD = 6.0


@dataclass(frozen=True)
class PaletteMatch:
    truth_hex: str
    predicted_hex: str
    delta_e: float


def hex_to_rgb(hex_color: str) -> list[int]:
    value = hex_color.strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) != 6:
        raise ValueError(f"invalid hex color: {hex_color!r}")
    return [int(value[index : index + 2], 16) for index in (0, 2, 4)]


def match_palettes(truth_hexes: Sequence[str], predicted_hexes: Sequence[str]) -> list[PaletteMatch]:
    if not truth_hexes:
        raise ValueError("truth palette must not be empty")
    if not predicted_hexes:
        return []

    truth_lab = _hexes_to_lab(truth_hexes)
    predicted_lab = _hexes_to_lab(predicted_hexes)
    distance = np.zeros((len(truth_lab), len(predicted_lab)), dtype=np.float64)

    for truth_index, truth_color in enumerate(truth_lab):
        truth_repeated = np.repeat(truth_color.reshape(1, 3), len(predicted_lab), axis=0)
        distance[truth_index] = deltaE_ciede2000(truth_repeated, predicted_lab)

    truth_indices, predicted_indices = linear_sum_assignment(distance)
    return [
        PaletteMatch(
            truth_hex=truth_hexes[truth_index].upper(),
            predicted_hex=predicted_hexes[predicted_index].upper(),
            delta_e=round(float(distance[truth_index, predicted_index]), 4),
        )
        for truth_index, predicted_index in zip(truth_indices, predicted_indices)
    ]


def evaluate_manifest(
    ground_truth_path: Path,
    images_dir: Path,
    n_colors: int,
    seed: int,
    delta_e_threshold: float,
    allow_missing: bool = False,
) -> dict:
    manifest = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    image_entries = manifest.get("images", [])
    if not image_entries:
        raise ValueError("ground truth manifest has no images")

    algorithm_results = {algorithm: _empty_algorithm_result(algorithm) for algorithm in ALGORITHMS}
    missing_files: list[str] = []

    for entry in image_entries:
        filename = entry["file"]
        truth_hexes = entry["expected_colors"]
        image_path = images_dir / filename
        if not image_path.exists():
            missing_files.append(filename)
            if allow_missing:
                continue
            raise FileNotFoundError(f"missing eval image: {image_path}")

        for algorithm in ALGORITHMS:
            colors = extract_colors(image_path, k=n_colors, algorithm=algorithm, seed=seed)
            predicted_hexes = [color["hex"] for color in colors]
            matches = match_palettes(truth_hexes, predicted_hexes)
            image_result = _image_result(filename, truth_hexes, predicted_hexes, matches, delta_e_threshold)
            _add_image_result(algorithm_results[algorithm], image_result)

    for result in algorithm_results.values():
        _finalize_algorithm_result(result, delta_e_threshold)

    primary_algorithm = choose_primary_algorithm(algorithm_results)
    return {
        "ground_truth": str(ground_truth_path),
        "images_dir": str(images_dir),
        "n_colors": n_colors,
        "seed": seed,
        "delta_e_threshold": delta_e_threshold,
        "target": "85%+ matched color pairs with Delta E < 6",
        "primary_algorithm": primary_algorithm,
        "style_features_source": "kmeans",
        "missing_files": missing_files,
        "results": algorithm_results,
    }


def choose_primary_algorithm(results: dict[str, dict]) -> Algorithm:
    ranked = sorted(
        ALGORITHMS,
        key=lambda algorithm: (
            -float(results[algorithm]["matched_delta_e_lt_threshold_rate"]),
            float(results[algorithm]["mean_delta_e"] or 9999.0),
            0 if algorithm == "kmeans" else 1,
        ),
    )
    return ranked[0]


def write_reports(report: dict, json_path: Path, md_path: Path, decision_path: Path) -> None:
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision = {
        "primary_algorithm": report["primary_algorithm"],
        "source_report": str(json_path),
        "decision_metric": "matched_delta_e_lt_threshold_rate",
        "style_features_source": "kmeans",
        "note": "Fixed offline decision. Do not switch per image at runtime.",
    }
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")


def render_markdown_report(report: dict) -> str:
    lines = [
        "# Color Evaluation Report",
        "",
        f"- Ground truth: `{report['ground_truth']}`",
        f"- Images dir: `{report['images_dir']}`",
        f"- Colors per image: `{report['n_colors']}`",
        f"- Delta E threshold: `{report['delta_e_threshold']}`",
        f"- Target: {report['target']}",
        f"- Fixed primary algorithm: `{report['primary_algorithm']}`",
        f"- Style feature source remains: `{report['style_features_source']}`",
        "",
        "## Summary",
        "",
        "| Algorithm | Images | Colors | Delta E < threshold | Mean Delta E |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for algorithm in ALGORITHMS:
        result = report["results"][algorithm]
        lines.append(
            "| {algorithm} | {images} | {colors} | {rate:.3f} | {mean_delta_e:.3f} |".format(
                algorithm=algorithm,
                images=result["image_count"],
                colors=result["truth_color_count"],
                rate=result["matched_delta_e_lt_threshold_rate"],
                mean_delta_e=result["mean_delta_e"] or 0.0,
            )
        )

    lines.extend(
        [
            "",
            "## Limitation",
            "",
            "Current committed color-eval fixtures are tiny deterministic swatches for repeatable smoke verification. Replace or extend with real manually labeled photos before making a final report claim.",
            "",
            "## Per Image",
            "",
        ]
    )
    for algorithm in ALGORITHMS:
        lines.append(f"### {algorithm}")
        lines.append("")
        for image in report["results"][algorithm]["images"]:
            lines.append(
                "- `{file}`: rate `{rate:.3f}`, mean Delta E `{mean_delta_e:.3f}`".format(
                    file=image["file"],
                    rate=image["matched_delta_e_lt_threshold_rate"],
                    mean_delta_e=image["mean_delta_e"] or 0.0,
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _hexes_to_lab(hexes: Sequence[str]) -> np.ndarray:
    rgb = np.array([hex_to_rgb(hex_color) for hex_color in hexes], dtype=np.float64) / 255.0
    return rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3)


def _empty_algorithm_result(algorithm: str) -> dict:
    return {
        "algorithm": algorithm,
        "image_count": 0,
        "truth_color_count": 0,
        "matched_color_count": 0,
        "matched_delta_e_lt_threshold_count": 0,
        "matched_delta_e_lt_threshold_rate": 0.0,
        "mean_delta_e": None,
        "images": [],
    }


def _image_result(
    filename: str,
    truth_hexes: Sequence[str],
    predicted_hexes: Sequence[str],
    matches: Sequence[PaletteMatch],
    delta_e_threshold: float,
) -> dict:
    under_threshold = sum(1 for match in matches if match.delta_e < delta_e_threshold)
    mean_delta = round(mean(match.delta_e for match in matches), 4) if matches else None
    return {
        "file": filename,
        "truth_hexes": list(truth_hexes),
        "predicted_hexes": list(predicted_hexes),
        "truth_color_count": len(truth_hexes),
        "matched_color_count": len(matches),
        "matched_delta_e_lt_threshold_count": under_threshold,
        "matched_delta_e_lt_threshold_rate": round(under_threshold / len(truth_hexes), 6),
        "mean_delta_e": mean_delta,
        "matches": [match.__dict__ for match in matches],
    }


def _add_image_result(algorithm_result: dict, image_result: dict) -> None:
    algorithm_result["image_count"] += 1
    algorithm_result["truth_color_count"] += image_result["truth_color_count"]
    algorithm_result["matched_color_count"] += image_result["matched_color_count"]
    algorithm_result["matched_delta_e_lt_threshold_count"] += image_result[
        "matched_delta_e_lt_threshold_count"
    ]
    algorithm_result["images"].append(image_result)


def _finalize_algorithm_result(algorithm_result: dict, delta_e_threshold: float) -> None:
    truth_count = algorithm_result["truth_color_count"]
    matched_under = algorithm_result["matched_delta_e_lt_threshold_count"]
    all_delta_e = [
        match["delta_e"]
        for image in algorithm_result["images"]
        for match in image["matches"]
    ]
    algorithm_result["delta_e_threshold"] = delta_e_threshold
    algorithm_result["matched_delta_e_lt_threshold_rate"] = (
        round(matched_under / truth_count, 6) if truth_count else 0.0
    )
    algorithm_result["mean_delta_e"] = round(mean(all_delta_e), 4) if all_delta_e else None


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline KMeans vs GMM color evaluation.")
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth_palettes.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("test_images/color_eval"))
    parser.add_argument("--output-json", type=Path, default=Path("evaluation_report.json"))
    parser.add_argument("--output-md", type=Path, default=Path("evaluation_report.md"))
    parser.add_argument("--decision-output", type=Path, default=Path("data/palette_algorithm_config.json"))
    parser.add_argument("--n-colors", type=int, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--delta-e-threshold", type=float, default=DEFAULT_DELTA_E_THRESHOLD)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate_manifest(
        ground_truth_path=args.ground_truth,
        images_dir=args.images_dir,
        n_colors=args.n_colors,
        seed=args.seed,
        delta_e_threshold=args.delta_e_threshold,
        allow_missing=args.allow_missing,
    )
    write_reports(report, args.output_json, args.output_md, args.decision_output)
    print(
        "primary_algorithm={algorithm} kmeans_rate={kmeans:.3f} gmm_rate={gmm:.3f}".format(
            algorithm=report["primary_algorithm"],
            kmeans=report["results"]["kmeans"]["matched_delta_e_lt_threshold_rate"],
            gmm=report["results"]["gmm"]["matched_delta_e_lt_threshold_rate"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
