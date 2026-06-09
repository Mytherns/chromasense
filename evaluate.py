# Evaluate ChromaSense color extraction and CLIP mood classification.

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clip_mood import MOOD_LABELS, classify_mood_with_metadata
from color_extract import extract_colors


DEFAULT_LABELS_PATH = Path("test_images") / "labels.json"
DEFAULT_REPORT_PATH = Path("evaluation_report.json")
STANDARD_N_COLORS = 5
DELTA_E_THRESHOLD = 6.0
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class EvaluationError(ValueError):
    # Raised when evaluation input is invalid.
    pass


@dataclass(frozen=True)
class ImageLabel:
    filename: str
    path: Path
    mood: str
    colors: list[str]


def main() -> int:
    args = parse_args()

    try:
        labels = load_and_validate_labels(
            args.labels,
            allow_missing=args.allow_missing,
        )
        report = evaluate_dataset(labels, args.output, n_colors=args.n_colors)
    except EvaluationError as exc:
        print(f"Evaluation failed: {exc}")
        return 1

    print_summary(report, args.output)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate KMeans color accuracy and CLIP mood accuracy.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="Path to test_images/labels.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path for evaluation_report.json.",
    )
    parser.add_argument(
        "--n-colors",
        type=int,
        default=STANDARD_N_COLORS,
        choices=range(3, 9),
        metavar="3-8",
        help="Number of extracted colors. Standard evaluation uses 5.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing images instead of failing validation.",
    )
    return parser.parse_args()


def load_and_validate_labels(
    labels_path: Path,
    allow_missing: bool = False,
) -> list[ImageLabel]:
    if not labels_path.exists():
        raise EvaluationError(
            f"labels file not found: {labels_path}. "
            "Create test_images/labels.json before running evaluation."
        )

    try:
        raw = json.loads(labels_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"labels file is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise EvaluationError("labels file must be a non-empty JSON object.")

    labels: list[ImageLabel] = []
    errors: list[str] = []
    image_dir = labels_path.parent

    for filename, item in raw.items():
        item_errors: list[str] = []

        if not isinstance(filename, str) or not filename.strip():
            errors.append("image filename keys must be non-empty strings")
            continue

        if Path(filename).name != filename:
            errors.append(f"{filename}: filename must not include directories")
            continue

        image_path = image_dir / filename
        suffix = image_path.suffix.lower()
        if suffix not in VALID_IMAGE_EXTENSIONS:
            item_errors.append("image extension must be JPEG, PNG, or WebP")

        if not image_path.exists() and not allow_missing:
            item_errors.append("image file is missing")

        if not isinstance(item, dict):
            errors.append(f"{filename}: label value must be an object")
            continue

        mood = item.get("mood")
        colors = item.get("colors")

        if mood not in MOOD_LABELS:
            item_errors.append("mood must be one of canonical mood labels")

        if not isinstance(colors, list) or len(colors) != STANDARD_N_COLORS:
            item_errors.append("colors must contain exactly 5 hex codes")
        elif not all(is_valid_hex_color(color) for color in colors):
            item_errors.append("colors must be valid #rrggbb hex codes")

        if item_errors:
            errors.extend(f"{filename}: {error}" for error in item_errors)
        else:
            labels.append(
                ImageLabel(
                    filename=filename,
                    path=image_path,
                    mood=str(mood),
                    colors=[normalize_hex_color(color) for color in colors],
                )
            )

    if errors:
        detail = "\n  - ".join(errors)
        raise EvaluationError(f"label validation failed:\n  - {detail}")

    return labels


def evaluate_dataset(
    labels: list[ImageLabel],
    output_path: Path,
    n_colors: int = STANDARD_N_COLORS,
) -> dict[str, Any]:
    started = time.perf_counter()
    details = []
    color_match_count = 0
    color_total_count = 0
    mood_correct_count = 0
    evaluated_count = 0
    skipped = []
    errors = []

    for label in labels:
        if not label.path.exists():
            skipped.append({"image": label.filename, "reason": "missing image"})
            continue

        try:
            color_result = extract_colors(label.path, n_colors=n_colors)
            extracted_hex = [color["hex"] for color in color_result["colors"]]
            color_eval = evaluate_color_matches(extracted_hex, label.colors)

            mood_result = classify_mood_with_metadata(label.path)
            clip_result = mood_result["classification"]
            predicted_mood = clip_result["primary_mood"]
            mood_correct = predicted_mood == label.mood

            color_match_count += color_eval["matched_count"]
            color_total_count += len(label.colors)
            mood_correct_count += int(mood_correct)
            evaluated_count += 1

            details.append(
                {
                    "image": label.filename,
                    "reference": {
                        "mood": label.mood,
                        "colors": label.colors,
                    },
                    "predicted": {
                        "mood": predicted_mood,
                        "confidence": clip_result["confidence"],
                        "top3_moods": clip_result["top3_moods"],
                        "colors": extracted_hex,
                    },
                    "color_accuracy": color_eval,
                    "mood_correct": mood_correct,
                    "warnings": mood_result.get("warnings", []),
                }
            )
        except Exception as exc:
            errors.append({"image": label.filename, "error": str(exc)})

    average_delta_e = average_detail_delta_e(details)
    color_accuracy = percentage(color_match_count, color_total_count)
    mood_accuracy = percentage(mood_correct_count, evaluated_count)

    report = {
        "summary": {
            "total_labels": len(labels),
            "evaluated_images": evaluated_count,
            "skipped_images": len(skipped),
            "error_images": len(errors),
            "n_colors": n_colors,
            "delta_e_threshold": DELTA_E_THRESHOLD,
            "color_matches": color_match_count,
            "color_total": color_total_count,
            "color_accuracy": round(color_accuracy, 1),
            "average_delta_e": round(average_delta_e, 2),
            "mood_correct": mood_correct_count,
            "mood_total": evaluated_count,
            "mood_accuracy": round(mood_accuracy, 1),
            "seconds": round(time.perf_counter() - started, 2),
        },
        "details": details,
        "skipped": skipped,
        "errors": errors,
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate_color_matches(
    extracted_hex: list[str],
    reference_hex: list[str],
) -> dict[str, Any]:
    extracted_lab = [rgb_to_lab(hex_to_rgb(color)) for color in extracted_hex]
    reference_lab = [rgb_to_lab(hex_to_rgb(color)) for color in reference_hex]
    matrix = [
        [delta_e_ciede2000(extracted, reference) for reference in reference_lab]
        for extracted in extracted_lab
    ]
    assignments = best_assignments(matrix)
    pair_details = []

    for extracted_index, reference_index in assignments:
        delta_e = matrix[extracted_index][reference_index]
        pair_details.append(
            {
                "extracted": extracted_hex[extracted_index],
                "reference": reference_hex[reference_index],
                "delta_e": round(delta_e, 2),
                "matched": delta_e < DELTA_E_THRESHOLD,
            }
        )

    matched_count = sum(1 for pair in pair_details if pair["matched"])
    average_delta_e = (
        sum(pair["delta_e"] for pair in pair_details) / len(pair_details)
        if pair_details
        else 0.0
    )

    return {
        "matched_count": matched_count,
        "total": len(reference_hex),
        "accuracy": round(percentage(matched_count, len(reference_hex)), 1),
        "average_delta_e": round(average_delta_e, 2),
        "matches": pair_details,
    }


def best_assignments(matrix: list[list[float]]) -> list[tuple[int, int]]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return greedy_assignments(matrix)

    row_indices, column_indices = linear_sum_assignment(matrix)
    return [(int(row), int(column)) for row, column in zip(row_indices, column_indices)]


def greedy_assignments(matrix: list[list[float]]) -> list[tuple[int, int]]:
    pairs = []
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            pairs.append((value, row_index, column_index))

    assignments = []
    used_rows = set()
    used_columns = set()

    for _, row_index, column_index in sorted(pairs):
        if row_index in used_rows or column_index in used_columns:
            continue
        assignments.append((row_index, column_index))
        used_rows.add(row_index)
        used_columns.add(column_index)

    return assignments


def rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    red, green, blue = [srgb_to_linear(channel / 255.0) for channel in rgb]

    x = red * 0.4124564 + green * 0.3575761 + blue * 0.1804375
    y = red * 0.2126729 + green * 0.7151522 + blue * 0.0721750
    z = red * 0.0193339 + green * 0.1191920 + blue * 0.9503041

    return xyz_to_lab(x * 100, y * 100, z * 100)


def srgb_to_linear(value: float) -> float:
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def xyz_to_lab(x: float, y: float, z: float) -> tuple[float, float, float]:
    x_ref, y_ref, z_ref = 95.047, 100.000, 108.883
    fx = lab_f(x / x_ref)
    fy = lab_f(y / y_ref)
    fz = lab_f(z / z_ref)
    return ((116 * fy) - 16, 500 * (fx - fy), 200 * (fy - fz))


def lab_f(value: float) -> float:
    delta = 6 / 29
    if value > delta**3:
        return value ** (1 / 3)
    return (value / (3 * delta**2)) + (4 / 29)


def delta_e_ciede2000(
    lab1: tuple[float, float, float],
    lab2: tuple[float, float, float],
) -> float:
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    c1 = math.hypot(a1, b1)
    c2 = math.hypot(a2, b2)
    c_bar = (c1 + c2) / 2
    c_bar7 = c_bar**7
    g = 0.5 * (1 - math.sqrt(c_bar7 / (c_bar7 + 25**7))) if c_bar else 0.0
    a1_prime = (1 + g) * a1
    a2_prime = (1 + g) * a2
    c1_prime = math.hypot(a1_prime, b1)
    c2_prime = math.hypot(a2_prime, b2)
    h1_prime = hue_degrees(a1_prime, b1)
    h2_prime = hue_degrees(a2_prime, b2)

    delta_l_prime = l2 - l1
    delta_c_prime = c2_prime - c1_prime
    delta_h_prime = delta_h_degrees(c1_prime, c2_prime, h1_prime, h2_prime)
    delta_h_term = 2 * math.sqrt(c1_prime * c2_prime) * math.sin(
        math.radians(delta_h_prime / 2)
    )

    l_bar_prime = (l1 + l2) / 2
    c_bar_prime = (c1_prime + c2_prime) / 2
    h_bar_prime = mean_hue_degrees(c1_prime, c2_prime, h1_prime, h2_prime)
    t = (
        1
        - 0.17 * math.cos(math.radians(h_bar_prime - 30))
        + 0.24 * math.cos(math.radians(2 * h_bar_prime))
        + 0.32 * math.cos(math.radians((3 * h_bar_prime) + 6))
        - 0.20 * math.cos(math.radians((4 * h_bar_prime) - 63))
    )
    delta_theta = 30 * math.exp(-(((h_bar_prime - 275) / 25) ** 2))
    c_bar_prime7 = c_bar_prime**7
    r_c = 2 * math.sqrt(c_bar_prime7 / (c_bar_prime7 + 25**7))
    s_l = 1 + (
        (0.015 * ((l_bar_prime - 50) ** 2))
        / math.sqrt(20 + ((l_bar_prime - 50) ** 2))
    )
    s_c = 1 + (0.045 * c_bar_prime)
    s_h = 1 + (0.015 * c_bar_prime * t)
    r_t = -math.sin(math.radians(2 * delta_theta)) * r_c

    l_term = delta_l_prime / s_l
    c_term = delta_c_prime / s_c
    h_term = delta_h_term / s_h
    return math.sqrt(
        (l_term**2)
        + (c_term**2)
        + (h_term**2)
        + (r_t * c_term * h_term)
    )


def hue_degrees(a_value: float, b_value: float) -> float:
    hue = math.degrees(math.atan2(b_value, a_value))
    return hue + 360 if hue < 0 else hue


def delta_h_degrees(
    c1: float,
    c2: float,
    h1: float,
    h2: float,
) -> float:
    if c1 * c2 == 0:
        return 0.0
    difference = h2 - h1
    if abs(difference) <= 180:
        return difference
    if difference > 180:
        return difference - 360
    return difference + 360


def mean_hue_degrees(
    c1: float,
    c2: float,
    h1: float,
    h2: float,
) -> float:
    if c1 * c2 == 0:
        return h1 + h2
    if abs(h1 - h2) <= 180:
        return (h1 + h2) / 2
    if h1 + h2 < 360:
        return (h1 + h2 + 360) / 2
    return (h1 + h2 - 360) / 2


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = normalize_hex_color(color)
    return (
        int(value[1:3], 16),
        int(value[3:5], 16),
        int(value[5:7], 16),
    )


def normalize_hex_color(color: str) -> str:
    return color.lower()


def is_valid_hex_color(color: object) -> bool:
    if not isinstance(color, str) or len(color) != 7 or not color.startswith("#"):
        return False
    try:
        int(color[1:], 16)
    except ValueError:
        return False
    return True


def percentage(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100


def average_detail_delta_e(details: list[dict[str, Any]]) -> float:
    values = [
        match["delta_e"]
        for detail in details
        for match in detail["color_accuracy"]["matches"]
    ]
    return sum(values) / len(values) if values else 0.0


def print_summary(report: dict[str, Any], output_path: Path) -> None:
    summary = report["summary"]
    print("Evaluation complete")
    print(f"Images evaluated: {summary['evaluated_images']}/{summary['total_labels']}")
    print(f"Color accuracy: {summary['color_accuracy']}%")
    print(f"Average Delta E: {summary['average_delta_e']}")
    print(f"Mood accuracy: {summary['mood_accuracy']}%")
    print(f"Report saved: {output_path}")
    if summary["error_images"]:
        print(f"Images with errors: {summary['error_images']}")
    if summary["skipped_images"]:
        print(f"Skipped images: {summary['skipped_images']}")


if __name__ == "__main__":
    raise SystemExit(main())
