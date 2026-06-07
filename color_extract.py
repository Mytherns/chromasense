# Dominant color extraction for Chromasense.
#
# This module is intentionally framework-agnostic. FastAPI should read the
# uploaded file bytes, call ``save_upload_to_temp(...)``, run ``extract_colors``,
# then call ``cleanup_temp_file(...)`` in a ``finally`` block.

from __future__ import annotations

import colorsys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.cluster import KMeans


SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_SIDE = 1024
MAX_SAMPLE_PIXELS = 50_000
RANDOM_STATE = 42
ACCENT_MIN_PERCENTAGE = 1.0
ACCENT_MIN_SCORE = 0.12
AREA_WEIGHT = 0.15
ACCENT_WEIGHT = 0.15
DIVERSITY_WEIGHT = 0.70
SATURATION_IMPORTANCE = 1.2
CENTER_IMPORTANCE = 0.6
MAX_PIXEL_WEIGHT = 2.5
MAX_CANDIDATE_CLUSTERS = 12
CANDIDATE_EXTRA_CLUSTERS = 7

COLOR_ROLES = [
    "Dominant",
    "Secondary",
    "Accent",
    "Support",
    "Neutral/Depth",
    "Highlight",
    "Contrast",
    "Detail",
]

BASIC_COLOR_NAMES = [
    ("Black", (0, 0, 0)),
    ("White", (255, 255, 255)),
    ("Gray", (128, 128, 128)),
    ("Dark Gray", (64, 64, 64)),
    ("Dark Brown", (65, 42, 24)),
    ("Blue Gray", (54, 71, 90)),
    ("Navy", (0, 0, 128)),
    ("Olive Green", (85, 86, 23)),
    ("Dark Green", (22, 101, 52)),
    ("Forest Green", (40, 92, 35)),
    ("Grass Green", (83, 141, 34)),
    ("Lime Green", (132, 204, 22)),
    ("Tan", (180, 140, 92)),
    ("Beige", (214, 196, 162)),
    ("Red", (220, 38, 38)),
    ("Orange", (249, 115, 22)),
    ("Yellow", (234, 179, 8)),
    ("Green", (34, 197, 94)),
    ("Cyan", (6, 182, 212)),
    ("Blue", (0, 0, 255)),
    ("Sky Blue", (59, 130, 246)),
    ("Purple", (147, 51, 234)),
    ("Pink", (236, 72, 153)),
    ("Brown", (120, 72, 38)),
]


class ColorExtractionError(ValueError):
    # Raised for invalid image input or failed color extraction.

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_error(self) -> dict[str, dict[str, str]]:
        return {"error": {"code": self.code, "message": self.message}}


def validate_n_colors(n_colors: int) -> int:
    # Validate KMeans cluster count.
    if not isinstance(n_colors, int):
        raise ColorExtractionError(
            "invalid_n_colors",
            "n_colors must be an integer from 3 to 8.",
        )
    if n_colors < 3 or n_colors > 8:
        raise ColorExtractionError(
            "invalid_n_colors",
            "n_colors must be between 3 and 8.",
        )
    return n_colors


def validate_upload(
    file_bytes: bytes,
    filename: str | None = None,
    content_type: str | None = None,
) -> None:
    # Validate upload metadata and size before writing a temp file.
    if not file_bytes:
        raise ColorExtractionError("empty_file", "Uploaded file is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ColorExtractionError(
            "file_too_large",
            "Uploaded image must be 10MB or smaller.",
        )
    if content_type and content_type.lower() not in SUPPORTED_CONTENT_TYPES:
        raise ColorExtractionError(
            "invalid_file_type",
            "Only JPEG, PNG, and WebP images are supported.",
        )
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix and suffix not in SUPPORTED_EXTENSIONS:
            raise ColorExtractionError(
                "invalid_file_type",
                "Only JPEG, PNG, and WebP images are supported.",
            )


def save_upload_to_temp(
    file_bytes: bytes,
    filename: str | None = None,
    content_type: str | None = None,
    temp_dir: str | Path | None = None,
) -> Path:
    # Validate and save upload bytes to a unique temp path.
    validate_upload(file_bytes, filename, content_type)
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        suffix = ".img"

    directory = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
    directory.mkdir(parents=True, exist_ok=True)
    temp_path = directory / f"chromasense_{uuid.uuid4().hex}{suffix}"
    temp_path.write_bytes(file_bytes)
    return temp_path


def cleanup_temp_file(path: str | Path | None) -> None:
    # Remove a temp file if it exists.
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def extract_colors(image_path: str | Path, n_colors: int = 5) -> dict[str, Any]:
    # Extract dominant colors and return records matching the API response shape.
    n_colors = validate_n_colors(n_colors)
    image = _load_resized_rgb_image(image_path)
    pixels, coordinates = _sample_pixels(np.asarray(image), MAX_SAMPLE_PIXELS)
    pixel_weights = _pixel_importance_weights(pixels, coordinates)

    if len(pixels) < n_colors:
        raise ColorExtractionError(
            "image_too_small",
            "Image does not contain enough pixels for color extraction.",
        )

    centers, counts, accent_index = _cluster_palette(pixels, pixel_weights, n_colors)

    order = np.argsort(counts)[::-1]
    if accent_index is not None and accent_index in order[:n_colors]:
        order = _promote_accent_role(order, accent_index)
    total = float(counts.sum())
    colors = []

    for output_index, cluster_index in enumerate(order):
        rgb = tuple(int(value) for value in centers[cluster_index])
        percentage = round((int(counts[cluster_index]) / total) * 100, 1)
        colors.append(
            {
                "hex": _rgb_to_hex(rgb),
                "rgb": list(rgb),
                "percentage": percentage,
                "name": _nearest_basic_color_name(rgb),
                "role": COLOR_ROLES[output_index],
            }
        )

    return {
        "image_size": [image.width, image.height],
        "colors": colors,
    }


def _cluster_palette(
    pixels: np.ndarray,
    pixel_weights: np.ndarray,
    n_colors: int,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    unique_pixels, unique_counts = np.unique(pixels, axis=0, return_counts=True)
    if len(unique_pixels) <= n_colors:
        order = np.argsort(unique_counts)[::-1]
        return unique_pixels[order].astype(int), unique_counts[order], None

    candidate_count = min(
        len(unique_pixels),
        max(n_colors, min(MAX_CANDIDATE_CLUSTERS, n_colors + CANDIDATE_EXTRA_CLUSTERS)),
    )
    kmeans = KMeans(n_clusters=candidate_count, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(pixels, sample_weight=pixel_weights)
    labels = kmeans.predict(pixels)
    candidate_centers = np.clip(np.rint(kmeans.cluster_centers_), 0, 255).astype(int)
    candidate_counts = np.bincount(labels, minlength=candidate_count)

    selected_indices, accent_index = _select_candidate_clusters(
        candidate_centers,
        candidate_counts,
        n_colors,
    )
    selected_centers = candidate_centers[selected_indices]
    selected_counts = _count_nearest_selected_centers(pixels, selected_centers)

    selected_accent_index = None
    if accent_index is not None and accent_index in selected_indices:
        selected_accent_index = selected_indices.index(accent_index)

    return selected_centers, selected_counts, selected_accent_index


def _select_candidate_clusters(
    centers: np.ndarray,
    counts: np.ndarray,
    n_colors: int,
) -> tuple[list[int], int | None]:
    order = list(np.argsort(counts)[::-1])
    selected = [order[0]]
    total = float(counts.sum())
    percentages = [(int(count) / total) * 100 for count in counts]
    max_percentage = max(percentages)

    while len(selected) < n_colors:
        remaining = [index for index in order if index not in selected]
        next_index = max(
            remaining,
            key=lambda index: _candidate_score(
                tuple(int(value) for value in centers[index]),
                percentages[index],
                max_percentage,
                centers[selected],
            ),
        )
        selected.append(next_index)

    accent_index = max(
        range(len(centers)),
        key=lambda index: _accent_score(tuple(int(value) for value in centers[index])),
    )
    accent_score = _accent_score(tuple(int(value) for value in centers[accent_index]))

    if (
        accent_index not in selected
        and percentages[accent_index] >= ACCENT_MIN_PERCENTAGE
        and accent_score >= ACCENT_MIN_SCORE
    ):
        selected[-1] = accent_index

    return selected, accent_index if accent_index in selected else None


def _candidate_score(
    rgb: tuple[int, int, int],
    percentage: float,
    max_percentage: float,
    selected_centers: np.ndarray,
) -> float:
    area_score = percentage / max_percentage if max_percentage else 0.0
    accent_score = _accent_score(rgb)
    diversity_score = _nearest_color_distance(rgb, selected_centers)
    return (
        (AREA_WEIGHT * area_score)
        + (ACCENT_WEIGHT * accent_score)
        + (DIVERSITY_WEIGHT * diversity_score)
    )


def _nearest_color_distance(
    rgb: tuple[int, int, int],
    selected_centers: np.ndarray,
) -> float:
    color_feature = _color_feature(rgb)
    selected_features = np.array(
        [_color_feature(tuple(int(value) for value in center)) for center in selected_centers]
    )
    distances = np.linalg.norm(color_feature - selected_features, axis=1)
    return float(np.clip(np.min(distances) / 1.8, 0, 1))


def _color_feature(rgb: tuple[int, int, int]) -> np.ndarray:
    red, green, blue = (channel / 255 for channel in rgb)
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    angle = 2 * np.pi * hue
    return np.array(
        [
            red * 0.45,
            green * 0.45,
            blue * 0.45,
            np.cos(angle) * saturation * 0.65,
            np.sin(angle) * saturation * 0.65,
            saturation * 0.35,
            value * 0.15,
        ]
    )


def _count_nearest_selected_centers(
    pixels: np.ndarray,
    selected_centers: np.ndarray,
) -> np.ndarray:
    distances = np.linalg.norm(
        pixels[:, np.newaxis, :] - selected_centers[np.newaxis, :, :],
        axis=2,
    )
    nearest = np.argmin(distances, axis=1)
    return np.bincount(nearest, minlength=len(selected_centers))


def _promote_accent_role(order: np.ndarray, accent_index: int) -> np.ndarray:
    without_accent = [int(index) for index in order if int(index) != accent_index]
    if len(without_accent) < 2:
        return order
    promoted = without_accent[:2] + [accent_index] + without_accent[2:]
    return np.array(promoted, dtype=int)


def _load_resized_rgb_image(image_path: str | Path) -> Image.Image:
    path = Path(image_path)
    if not path.exists():
        raise ColorExtractionError("file_not_found", "Uploaded image was not found.")

    try:
        with Image.open(path) as opened:
            opened.verify()
        with Image.open(path) as opened:
            image = _convert_to_rgb(opened)
    except (UnidentifiedImageError, OSError) as exc:
        raise ColorExtractionError(
            "invalid_image",
            "Uploaded file could not be opened as a valid image.",
        ) from exc

    if image.width <= 0 or image.height <= 0:
        raise ColorExtractionError("invalid_image", "Uploaded image has invalid size.")

    longest_side = max(image.width, image.height)
    if longest_side > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest_side
        new_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def _convert_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def _sample_pixels(image_array: np.ndarray, max_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_array.shape[:2]
    pixels = image_array.reshape(-1, 3)
    flat_indices = np.arange(len(pixels))

    if len(pixels) <= max_pixels:
        return pixels, _normalized_coordinates(flat_indices, width, height)

    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(len(pixels), size=max_pixels, replace=False)
    return pixels[indices], _normalized_coordinates(indices, width, height)


def _normalized_coordinates(
    flat_indices: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    x_coordinates = (flat_indices % width) / max(1, width - 1)
    y_coordinates = (flat_indices // width) / max(1, height - 1)
    return np.column_stack((x_coordinates, y_coordinates))


def _pixel_importance_weights(
    pixels: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    colors = pixels.astype(float) / 255
    value = np.max(colors, axis=1)
    chroma = value - np.min(colors, axis=1)
    saturation = np.divide(
        chroma,
        value,
        out=np.zeros_like(chroma),
        where=value > 0,
    )
    saturation_score = saturation * value
    center_score = _composition_center_score(coordinates)
    weights = (
        1.0
        + (SATURATION_IMPORTANCE * saturation_score)
        + (CENTER_IMPORTANCE * center_score)
    )
    return np.clip(weights, 1.0, MAX_PIXEL_WEIGHT)


def _composition_center_score(coordinates: np.ndarray) -> np.ndarray:
    x_coordinates = coordinates[:, 0]
    y_coordinates = coordinates[:, 1]
    x_center = 0.50
    y_center = 0.58
    x_spread = 0.36
    y_spread = 0.34
    distance = (
        ((x_coordinates - x_center) / x_spread) ** 2
        + ((y_coordinates - y_center) / y_spread) ** 2
    )
    return np.exp(-0.5 * distance)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _accent_score(rgb: tuple[int, int, int]) -> float:
    red, green, blue = (channel / 255 for channel in rgb)
    value = max(red, green, blue)
    chroma = value - min(red, green, blue)
    saturation = 0.0 if value == 0 else chroma / value
    return saturation * value


def _nearest_basic_color_name(rgb: tuple[int, int, int]) -> str:
    rgb_array = np.array(rgb)
    best_name = min(
        BASIC_COLOR_NAMES,
        key=lambda item: float(np.linalg.norm(rgb_array - np.array(item[1]))),
    )[0]
    return best_name
