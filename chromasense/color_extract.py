from __future__ import annotations

import colorsys
import warnings
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

Algorithm = Literal["kmeans", "gmm"]

DEFAULT_K = 5
DEFAULT_SEED = 42
MAX_LONG_SIDE = 1024
MAX_SAMPLE_PIXELS = 50_000
MIN_CLUSTER_SHARE = 0.01
AREA_WEIGHT = 0.15
ACCENT_WEIGHT = 0.15
DIVERSITY_WEIGHT = 0.70
SATURATION_IMPORTANCE = 1.2
MAX_PIXEL_WEIGHT = 2.5


def extract_colors(
    image: str | Path | Image.Image,
    k: int = DEFAULT_K,
    algorithm: Algorithm = "kmeans",
    seed: int = DEFAULT_SEED,
    max_sample_pixels: int = MAX_SAMPLE_PIXELS,
    resize_long_side: int = MAX_LONG_SIDE,
) -> list[dict]:
    # Extract weighted dominant colors from image using LAB clustering.
    if algorithm not in {"kmeans", "gmm"}:
        raise ValueError("algorithm must be 'kmeans' or 'gmm'")
    if k < 1:
        raise ValueError("k must be >= 1")

    rgb_image = load_rgb_image(image)
    resized = resize_to_long_side(rgb_image, resize_long_side)
    pixels_rgb = _image_pixels_rgb(resized)
    if pixels_rgb.size == 0:
        raise ValueError("image has no pixels")

    pixels_lab = rgb2lab(pixels_rgb.reshape(1, -1, 3)).reshape(-1, 3)
    weights = saturation_weights(pixels_rgb)
    output_k = min(k, len(pixels_lab))
    fit_k = min(max(output_k * 2, output_k), len(pixels_lab))

    if algorithm == "kmeans":
        centers_lab, labels = _fit_kmeans(
            pixels_lab=pixels_lab,
            weights=weights,
            k=fit_k,
            seed=seed,
            max_sample_pixels=max_sample_pixels,
        )
    else:
        centers_lab, labels = _fit_gmm(
            pixels_lab=pixels_lab,
            weights=weights,
            k=fit_k,
            seed=seed,
            max_sample_pixels=max_sample_pixels,
        )

    return _palette_from_clusters(
        pixels_rgb=pixels_rgb,
        pixels_lab=pixels_lab,
        weights=weights,
        centers_lab=centers_lab,
        labels=labels,
        k=output_k,
    )


def extract_palette(
    image: str | Path | Image.Image,
    k: int = DEFAULT_K,
    algorithm: Algorithm = "kmeans",
    seed: int = DEFAULT_SEED,
) -> dict:
    # Return API-friendly palette envelope for one algorithm.
    return {
        "algorithm": algorithm,
        "colors": extract_colors(image=image, k=k, algorithm=algorithm, seed=seed),
    }


def load_rgb_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as opened:
        return opened.convert("RGB")


def resize_to_long_side(image: Image.Image, max_long_side: int = MAX_LONG_SIDE) -> Image.Image:
    if max_long_side < 1:
        raise ValueError("max_long_side must be >= 1")

    width, height = image.size
    longest = max(width, height)
    if longest <= max_long_side:
        return image.copy()

    scale = max_long_side / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def saturation_weights(pixels_rgb: np.ndarray) -> np.ndarray:
    colors = pixels_rgb.astype(np.float64) / 255.0
    value = np.max(colors, axis=1)
    chroma = value - np.min(colors, axis=1)
    saturation = np.divide(chroma, value, out=np.zeros_like(chroma), where=value > 0)
    saturation_score = saturation * value
    weights = 1.0 + (SATURATION_IMPORTANCE * saturation_score)
    return np.clip(weights, 1.0, MAX_PIXEL_WEIGHT)


def _fit_kmeans(
    pixels_lab: np.ndarray,
    weights: np.ndarray,
    k: int,
    seed: int,
    max_sample_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sample_idx = _sample_indices(len(pixels_lab), max_sample_pixels, rng)
    sample_lab = pixels_lab[sample_idx]
    sample_weights = weights[sample_idx]

    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(sample_lab, sample_weight=sample_weights)
    labels = model.predict(pixels_lab)
    return model.cluster_centers_, labels


def _fit_gmm(
    pixels_lab: np.ndarray,
    weights: np.ndarray,
    k: int,
    seed: int,
    max_sample_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sample_idx = _weighted_sample_indices(weights, max_sample_pixels, rng)
    sample_lab = pixels_lab[sample_idx]

    model = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=seed,
        max_iter=200,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(sample_lab)

    labels = model.predict(pixels_lab)
    return model.means_, labels


def _palette_from_clusters(
    pixels_rgb: np.ndarray,
    pixels_lab: np.ndarray,
    weights: np.ndarray,
    centers_lab: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> list[dict]:
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("pixel weights sum to zero")

    clusters = []
    for label in range(len(centers_lab)):
        mask = labels == label
        if not np.any(mask):
            continue
        share = float(weights[mask].sum() / total_weight)

        rgb, center_lab = _representative_cluster_color(
            pixels_rgb=pixels_rgb[mask],
            pixels_lab=pixels_lab[mask],
            weights=weights[mask],
            center_lab=centers_lab[label],
        )
        hue, lightness, saturation = _rgb_to_hsl(rgb)
        if share < MIN_CLUSTER_SHARE and saturation < 0.35 and len(centers_lab) > 1:
            continue
        clusters.append(
            {
                "rgb": rgb,
                "hex": rgb_to_hex(rgb),
                "percentage": round(share, 6),
                "lab": [round(float(v), 4) for v in center_lab],
                "hue": round(hue, 4),
                "lightness": round(lightness, 6),
                "saturation": round(saturation, 6),
                "_score": _palette_score(share, lightness, saturation),
            }
        )

    clusters = _select_diverse_colors(clusters, k)
    if len(clusters) < k:
        clusters = _refill_palette(clusters, pixels_rgb, weights, k)
    clusters = _renormalize_percentages(clusters[:k])
    for color in clusters:
        color.pop("_score", None)
    return clusters


def _palette_score(share: float, lightness: float, saturation: float) -> float:
    lightness_score = 0.35 + 0.65 * (1.0 - abs(lightness - 0.55) / 0.55)
    lightness_score = max(0.15, min(1.0, lightness_score))
    if lightness < 0.2:
        lightness_score *= 0.25
    if saturation < 0.2:
        lightness_score *= 0.35
    return share * (0.4 + saturation * 2.6) * lightness_score


def _select_diverse_colors(clusters: list[dict], k: int) -> list[dict]:
    if not clusters:
        return []

    selected = [max(clusters, key=lambda color: color["percentage"])]
    while len(selected) < k and len(selected) < len(clusters):
        remaining = [color for color in clusters if color not in selected]
        selected.append(
            max(
                remaining,
                key=lambda color: _candidate_score(
                    color=color,
                    selected=selected,
                    max_percentage=max(item["percentage"] for item in clusters),
                ),
            )
        )
    selected = _promote_color_accents(selected, clusters, k)
    return sorted(selected, key=lambda item: item["percentage"], reverse=True)


def _promote_color_accents(selected: list[dict], clusters: list[dict], k: int) -> list[dict]:
    accent_candidates = sorted(
        [
            color
            for color in clusters
            if float(color["saturation"]) >= 0.28 and float(color["percentage"]) >= 0.003
        ],
        key=lambda color: (
            float(color["saturation"]) * max(float(color["lightness"]), 0.0),
            float(color["percentage"]),
        ),
        reverse=True,
    )
    if not accent_candidates:
        return selected

    promoted = list(selected)
    max_accents = min(2, k)
    for accent in accent_candidates[:max_accents]:
        if accent in promoted:
            continue
        replace_index = _replaceable_neutral_index(promoted)
        if replace_index is None:
            break
        promoted[replace_index] = accent
    return promoted


def _replaceable_neutral_index(colors: list[dict]) -> int | None:
    candidates = [
        (index, color)
        for index, color in enumerate(colors)
        if index != 0 and float(color["saturation"]) < 0.18
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (float(item[1]["percentage"]), -float(item[1]["lightness"])))[0]


def _candidate_score(color: dict, selected: list[dict], max_percentage: float) -> float:
    area_score = color["percentage"] / max_percentage if max_percentage else 0.0
    accent_score = float(color["saturation"]) * max(float(color["lightness"]), 0.0)
    diversity_score = _nearest_color_distance(color, selected)
    return (
        (AREA_WEIGHT * area_score)
        + (ACCENT_WEIGHT * accent_score)
        + (DIVERSITY_WEIGHT * diversity_score)
    )


def _nearest_color_distance(color: dict, selected: list[dict]) -> float:
    color_feature = _color_feature(color)
    selected_features = np.array([_color_feature(item) for item in selected])
    distances = np.linalg.norm(color_feature - selected_features, axis=1)
    return float(np.clip(np.min(distances) / 1.8, 0.0, 1.0))


def _color_feature(color: dict) -> np.ndarray:
    hue = np.deg2rad(float(color["hue"]))
    saturation = float(color["saturation"])
    lightness = float(color["lightness"])
    red, green, blue = [channel / 255.0 for channel in color["rgb"]]
    return np.array(
        [
            red * 0.45,
            green * 0.45,
            blue * 0.45,
            np.cos(hue) * saturation * 0.65,
            np.sin(hue) * saturation * 0.65,
            saturation * 0.35,
            lightness * 0.15,
        ],
        dtype=np.float64,
    )


def _representative_cluster_color(
    pixels_rgb: np.ndarray,
    pixels_lab: np.ndarray,
    weights: np.ndarray,
    center_lab: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    distances = np.linalg.norm(pixels_lab - center_lab, axis=1)
    closeness = 1.0 / (1.0 + distances / 25.0)
    salience = weights * closeness
    best_index = int(np.argmax(salience))
    rgb = [int(value) for value in pixels_rgb[best_index]]
    lab = pixels_lab[best_index]
    return rgb, lab


def _refill_palette(
    clusters: list[dict],
    pixels_rgb: np.ndarray,
    weights: np.ndarray,
    k: int,
) -> list[dict]:
    if len(clusters) >= k:
        return clusters

    used_hex = {color["hex"] for color in clusters}
    order = np.argsort(weights)[::-1]
    for idx in order:
        rgb = [int(v) for v in pixels_rgb[idx]]
        hex_color = rgb_to_hex(rgb)
        if hex_color in used_hex:
            continue
        hue, lightness, saturation = _rgb_to_hsl(rgb)
        clusters.append(
            {
                "rgb": rgb,
                "hex": hex_color,
                "percentage": 0.0,
                "lab": [round(float(v), 4) for v in rgb2lab(np.array([[np.array(rgb) / 255.0]])).reshape(3)],
                "hue": round(hue, 4),
                "lightness": round(lightness, 6),
                "saturation": round(saturation, 6),
            }
        )
        used_hex.add(hex_color)
        if len(clusters) >= k:
            break
    return clusters


def _renormalize_percentages(colors: list[dict]) -> list[dict]:
    total = sum(float(color["percentage"]) for color in colors)
    if total <= 0:
        even = round(1 / len(colors), 6) if colors else 0.0
        for color in colors:
            color["percentage"] = even
        return colors

    normalized = []
    running = 0.0
    for color in colors:
        current = dict(color)
        current["percentage"] = round(float(current["percentage"]) / total, 6)
        running += current["percentage"]
        normalized.append(current)
    if normalized:
        normalized[-1]["percentage"] = round(normalized[-1]["percentage"] + (1.0 - running), 6)
    return normalized


def _image_pixels_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8).reshape(-1, 3)


def _sample_indices(size: int, max_sample_pixels: int, rng: np.random.Generator) -> np.ndarray:
    if size <= max_sample_pixels:
        return np.arange(size)
    return rng.choice(size, size=max_sample_pixels, replace=False)


def _weighted_sample_indices(
    weights: np.ndarray,
    max_sample_pixels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    size = len(weights)
    sample_size = min(size, max_sample_pixels)
    probabilities = weights / weights.sum()
    return rng.choice(size, size=sample_size, replace=size < max_sample_pixels, p=probabilities)


def _lab_to_rgb255(lab: Iterable[float]) -> list[int]:
    lab_array = np.array(lab, dtype=np.float64).reshape(1, 1, 3)
    rgb_float = lab2rgb(lab_array).reshape(3)
    rgb = np.clip(np.round(rgb_float * 255), 0, 255).astype(np.uint8)
    return [int(value) for value in rgb]


def _rgb_to_hsl(rgb: Iterable[float]) -> tuple[float, float, float]:
    r, g, b = [float(value) / 255.0 for value in rgb]
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    return hue * 360.0, lightness, saturation


def rgb_to_hex(rgb: Iterable[int]) -> str:
    values = [max(0, min(255, int(round(value)))) for value in rgb]
    return "#" + "".join(f"{value:02X}" for value in values)
