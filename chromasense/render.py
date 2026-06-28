from __future__ import annotations

import base64
import io
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

PREVIEW_PHRASE = "Color speaks before words"
DEFAULT_SIZE = (600, 300)
DEFAULT_FONTS_DIR = Path("fonts")

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_FONT_EXTENSIONS = {".ttf", ".otf"}
_FALLBACK_FONT_NAMES = (
    "DejaVuSans.ttf",
    "Arial.ttf",
    "arial.ttf",
    "segoeui.ttf",
)


def render_font_preview(
    palette: Sequence[Mapping[str, object] | str],
    font_name: str | None = None,
    *,
    phrase: str = PREVIEW_PHRASE,
    size: tuple[int, int] = DEFAULT_SIZE,
    fonts_dir: Path | str = DEFAULT_FONTS_DIR,
) -> str:
    """Render font preview to a base64-encoded PNG string."""
    width, height = _validated_size(size)
    palette_hexes = _palette_hexes(palette)
    background_hex = palette_hexes[0]
    text_hex = pick_contrasting_color(background_hex, [*palette_hexes[1:], "#FFFFFF", "#111111"])

    image = Image.new("RGB", (width, height), _hex_to_rgb(background_hex))
    draw = ImageDraw.Draw(image)
    font = _fit_font(
        draw=draw,
        text=phrase,
        font_name=font_name,
        fonts_dir=Path(fonts_dir),
        max_width=int(width * 0.86),
        max_size=54,
        min_size=22,
    )
    _draw_centered_text(draw, phrase, font, (0, 0, width, height), text_hex)

    if len(palette_hexes) > 1:
        _draw_palette_strip(draw, palette_hexes, width, height)

    return _encode_png_base64(image)


def render_mockup(
    palette: Sequence[Mapping[str, object] | str],
    font_name: str | None = None,
    *,
    size: tuple[int, int] = DEFAULT_SIZE,
    fonts_dir: Path | str = DEFAULT_FONTS_DIR,
) -> str:
    """Render simple brand mockup to a base64-encoded PNG string."""
    width, height = _validated_size(size)
    palette_hexes = _palette_hexes(palette)
    primary_hex = palette_hexes[0]
    accent_hex = palette_hexes[1] if len(palette_hexes) > 1 else primary_hex
    canvas_hex = pick_contrasting_color(primary_hex, ["#FAFAF7", "#151515"])
    header_text_hex = pick_contrasting_color(primary_hex, ["#FFFFFF", "#111111"])
    button_text_hex = pick_contrasting_color(accent_hex, ["#FFFFFF", "#111111"])
    body_text_hex = pick_contrasting_color(canvas_hex, ["#151515", "#FFFFFF"])

    image = Image.new("RGB", (width, height), _hex_to_rgb(canvas_hex))
    draw = ImageDraw.Draw(image)

    header_h = int(height * 0.30)
    draw.rectangle((0, 0, width, header_h), fill=_hex_to_rgb(primary_hex))

    title_font = _load_font(font_name, 30, Path(fonts_dir))
    body_font = _load_font(font_name, 18, Path(fonts_dir))
    button_font = _load_font(font_name, 17, Path(fonts_dir))

    draw.text((34, 24), "Chromasense", fill=_hex_to_rgb(header_text_hex), font=title_font)
    draw.text((36, 112), "Palette-led identity", fill=_hex_to_rgb(body_text_hex), font=title_font)
    draw.text((38, 154), "Color, typography, and mood aligned", fill=_hex_to_rgb(body_text_hex), font=body_font)
    draw.text((38, 178), "from one image.", fill=_hex_to_rgb(body_text_hex), font=body_font)

    button_box = (38, 207, 194, 252)
    draw.rounded_rectangle(button_box, radius=8, fill=_hex_to_rgb(accent_hex))
    _draw_centered_text(draw, "Preview", button_font, button_box, button_text_hex)

    swatch_x = width - 178
    swatch_y = 112
    for index, hex_color in enumerate(palette_hexes[:5]):
        x0 = swatch_x + (index % 2) * 72
        y0 = swatch_y + (index // 2) * 52
        draw.rounded_rectangle((x0, y0, x0 + 52, y0 + 34), radius=6, fill=_hex_to_rgb(hex_color))
        draw.rectangle((x0, y0, x0 + 52, y0 + 34), outline=_hex_to_rgb(body_text_hex), width=1)

    return _encode_png_base64(image)


def pick_contrasting_color(background_hex: str, candidate_hexes: Iterable[str]) -> str:
    """Pick candidate with highest gamma-corrected WCAG contrast ratio."""
    background_rgb = _hex_to_rgb(background_hex)
    candidates = list(candidate_hexes)
    if not candidates:
        raise ValueError("candidate_hexes must not be empty")
    return max(candidates, key=lambda candidate: _contrast_ratio(background_rgb, _hex_to_rgb(candidate)))


def _validated_size(size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    if width < 1 or height < 1:
        raise ValueError("render size must be positive")
    if width > 800 or height > 800:
        raise ValueError("render size must be capped at 800px per side")
    return int(width), int(height)


def _palette_hexes(palette: Sequence[Mapping[str, object] | str]) -> list[str]:
    if not palette:
        raise ValueError("palette must contain at least one color")

    hexes = []
    for item in palette:
        if isinstance(item, str):
            hex_color = item
        elif isinstance(item, Mapping) and "hex" in item:
            hex_color = str(item["hex"])
        else:
            raise ValueError("palette items must be hex strings or mappings with hex")
        _hex_to_rgb(hex_color)
        hexes.append(hex_color.upper())
    return hexes


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    if not _HEX_RE.match(str(hex_color)):
        raise ValueError(f"invalid hex color: {hex_color}")
    color = str(hex_color).lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    channels = []
    for value in rgb:
        channel = value / 255.0
        if channel <= 0.03928:
            channels.append(channel / 12.92)
        else:
            channels.append(((channel + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast_ratio(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    left_luminance = _relative_luminance(left)
    right_luminance = _relative_luminance(right)
    lighter = max(left_luminance, right_luminance)
    darker = min(left_luminance, right_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def _draw_palette_strip(draw: ImageDraw.ImageDraw, palette_hexes: Sequence[str], width: int, height: int) -> None:
    strip_h = 22
    swatch_w = max(1, width // len(palette_hexes))
    y0 = height - strip_h
    for index, hex_color in enumerate(palette_hexes):
        x0 = index * swatch_w
        x1 = width if index == len(palette_hexes) - 1 else (index + 1) * swatch_w
        draw.rectangle((x0, y0, x1, height), fill=_hex_to_rgb(hex_color))


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    box: tuple[int, int, int, int],
    hex_color: str,
) -> None:
    text_box = draw.textbbox((0, 0), text, font=font)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    x0, y0, x1, y1 = box
    x = x0 + ((x1 - x0) - text_w) / 2
    y = y0 + ((y1 - y0) - text_h) / 2 - text_box[1]
    draw.text((x, y), text, fill=_hex_to_rgb(hex_color), font=font)


def _fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_name: str | None,
    fonts_dir: Path,
    max_width: int,
    max_size: int,
    min_size: int,
) -> ImageFont.ImageFont:
    for size in range(max_size, min_size - 1, -2):
        font = _load_font(font_name, size, fonts_dir)
        text_box = draw.textbbox((0, 0), text, font=font)
        if text_box[2] - text_box[0] <= max_width:
            return font
    return _load_font(font_name, min_size, fonts_dir)


@lru_cache(maxsize=128)
def _load_font(font_name: str | None, size: int, fonts_dir: Path) -> ImageFont.ImageFont:
    font_path = _find_local_font(font_name, fonts_dir)
    if font_path is not None:
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except OSError:
            pass

    for fallback_name in _FALLBACK_FONT_NAMES:
        try:
            return ImageFont.truetype(fallback_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _find_local_font(font_name: str | None, fonts_dir: Path) -> Path | None:
    if not font_name or not fonts_dir.is_dir():
        return None

    target = _normalize_font_name(font_name)
    candidates = [
        path
        for path in fonts_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _FONT_EXTENSIONS
    ]
    for path in sorted(candidates):
        stem = _normalize_font_name(path.stem)
        if target in stem or stem in target:
            return path
    return None


def _normalize_font_name(font_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", font_name.lower())


def _encode_png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
