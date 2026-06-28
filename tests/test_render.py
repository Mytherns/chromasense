import base64
import io

import pytest
from PIL import Image

from chromasense.render import (
    _contrast_ratio,
    _relative_luminance,
    pick_contrasting_color,
    render_font_preview,
    render_mockup,
)


def _palette():
    return [
        {"hex": "#123456", "percentage": 0.4},
        {"hex": "#F2C14E", "percentage": 0.25},
        {"hex": "#E94F37", "percentage": 0.2},
        {"hex": "#4F7CAC", "percentage": 0.1},
        {"hex": "#111111", "percentage": 0.05},
    ]


def _decode_png(encoded):
    raw = base64.b64decode(encoded, validate=True)
    image = Image.open(io.BytesIO(raw))
    image.load()
    return image


def test_contrast_helpers_use_expected_luminance_extremes():
    assert _relative_luminance((0, 0, 0)) == pytest.approx(0.0)
    assert _relative_luminance((255, 255, 255)) == pytest.approx(1.0)
    assert _contrast_ratio((0, 0, 0), (255, 255, 255)) == pytest.approx(21.0)


def test_pick_contrasting_color_selects_highest_contrast_candidate():
    assert pick_contrasting_color("#000000", ["#222222", "#FFFFFF"]) == "#FFFFFF"
    assert pick_contrasting_color("#FFFFFF", ["#F7F7F7", "#111111"]) == "#111111"


def test_render_font_preview_returns_capped_png_base64():
    image = _decode_png(render_font_preview(_palette(), font_name="Missing Font"))

    assert image.format == "PNG"
    assert image.size == (600, 300)


def test_render_mockup_returns_capped_png_base64():
    image = _decode_png(render_mockup(_palette(), font_name="Missing Font"))

    assert image.format == "PNG"
    assert image.size == (600, 300)


def test_render_rejects_empty_palette_or_uncapped_size():
    with pytest.raises(ValueError, match="palette"):
        render_font_preview([])

    with pytest.raises(ValueError, match="capped"):
        render_mockup(_palette(), size=(900, 300))
