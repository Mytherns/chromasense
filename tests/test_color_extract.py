from PIL import Image
import numpy as np

from chromasense.color_extract import extract_colors, resize_to_long_side, rgb_to_hex, saturation_weights


def test_kmeans_extracts_sorted_palette_from_simple_image():
    image = Image.new("RGB", (20, 10), "#ff0000")
    for x in range(10, 20):
        for y in range(10):
            image.putpixel((x, y), (0, 0, 255))

    colors = extract_colors(image, k=2, algorithm="kmeans", max_sample_pixels=200)

    assert len(colors) == 2
    assert sum(color["percentage"] for color in colors) == 1.0
    assert all(color["hex"].startswith("#") and len(color["hex"]) == 7 for color in colors)
    assert colors[0]["percentage"] >= colors[1]["percentage"]


def test_resize_to_long_side_preserves_aspect_ratio():
    resized = resize_to_long_side(Image.new("RGB", (2000, 1000)), max_long_side=1000)

    assert resized.size == (1000, 500)


def test_rgb_to_hex_clamps_and_formats_uppercase():
    assert rgb_to_hex([-5, 16, 300]) == "#0010FF"


def test_saturation_weights_prefer_colorful_midtones_over_gray_and_shadow():
    pixels = np.array(
        [
            [210, 40, 120],
            [190, 190, 190],
            [30, 28, 22],
        ],
        dtype=np.uint8,
    )

    weights = saturation_weights(pixels)

    assert weights[0] > weights[1]
    assert weights[0] > weights[2]
