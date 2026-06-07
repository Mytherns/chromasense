# Qwen vision-language guidance for Chromasense.
#
# This module stays framework-agnostic. FastAPI should pass the saved image path,
# extracted palette, and CLIP classification, then return this schema directly.

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_NEW_TOKENS = 900
GENERATION_TEMPERATURE = 0.35
GENERATION_TOP_P = 0.9

_load_lock = threading.Lock()
_model: Any | None = None
_processor: Any | None = None
_process_vision_info: Any | None = None
_device: str | None = None
_model_load_seconds = 0.0


BASIC_COLOR_NAMES = [
    ("Black", (0, 0, 0)),
    ("White", (255, 255, 255)),
    ("Gray", (128, 128, 128)),
    ("Dark Gray", (64, 64, 64)),
    ("Navy", (0, 0, 128)),
    ("Blue Gray", (54, 71, 90)),
    ("Brown", (120, 72, 38)),
    ("Tan", (180, 140, 92)),
    ("Beige", (214, 196, 162)),
    ("Red", (220, 38, 38)),
    ("Orange", (249, 115, 22)),
    ("Yellow", (234, 179, 8)),
    ("Olive Green", (85, 86, 23)),
    ("Green", (34, 197, 94)),
    ("Cyan", (6, 182, 212)),
    ("Sky Blue", (59, 130, 246)),
    ("Blue", (0, 0, 255)),
    ("Purple", (147, 51, 234)),
    ("Pink", (236, 72, 153)),
]


class QwenAnalysisError(RuntimeError):
    """Raised when Qwen generation or schema handling fails."""


def load_qwen_model() -> dict[str, Any]:
    """Lazy-load Qwen and return compact readiness metadata."""
    global _model, _processor, _process_vision_info, _device, _model_load_seconds

    if _model is not None and _processor is not None and _process_vision_info is not None:
        return _loaded_metadata()

    with _load_lock:
        if (
            _model is not None
            and _processor is not None
            and _process_vision_info is not None
        ):
            return _loaded_metadata()

        started = time.perf_counter()
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise QwenAnalysisError(
                "Qwen dependencies are missing. Install torch, transformers, accelerate, and qwen-vl-utils."
            ) from exc

        _device = _best_torch_device(torch)
        _processor = AutoProcessor.from_pretrained(MODEL_NAME)
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
        )
        _model.eval()
        _process_vision_info = process_vision_info
        _model_load_seconds = round(time.perf_counter() - started, 2)

    return _loaded_metadata()


def is_qwen_loaded() -> bool:
    """Return whether Qwen model, processor, and vision helper are already loaded."""
    return _model is not None and _processor is not None and _process_vision_info is not None


def analyze_design(
    image: str | Path | Any,
    colors: list[dict[str, Any]],
    clip_classification: dict[str, Any],
    n_colors: int | None = None,
    image_size: list[int] | tuple[int, int] | None = None,
    clip_used_fallback: bool = False,
) -> dict[str, Any]:
    """Generate beginner design guidance and always return the public schema."""
    started = time.perf_counter()

    if os.getenv("CHROMASENSE_SKIP_QWEN") == "1":
        return fallback_analysis(
            colors=colors,
            clip_classification=clip_classification,
            n_colors=n_colors,
            image_size=image_size,
            seconds=time.perf_counter() - started,
            warning="Qwen skipped because CHROMASENSE_SKIP_QWEN=1.",
            clip_used_fallback=clip_used_fallback,
        )

    try:
        raw_text = _generate_qwen_text(image, colors, clip_classification)
        parsed = parse_qwen_json(raw_text)
        result = enforce_analysis_schema(
            parsed,
            colors=colors,
            clip_classification=clip_classification,
            n_colors=n_colors,
            image_size=image_size,
            seconds=time.perf_counter() - started,
            qwen_used_fallback=False,
            clip_used_fallback=clip_used_fallback,
            warnings=[],
        )
        return result
    except Exception as exc:
        return fallback_analysis(
            colors=colors,
            clip_classification=clip_classification,
            n_colors=n_colors,
            image_size=image_size,
            seconds=time.perf_counter() - started,
            warning=f"Qwen guidance failed: {exc}",
            clip_used_fallback=clip_used_fallback,
        )


def warmup_qwen() -> dict[str, Any]:
    """Load Qwen for the future `/warmup` endpoint."""
    if os.getenv("CHROMASENSE_SKIP_QWEN") == "1":
        raise QwenAnalysisError("Qwen skipped because CHROMASENSE_SKIP_QWEN=1.")
    return load_qwen_model()


def parse_qwen_json(raw_text: str) -> dict[str, Any]:
    """Parse Qwen JSON, repairing common Markdown/object-wrapper output."""
    if not raw_text or not raw_text.strip():
        raise QwenAnalysisError("Qwen returned empty output.")

    stripped = _strip_code_fence(raw_text.strip())
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        block = _extract_first_json_object(stripped)
        if block is None:
            raise QwenAnalysisError("Qwen output did not contain a JSON object.")
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError as exc:
            raise QwenAnalysisError("Qwen JSON repair failed.") from exc

    if not isinstance(parsed, dict):
        raise QwenAnalysisError("Qwen output JSON must be an object.")
    return parsed


def enforce_analysis_schema(
    data: dict[str, Any],
    colors: list[dict[str, Any]],
    clip_classification: dict[str, Any],
    n_colors: int | None = None,
    image_size: list[int] | tuple[int, int] | None = None,
    seconds: float = 0.0,
    qwen_used_fallback: bool = False,
    clip_used_fallback: bool = False,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Coerce parsed Qwen output into the project response schema."""
    safe_colors = _coerce_colors(data.get("colors"), colors)
    safe_clip = _coerce_clip_classification(clip_classification)
    safe_tags = _coerce_string_list(data.get("tags"), _fallback_tags(safe_clip, safe_colors))

    result = {
        "mood": _coerce_string(data.get("mood"), _fallback_mood_name(safe_clip, safe_colors)),
        "vibe": _coerce_string(data.get("vibe"), _fallback_vibe(safe_colors, safe_clip)),
        "clip_classification": safe_clip,
        "colors": safe_colors,
        "tags": safe_tags,
        "use_case": _coerce_string(data.get("use_case"), _fallback_use_case(safe_clip, safe_colors)),
        "feedback": _coerce_feedback(data.get("feedback"), safe_colors, safe_clip),
        "pairing": _coerce_pairing(data.get("pairing"), safe_colors),
        "warnings": _coerce_string_list(warnings, []),
        "metadata": _build_metadata(
            n_colors=n_colors or len(safe_colors),
            image_size=image_size,
            seconds=seconds,
            qwen_used_fallback=qwen_used_fallback,
            clip_used_fallback=clip_used_fallback,
        ),
    }
    return result


def fallback_analysis(
    colors: list[dict[str, Any]],
    clip_classification: dict[str, Any],
    n_colors: int | None = None,
    image_size: list[int] | tuple[int, int] | None = None,
    seconds: float = 0.0,
    warning: str = "Qwen guidance fallback used.",
    clip_used_fallback: bool = False,
) -> dict[str, Any]:
    """Return deterministic design guidance when Qwen cannot run."""
    safe_colors = _fallback_color_names(colors)
    safe_clip = _coerce_clip_classification(clip_classification)

    return {
        "mood": _fallback_mood_name(safe_clip, safe_colors),
        "vibe": _fallback_vibe(safe_colors, safe_clip),
        "clip_classification": safe_clip,
        "colors": safe_colors,
        "tags": _fallback_tags(safe_clip, safe_colors),
        "use_case": _fallback_use_case(safe_clip, safe_colors),
        "feedback": _coerce_feedback(None, safe_colors, safe_clip),
        "pairing": _coerce_pairing(None, safe_colors),
        "warnings": [warning],
        "metadata": _build_metadata(
            n_colors=n_colors or len(safe_colors),
            image_size=image_size,
            seconds=seconds,
            qwen_used_fallback=True,
            clip_used_fallback=clip_used_fallback,
        ),
    }


def _generate_qwen_text(
    image: str | Path | Any,
    colors: list[dict[str, Any]],
    clip_classification: dict[str, Any],
) -> str:
    metadata = load_qwen_model()
    prompt = _build_prompt(colors, clip_classification)
    messages = [
        {
            "role": "user",
            "content": [
                _image_content(image),
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = _processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = _process_vision_info(messages)
    inputs = _processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = _move_inputs_to_device(inputs, metadata["device"])

    generated_ids = _model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=GENERATION_TEMPERATURE,
        top_p=GENERATION_TOP_P,
    )
    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    generated_trimmed = [
        output_ids[len(input_ids[index]) :]
        for index, output_ids in enumerate(generated_ids)
    ]
    decoded = _processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0] if decoded else ""


def _build_prompt(
    colors: list[dict[str, Any]],
    clip_classification: dict[str, Any],
) -> str:
    input_payload = {
        "colors": colors,
        "clip_classification": clip_classification,
    }
    return (
        "You are Chromasense, a beginner-focused design tutor. "
        "Analyze the uploaded image and the extracted palette. "
        "Return JSON only, no Markdown, no commentary. "
        "Use snake_case keys only. Do not use useCase or similar_brand. "
        "Preserve each color hex, rgb, role, and percentage from the input; improve only the color name. "
        "Avoid exact brand claims; use broad style references.\n\n"
        "Make the guidance specific to this image, not generic palette advice. "
        "Mention visible subject matter, material, lighting, composition, or scene context when clearly visible. "
        "Avoid repeated sentence starters across fields. "
        "Use concise beginner-friendly sentences with varied wording.\n\n"
        "Required JSON schema:\n"
        "{"
        '"mood": string, '
        '"vibe": string, '
        '"colors": [{"hex": string, "rgb": [number, number, number], "name": string, "role": string, "percentage": number}], '
        '"tags": [string, string, string], '
        '"use_case": string, '
        '"feedback": {"why_it_works": string, "emotion": string, "mistakes_to_avoid": string, "brand_style_reference": string}, '
        '"pairing": {"font_style": string, "texture": string, "secondary_palette": string, "layout_style": string}'
        "}\n\n"
        f"Input data:\n{json.dumps(input_payload, ensure_ascii=True)}"
    )


def _image_content(image: str | Path | Any) -> dict[str, Any]:
    if isinstance(image, (str, Path)):
        image_path = Path(image).expanduser().resolve()
        image_ref = str(image_path) if image_path.exists() else str(Path(image))
        return {"type": "image", "image": image_ref}
    return {"type": "image", "image": image}


def _move_inputs_to_device(inputs: Any, device: str | None) -> Any:
    if not device or device == "unknown":
        return inputs
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return inputs


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _coerce_colors(
    generated_colors: Any,
    source_colors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    generated_list = generated_colors if isinstance(generated_colors, list) else []
    output = []

    for index, source in enumerate(source_colors):
        generated = generated_list[index] if index < len(generated_list) else {}
        if not isinstance(generated, dict):
            generated = {}

        rgb = _coerce_rgb(source.get("rgb"))
        role = _coerce_string(source.get("role"), f"Color {index + 1}")
        output.append(
            {
                "hex": _coerce_hex(source.get("hex"), rgb),
                "rgb": rgb,
                "name": _coerce_string(
                    generated.get("name"),
                    _fallback_color_name(rgb, role),
                ),
                "role": role,
                "percentage": _coerce_percentage(source.get("percentage")),
            }
        )
    return output


def _fallback_color_names(colors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _coerce_colors([], colors)


def _coerce_clip_classification(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        data = {}

    top3 = data.get("top3_moods")
    safe_top3 = []
    if isinstance(top3, list):
        for item in top3[:3]:
            if isinstance(item, dict):
                safe_top3.append(
                    {
                        "mood": _coerce_string(item.get("mood"), "unknown"),
                        "score": _coerce_float(item.get("score"), 0.0),
                    }
                )

    primary = _coerce_string(data.get("primary_mood"), "unknown")
    confidence = _coerce_float(data.get("confidence"), 0.0)

    return {
        "primary_mood": primary,
        "confidence": confidence,
        "top3_moods": safe_top3,
    }


def _coerce_feedback(
    data: Any,
    colors: list[dict[str, Any]],
    clip_classification: dict[str, Any],
) -> dict[str, str]:
    if not isinstance(data, dict):
        data = {}
    mood = clip_classification.get("primary_mood", "unknown")
    profile = _palette_profile(colors)
    color_names = _color_name_phrase(colors)
    return {
        "why_it_works": _coerce_string(
            data.get("why_it_works"),
            _feedback_why_it_works(profile, color_names),
        ),
        "emotion": _coerce_string(
            data.get("emotion"),
            _feedback_emotion(profile, mood),
        ),
        "mistakes_to_avoid": _coerce_string(
            data.get("mistakes_to_avoid"),
            _feedback_mistakes(profile),
        ),
        "brand_style_reference": _coerce_string(
            data.get("brand_style_reference"),
            _brand_style_reference(profile, mood),
        ),
    }


def _coerce_pairing(data: Any, colors: list[dict[str, Any]]) -> dict[str, str]:
    if not isinstance(data, dict):
        data = {}
    profile = _palette_profile(colors)
    return {
        "font_style": _coerce_string(
            data.get("font_style"),
            _pairing_font_style(profile),
        ),
        "texture": _coerce_string(
            data.get("texture"),
            _pairing_texture(profile),
        ),
        "secondary_palette": _coerce_string(
            data.get("secondary_palette"),
            _pairing_secondary_palette(profile, colors),
        ),
        "layout_style": _coerce_string(
            data.get("layout_style"),
            _pairing_layout_style(profile),
        ),
    }


def _build_metadata(
    n_colors: int,
    image_size: list[int] | tuple[int, int] | None,
    seconds: float,
    qwen_used_fallback: bool,
    clip_used_fallback: bool,
) -> dict[str, Any]:
    return {
        "n_colors": int(n_colors),
        "image_size": _coerce_image_size(image_size),
        "models": {
            "color": "Weighted KMeans",
            "mood": "openai/clip-vit-base-patch32",
            "generation": MODEL_NAME,
        },
        "runtime": {
            "device": _device or "unknown",
            "seconds": round(float(seconds), 2),
            "model_load_seconds": _model_load_seconds,
            "used_fallback": {
                "clip": bool(clip_used_fallback),
                "qwen": bool(qwen_used_fallback),
            },
        },
    }


def _palette_profile(colors: list[dict[str, Any]]) -> dict[str, Any]:
    weighted = []
    for color in colors:
        if not isinstance(color, dict):
            continue
        rgb = _coerce_rgb(color.get("rgb"))
        weight = max(_coerce_percentage(color.get("percentage")), 1.0)
        hue, saturation, value = _rgb_to_hsv(rgb)
        weighted.append(
            {
                "rgb": rgb,
                "weight": weight,
                "hue": hue,
                "saturation": saturation,
                "value": value,
            }
        )

    if not weighted:
        return {
            "family": "neutral",
            "mood": "Practical Palette",
            "temperature": "balanced",
            "saturation": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
        }

    total = sum(item["weight"] for item in weighted)
    saturation = sum(item["saturation"] * item["weight"] for item in weighted) / total
    brightness = sum(item["value"] * item["weight"] for item in weighted) / total
    values = [item["value"] for item in weighted]
    contrast = max(values) - min(values)
    warm_weight = sum(
        item["weight"]
        for item in weighted
        if item["saturation"] >= 0.18 and (item["hue"] <= 70 or item["hue"] >= 330)
    )
    cool_weight = sum(
        item["weight"]
        for item in weighted
        if item["saturation"] >= 0.18 and 170 <= item["hue"] <= 270
    )
    green_weight = sum(
        item["weight"]
        for item in weighted
        if item["saturation"] >= 0.18 and 75 <= item["hue"] < 170
    )
    warm_ratio = warm_weight / total
    cool_ratio = cool_weight / total
    green_ratio = green_weight / total
    strongest_temperature_ratio = max(warm_ratio, cool_ratio, green_ratio)

    if brightness < 0.34 and contrast > 0.25:
        family = "dark"
        mood = "Mysterious Depth"
        temperature = "shadowed"
    elif saturation < 0.16 and brightness > 0.72:
        family = "minimal"
        mood = "Soft Minimal"
        temperature = "quiet"
    elif warm_weight >= cool_weight and warm_weight >= green_weight and warm_ratio >= 0.32:
        family = "warm"
        mood = "Warm Focus"
        temperature = "warm"
    elif cool_weight >= warm_weight and cool_weight >= green_weight and cool_ratio >= 0.28:
        family = "cool"
        mood = "Cool Calm"
        temperature = "cool"
    elif green_ratio >= 0.28:
        family = "natural"
        mood = "Fresh Natural"
        temperature = "organic"
    elif saturation > 0.48 and brightness > 0.55 and strongest_temperature_ratio < 0.32:
        family = "vibrant"
        mood = "Vivid Energy"
        temperature = "bright"
    elif brightness < 0.48:
        family = "moody"
        mood = "Muted Mood"
        temperature = "muted"
    else:
        family = "neutral"
        mood = "Balanced Palette"
        temperature = "balanced"

    return {
        "family": family,
        "mood": mood,
        "temperature": temperature,
        "saturation": saturation,
        "brightness": brightness,
        "contrast": contrast,
    }


def _rgb_to_hsv(rgb: list[int]) -> tuple[float, float, float]:
    red, green, blue = [channel / 255 for channel in rgb]
    maximum = max(red, green, blue)
    minimum = min(red, green, blue)
    delta = maximum - minimum

    if delta == 0:
        hue = 0.0
    elif maximum == red:
        hue = (60 * ((green - blue) / delta) + 360) % 360
    elif maximum == green:
        hue = 60 * ((blue - red) / delta) + 120
    else:
        hue = 60 * ((red - green) / delta) + 240

    saturation = 0.0 if maximum == 0 else delta / maximum
    return hue, saturation, maximum


def _coerce_string(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _coerce_string_list(value: Any, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return fallback
    strings = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return strings or fallback


def _coerce_rgb(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return [max(0, min(255, int(channel))) for channel in value]
        except (TypeError, ValueError):
            pass
    return [0, 0, 0]


def _coerce_hex(value: Any, rgb: list[int]) -> str:
    if isinstance(value, str) and len(value) == 7 and value.startswith("#"):
        try:
            int(value[1:], 16)
            return value.lower()
        except ValueError:
            pass
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _coerce_percentage(value: Any) -> float:
    return round(_coerce_float(value, 0.0), 1)


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(number) or math.isinf(number):
        return fallback
    return round(number, 1)


def _coerce_image_size(value: list[int] | tuple[int, int] | None) -> list[int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return [int(value[0]), int(value[1])]
        except (TypeError, ValueError):
            pass
    return []


def _fallback_mood_name(
    clip_classification: dict[str, Any],
    colors: list[dict[str, Any]] | None = None,
) -> str:
    primary = _coerce_string(clip_classification.get("primary_mood"), "unknown")
    if primary == "unknown":
        return _palette_profile(colors or [])["mood"]
    return primary.title()


def _fallback_vibe(
    colors: list[dict[str, Any]],
    clip_classification: dict[str, Any],
) -> str:
    mood = _coerce_string(clip_classification.get("primary_mood"), "balanced")
    profile = _palette_profile(colors)
    if mood == "unknown":
        return f"A {profile['temperature']} palette led by {_color_name_phrase(colors)}."
    return f"A {mood} palette led by {_color_name_phrase(colors)}."


def _fallback_tags(
    clip_classification: dict[str, Any],
    colors: list[dict[str, Any]] | None = None,
) -> list[str]:
    primary = _coerce_string(clip_classification.get("primary_mood"), "unknown")
    if primary == "unknown":
        profile = _palette_profile(colors or [])
        return [profile["family"], profile["temperature"], "palette", "design"]
    tags = [word for word in primary.replace("and", " ").split() if len(word) > 2]
    return (tags + ["palette", "design"])[:5]


def _fallback_use_case(
    clip_classification: dict[str, Any],
    colors: list[dict[str, Any]] | None = None,
) -> str:
    primary = _coerce_string(clip_classification.get("primary_mood"), "balanced")
    if primary == "unknown":
        profile = _palette_profile(colors or [])
        return {
            "warm": "Seasonal posters, food visuals, event promos, and friendly brand accents.",
            "cool": "Wellness pages, tech layouts, portfolio sections, and calm presentation slides.",
            "natural": "Eco campaigns, outdoor content, organic packaging, and lifestyle visuals.",
            "vibrant": "Festival posters, social media graphics, youth campaigns, and launch visuals.",
            "dark": "Music covers, cinematic posters, premium promos, and dramatic landing sections.",
            "moody": "Editorial layouts, atmospheric campaigns, and image-led portfolio pages.",
            "minimal": "Clean portfolios, product mockups, architecture pages, and premium stationery.",
            "neutral": "Beginner-friendly posters, social posts, and simple branding systems.",
        }.get(profile["family"], "Beginner-friendly posters, social posts, and simple branding systems.")
    return f"Beginner-friendly posters, social posts, and simple branding with a {primary} tone."


def _feedback_why_it_works(profile: dict[str, Any], color_names: str) -> str:
    reasons = {
        "warm": [
            "Warm hues make the design feel close, active, and easy to notice.",
            "Orange and brown warmth gives the palette instant human appeal.",
            "The warmer range creates a clear focal tone before details compete.",
        ],
        "cool": [
            "Cool hues create breathing room and keep the composition calm.",
            "Blue-leaning values make the palette feel orderly and easy to scan.",
            "The cool temperature lowers visual noise and supports a focused layout.",
        ],
        "natural": [
            "Green and earthy notes make the palette feel organic and grounded.",
            "Natural color cues help the palette feel fresh without becoming loud.",
            "Earthy balance gives the design a relaxed outdoor character.",
        ],
        "vibrant": [
            "High saturation gives the palette strong attention and quick visual impact.",
            "Bright color contrast creates energy before the viewer reads any text.",
            "The saturated accents make the palette useful for bold visual hierarchy.",
        ],
        "dark": [
            "Low brightness and contrast create depth, focus, and drama.",
            "The darker range frames bright details and gives the design weight.",
            "Shadow-heavy values make the palette feel cinematic and controlled.",
        ],
        "moody": [
            "Muted depth keeps the palette expressive without becoming loud.",
            "Soft contrast gives the design atmosphere while staying readable.",
            "The restrained values make the palette feel mature and reflective.",
        ],
        "minimal": [
            "Low saturation and light values make the palette clean and controlled.",
            "Soft neutrals reduce distraction and leave room for typography.",
            "The quiet value range keeps the design polished and easy to arrange.",
        ],
        "neutral": [
            "Balanced values make the palette flexible across many layouts.",
            "Neutral structure lets one accent color carry the main message.",
            "The palette has enough restraint to support many design directions.",
        ],
    }
    reason = _variant_choice(reasons.get(profile["family"], ["The palette has a clear visual direction."]), profile)
    return f"{reason} {color_names} create the main hierarchy."


def _feedback_emotion(profile: dict[str, Any], mood: str) -> str:
    if mood != "unknown":
        choices = [
            f"The palette reads as {mood}, with enough restraint for beginner layouts.",
            f"It gives a {mood} impression before extra graphics are added.",
            f"The color balance supports a {mood} tone without needing heavy effects.",
        ]
        return _variant_choice(choices, profile)
    choices_by_family = {
        "warm": ["The palette feels inviting, energetic, and human.", "It gives a friendly, active mood with strong warmth.", "The emotional tone feels close, seasonal, and approachable."],
        "cool": ["The palette feels calm, focused, and trustworthy.", "It creates a quieter mood suited to careful reading.", "The color temperature feels composed and steady."],
        "natural": ["The palette feels fresh, balanced, and outdoors-oriented.", "It suggests organic growth, openness, and comfort.", "The mood feels grounded, healthy, and relaxed."],
        "vibrant": ["The palette feels playful, bold, and high-energy.", "It creates excitement quickly and suits attention-heavy work.", "The mood feels expressive, young, and lively."],
        "dark": ["The palette feels serious, cinematic, and mysterious.", "It creates tension, depth, and a premium dramatic tone.", "The emotion is heavier, sharper, and more immersive."],
        "moody": ["The palette feels reflective, mature, and atmospheric.", "It creates a thoughtful mood with soft dramatic weight.", "The emotional tone feels quiet, layered, and editorial."],
        "minimal": ["The palette feels clean, quiet, and premium.", "It creates a polished mood with little visual clutter.", "The emotion feels calm, precise, and restrained."],
        "neutral": ["The palette feels stable, practical, and easy to reuse.", "It creates a flexible mood that can support many messages.", "The emotional tone is balanced and dependable."],
    }
    return _variant_choice(choices_by_family.get(profile["family"], ["The palette creates a clear emotional tone."]), profile)


def _feedback_mistakes(profile: dict[str, Any]) -> str:
    if profile["family"] == "vibrant":
        return _variant_choice([
            "Do not place all bright colors at full strength; use one dominant color and leave enough neutral space.",
            "Avoid making every saturated color compete; reserve the strongest hue for the main action.",
            "Do not pair bright backgrounds with small low-contrast text.",
        ], profile)
    if profile["family"] in {"dark", "moody"}:
        return _variant_choice([
            "Do not let text sit on similar dark values; add enough contrast for readability.",
            "Avoid hiding key information inside shadows; give text a clear light anchor.",
            "Do not overuse black overlays, or the palette may lose its color character.",
        ], profile)
    if profile["family"] == "minimal":
        return _variant_choice([
            "Do not make every element pale; keep one darker anchor for structure.",
            "Avoid removing all contrast; minimal designs still need hierarchy.",
            "Do not rely only on whitespace; use one firm value to guide the eye.",
        ], profile)
    return _variant_choice([
        "Avoid using every color at equal strength; keep one dominant color and use accents sparingly.",
        "Do not spread accents evenly across the page; group them around the main message.",
        "Avoid matching text too closely to the background; check contrast before final use.",
    ], profile)


def _brand_style_reference(profile: dict[str, Any], mood: str) -> str:
    if mood != "unknown":
        return _variant_choice([
            f"general {mood} visual identity and campaign design",
            f"{mood} editorial, poster, and social campaign systems",
            f"broad {mood} branding for image-led digital campaigns",
        ], profile)
    return {
        "warm": "friendly seasonal branding and hospitality campaigns",
        "cool": "wellness, tech, and calm editorial design",
        "natural": "organic lifestyle and eco-focused branding",
        "vibrant": "youth campaigns, event visuals, and launch graphics",
        "dark": "cinematic posters and premium entertainment visuals",
        "moody": "editorial campaigns and atmospheric portfolio design",
        "minimal": "premium product, architecture, and clean portfolio design",
        "neutral": "general visual identity and campaign design",
    }.get(profile["family"], "general visual identity and campaign design")


def _pairing_font_style(profile: dict[str, Any]) -> str:
    choices = {
        "warm": ["Use a friendly rounded sans-serif with a sturdy display heading.", "Try a soft display serif for headings with a readable rounded sans-serif body.", "Use warm, sturdy letterforms rather than thin elegant type."],
        "cool": ["Use a clean geometric sans-serif with light weights and generous spacing.", "Pair a precise sans-serif heading with calm regular-weight body text.", "Use airy typography with wider line spacing and simple numerals."],
        "natural": ["Use a humanist sans-serif or soft serif with relaxed line spacing.", "Pair an organic serif heading with a clean humanist sans-serif body.", "Use approachable type with open counters and medium weight."],
        "vibrant": ["Use a bold display font for headlines and a simple sans-serif for body text.", "Choose a chunky headline face, then keep body text plain.", "Use expressive type only in headlines so the colors stay readable."],
        "dark": ["Use a sharp serif or condensed display font with clean sans-serif body text.", "Try a high-contrast serif heading with compact sans-serif labels.", "Use narrow display type for drama, balanced by readable body copy."],
        "moody": ["Use a refined serif heading with a quiet sans-serif body font.", "Pair editorial serif headlines with restrained sans-serif captions.", "Use medium-weight type with enough spacing to avoid visual heaviness."],
        "minimal": ["Use a neutral sans-serif with medium weights and precise spacing.", "Use one clean sans-serif family and vary only weight and size.", "Choose restrained typography with clear hierarchy and no decorative extras."],
        "neutral": ["Use a clean sans-serif for body text and a slightly expressive display font for headings.", "Use practical sans-serif type with one bolder heading style.", "Pair simple body typography with a modest display headline."],
    }
    return _variant_choice(choices.get(profile["family"], choices["neutral"]), profile)


def _pairing_texture(profile: dict[str, Any]) -> str:
    choices = {
        "warm": ["Use paper grain, soft shadows, or subtle fabric texture.", "Add light paper fiber or warm shadow texture for a handmade feel.", "Use matte surfaces instead of glossy effects to keep warmth natural."],
        "cool": ["Use glass blur, fine gradients, or smooth matte surfaces.", "Use smooth gradients and low-noise backgrounds for a calm finish.", "Try translucent panels or soft blue-gray surfaces."],
        "natural": ["Use recycled paper, leaf grain, or soft organic texture.", "Use soft fiber, stone, or botanical texture in small amounts.", "Keep texture irregular and subtle so it feels organic."],
        "vibrant": ["Use crisp shapes, light grain, or glossy highlight accents.", "Use clean vector shapes with small highlight details.", "Try a tiny amount of grain to stop bright colors feeling flat."],
        "dark": ["Use film grain, low-key shadows, or matte noise texture.", "Use controlled shadow texture and avoid busy patterns.", "Try subtle noise or cinematic grain for depth."],
        "moody": ["Use soft grain, rain-like texture, or muted paper noise.", "Use atmospheric grain or blurred texture behind large type.", "Keep texture soft so the palette stays mature."],
        "minimal": ["Use very subtle paper texture or flat matte surfaces.", "Use almost-flat surfaces with only slight material variation.", "Keep texture barely visible and let spacing do the work."],
        "neutral": ["Use subtle grain, paper, or soft shadow texture so the palette feels intentional.", "Use matte texture and light shadows for quiet depth.", "Keep surfaces simple, with texture used only as support."],
    }
    return _variant_choice(choices.get(profile["family"], choices["neutral"]), profile)


def _pairing_secondary_palette(
    profile: dict[str, Any],
    colors: list[dict[str, Any]],
) -> str:
    anchor = _first_color_name(colors)
    extra = {
        "warm": "cream, deep brown, and one muted red accent",
        "cool": "off-white, slate, and one pale blue accent",
        "natural": "warm white, charcoal, and one muted sage accent",
        "vibrant": "white, near-black, and one low-saturation support color",
        "dark": "bone white, charcoal, and one metallic muted accent",
        "moody": "fog gray, deep charcoal, and one dusty accent",
        "minimal": "warm white, graphite, and one restrained accent",
        "neutral": "off-white, charcoal, and one muted accent",
    }.get(profile["family"], "off-white, charcoal, and one muted accent")
    return f"Pair {anchor} with {extra}."


def _pairing_layout_style(profile: dict[str, Any]) -> str:
    choices = {
        "warm": ["Use large welcoming blocks, rounded spacing, and small high-contrast accent areas.", "Build a clear hero block, then repeat warm accents in smaller callouts.", "Use generous spacing and keep warm accents near the main message."],
        "cool": ["Use open spacing, aligned columns, and calm image-led sections.", "Use a clean grid with wide margins and one quiet focal image.", "Keep layout rhythm steady with aligned text and low-density sections."],
        "natural": ["Use asymmetry, generous margins, and organic image crops.", "Use relaxed spacing with image crops that feel less rigid.", "Let one large natural color area lead, then add smaller supporting blocks."],
        "vibrant": ["Use bold hierarchy, large type, and controlled accent bursts.", "Use one oversized headline and cluster bright accents around it.", "Keep the grid simple so saturated colors can carry motion."],
        "dark": ["Use dramatic contrast, centered hero areas, and sparse supporting text.", "Use one strong focal area with plenty of dark negative space.", "Keep secondary content minimal so the drama stays focused."],
        "moody": ["Use layered imagery, quieter type scale, and strong negative space.", "Use overlapping image and text blocks with careful contrast.", "Build a slower editorial layout with fewer but stronger elements."],
        "minimal": ["Use strict grid alignment, plenty of whitespace, and one clear focal point.", "Use precise margins, few elements, and one deliberate accent placement.", "Let whitespace carry structure, with color used only for priority."],
        "neutral": ["Use strong spacing, one hero color block, and small accent areas for contrast.", "Use a simple grid with one dominant block and restrained accents.", "Keep layout practical: clear title, clear image, and one accent path."],
    }
    return _variant_choice(choices.get(profile["family"], choices["neutral"]), profile)


def _variant_choice(options: list[str], profile: dict[str, Any]) -> str:
    if not options:
        return ""
    payload = json.dumps(
        {
            "family": profile.get("family"),
            "temperature": profile.get("temperature"),
            "saturation": round(float(profile.get("saturation", 0.0)), 2),
            "brightness": round(float(profile.get("brightness", 0.0)), 2),
            "contrast": round(float(profile.get("contrast", 0.0)), 2),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def _fallback_color_name(rgb: list[int], role: str) -> str:
    nearest = min(
        BASIC_COLOR_NAMES,
        key=lambda item: _rgb_distance(rgb, item[1]),
    )[0]
    return f"{nearest} {role}"


def _rgb_distance(rgb: list[int], other: tuple[int, int, int]) -> float:
    return math.sqrt(sum((rgb[index] - other[index]) ** 2 for index in range(3)))


def _color_name_phrase(colors: list[dict[str, Any]]) -> str:
    names = [
        _coerce_string(color.get("name"), "the main colors")
        for color in colors[:3]
        if isinstance(color, dict)
    ]
    if not names:
        return "the main colors"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _first_color_name(colors: list[dict[str, Any]]) -> str:
    if not colors:
        return "the dominant color"
    return _coerce_string(colors[0].get("name"), "the dominant color")


def _best_torch_device(torch_module: Any) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if (
        hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def _loaded_metadata() -> dict[str, Any]:
    return {
        "loaded": True,
        "model": MODEL_NAME,
        "device": _device,
        "model_load_seconds": _model_load_seconds,
    }
