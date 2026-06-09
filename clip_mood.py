# CLIP zero-shot mood classification

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


MODEL_NAME = "openai/clip-vit-base-patch32"
PROMPT_TEMPLATE = "a design color palette that feels {label}"
TOP_K = 3

MOOD_LABELS = [
    "warm and cozy",
    "dark and eerie",
    "cool and calm",
    "energetic and vibrant",
    "melancholic and moody",
    "fresh and natural",
    "luxurious and elegant",
    "minimal and clean",
    "playful and fun",
    "mysterious and dramatic",
]

_load_lock = threading.Lock()
_processor: Any | None = None
_model: Any | None = None
_device: str | None = None


class ClipMoodError(RuntimeError):
    # Raised when CLIP mood classification cannot run.
    pass


def load_clip_model() -> dict[str, Any]:
    # Lazy-load CLIP and return compact readiness metadata.
    global _processor, _model, _device

    if _processor is not None and _model is not None and _device is not None:
        return _loaded_metadata()

    with _load_lock:
        if _processor is not None and _model is not None and _device is not None:
            return _loaded_metadata()

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ClipMoodError(
                "CLIP dependencies are missing. Install torch and transformers."
            ) from exc

        _device = _best_torch_device(torch)
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        _model = CLIPModel.from_pretrained(MODEL_NAME)
        _model.to(_device)
        _model.eval()

    return _loaded_metadata()


def is_clip_loaded() -> bool:
    # Return whether CLIP model and processor are already in memory.
    return _processor is not None and _model is not None and _device is not None


def classify_mood(
    image: str | Path | Image.Image,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    # Classify image mood, returning only classifier output fields.
    return classify_mood_with_metadata(image, top_k)["classification"]


def classify_mood_with_metadata(
    image: str | Path | Image.Image,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    # Classify image mood and include fallback/warning metadata.
    try:
        classification = _classify_mood(image, top_k)
        return {
            "classification": classification,
            "warnings": [],
            "used_fallback": False,
            "model": MODEL_NAME,
            "device": _device,
        }
    except Exception as exc:
        return {
            "classification": fallback_mood_classification(),
            "warnings": [f"CLIP mood classification failed: {exc}"],
            "used_fallback": True,
            "model": MODEL_NAME,
            "device": _device,
        }


def fallback_mood_classification() -> dict[str, Any]:
    # Deterministic response used when CLIP cannot classify.
    return {
        "primary_mood": "unknown",
        "confidence": 0.0,
        "top3_moods": [],
    }


def _classify_mood(
    image: str | Path | Image.Image,
    top_k: int,
) -> dict[str, Any]:
    metadata = load_clip_model()
    device = metadata["device"]

    try:
        import torch
    except ImportError as exc:
        raise ClipMoodError("PyTorch is not available.") from exc

    pil_image = _load_rgb_image(image)
    prompts = [PROMPT_TEMPLATE.format(label=label) for label in MOOD_LABELS]
    inputs = _processor(
        text=prompts,
        images=pil_image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = _model(**inputs)
        scores = outputs.logits_per_image.softmax(dim=1)[0]

    ranked = torch.argsort(scores, descending=True)
    safe_top_k = max(1, min(int(top_k), len(MOOD_LABELS)))
    top_scores = []

    for index in ranked[:safe_top_k]:
        label_index = int(index.item())
        score = round(float(scores[label_index].item()) * 100, 1)
        top_scores.append({"mood": MOOD_LABELS[label_index], "score": score})

    return {
        "primary_mood": top_scores[0]["mood"],
        "confidence": top_scores[0]["score"],
        "top3_moods": top_scores[:TOP_K],
    }


def _load_rgb_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return _convert_to_rgb(image)

    path = Path(image)
    if not path.exists():
        raise ClipMoodError("Image file was not found.")

    try:
        with Image.open(path) as opened:
            return _convert_to_rgb(opened)
    except (UnidentifiedImageError, OSError) as exc:
        raise ClipMoodError("Image file could not be opened as a valid image.") from exc


def _convert_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


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
    }
