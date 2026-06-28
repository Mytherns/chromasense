from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from chromasense.color_extract import DEFAULT_K, extract_palette, resize_to_long_side
from chromasense.features import palette_to_features
from chromasense.generate import generate_mood
from chromasense.style_classifier import StyleClassifier, load_style_classifier

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "data" / "palette_algorithm_config.json"
STYLE_MODEL_PATH = BASE_DIR / "models" / "style_classifier.joblib"
MOOD_LEXICON_PATH = BASE_DIR / "data" / "mood_lexicon.json"
STATIC_INDEX_PATH = BASE_DIR / "static" / "index.html"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALGORITHMS = ("kmeans", "gmm")

app = FastAPI(title="Chromasense API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STYLE_CLASSIFIER: StyleClassifier | None = None
_STYLE_LOAD_ATTEMPTED = False
_STYLE_LOAD_STATUS: dict[str, Any] = {
    "status": "not_loaded",
    "model_path": str(STYLE_MODEL_PATH),
}


@app.get("/", include_in_schema=False, response_model=None)
def index():
    if STATIC_INDEX_PATH.is_file():
        return FileResponse(STATIC_INDEX_PATH)
    return {"ok": True, "service": "chromasense", "ui": "static/index.html not found"}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "chromasense"}


@app.get("/warmup")
def warmup() -> dict:
    classifier = _load_style_classifier_cached(force=True)
    style_status = dict(_STYLE_LOAD_STATUS)
    if classifier is not None:
        style_status["labels"] = list(classifier.labels)
    return {
        "ok": True,
        "style_classifier": style_status,
        "mood_lexicon": _file_status(MOOD_LEXICON_PATH),
    }


@app.post("/api/analyze")
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    temp_path: Path | None = None
    try:
        upload = await _read_and_validate_upload(file)
        temp_path = upload["path"]
        payload = _analyze_resized_image(temp_path, upload["image"])
        return JSONResponse(status_code=200, content=payload)
    except _ApiError as exc:
        return _error_response(exc.status_code, exc.code, exc.message)
    except Exception:
        return _error_response(
            status_code=500,
            code="palette_extraction_failed",
            message="Could not extract palette from image.",
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


async def _read_and_validate_upload(file: UploadFile) -> dict[str, Any]:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise _ApiError(413, "file_too_large", "Upload must be 10MB or smaller.")
    if not data:
        raise _ApiError(400, "invalid_image", "Upload must be a valid image.")

    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.verify()
        with Image.open(io.BytesIO(data)) as opened:
            rgb_image = opened.convert("RGB")
            rgb_image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        raise _ApiError(400, "invalid_image", "Upload must be a valid image.") from None

    resized = resize_to_long_side(rgb_image)
    temp_handle = tempfile.NamedTemporaryFile(prefix="chromasense_", suffix=".png", delete=False)
    temp_path = Path(temp_handle.name)
    temp_handle.close()
    try:
        resized.save(temp_path, format="PNG", optimize=True)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return {
        "path": temp_path,
        "image": {
            "filename": file.filename,
            "content_type": file.content_type,
            "width": resized.width,
            "height": resized.height,
            "resized": resized.size != rgb_image.size,
        },
    }


def _analyze_resized_image(image_path: Path, image_info: dict[str, Any]) -> dict:
    primary_algorithm = _load_primary_algorithm()
    palettes = {
        algorithm: extract_palette(
            image=image_path,
            k=DEFAULT_K,
            algorithm=algorithm,
        )
        for algorithm in ALGORITHMS
    }
    primary_colors = palettes[primary_algorithm]["colors"]
    style_palette = palettes["kmeans"]["colors"]
    features = palette_to_features(style_palette)

    art_style = _predict_art_style(features)
    style_name = art_style.get("prediction") if art_style.get("status") == "ok" else None

    return {
        "ok": True,
        "image": image_info,
        "palette": {
            "algorithm_used": primary_algorithm,
            "primary": palettes[primary_algorithm],
            "kmeans": palettes["kmeans"],
            "gmm": palettes["gmm"],
        },
        "features": features,
        "style_features_source": "kmeans",
        "art_style": art_style,
        "mood": _component_or_error(
            lambda: generate_mood(style_name, features, lexicon_path=MOOD_LEXICON_PATH),
            "mood_unavailable",
        ),
    }


def _load_primary_algorithm() -> str:
    try:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        algorithm = str(payload.get("primary_algorithm", "kmeans")).lower()
    except (OSError, json.JSONDecodeError):
        algorithm = "kmeans"
    if algorithm not in ALGORITHMS:
        return "kmeans"
    return algorithm


def _predict_art_style(features: dict[str, float]) -> dict:
    classifier = _load_style_classifier_cached()
    if classifier is None:
        return {
            "status": "model_missing",
            "prediction": None,
            "confidence": None,
            "all_scores": {},
            "explanation": "Style classifier unavailable; palette analysis still completed.",
            "nearest_training_examples": [],
            "style_features_source": "kmeans",
        }
    return classifier.predict(features)


def _component_or_error(factory, code: str) -> dict:
    try:
        return factory()
    except Exception:
        return {"status": "error", "code": code}


def _load_style_classifier_cached(force: bool = False) -> StyleClassifier | None:
    global _STYLE_CLASSIFIER, _STYLE_LOAD_ATTEMPTED, _STYLE_LOAD_STATUS
    if _STYLE_LOAD_ATTEMPTED and not force:
        return _STYLE_CLASSIFIER

    _STYLE_LOAD_ATTEMPTED = True
    try:
        _STYLE_CLASSIFIER = load_style_classifier(STYLE_MODEL_PATH)
        _STYLE_LOAD_STATUS = {
            "status": "ok",
            "model_path": str(STYLE_MODEL_PATH),
        }
    except FileNotFoundError:
        _STYLE_CLASSIFIER = None
        _STYLE_LOAD_STATUS = {
            "status": "model_missing",
            "model_path": str(STYLE_MODEL_PATH),
        }
    except Exception:
        _STYLE_CLASSIFIER = None
        _STYLE_LOAD_STATUS = {
            "status": "model_missing",
            "model_path": str(STYLE_MODEL_PATH),
        }
    return _STYLE_CLASSIFIER


def _file_status(path: Path) -> dict:
    return {
        "status": "ok" if path.is_file() else "missing",
        "path": str(path),
    }


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "error": {
                "code": code,
                "message": message,
            },
        },
    )


class _ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(code)
