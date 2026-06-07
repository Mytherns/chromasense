# FastAPI entrypoint for Chromasense.

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from clip_mood import classify_mood_with_metadata, is_clip_loaded, load_clip_model
from color_extract import (
    MAX_UPLOAD_BYTES,
    ColorExtractionError,
    cleanup_temp_file,
    extract_colors,
    save_upload_to_temp,
    validate_n_colors,
)
from qwen_analyze import analyze_design, is_qwen_loaded, warmup_qwen


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

inference_lock = asyncio.Lock()

app = FastAPI(
    title="Chromasense API",
    description="Upload an image to extract dominant colors, classify mood, and generate beginner design guidance.",
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(ColorExtractionError)
async def color_error_handler(
    _request: Any,
    exc: ColorExtractionError,
) -> JSONResponse:
    return _error_response(exc.code, exc.message, status_code=400)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    _request: Any,
    exc: RequestValidationError,
) -> JSONResponse:
    code = "invalid_request"
    message = "Request is invalid."

    for error in exc.errors():
        location = error.get("loc", ())
        if "n_colors" in location:
            code = "invalid_n_colors"
            message = "n_colors must be an integer from 3 to 8."
            break
        if "file" in location:
            code = "missing_file"
            message = "Upload field 'file' is required."
            break

    return _error_response(code, message, status_code=422)


@app.exception_handler(HTTPException)
async def http_error_handler(_request: Any, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return _error_response("http_error", str(exc.detail), status_code=exc.status_code)


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "status": "ok",
        "app": "Chromasense",
        "endpoints": {
            "warmup": "POST /warmup",
            "analyze": "POST /analyze?n_colors=5",
            "frontend": "GET /app",
        },
        "loaded": {
            "clip": is_clip_loaded(),
            "qwen": is_qwen_loaded(),
        },
    }


@app.post("/warmup")
async def warmup() -> dict[str, Any]:
    async with inference_lock:
        clip_meta, clip_warning = await run_in_threadpool(_warmup_clip)
        qwen_meta, qwen_warning = await run_in_threadpool(_warmup_qwen)

    warnings = [warning for warning in (clip_warning, qwen_warning) if warning]
    loaded = {
        "clip": bool(clip_meta),
        "qwen": bool(qwen_meta),
    }
    return {
        "status": "ready" if all(loaded.values()) else "degraded",
        "loaded": loaded,
        "device": _select_device(qwen_meta, clip_meta),
        "warnings": warnings,
        "metadata": {
            "clip": clip_meta or {},
            "qwen": qwen_meta or {},
        },
    }


@app.post(
    "/analyze",
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "example": {
                        "file": "image.png",
                    }
                }
            }
        }
    },
)
async def analyze(
    file: UploadFile = File(...),
    n_colors: int = Query(5, description="Palette size, from 3 to 8."),
) -> dict[str, Any]:
    n_colors = validate_n_colors(n_colors)
    file_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    temp_path = save_upload_to_temp(
        file_bytes,
        file.filename,
        _upload_content_type(file.content_type),
    )

    try:
        async with inference_lock:
            return await run_in_threadpool(_run_analysis, temp_path, n_colors)
    finally:
        cleanup_temp_file(temp_path)
        await file.close()


@app.get("/app", include_in_schema=False)
async def frontend():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)

    return HTMLResponse(
        """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Chromasense</title>
            <style>
              body { font-family: system-ui, sans-serif; margin: 2rem; max-width: 720px; }
              code { background: #f2f2f2; padding: 0.1rem 0.25rem; }
            </style>
          </head>
          <body>
            <h1>Chromasense</h1>
            <p>Static frontend not added yet. Use <code>POST /analyze?n_colors=5</code> with multipart field <code>file</code>.</p>
          </body>
        </html>
        """.strip()
    )


def _run_analysis(temp_path: Path, n_colors: int) -> dict[str, Any]:
    started = time.perf_counter()
    color_result = extract_colors(temp_path, n_colors=n_colors)
    clip_result = classify_mood_with_metadata(temp_path)
    result = analyze_design(
        image=temp_path,
        colors=color_result["colors"],
        clip_classification=clip_result["classification"],
        n_colors=n_colors,
        image_size=color_result["image_size"],
        clip_used_fallback=clip_result["used_fallback"],
    )

    warnings = [
        *clip_result.get("warnings", []),
        *result.get("warnings", []),
    ]
    result["warnings"] = warnings
    result.setdefault("metadata", {})
    result["metadata"]["n_colors"] = n_colors
    result["metadata"]["image_size"] = color_result["image_size"]
    result["metadata"].setdefault("runtime", {})
    result["metadata"]["runtime"]["seconds"] = round(time.perf_counter() - started, 2)
    result["metadata"]["runtime"].setdefault("used_fallback", {})
    result["metadata"]["runtime"]["used_fallback"]["clip"] = bool(
        clip_result["used_fallback"]
    )
    return result


def _warmup_clip() -> tuple[dict[str, Any] | None, str | None]:
    try:
        return load_clip_model(), None
    except Exception as exc:
        return None, f"CLIP warmup failed: {exc}"


def _warmup_qwen() -> tuple[dict[str, Any] | None, str | None]:
    try:
        return warmup_qwen(), None
    except Exception as exc:
        return None, f"Qwen warmup failed: {exc}"


def _select_device(*metadata_items: dict[str, Any] | None) -> str:
    for metadata in metadata_items:
        if metadata and metadata.get("device"):
            return str(metadata["device"])
    return "unknown"


def _upload_content_type(content_type: str | None) -> str | None:
    if not content_type or content_type.lower() == "application/octet-stream":
        return None
    return content_type


def _error_response(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )
