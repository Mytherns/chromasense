# Chromasense

Chromasense is an AI image palette analyzer for beginner designers. Upload an image, choose 3-8 colors, and the app returns dominant palette colors, CLIP mood classification, and practical design guidance from Qwen.

Core logic lives in repo Python files. The Colab notebook is only a runner for GPU demo setup.

## What It Does

- Validates JPEG, PNG, and WebP uploads up to 10MB.
- Resizes images to a maximum side length of 1024px.
- Extracts 3-8 dominant colors with weighted KMeans.
- Classifies image mood with CLIP zero-shot prompts.
- Generates beginner-friendly design guidance with Qwen2.5-VL-3B-Instruct.
- Serves a small built-in frontend at `/app`.
- Evaluates KMeans color accuracy and CLIP mood accuracy with labeled test images.

## Architecture

```text
Image upload
  -> FastAPI validation and temp file save
  -> Weighted KMeans palette extraction
  -> CLIP zero-shot mood classification
  -> Qwen2.5-VL structured design guidance
  -> JSON response + static frontend render
```

Main files:

| File | Responsibility |
|---|---|
| `main.py` | FastAPI app, routes, upload limits, inference lock, error schema |
| `color_extract.py` | Image validation, resize, weighted KMeans palette extraction, temp cleanup helpers |
| `clip_mood.py` | Lazy CLIP loading, 10-label mood classification, fallback result |
| `qwen_analyze.py` | Lazy Qwen loading, one structured call, schema repair, deterministic fallback |
| `static/` | Built-in upload and result UI |
| `evaluate.py` | Label validation, Delta E color scoring, CLIP mood scoring |
| `notebooks/chromasense_colab_runner.ipynb` | Colab GPU runner only |

## Models And Methods

| Step | Tool | Output |
|---|---|---|
| Color extraction | Weighted KMeans (`scikit-learn`) | Hex, RGB, percentage, role, fallback color name |
| Mood classification | `openai/clip-vit-base-patch32` | Primary mood, confidence, top 3 scores |
| Design guidance | `Qwen/Qwen2.5-VL-3B-Instruct` | Mood name, vibe, tags, use case, feedback, pairing |

Mood labels:

```text
warm and cozy
dark and eerie
cool and calm
energetic and vibrant
melancholic and moody
fresh and natural
luxurious and elegant
minimal and clean
playful and fun
mysterious and dramatic
```

## Local Setup

Recommended local Python: 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Start API:

```powershell
.\.venv\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000/app
```

Useful endpoints:

```text
GET  http://127.0.0.1:8000/
POST http://127.0.0.1:8000/warmup
POST http://127.0.0.1:8000/analyze?n_colors=5
GET  http://127.0.0.1:8000/docs
```

For local CPU-only testing where Qwen is too heavy, use fallback mode:

```powershell
$env:CHROMASENSE_SKIP_QWEN = "1"
.\.venv\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

In this mode, KMeans and CLIP still run. Qwen guidance uses deterministic fallback and response warnings show that fallback was used.

## Frontend Usage

1. Start FastAPI.
2. Open `/app`.
3. Upload a JPEG, PNG, or WebP image.
4. Choose `n_colors` from 3 to 8.
5. Click `Analyze`.
6. Review palette, mood, CLIP top 3, use case, feedback, pairing, warnings, and metadata.

## Colab GPU Demo

Use `notebooks/chromasense_colab_runner.ipynb`.

Recommended flow:

1. Open notebook in Google Colab.
2. Set runtime to T4 GPU.
3. Set `REPO_URL` if repo is on GitHub, or leave blank and upload a repo zip.
4. Run dependency install cells.
5. Add optional Colab secret `NGROK_AUTHTOKEN` if ngrok requires auth.
6. Start FastAPI with uvicorn.
7. Open ngrok public tunnel.
8. Run `/warmup`.
9. Open printed `/app` URL and demo upload flow.

Notebook cells only clone/upload, install, start server, expose ngrok, warm up models, and show logs. They do not duplicate app logic.

## API Contract

### `GET /`

Health and endpoint metadata.

Example response:

```json
{
  "status": "ok",
  "app": "Chromasense",
  "endpoints": {
    "warmup": "POST /warmup",
    "analyze": "POST /analyze?n_colors=5",
    "frontend": "GET /app"
  },
  "loaded": {
    "clip": false,
    "qwen": false
  }
}
```

### `POST /warmup`

Lazy-loads CLIP and Qwen before demo.

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/warmup
```

Example response:

```json
{
  "status": "ready",
  "loaded": {
    "clip": true,
    "qwen": true
  },
  "device": "cuda",
  "warnings": [],
  "metadata": {
    "clip": {},
    "qwen": {}
  }
}
```

If a model cannot load, `status` becomes `degraded` and `warnings` explains why.

### `POST /analyze?n_colors=5`

Request:

- Method: `POST`
- Content type: `multipart/form-data`
- File field: `file`
- Query param: `n_colors`, integer from 3 to 8, default `5`

Example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/analyze?n_colors=5" -F "file=@test_images/sample.jpg"
```

Response shape:

```json
{
  "mood": "Warm Focus",
  "vibe": "A warm and cozy palette led by Orange Dominant, Brown Secondary, and Beige Accent.",
  "clip_classification": {
    "primary_mood": "warm and cozy",
    "confidence": 91.4,
    "top3_moods": [
      {
        "mood": "warm and cozy",
        "score": 91.4
      }
    ]
  },
  "colors": [
    {
      "hex": "#ff6b00",
      "rgb": [255, 107, 0],
      "name": "Ember Glow",
      "role": "Dominant",
      "percentage": 40.0
    }
  ],
  "tags": ["warm", "cozy", "palette", "design"],
  "use_case": "Beginner-friendly posters, social posts, and simple branding with a warm and cozy tone.",
  "feedback": {
    "why_it_works": "Warm hues make the design feel close, active, and easy to notice.",
    "emotion": "It gives a warm and cozy impression before extra graphics are added.",
    "mistakes_to_avoid": "Avoid using every color at equal strength; keep one dominant color and use accents sparingly.",
    "brand_style_reference": "general warm and cozy visual identity and campaign design"
  },
  "pairing": {
    "font_style": "Use a friendly rounded sans-serif with a sturdy display heading.",
    "texture": "Use paper grain, soft shadows, or subtle fabric texture.",
    "secondary_palette": "Pair Orange Dominant with cream, deep brown, and one muted red accent.",
    "layout_style": "Use large welcoming blocks, rounded spacing, and small high-contrast accent areas."
  },
  "warnings": [],
  "metadata": {
    "n_colors": 5,
    "image_size": [1024, 768],
    "models": {
      "color": "Weighted KMeans",
      "mood": "openai/clip-vit-base-patch32",
      "generation": "Qwen/Qwen2.5-VL-3B-Instruct"
    },
    "runtime": {
      "device": "cuda",
      "seconds": 12.4,
      "model_load_seconds": 38.7,
      "used_fallback": {
        "clip": false,
        "qwen": false
      }
    }
  }
}
```

### Error Response

Errors use one shape:

```json
{
  "error": {
    "code": "invalid_file_type",
    "message": "Only JPEG, PNG, and WebP images are supported."
  }
}
```

Common error codes:

| Code | Cause |
|---|---|
| `empty_file` | Upload has no bytes |
| `file_too_large` | Upload is over 10MB |
| `invalid_file_type` | File is not JPEG, PNG, or WebP |
| `invalid_image` | File cannot be opened as an image |
| `invalid_n_colors` | `n_colors` is outside 3-8 |
| `missing_file` | Multipart field `file` is missing |

## Evaluation

Evaluation uses saved labels only. It evaluates KMeans and CLIP, not Qwen.

Create:

```text
test_images/
  labels.json
  autumn_01.jpg
  ocean_01.jpg
  ...
```

`labels.json` format:

```json
{
  "autumn_01.jpg": {
    "mood": "warm and cozy",
    "colors": ["#a65a2b", "#d88a3d", "#f1c27d", "#4a2d1f", "#efe2c6"]
  }
}
```

Rules:

- Each key is an image filename in `test_images/`.
- Image extension must be `.jpg`, `.jpeg`, `.png`, or `.webp`.
- `mood` must be one canonical mood label.
- `colors` must contain exactly five `#rrggbb` values.
- Standard evaluation uses 5 extracted colors.
- Missing images fail validation unless `--allow-missing` is used.

Recommended dataset balance: 20 images total.

| Category | Count |
|---|---:|
| Warm tones | 4 |
| Cool tones | 4 |
| Dark / moody | 4 |
| Bright / vibrant | 4 |
| Neutral / minimal | 4 |

Run evaluation:

```powershell
.\.venv\Scripts\python evaluate.py --labels test_images\labels.json --output evaluation_report.json
```

Optional:

```powershell
.\.venv\Scripts\python evaluate.py --labels test_images\labels.json --output evaluation_report.json --n-colors 5
```

Metrics:

- Color accuracy uses Delta E CIE2000.
- Match counts as accurate when Delta E is below `6.0`.
- Extracted and reference colors are matched by best assignment.
- Mood accuracy compares CLIP `primary_mood` against manual label.
- Report is saved to `evaluation_report.json`.

Target scores:

- 85%+ colors matched within Delta E < 6.
- 80%+ mood classification accuracy.

## Dependencies

Core packages are listed in `requirements.txt`:

- FastAPI, Uvicorn, python-multipart
- Pillow, NumPy, scikit-learn
- PyTorch, Transformers, Accelerate
- qwen-vl-utils
- SciPy, colormath
- ftfy, regex, tqdm
- pyngrok for Colab tunnel

Use version floor pins from `requirements.txt`, not exact lockfile pins.

## Known Limitations

- Qwen2.5-VL-3B is heavy. Colab T4 GPU is the intended full-demo runtime.
- First `/warmup` can take several minutes because models download and load.
- Local CPU runs may be slow or need `CHROMASENSE_SKIP_QWEN=1`.
- Qwen output is schema-repaired, but generation is not fully deterministic.
- CLIP mood labels are fixed to 10 broad categories.
- Color extraction is deterministic, but palette quality still depends on image lighting, compression, and subject framing.
- The demo has no auth, database, saved gallery, or multi-user dashboard.
- CORS allows all origins for class demo convenience.
