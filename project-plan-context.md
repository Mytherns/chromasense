# Chromasense - Grill-Me Decision Reference

Purpose: implementation reference for decisions made after reviewing `project-plan.md`.

## 1. Project Overview

- Chromasense is an AI-powered color mood extractor for beginner designers.
- User uploads an image; system extracts dominant colors, classifies mood, and generates practical design guidance.
- Goal is teaching, not only displaying:
  - why colors work
  - what emotion they create
  - how to use palette professionally
- Presentation pitch:
  - KMeans provides unsupervised color extraction.
  - CLIP provides zero-shot vision mood classification.
  - Qwen2.5-VL-3B provides vision-language semantic guidance.

## 2. Runtime Target

- Primary demo target: Google Colab with T4 GPU.
- Local machine may run partial/fallback mode.
- Repo-first implementation.
- Colab notebook is backup runner only.
- Notebook must not contain core project logic.

## 3. Architecture

- Keep three AI/ML techniques:
  - KMeans for dominant color extraction.
  - CLIP zero-shot classification for mood detection.
  - Qwen2.5-VL-3B-Instruct for semantic design guidance.
- Use one Qwen call, not three.
- Reason: same VLM technique, faster demo, fewer failure points.

### AI Techniques Summary

| Technique | Tool | Type | Job |
|---|---|---|---|
| Unsupervised ML | Weighted KMeans (scikit-learn) | Algorithm | Extract palette colors and area percentages with perceptual importance weighting |
| Zero-shot classification | CLIP (`openai/clip-vit-base-patch32`) | Neural network | Classify image mood from fixed vocabulary |
| Vision-language generation | Qwen2.5-VL-3B-Instruct | VLM | Generate color names, mood name, vibe, tags, use case, feedback, and pairing guidance in one structured call |

### Model Links

| Model | Source | URL |
|---|---|---|
| KMeans | scikit-learn | https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html |
| CLIP | Hugging Face | https://huggingface.co/openai/clip-vit-base-patch32 |
| Qwen2.5-VL-3B | Hugging Face | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct |

### Full Pipeline

```text
Image uploaded by user
        |
[FastAPI] validate JPEG/PNG/WebP, max 10MB, resize max 1024px, save temp
        |
[Weighted KMeans] sample max 50k pixels, weight focal/saturated pixels, select 3-8 palette colors
        |
[CLIP] zero-shot mood classification with 10 mood labels, return top 3
        |
[Qwen2.5-VL-3B] one structured vision-language call
        |  generate color names, mood name, vibe, tags, use case,
        |  beginner feedback, pairing suggestions
        |
[FastAPI] enforce schema, fallback if Qwen fails, add metadata/warnings
        |
[Static frontend /app] render image, palette, mood, feedback, pairing
```

## 4. File Responsibilities

### `color_extract.py`

- Validate image type and size.
- Save upload to temp path with unique filename.
- Resize image to max 1024px.
- Extract dominant colors with perceptual-importance weighted KMeans.
- Support `n_colors` range 3-8.
- Sample max 50,000 pixels with `random_state=42`.
- Weight sampled pixels by saturation/value and broad lower-center composition.
- Use extra internal KMeans candidates, then select final palette by area, accent strength, and color diversity.
- Keep final percentages unweighted so percentages still represent image area.
- Return hex, RGB, percentage, role, and fallback name.
- Clean up temp files after analysis.

### `clip_mood.py`

- Define canonical 10 mood labels.
- Lazy-load CLIP model and processor.
- Use prompt format: `a design color palette that feels {label}`.
- Return primary mood, confidence, and top 3 mood scores.

### `qwen_analyze.py`

- Lazy-load Qwen2.5-VL-3B-Instruct.
- Use `torch_dtype="auto"` and `device_map="auto"`.
- Run one structured vision-language call.
- Enforce JSON schema with parse repair.
- Return deterministic fallback with warnings if Qwen fails.

### `main.py`

- Define FastAPI app.
- Serve `GET /`, `POST /warmup`, `POST /analyze`, and `/app`.
- Apply CORS allow-all for demo.
- Enforce upload limits and error schema.
- Run model inference under one-at-a-time lock.
- Return combined response with warnings and metadata.

### `evaluate.py`

- Load `test_images/labels.json`.
- Evaluate KMeans color accuracy with Delta E CIE2000.
- Match extracted/reference colors by best assignment.
- Evaluate CLIP mood accuracy against manual labels.
- Print summary and save `evaluation_report.json`.
- Do not evaluate Qwen output.

### `static/`

- Provide minimal built-in frontend.
- Upload image, choose `n_colors`, render palette, mood, feedback, pairing, warnings, and metadata.

### `notebooks/chromasense_colab_runner.ipynb`

- Clone/upload repo.
- Install dependencies.
- Start FastAPI with uvicorn.
- Expose public ngrok URL.
- Call `/warmup`.
- Keep all core logic inside repo files, not notebook cells.

## 5. API Runtime

- Framework: FastAPI.
- Endpoints:
  - `GET /` health check.
  - `POST /warmup` lazy-load CLIP/Qwen models.
  - `POST /analyze?n_colors=5` full image analysis.
  - `/app` serves built-in static frontend.
- Use one-at-a-time inference lock.
- Queued behavior preferred over immediate rejection.
- Include lightweight OpenAPI metadata and examples.

### Request Contracts

`POST /analyze?n_colors=5`

- Content type: `multipart/form-data`.
- File field name: `file`.
- Query params:
  - `n_colors`: integer, optional, default 5, allowed range 3-8.
- Invalid file/type/size/range returns consistent JSON error.

`POST /warmup`

- Loads CLIP and Qwen lazily.
- Returns readiness metadata.

```json
{
  "status": "ready",
  "loaded": {
    "clip": true,
    "qwen": true
  },
  "device": "cuda",
  "warnings": []
}
```

### Failure Modes

- KMeans/color extraction failure is hard error.
- Invalid image input is hard error.
- CLIP failure returns response with:
  - `primary_mood`: `unknown`
  - `confidence`: `0.0`
  - empty `top3_moods`
  - warning explaining CLIP failure
- Qwen failure returns deterministic fallback content with warning.
- Any fallback must be visible in `warnings` and `metadata.runtime.used_fallback`.

## 6. Public Demo Access

- Use ngrok public tunnel from Colab.
- No API key/auth for class demo.
- CORS: allow all origins in demo mode.
- Keep upload size/type limits and no persistence to reduce risk.

## 7. File Upload Rules

- Supported types: JPEG, PNG, WebP.
- Max upload size: 10MB.
- Resize internally to max 1024px on longest side.
- Delete temp upload after analysis.
- Do not persist results server-side.

## 8. Color Extraction

- `n_colors` adjustable through query param.
- Allowed range: 3-8.
- Default: 5.
- Evaluation uses fixed 5 colors.
- KMeans includes sampled pixels from the full image, including near-white and near-black.
- Resize first, then sample max 50,000 pixels with fixed NumPy seed.
- Use `random_state=42` for KMeans.
- Use perceptual-importance weighted KMeans:
  - Base color points remain RGB pixels.
  - `sample_weight` favors saturated/high-value pixels.
  - `sample_weight` also favors a broad lower-center region where focal subjects often appear.
  - Cap pixel weights so tiny bright details cannot dominate the palette.
- Use more internal candidate clusters than requested:
  - final request still returns `n_colors` only
  - internal candidates can go up to 12
  - reason: focal colors may disappear if KMeans is forced directly into 5 clusters
- Select final colors by combined score:
  - area coverage
  - accent strength from saturation/value
  - color diversity using RGB plus hue/saturation/value feature distance
- Final percentages are recomputed unweighted from nearest selected centers.
- Avoid hardcoded object/color special cases; keep method general and explainable as weighted KMeans.
- Return one-decimal percentages.
- Return fields:
  - `hex`
  - `rgb`
  - `percentage`
  - `name`
  - `role`
- Suggested roles for 5 colors:
  - Dominant
  - Secondary
  - Accent
  - Support
  - Neutral/Depth

## 9. Mood Classification

- Keep 10 existing mood labels:
  - warm and cozy
  - dark and eerie
  - cool and calm
  - energetic and vibrant
  - melancholic and moody
  - fresh and natural
  - luxurious and elegant
  - minimal and clean
  - playful and fun
  - mysterious and dramatic
- Improve CLIP prompt format:
  - `a design color palette that feels {label}`
- Return primary mood, confidence, and top 3 scores.
- Qwen may generate nicer display mood name, but CLIP label remains raw classifier result.

## 10. Qwen Generation

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`.
- Load lazily, preferably through `/warmup`.
- Use `torch_dtype="auto"` and `device_map="auto"`.
- Document optional 4-bit fallback for Colab memory issues.
- Use one structured prompt to generate:
  - poetic color names
  - mood name
  - vibe
  - tags
  - use case
  - beginner design feedback
  - font/texture/palette/layout pairing suggestions
- Tone: beginner-focused, educational, practical.

### Generation Config

- Use low temperature for JSON stability.
- Recommended:
  - `temperature=0.2`
  - `max_new_tokens=900`
  - `do_sample=True` only if required by model config; otherwise prefer deterministic generation.
- Qwen output is not guaranteed fully deterministic.
- Prompt must request JSON only, no Markdown.

## 11. Qwen Output Robustness

- Enforce strict schema.
- Parse with `json.loads`.
- If parse fails, extract first JSON object block and parse again.
- If still fails, return deterministic fallback.
- Fallback response must include warnings.
- Fallback color names use nearest basic/CSS color plus role.

## 12. Response Schema

- Use snake_case everywhere.
- No camelCase aliases.
- Remove `useCase`; use `use_case`.
- Replace `similar_brand` with `brand_style_reference`.
- Avoid exact brand claims unless clearly framed as style reference.
- Include `warnings` array.
- Include compact `metadata`.
- Include runtime fallback flags.

Core response shape:

```json
{
  "mood": "Haunted Harvest",
  "vibe": "A warm yet eerie atmosphere...",
  "clip_classification": {
    "primary_mood": "dark and eerie",
    "confidence": 91.4,
    "top3_moods": [
      { "mood": "dark and eerie", "score": 91.4 }
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
  "tags": ["warm", "eerie", "seasonal"],
  "use_case": "Halloween branding and event posters",
  "feedback": {
    "why_it_works": "...",
    "emotion": "...",
    "mistakes_to_avoid": "...",
    "brand_style_reference": "seasonal event branding"
  },
  "pairing": {
    "font_style": "...",
    "texture": "...",
    "secondary_palette": "...",
    "layout_style": "..."
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

## 13. Error Schema

- Use consistent JSON errors.

```json
{
  "error": {
    "code": "invalid_file_type",
    "message": "Only JPEG, PNG, and WebP images are supported."
  }
}
```

## 14. Built-In Frontend

- Include minimal static frontend in repo.
- Served by FastAPI at `/app`.
- Files:
  - `static/index.html`
  - `static/app.js`
  - `static/styles.css`
- Required UI:
  - upload image
  - choose `n_colors` from 3-8
  - show image preview
  - show palette swatches, hex, percentage, names
  - show mood, vibe, tags, use case
  - show CLIP top 3
  - show feedback and pairing
  - show warnings and metadata in collapsed/secondary area
- Scope: demo-complete, not overdesigned.

### Frontend Non-Goals

- No account system.
- No saved gallery.
- No database.
- No multi-user dashboard.
- No advanced image editing.

## 15. Evaluation Dataset

- User manually adds test images.
- Use 20 images total.
- Balanced categories:
  - 4 warm
  - 4 cool
  - 4 dark/moody
  - 4 bright/vibrant
  - 4 neutral/minimal
- Store image labels in `test_images/labels.json`.
- Evaluation source is saved labels only, not live Adobe/Coolors lookup.
- If image listed in labels is missing, strict mode fails.
- Use own photos or royalty-free images when possible.
- Optionally record image source/license in labels file.

### Label Validation

- `mood` must be one of the 10 canonical mood labels.
- `colors` must be valid hex codes.
- Each image should have 5 reference colors for standard evaluation.
- Invalid labels fail fast before running models.

### Test Image Search Terms

| Category | Count | Search Terms | Expected Mood Labels |
|---|---:|---|---|
| Warm tones | 4 | halloween, autumn, sunset | warm and cozy / dark and eerie |
| Cool tones | 4 | ocean, winter, blue sky | cool and calm |
| Dark / moody | 4 | night city, gothic, rain | dark and eerie / mysterious and dramatic / melancholic and moody |
| Bright / vibrant | 4 | neon, flowers, festival | energetic and vibrant / playful and fun |
| Neutral / minimal | 4 | coffee, concrete, beige, architecture | minimal and clean / luxurious and elegant |

Example:

```json
{
  "autumn_01.jpg": {
    "mood": "warm and cozy",
    "colors": ["#a65a2b", "#d88a3d", "#f1c27d", "#4a2d1f", "#efe2c6"]
  }
}
```

## 16. Evaluation Metrics

- Evaluate KMeans and CLIP only.
- Do not evaluate Qwen output.
- Color accuracy:
  - Convert hex to LAB.
  - Compute Delta E CIE2000.
  - Count match as accurate when Delta E < 6.
  - Match extracted colors to reference colors by best assignment, not same index.
  - Use SciPy Hungarian algorithm if available.
  - Use greedy nearest-match fallback otherwise.
- Mood accuracy:
  - Compare CLIP `primary_mood` to manual label.
  - Score = correct / total * 100.
- Determinism:
  - KMeans uses `random_state=42`.
  - Pixel sampling uses fixed NumPy random seed.
  - Pixel importance weighting is deterministic.
  - Qwen generation is not used in evaluation.

### Delta E Interpretation

| Delta E | Perception | Meaning for Accuracy |
|---:|---|---|
| < 1 | Imperceptible | Perfect match |
| 1-3 | Slight difference | Excellent match |
| 3-6 | Noticeable | Acceptable match; counts as accurate |
| > 6 | Different color | Miss; does not count |

- Targets:
  - 85%+ colors matched within Delta E < 6.
  - 80%+ mood classification accuracy.
- Output:
  - print console summary
  - save `evaluation_report.json`
  - include per-image details, skipped/errors, average Delta E, mood accuracy

## 17. Dependencies

- Use version floor pins, not exact lockfile.
- Include core:
  - fastapi
  - uvicorn
  - python-multipart
  - pillow
  - transformers
  - accelerate
  - qwen-vl-utils
  - torch
  - scikit-learn
  - colormath
  - scipy
  - ftfy
  - regex
  - tqdm
  - pyngrok for Colab runner
- Colab notebook may include fallback install/upgrade commands.

## 18. Colab Notebook

- Notebook path: `notebooks/chromasense_colab_runner.ipynb`.
- Primary flow:
  - clone repo from `<YOUR_REPO_URL>`
  - install requirements
  - optionally install ngrok token
  - run FastAPI with uvicorn
  - expose via ngrok
  - call `/warmup`
- Include upload fallback note if repo is not on GitHub yet.
- Do not duplicate application logic in notebook cells.

## 19. README

- Include full README, not install-only.
- Must cover:
  - project overview
  - architecture/pipeline
  - local setup
  - Colab GPU demo
  - API contract
  - frontend usage
  - evaluation dataset format
  - evaluation command
  - known limitations

## 20. Planned Repo Files

```text
main.py
color_extract.py
clip_mood.py
qwen_analyze.py
evaluate.py
requirements.txt
README.md
project-plan.md
project-plan-grill-me.md
static/
  index.html
  app.js
  styles.css
notebooks/
  chromasense_colab_runner.ipynb
test_images/
  labels.example.json
```

## 21. Implementation Order

1. `color_extract.py`
2. `clip_mood.py`
3. `qwen_analyze.py`
4. `main.py`
5. `static/` frontend
6. `evaluate.py`
7. Colab runner notebook
8. `README.md`

## 22. Manual Demo Checklist

1. Start Colab T4 runtime.
2. Clone/upload repo.
3. Install dependencies.
4. Start FastAPI and ngrok tunnel.
5. Call `POST /warmup`.
6. Open `/app`.
7. Upload sample image.
8. Change `n_colors` and rerun analysis.
9. Show palette, CLIP top 3, Qwen guidance, warnings, and metadata.
10. Run evaluation after `test_images/labels.json` and 20 images are prepared.
