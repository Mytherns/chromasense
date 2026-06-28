# Chromasense

Chromasense analyzes an uploaded image and returns a color palette, art-style prediction, mood metadata, and style scores.

## What Counts As ML

- Main supervised ML component: art-style classifier trained on real labeled WikiArt-derived feature rows.
- Color extraction uses KMeans and GMM clustering; these are algorithmic ML methods, not the supervised project centerpiece.
- Mood generation is a deterministic support feature, not trained ML.
- This project does not use pretrained vision/language stacks such as `torch`, `transformers`, Qwen, CLIP, or pretrained vision models.

## Current Model

- Style labels: `art_nouveau`, `baroque`, `impressionism`, `pop_art`
- Training data: 600 feature rows, 150 per class
- Selected model: `ExtraTreesClassifier`
- Validation split: stratified 80/20 by path
- Validation accuracy: `0.566667`
- Random baseline: `0.25`
- Style feature source: KMeans palette features

Limitation: artist grouping was not used because cheap artist metadata was unavailable in the extracted rows. Report this as possible leakage risk.

## Setup

PowerShell:

```powershell
py -m pip install -r requirements.txt
```

## Run

```powershell
py -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

Useful endpoints:

- `GET /` serves the one-page UI.
- `GET /health` returns liveness.
- `GET /warmup` loads/checks models and data files.
- `POST /api/analyze` analyzes an uploaded image.
- `POST /analyze` is a compatibility alias.

## API Response Shape

Successful analysis returns:

- `palette`: fixed primary palette plus both KMeans and GMM outputs
- `features`: palette feature vector
- `style_features_source`: currently `kmeans`
- `art_style`: prediction, confidence, all style scores, explanation, nearest examples
- `mood`: deterministic mood name, tagline, tags

Missing optional model/data pieces degrade to partial success where possible. Invalid images, oversized uploads, and palette extraction failure return structured errors.

## Training

From existing extracted features:

```powershell
py scripts\train_style_classifier.py --min-per-class 150
```

Extract features from local WikiArt-style folders:

```powershell
py scripts\extract_features_style.py --input-dir data\raw\wikiart --output data\training_features_style.csv --seed 42
```

Low-storage Hugging Face streaming extraction:

```powershell
py scripts\extract_features_style_hf.py --output data\training_features_style.csv --seed 42
```

The Hugging Face dataset is used only as data loading. It is not a pretrained vision model.

## Color Evaluation

Run offline evaluation:

```powershell
py evaluate.py
```

Current fixed primary palette algorithm is `kmeans`, saved in `data/palette_algorithm_config.json`.

Limitation: committed color-eval fixtures are tiny deterministic swatches. Replace or extend them with real manually labeled photos before making a final real-photo color accuracy claim.

## Tests

```powershell
py -m compileall chromasense tests main.py
py -m pytest tests -q
```

Current verified result after installing API dependencies: `43 passed`.

## Data And Git Hygiene

Commit:

- source code
- `data/training_features_style.csv`
- `models/*.joblib`
- curated JSON files under `data/`
- `test_images/`
- `static/`
- `README.md`

Do not commit:

- `data/raw/wikiart/`
- uploads
- credentials such as `kaggle.json`
- `.venv/`
- `__pycache__/`
- `*.pyc`
