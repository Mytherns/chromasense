# Agent Context

## Communication

- Always use `caveman` skill.
- Default level: `full`.
- Stay terse. No filler.
- Stop only if user says `stop caveman` or `normal mode`.

## Source Of Truth

- Use `project-plan-context.md` for implementation decisions.
- Keep implementation repo-first.
- Keep Colab notebook as runner only; no core logic inside notebook.

## Implementation Checklist

Update checkbox when each phase completed.

- [x] Phase 1: `color_extract.py`
  - image validation
  - resize max 1024px
  - KMeans 3-8 colors
  - sample max 50k pixels
  - temp cleanup

- [x] Phase 2: `clip_mood.py`
  - mood labels
  - lazy CLIP load
  - top 3 scores
  - failure fallback

- [x] Phase 3: `qwen_analyze.py`
  - lazy Qwen load
  - one structured call
  - schema parse/repair
  - deterministic fallback

- [x] Phase 4: `main.py`
  - FastAPI app
  - `/`, `/warmup`, `/analyze`, `/app`
  - upload limits
  - inference lock
  - error schema
  - metadata/warnings

- [x] Phase 5: `static/` frontend
  - upload UI
  - `n_colors` control
  - preview
  - palette/mood/feedback/pairing
  - warnings/metadata

- [x] Phase 6: `evaluate.py`
  - `labels.json` validation
  - Delta E color accuracy
  - CLIP mood accuracy
  - `evaluation_report.json`

- [x] Phase 7: Colab runner notebook
  - clone/upload repo
  - install deps
  - uvicorn
  - ngrok
  - `/warmup`

- [ ] Phase 8: `README.md`
  - overview
  - setup
  - Colab demo
  - API contract
  - evaluation guide
  - limitations
