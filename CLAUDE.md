# Shelfscape (formerly Semantic Story Atlas)

## What This Is
Flask + vanilla JS: embeddings index of a text collection, 2D projection, natural-language search with animated results.

## Non-Negotiables
- Never commit secrets from `.env`.

## Commands
- Run: `python main.py`
- Deps: `pip install -r requirements.txt`
- Sample content: `python generate_stories.py`

## Cache and library contract

- CSV IDs are stable when supplied; reject duplicates. `calibre_id` is source metadata.
- Hash embedding inputs per record and fingerprint projections by ordered dataset.
- `--stories` selects a collection; never combine exports with an existing collection implicitly.
- Check: `python -B -m unittest -q test_cache test_startup` (no model/network needed).
- Embedding vectors are cached as float32 `.npz` (keys + matrix); a legacy JSON cache of the same
  stem is converted once, then deleted. The cache is only rewritten when its key set changes.
- Warm boot never fits UMAP: cached positions are used as-is. The query is drawn at the centre of
  the radial view, so it is never projected (no fitted reducer is kept).
- `/api/stories` ships a short plain `excerpt`, not the summary; `/api/story/<id>` has the full one.
- `/api/search` returns every story as `{id, similarity, rank, radialPosition}`, best first; the
  frontend animates the wave itself. Frontend interpolates data into HTML only through `esc()`.

## Embeddings providers

- `sentence_transformers` runs in-process. `lm_studio` (any OpenAI-compatible `/embeddings`)
  goes through the shared ai-suite AIService (`shared_embedding_service()`, `AIService.embed`;
  sibling `ai-suite` checkout or `AI_SUITE_DIR`, else the vendored `ai_suite/` copy synced by
  ai-suite's `sync.py` -- never edit it here) — no provider client of its own.

- main.py refreshes Calibre read-only via sibling book-watch/library_exchange.py. Never mix its data/ snapshot into stories/.
- Cache-only browsing loads no model; startup failures are sticky until restart.
