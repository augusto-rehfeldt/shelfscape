# Shelfscape

Shelfscape (formerly Semantic Story Atlas) is a Flask + vanilla JavaScript app for exploring a text collection with embeddings. It indexes `.txt` and `.csv` content, projects embeddings into 2D, and lets you search the library with natural-language queries while the results animate in the UI.

## What it does

- Indexes your text collection into embeddings
- Projects stories into 2D for visualization
- Searches by semantic similarity using a natural-language query
- Shows the library as a 2D map; a search moves every book into similarity rings around the query, best matches first
- Supports series-aware grouping and story detail cards
- Press `/` to focus the search box; clearing it returns to the map view

## Repository layout

```text
shelfscape/
├── backend/
│   ├── app.py          # Flask server and API routes
│   └── embeddings.py   # Loading, caching, embedding, and projection logic
├── frontend/
│   ├── index.html      # UI shell
│   ├── styles.css      # App styling
│   └── script.js       # Canvas graph, search UX, and interaction logic
├── examples/
│   ├── sample_book.txt
│   └── sample_chapter.txt
├── generate_stories.py # Helper to generate sample stories into stories/
├── requirements.txt    # Python dependencies
└── README.md
```

## Requirements

- Python 3.10+
- Packages from `requirements.txt`
- Optional: a GPU for faster local encoding
- Optional: `umap-learn` for better 2D projections; the app falls back to PCA if it is unavailable

## Quick start

From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your content to a `stories/` folder at the repo root. The app expects plain text files there, and optionally CSV files and cover images:

```text
stories/
├── 001_my_story.txt
├── 002_another_story.txt
└── covers/
    └── my_cover.jpg
```

You can also generate sample data:

```bash
python generate_stories.py
```

Start everything with one command:

```powershell
python main.py
```

The browser opens after the local server binds. Do not open `frontend/index.html`
as a file. `--no-open` keeps the browser closed.

If `~/Calibre Library/metadata.db` exists, startup refreshes a read-only snapshot
through the sibling `book-watch` checkout. Set `CALIBRE_LIBRARY` in `.env` or use
`--library "path/to/library"` for another library. `--stories path/to/collection`
selects TXT/CSV input instead. Without a detected library, `stories/` is used.
Library errors stop startup rather than silently using a stale snapshot.

## Text formats

### TXT

Each `.txt` file should start with a title line:

```text
# Title of Your Text

The content goes here.
```

You can optionally add frontmatter right after the title:

```text
# Moby Dick
---
author: Herman Melville
summary: A whaling captain's obsessive quest for a white whale.
cover: moby_dick.jpg
genre: Adventure
tags: sea, whaling, obsession
year: 1851
---

Call me Ishmael...
```

### CSV

Drop `.csv` files into `stories/` for bulk import.

Required column:

- `content`

Optional columns:

- `title`
- `author` or `authors`
- `summary` or `#summary`
- `cover`
- `series`
- `series_index`
- `genre`
- `tags`
- `year`

## Embedding providers

By default, the backend uses `sentence_transformers` with `Qwen/Qwen3-Embedding-0.6B`.

To use LM Studio instead, set these environment variables before starting the backend:

```bash
export EMBEDDING_PROVIDER=lm_studio
export EMBEDDING_MODEL=text-embedding-qwen3-embedding-0.6b
export LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
export LM_STUDIO_API_KEY=lm-studio
python main.py
```

Useful tuning variables:

- `EMBEDDING_ENCODING_MODE` — `auto`, `single`, or `multi`
- `EMBEDDING_MAX_BATCH_SIZE`
- `EMBEDDING_MIN_BATCH_SIZE`
- `CUDA_MEMORY_SAFETY_MB`
- `EMBEDDING_BENCHMARK`
- `EMBEDDING_BENCHMARK_SAMPLE_SIZE`
- `EMBEDDING_QUERY_INSTRUCTION`

## API endpoints

The backend currently exposes:

- `GET /` — frontend
- `GET /api/health` — health check
- `GET /api/stories` — all indexed stories with 2D positions and a short excerpt
- `GET /api/story/<story_id>` — one story, with its full summary
- `GET /api/story/<story_id>/cover` — story cover lookup
- `GET /api/covers/<filename>` — cover image from `stories/covers/`
- `POST /api/search` with `{"query": "..."}` — every story ranked best first as `{id, similarity, rank, radialPosition}`

## Caches

Collection-specific embedding (`.npz`) and projection (`.json`) caches live under
ignored `data/cache/`. The original `stories/` collection retains its `backend/` caches.
An older JSON embedding cache is converted on first start.
They are keyed by provider/model so different embedding setups do not overwrite each other.
If you change your dataset and want a clean rebuild, delete the cache files and restart the server.

## Notes

- Startup chooses Calibre when available; `--stories stories` forces the original collection.
- Cover images can live in `stories/covers/` or be referenced by path in story metadata.
- `examples/` contains sample text files you can copy into your own dataset.
- If port 5000 is busy, the OS selects an available port; the browser opens that address.

## Calibre/book-watch integration

From book-watch, run `python book_watch.py export-atlas --output data/atlas/library.csv`.
Then, from this project, run `python main.py --stories ../book-watch/data/atlas`.
This selects the exported library without mixing it into the original stories folder.
The summarizer plugin's custom column feeds the export; book comments are a fallback.

CSV may include `id` and `calibre_id`. Explicit IDs survive sorting and re-export;
duplicate IDs are rejected. Files without explicit IDs retain their legacy row IDs.
Embedding caches use the actual per-record embedding text, provider and model, so an
edited CSV row does not re-encode every other row. Projection caches fingerprint the
ordered dataset; replacing/editing stories rebuilds positions automatically. Compatible legacy CSV cache entries are migrated; changed embedding inputs rebuild automatically. Restart the backend after updating a collection.

Offline cache checks (tiny synthetic encoder, no downloads):
`python -B -m unittest -q test_cache test_startup`.

## Automatic Calibre refresh

`python main.py` refreshes the snapshot on every launch. It indexes new books and
changed summaries; unchanged embedding inputs reuse their vectors. Deleted books
leave the current index. Calibre's database and books are never modified. Snapshots
are separate from `stories/`, with stable UUID-based IDs.

Retrieval uses the complete `#summary`, or comments when no summary exists, plus
metadata and tags. It does **not** read entire EPUB/PDF files or generate missing
summaries. Startup reports how many books have summaries. Use the existing Calibre
summarizer to improve missing or weak descriptions.

LM Studio starts through `lms server start` when a local embedding endpoint is down.
Remote endpoints never launch a local server. Fully cached browsing does not need
LM Studio; searching or indexing new content does. A failed startup is reported by
`/api/health` without retrying indefinitely on every poll; restart after fixing it.

## Local model review (2026-09-27)

Keep the installed Qwen3-Embedding-0.6B for now: no new model download was requested
after reviewing the approximately 4.9 GB upgrade. A stronger model needs a separate
index; vectors from different models must not be mixed.

- **[Nemotron-3-Embed-8B](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16)**:
  leading upgrade candidate. July 2026 model card reports RTEB-16 78.46 and MMTEB
  retrieval 75.45. OpenMDW 1.1 license. Community GGUFs exist, but their LM Studio
  pooling/runtime compatibility and quantized retrieval quality remain unverified.
  Shelfscape supports its documented `query: ` / `passage: ` input prefixes.
- **[Nemotron-3-Embed-1B](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16)**:
  lower-memory candidate, reported RTEB-16 72.38. Also requires runtime validation.
- **[Octen-Embedding-8B](https://huggingface.co/Octen/Octen-Embedding-8B)**:
  Apache-2.0 Qwen3 derivative, strong January 2026 RTEB results; 4B variant available.
  Its overall RTEB score is not directly comparable to NVIDIA's RTEB-16 subset.
- **[Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)**:
  established Apache-2.0 upgrade; 4B offers a memory/speed compromise. Both support
  long-context text and instruction-aware queries.
- **[Jina Embeddings v4](https://huggingface.co/jinaai/jina-embeddings-v4)**:
  visual-document and multi-vector features add little to this summary-only index.
- **[EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m)**:
  lightweight candidate, not a demonstrated quality upgrade for this library.

These are published model-card claims, not an exhaustive independently verified
leaderboard or a library-specific benchmark. Bigger models cannot recover topics
missing from summaries. Test representative complex queries before switching.

After installing and validating a model in LM Studio, pass its exact identifier:
`python main.py --model MODEL_IDENTIFIER`. This does not download models or silently
fall back to another model. All embedding API calls still go through ai-suite.
