print("Booting Shelfscape backend...")

import os
import threading
import logging
from datetime import datetime
from pathlib import Path

# ponytail: minimal .env reader instead of python-dotenv; no multiline/export syntax.
_env = Path(__file__).resolve().parent.parent / ".env"
for _line in (_env.read_text(encoding="utf-8").splitlines() if _env.exists() else []):
    _key, _sep, _value = _line.strip().partition("=")
    _value = _value.strip()
    if len(_value) > 1 and _value[0] == _value[-1] and _value[0] in "\"'":
        _value = _value[1:-1]
    if _sep and _key.strip() and not _key.startswith("#"):
        os.environ.setdefault(_key.strip(), _value)
from flask import Flask, jsonify, request, send_from_directory, send_file
if __package__:
    from .embeddings import EmbeddingsManager, build_cache_stem
else:
    from embeddings import EmbeddingsManager, build_cache_stem


# Suppress verbose Werkzeug request logs for health checks
class HealthCheckFilter(logging.Filter):
    """Filter out health check request logs."""

    def filter(self, record):
        if not hasattr(record, 'msg') or not isinstance(record.msg, str):
            return True
        # Filter out health check GET requests
        if 'GET /api/health' in record.msg:
            return False
        return True


# Apply filter to Werkzeug logger
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.addFilter(HealthCheckFilter())
# Set logging level to WARNING to reduce noise, but keep INFO for startup messages
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Serve frontend static files
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app = Flask(__name__, static_folder=frontend_path, static_url_path="")

# Initialize embeddings manager
stories_path = os.path.join(os.path.dirname(__file__), "..", "stories")
covers_path = os.path.join(stories_path, "covers")
backend_path = os.path.dirname(__file__)
encoding_mode = os.getenv("EMBEDDING_ENCODING_MODE", "auto")
max_batch_size = int(os.getenv("EMBEDDING_MAX_BATCH_SIZE", "32"))
min_batch_size = int(os.getenv("EMBEDDING_MIN_BATCH_SIZE", "4"))
cuda_memory_safety_mb = int(os.getenv("CUDA_MEMORY_SAFETY_MB", "1536"))
benchmark_encoding = os.getenv("EMBEDDING_BENCHMARK", "0").lower() in ("1", "true", "yes", "on")
benchmark_sample_size = int(os.getenv("EMBEDDING_BENCHMARK_SAMPLE_SIZE", "8"))
embedding_provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").strip().lower()
embedding_model = os.getenv("EMBEDDING_MODEL", "").strip() or None
lm_studio_base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
lm_studio_api_key = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
query_instruction = os.getenv(
    "EMBEDDING_QUERY_INSTRUCTION",
    "Given a book search query, retrieve relevant passages that answer the query.",
)

default_model_name = (
    "text-embedding-qwen3-embedding-0.6b"
    if embedding_provider == "lm_studio"
    else "Qwen/Qwen3-Embedding-0.6B"
)
model_name = embedding_model or default_model_name

if embedding_provider == "sentence_transformers" and model_name == "Qwen/Qwen3-Embedding-0.6B":
    embeddings_cache_file = os.path.join(backend_path, "embeddings_cache_qwen3_embedding_0_6b.json")
    projection_cache_file = os.path.join(
        backend_path, "projection_cache_qwen3_embedding_0_6b.json"
    )
else:
    cache_stem = build_cache_stem(embedding_provider, model_name)
    embeddings_cache_file = os.path.join(backend_path, f"embeddings_cache_{cache_stem}.json")
    projection_cache_file = os.path.join(backend_path, f"projection_cache_{cache_stem}.json")

embeddings_manager = EmbeddingsManager(
    stories_folder=stories_path,
    cache_file=embeddings_cache_file,
    projection_cache_file=projection_cache_file,
    encoding_mode=encoding_mode,
    max_batch_size=max_batch_size,
    min_batch_size=min_batch_size,
    cuda_memory_safety_mb=cuda_memory_safety_mb,
    benchmark_encoding=benchmark_encoding,
    benchmark_sample_size=benchmark_sample_size,
    embedding_provider=embedding_provider,
    model_name=model_name,
    lm_studio_base_url=lm_studio_base_url,
    lm_studio_api_key=lm_studio_api_key,
    query_instruction=query_instruction,
)

_load_thread = None
_load_thread_lock = threading.Lock()


def _load_stories():
    try:
        embeddings_manager.load_stories()
    except Exception:
        pass  # The manager records the actionable error for /api/health.


def ensure_story_loading_started():
    global _load_thread
    if embeddings_manager.is_loading or embeddings_manager.is_ready or embeddings_manager.load_error:
        return

    with _load_thread_lock:
        if _load_thread is not None and _load_thread.is_alive():
            return
        if embeddings_manager.is_loading or embeddings_manager.is_ready or embeddings_manager.load_error:
            return

        print(f"[{datetime.now().strftime('%H:%M:%S')}][BOOT] Starting story loading on demand...")
        _load_thread = threading.Thread(target=_load_stories, daemon=True)
        _load_thread.start()




@app.route("/")
def index():
    """Serve the frontend."""
    return send_from_directory(frontend_path, "index.html")


@app.route("/api/stories", methods=["GET"])
def get_stories():
    """Get all stories with positions."""
    ensure_story_loading_started()
    if not embeddings_manager.is_ready:
        return jsonify(
            {
                "stories": [],
                "count": 0,
                "loading": embeddings_manager.is_loading,
                "ready": embeddings_manager.is_ready,
                "stories_loaded": len(embeddings_manager.stories),
            }
        )
    stories = embeddings_manager.get_all_stories()
    return jsonify(
        {
            "stories": stories,
            "count": len(stories),
            "loading": embeddings_manager.is_loading,
            "ready": embeddings_manager.is_ready,
            "stories_loaded": len(embeddings_manager.stories),
        }
    )


@app.route("/api/covers/<path:filename>", methods=["GET"])
def get_cover(filename):
    """Serve cover images from stories/covers/."""
    return send_from_directory(covers_path, filename)


@app.route("/api/story/<path:story_id>/cover", methods=["GET"])
def get_story_cover(story_id):
    """Serve a story cover from either stories/covers/ or a local file path."""
    if story_id not in embeddings_manager.stories:
        return jsonify({"error": "Story not found"}), 404

    story = embeddings_manager.stories[story_id]
    cover = story.get("cover", "")
    if not cover:
        return jsonify({"error": "Cover not found"}), 404

    if os.path.isabs(cover) and os.path.exists(cover):
        return send_file(cover)

    if os.path.exists(cover):
        return send_file(cover)

    cover_name = os.path.basename(cover)
    local_cover = os.path.join(covers_path, cover_name)
    if os.path.exists(local_cover):
        return send_file(local_cover)

    fallback = os.path.join(stories_path, cover)
    if os.path.exists(fallback):
        return send_file(fallback)

    return jsonify({"error": "Cover not found"}), 404


@app.route("/api/search", methods=["POST"])
def search():
    """Rank every story against a query: id, similarity, rank and radial position, best first.
    Titles and metadata are already on the client from /api/stories."""
    ensure_story_loading_started()
    if not embeddings_manager.is_ready:
        return jsonify(
            {
                "error": "Stories are still loading",
                "loading": embeddings_manager.is_loading,
                "stories_loaded": len(embeddings_manager.stories),
            }
        ), 503
    query = ((request.get_json(silent=True) or {}).get("query") or "").strip()
    if not query:
        return jsonify({"error": "Query is required"}), 400
    return jsonify({"query": query, "results": embeddings_manager.search(query)})


@app.route("/api/story/<story_id>", methods=["GET"])
def get_story(story_id):
    """Get a specific story."""
    ensure_story_loading_started()
    if not embeddings_manager.is_ready:
        return jsonify(
            {
                "error": "Stories are still loading",
                "loading": embeddings_manager.is_loading,
                "stories_loaded": len(embeddings_manager.stories),
            }
        ), 503
    if story_id not in embeddings_manager.stories:
        return jsonify({"error": "Story not found"}), 404

    story = embeddings_manager.stories[story_id]
    result = {
        "id": story_id,
        "title": story["title"],
        "content": story["content"],
        "filename": story["filename"],
        "position": embeddings_manager._get_story_position(story_id),
    }
    result.update(embeddings_manager._story_metadata(story_id))
    return jsonify(result)


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    ensure_story_loading_started()
    return jsonify(
        {
            "status": "healthy",
            "stories_loaded": len(embeddings_manager.stories),
            "model": embeddings_manager.model_name,
            "has_projections": embeddings_manager.projections_2d is not None,
            "loading": embeddings_manager.is_loading,
            "ready": embeddings_manager.is_ready,
            "error": embeddings_manager.load_error,
        }
    )


def run_server(open_browser=True):
    import webbrowser
    from werkzeug.serving import make_server
    # Bind first; port zero lets the OS select a free port if 5000 is occupied.
    try:
        server = make_server("127.0.0.1", 5000, app, threaded=True)
    except SystemExit:
        server = make_server("127.0.0.1", 0, app, threaded=True)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Shelfscape: {url}")
    ensure_story_loading_started()
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from main import main
    main()
