import os
import json
import hashlib
import csv
import re
import time
import tempfile
from urllib.parse import quote
import numpy as np
from typing import List, Dict, Optional
import threading

# LM Studio (and any OpenAI-compatible /embeddings endpoint) is reached through
# the shared ai-suite package, like every AI call in the workspace: the sibling
# checkout when present (AI_SUITE_DIR overrides it), else the copy vendored into this
# repository. The local sentence-transformers path needs no provider at all.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_SUITE = os.environ.get("AI_SUITE_DIR") or os.path.join(os.path.dirname(_REPO), "ai-suite")
SUITE_PATH = AI_SUITE if os.path.isdir(AI_SUITE) else _REPO


def _suite_service_module():
    import sys
    if SUITE_PATH not in sys.path:
        sys.path.insert(0, SUITE_PATH)
    from ai_suite import service
    return service


def shared_embedding_service(base_url: str, api_key: str, model_name: str):
    """The shared AIService pointed at an OpenAI-compatible embeddings endpoint."""
    state = tempfile.gettempdir()
    overrides = {
        "provider": "openrouter", "base_url": base_url, "api_key": api_key, "writing_model": model_name,
        "groq_rate_state_path": os.path.join(state, "shelfscape_groq.json"),
    }
    return _suite_service_module().AIService(
        None, os.path.join(state, "shelfscape_ai_usage.json"), allow_auth_prompt=False,
        client_max_retries=2, config_overrides=overrides)

# umap drags numba in (seconds of import); only look for it here, import it when fitting.
try:
    import importlib.util
    HAS_UMAP = importlib.util.find_spec("umap") is not None
except (ImportError, ValueError):
    HAS_UMAP = False
if not HAS_UMAP:
    print("UMAP not available, falling back to PCA")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("tqdm not available, progress bar disabled")


_TORCH = None
_SENTENCE_TRANSFORMER = None
_PCA = None


def _get_torch():
    global _TORCH
    if _TORCH is None:
        import torch as _torch
        _TORCH = _torch
    return _TORCH


def _get_sentence_transformer():
    global _SENTENCE_TRANSFORMER
    if _SENTENCE_TRANSFORMER is None:
        from sentence_transformers import SentenceTransformer as _SentenceTransformer

        _SENTENCE_TRANSFORMER = _SentenceTransformer
    return _SENTENCE_TRANSFORMER


def _get_pca():
    global _PCA
    if _PCA is None:
        from sklearn.decomposition import PCA as _PCAClass

        _PCA = _PCAClass
    return _PCA


def build_cache_stem(provider: str, model_name: str) -> str:
    """Build a stable filename stem for cache files."""
    provider_slug = re.sub(r"[^a-z0-9]+", "_", provider.lower()).strip("_")
    model_slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    return f"{provider_slug}_{model_slug}"


def atomic_write(path, write, mode='w'):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode=mode, encoding=None if 'b' in mode else 'utf-8',
                                         dir=os.path.dirname(os.path.abspath(path)), delete=False) as stream:
            temporary = stream.name
            write(stream)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_write(path, lambda stream: json.dump(value, stream))


# Similarity bands for the radial layout: band 0 is 90%+, band 9 is under 10%.
BAND_RADII = np.array([0.08, 0.18, 0.28, 0.38, 0.48, 0.58, 0.68, 0.78, 0.88, 0.96])


def radial_layout(similarities: np.ndarray) -> np.ndarray:
    """Place descending similarities on concentric rings, spread evenly around each ring."""
    band = np.clip(9 - np.floor(similarities * 10 + 1e-9).astype(int), 0, 9)
    ordinal = np.arange(len(band)) - np.searchsorted(band, band)  # position inside its band
    count = np.bincount(band, minlength=10)[band]
    angle = np.where(count > 1, 2 * np.pi * ordinal / np.maximum(count, 1), 0.0) + band * 0.4
    radius = BAND_RADII[band] + 0.03 * np.sin(ordinal * 2.5)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)


class EmbeddingsManager:
    def __init__(
        self,
        stories_folder: str = "../stories",
        cache_file: str = "embeddings_cache_qwen3_embedding_0_6b.json",
        projection_cache_file: str = "projection_cache_qwen3_embedding_0_6b.json",
        encoding_mode: str = "auto",
        max_batch_size: int = 32,
        min_batch_size: int = 4,
        cuda_memory_safety_mb: int = 1536,
        benchmark_encoding: bool = False,
        benchmark_sample_size: int = 8,
        embedding_provider: str = "sentence_transformers",
        model_name: Optional[str] = None,
        lm_studio_base_url: str = "http://127.0.0.1:1234/v1",
        lm_studio_api_key: str = "lm-studio",
        query_instruction: str = (
            "Given a book search query, retrieve relevant passages that answer the query."
        ),
    ):
        self.stories_folder = stories_folder
        # Embeddings live in a float32 .npz; a legacy JSON cache of the same stem is read once.
        stem = os.path.splitext(cache_file)[0]
        self.cache_file = stem + ".npz"
        self.legacy_cache_file = stem + ".json"
        self.projection_cache_file = projection_cache_file
        self.encoding_mode = encoding_mode
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.cuda_memory_safety_mb = cuda_memory_safety_mb
        self.benchmark_encoding = benchmark_encoding
        self.benchmark_sample_size = benchmark_sample_size
        self.embedding_provider = embedding_provider.lower().strip()
        if self.embedding_provider not in ("sentence_transformers", "lm_studio"):
            raise ValueError(
                f"Unsupported embedding_provider '{embedding_provider}'. "
                "Expected 'sentence_transformers' or 'lm_studio'."
            )
        self.model = None
        self.model_name = model_name or (
            "text-embedding-qwen3-embedding-0.6b"
            if self.embedding_provider == "lm_studio"
            else "Qwen/Qwen3-Embedding-0.6B"
        )
        self.lm_studio_base_url = lm_studio_base_url
        self.lm_studio_api_key = lm_studio_api_key
        self.query_instruction = query_instruction
        self.stories: Dict[str, Dict] = {}
        self.embeddings_matrix = None
        self.normalized_matrix = None  # Pre-normalized for fast cosine similarity
        self.story_keys = []
        self.story_index: Dict[str, int] = {}  # O(1) key -> index lookup
        self.projections_2d = None
        self._lock = threading.Lock()
        self.status_callback = None
        self.is_loading = False
        self.is_ready = False
        self.load_error = None
        self._lm_studio_client = None

    def set_status_callback(self, callback):
        self.status_callback = callback

    def _emit_status(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}][STATUS] {message}")
        if self.status_callback:
            self.status_callback(message)

    def load_model(self):
        if self.model is None:
            if self.embedding_provider == "lm_studio":
                self._ensure_lm_studio()
                self._emit_status(
                    f"Connecting to LM Studio embeddings at {self.lm_studio_base_url} "
                    f"using model {self.model_name}..."
                )
                self._lm_studio_client = shared_embedding_service(
                    self.lm_studio_base_url, self.lm_studio_api_key, self.model_name)
                self.model = self._lm_studio_client
                self._emit_status("LM Studio embedding client ready!")
            else:
                self._emit_status(
                    f"Loading embedding model ({self.model_name})..."
                )
                sentence_transformer_cls = _get_sentence_transformer()
                self.model = sentence_transformer_cls(self.model_name)
                self._emit_status("Embedding model loaded successfully!")
        return self.model

    def _ensure_lm_studio(self):
        """Start the LM Studio server through its `lms` CLI when nothing listens on the port.
        The embedding model loads on first request (LM Studio just-in-time loading)."""
        import shutil
        import socket
        import subprocess
        from urllib.parse import urlparse
        url = urlparse(self.lm_studio_base_url)
        port = url.port or (443 if url.scheme == "https" else 80)
        try:
            with socket.create_connection((url.hostname or "127.0.0.1", port), timeout=2):
                return
        except OSError:
            if url.hostname not in ("localhost", "127.0.0.1", "::1"):
                raise RuntimeError(f"Embedding server unavailable: {self.lm_studio_base_url}")
        lms = shutil.which("lms") or os.path.expanduser("~/.cache/lm-studio/bin/lms.exe")
        if not os.path.exists(lms) and not shutil.which("lms"):
            raise RuntimeError(f"LM Studio is not running at {self.lm_studio_base_url}; open it and start the server")
        self._emit_status("LM Studio server is not running, starting it...")
        subprocess.run([lms, "server", "start", "--port", str(port)], check=True, timeout=120)

    def _get_file_hash(self, filepath: str) -> str:
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _load_cache(self) -> Dict[str, np.ndarray]:
        """Cache key -> embedding vector."""
        try:
            if os.path.exists(self.cache_file):
                self._emit_status("Loading cached embeddings...")
                with np.load(self.cache_file) as data:
                    cache = dict(zip(data["keys"].tolist(), data["matrix"]))
            elif os.path.exists(self.legacy_cache_file):
                self._emit_status("Converting legacy JSON embedding cache...")
                with open(self.legacy_cache_file, "r") as f:
                    cache = {key: np.asarray(value["embedding"], dtype=np.float32)
                             for key, value in json.load(f).items()}
            else:
                self._emit_status("No embedding cache found, will generate fresh embeddings")
                return {}
        except Exception as e:
            self._emit_status(f"Cache load failed: {e}")
            return {}
        self._emit_status(f"Found {len(cache)} cached embeddings")
        return cache

    def _save_cache(self, cache: Dict[str, np.ndarray]):
        self._emit_status(f"Saving {len(cache)} embeddings to cache...")
        keys = list(cache)
        matrix = np.array([cache[key] for key in keys], dtype=np.float32)
        atomic_write(self.cache_file, lambda stream: np.savez(stream, keys=np.array(keys), matrix=matrix), 'wb')
        if os.path.exists(self.legacy_cache_file):
            os.remove(self.legacy_cache_file)
        self._emit_status("Embeddings cached successfully!")

    def _load_projection_cache(self) -> Optional[Dict]:
        if os.path.exists(self.projection_cache_file):
            try:
                self._emit_status("Loading cached 2D projections...")
                with open(self.projection_cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_projection_cache(self, projections: Dict):
        self._emit_status("Saving 2D projections to cache...")
        atomic_json(self.projection_cache_file, {
            'fingerprint': self.dataset_fingerprint, 'positions': projections})

    def _get_target_devices(self) -> List[str]:
        torch = _get_torch()
        if not torch.cuda.is_available():
            return []

        if self.encoding_mode == "single":
            return ["cuda:0"]

        device_count = torch.cuda.device_count()
        if device_count <= 1 and self.encoding_mode == "auto":
            return ["cuda:0"]

        return [f"cuda:{i}" for i in range(device_count)]

    def _estimate_batch_size(self, devices: List[str]) -> int:
        torch = _get_torch()
        if not torch.cuda.is_available():
            return self.min_batch_size

        device_ids = [int(device.split(":")[1]) for device in devices] if devices else [0]
        free_memory_mb = []

        for device_id in device_ids:
            try:
                free_bytes, _ = torch.cuda.mem_get_info(device_id)
            except TypeError:
                with torch.cuda.device(device_id):
                    free_bytes, _ = torch.cuda.mem_get_info()
            free_memory_mb.append(free_bytes / (1024 * 1024))

        available_mb = max(0, min(free_memory_mb) - self.cuda_memory_safety_mb)
        estimated = int(available_mb // 350)
        return max(self.min_batch_size, min(self.max_batch_size, estimated or self.min_batch_size))

    def _build_embedding_text(self, record: Dict) -> str:
        text = self._build_book_text(record)
        if record.get("tags"):
            text += ". Topics: " + record["tags"]
        if "nemotron-3-embed" in self.model_name.lower():
            text = "passage: " + text
        return text

    def _build_book_text(self, record: Dict) -> str:
        """Build the text used for embedding from available metadata."""
        if record["summary"]:
            parts = [record["title"]]
            if record["author"]:
                parts.append(record["author"])
            if record.get("series"):
                series_text = record["series"]
                if record.get("series_index"):
                    series_text = f"{series_text} #{record['series_index']}"
                parts.append(series_text)
            parts.append(record["summary"])
            return ". ".join(parts)

        author_prefix = f"{record['author']}. " if record["author"] else ""
        series_prefix = ""
        if record.get("series"):
            series_text = record["series"]
            if record.get("series_index"):
                series_text = f"{series_text} #{record['series_index']}"
            series_prefix = f"{series_text}. "
        return f"{record['title']}. {author_prefix}{series_prefix}{record['content'][:1000]}"

    def _encode_texts(self, texts: List[str]):
        """Encode a batch of texts, using multi-process encoding when available."""
        if self.embedding_provider == "lm_studio":
            if not texts:
                return np.array([])

            cleaned_texts = [text.replace("\n", " ") for text in texts]
            embeddings: List[List[float]] = []
            batch_size = max(1, self.max_batch_size)
            total_batches = (len(cleaned_texts) + batch_size - 1) // batch_size

            pbar = tqdm(total=total_batches, desc="Encoding via LM Studio") if HAS_TQDM else None
            for batch_index in range(0, len(cleaned_texts), batch_size):
                batch = cleaned_texts[batch_index : batch_index + batch_size]
                self._emit_status(
                    f"Encoding {len(batch)} texts via LM Studio "
                    f"(batch {(batch_index // batch_size) + 1}/{total_batches})..."
                )
                embeddings.extend(self.model.embed(batch, model=self.model_name))
                if pbar:
                    pbar.update(1)

            if pbar:
                pbar.close()

            return np.array(embeddings, dtype=np.float32)

        devices = self._get_target_devices()
        batch_size = self._estimate_batch_size(devices)

        if len(devices) > 1 and self.encoding_mode in ("auto", "multi"):
            self._emit_status(
                f"Encoding {len(texts)} stories on {len(devices)} GPUs (batch_size={batch_size})..."
            )
            pool = self.model.start_multi_process_pool(target_devices=devices)
            try:
                return self.model.encode(
                    texts,
                    pool=pool,
                    batch_size=batch_size,
                    show_progress_bar=True,
                )
            finally:
                self.model.stop_multi_process_pool(pool)

        if devices == ["cuda:0"]:
            self._emit_status(
                f"Encoding {len(texts)} stories on GPU with batch_size={batch_size}..."
            )
        else:
            self._emit_status(
                f"Encoding {len(texts)} stories on CPU with batch_size={batch_size}..."
            )

        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
        )

    def _count_tokens(self, texts: List[str]) -> int:
        """Best-effort token count for throughput reporting."""
        if not texts:
            return 0

        try:
            tokenized = self.model.tokenize(texts)
            if isinstance(tokenized, dict) and "attention_mask" in tokenized:
                attention_mask = tokenized["attention_mask"]
                if hasattr(attention_mask, "sum"):
                    return int(attention_mask.sum().item())
            if isinstance(tokenized, dict) and "input_ids" in tokenized:
                input_ids = tokenized["input_ids"]
                if hasattr(input_ids, "numel"):
                    return int(input_ids.numel())
        except Exception:
            pass

        # Fallback: rough heuristic based on character count.
        return max(1, sum(max(1, len(text) // 4) for text in texts))

    def _benchmark_encoding(self, texts: List[str]):
        """Run a small encode benchmark and log approximate throughput."""
        if not self.benchmark_encoding or not texts:
            return

        sample = texts[: self.benchmark_sample_size]
        if not sample:
            return

        devices = self._get_target_devices()
        batch_size = self._estimate_batch_size(devices)
        token_count = self._count_tokens(sample)

        self._emit_status(
            f"Benchmarking embedding throughput with {len(sample)} texts (batch_size={batch_size})..."
        )

        import time as _time

        start = _time.perf_counter()
        if len(devices) > 1 and self.encoding_mode in ("auto", "multi"):
            pool = self.model.start_multi_process_pool(target_devices=devices)
            try:
                self.model.encode(sample, pool=pool, batch_size=batch_size, show_progress_bar=False)
            finally:
                self.model.stop_multi_process_pool(pool)
        else:
            self.model.encode(sample, batch_size=batch_size, show_progress_bar=False)
        elapsed = max(_time.perf_counter() - start, 1e-6)
        tokens_per_second = token_count / elapsed
        mode = "multi-GPU" if len(devices) > 1 and self.encoding_mode in ("auto", "multi") else (
            "GPU" if devices == ["cuda:0"] else "CPU"
        )
        self._emit_status(
            f"Benchmark result: {tokens_per_second:,.0f} tokens/sec on {mode} with batch_size={batch_size} "
            f"({len(sample)} texts, {token_count} tokens, {elapsed:.2f}s)"
        )

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a search query using a retrieval-style instruction prompt."""
        self.load_model()
        query_text = (
            f"Instruct: {self.query_instruction}\n"
            f"Query: {query}"
        )
        if "nemotron-3-embed" in self.model_name.lower():
            query_text = "query: " + query
        if self.embedding_provider == "lm_studio":
            vector = self.model.embed([query_text.replace("\n", " ")], model=self.model_name)[0]
            return np.array(vector, dtype=np.float32)

        return self.model.encode(query_text, prompt="")

    def _cover_url_for_story(self, story_id: str) -> str:
        story = self.stories.get(story_id, {})
        if not story.get("cover"):
            return ""
        return f"/api/story/{quote(story_id, safe='')}/cover"

    def _create_reducer(self):
        if HAS_UMAP and len(self.story_keys) > 3:
            import umap
            return umap.UMAP(
                n_components=2,
                n_neighbors=min(15, len(self.story_keys) - 1),
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
        pca_cls = _get_pca()
        return pca_cls(n_components=2, random_state=42)

    @staticmethod
    def _normalize_projections(projections: np.ndarray) -> np.ndarray:
        """Normalize projections to [-1, 1] range."""
        min_vals = projections.min(axis=0)
        max_vals = projections.max(axis=0)
        span = np.where(max_vals > min_vals, max_vals - min_vals, 1)
        return 2 * (projections - min_vals) / span - 1

    def _compute_2d_projections(self):
        if len(self.story_keys) < 2:
            self.projections_2d = np.zeros((len(self.story_keys), 2))
            return
        method = "UMAP" if HAS_UMAP else "PCA"
        self._emit_status(f"Computing 2D projections using {method}...")

        self.projections_2d = self._normalize_projections(
            self._create_reducer().fit_transform(self.embeddings_matrix)
        )

        projection_data = {
            key: self.projections_2d[i].tolist()
            for i, key in enumerate(self.story_keys)
        }
        self._save_projection_cache(projection_data)
        self._emit_status("2D projections computed and cached!")

    def _build_index(self):
        """Build lookup structures after stories are loaded."""
        self.story_keys = list(self.stories.keys())
        self.story_index = {key: i for i, key in enumerate(self.story_keys)}
        self.embeddings_matrix = np.array(
            [self.stories[key]["embedding"] for key in self.story_keys]
        )
        if not self.story_keys:
            self.embeddings_matrix = np.zeros((0, 2))
        # Pre-normalize for fast cosine similarity
        norms = np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
        self.normalized_matrix = self.embeddings_matrix / np.where(norms > 0, norms, 1)

    @staticmethod
    def _parse_frontmatter(text):
        """Parse optional YAML-style frontmatter from text content.

        Expected format:
            # Title
            ---
            key: value
            key: value
            ---
            Content...

        Returns (metadata_dict, content_without_frontmatter).
        """
        lines = text.split("\n")
        metadata = {}

        # Find title line first
        title_end = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped:
                title_end = i + 1
                break

        # Skip blank lines after title
        fm_start = None
        for i in range(title_end, len(lines)):
            stripped = lines[i].strip()
            if stripped == "---":
                fm_start = i
                break
            elif stripped:
                # Non-empty, non-delimiter line means no frontmatter
                break

        if fm_start is not None:
            fm_end = None
            for i in range(fm_start + 1, len(lines)):
                if lines[i].strip() == "---":
                    fm_end = i
                    break

            if fm_end is not None:
                for line in lines[fm_start + 1 : fm_end]:
                    line = line.strip()
                    if not line:
                        continue
                    colon_idx = line.find(":")
                    if colon_idx > 0:
                        key = line[:colon_idx].strip().lower()
                        value = line[colon_idx + 1 :].strip()
                        metadata[key] = value

                # Remove frontmatter block from content
                content_lines = lines[:title_end] + lines[fm_end + 1 :]
                return metadata, "\n".join(content_lines)

        return metadata, text

    @staticmethod
    def _normalize_field_name(name: str) -> str:
        return name.strip().lower().lstrip("#").replace(" ", "_").replace("-", "_")

    def _iter_story_records(self):
        """Yield normalized story records from .txt and .csv sources."""
        source_files = sorted(
            [
                f
                for f in os.listdir(self.stories_folder)
                if f.endswith(".txt") or f.endswith(".csv")
            ]
        )

        for filename in source_files:
            filepath = os.path.join(self.stories_folder, filename)

            if filename.endswith(".txt"):
                yield from self._iter_txt_story_records(filename, filepath)
            else:
                yield from self._iter_csv_story_records(filename, filepath)

    def _iter_txt_story_records(self, filename: str, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        metadata, content = self._parse_frontmatter(content)
        lines = content.split("\n")
        title = lines[0].replace("#", "").strip() if lines else filename
        full_content = "\n".join(lines[1:]).strip() if len(lines) > 1 else content
        record = {
            "source_filename": filename,
            "story_id": filename,
            "file_hash": self._get_file_hash(filepath),
            "title": title,
            "author": metadata.get("author", ""),
            "summary": metadata.get("summary", ""),
            "cover": metadata.get("cover", ""),
            "series": metadata.get("series", ""),
            "series_index": metadata.get("series_index", ""),
            "genre": metadata.get("genre", ""),
            "tags": metadata.get("tags", ""),
            "year": metadata.get("year", ""),
            "content": full_content,
        }
        yield record

    def _iter_csv_story_records(self, filename: str, filepath: str):
        self._emit_status(f"Reading CSV file: {filename}")
        file_hash = self._get_file_hash(filepath)
        with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            base_name = os.path.splitext(filename)[0]
            for row_index, row in enumerate(reader, start=1):
                normalized = {
                    self._normalize_field_name(key): (value or "").strip()
                    for key, value in row.items()
                    if key is not None
                }
                content = (
                    normalized.get("content")
                    or normalized.get("summary")
                    or normalized.get("text")
                    or ""
                )
                if not content:
                    self._emit_status(
                        f"Skipping {filename} row {row_index}: no content or summary found"
                    )
                    continue

                title = normalized.get("title") or f"{base_name} {row_index}"
                cover = (
                    normalized.get("cover")
                    or normalized.get("cover_route")
                    or normalized.get("cover_path")
                    or ""
                )
                author = normalized.get("author") or normalized.get("authors") or ""
                series = normalized.get("series", "")
                series_index = (
                    normalized.get("series_index")
                    or normalized.get("seriesindex")
                    or normalized.get("series_no")
                    or ""
                )

                yield {
                    "source_filename": filename,
                    "story_id": normalized.get('id') or f"{base_name}_{row_index:04d}",
                    "calibre_id": normalized.get('calibre_id', ''),
                    "legacy_cache_key": f"{filename}_{file_hash}_row_{row_index}",
                    "title": title,
                    "author": author,
                    "summary": normalized.get("summary", ""),
                    "cover": cover,
                    "series": series,
                    "series_index": series_index,
                    "genre": normalized.get("genre", ""),
                    "tags": normalized.get("tags", ""),
                    "year": normalized.get("year", ""),
                    "content": content,
                }

    def load_stories(self):
        self.is_loading = True
        self.is_ready = False
        self.load_error = None

        try:
            self.stories = {}
            self.embeddings_matrix = None
            self.normalized_matrix = None
            self.story_keys = []
            self.story_index = {}
            self.projections_2d = None

            self._emit_status("Starting to load stories...")
            self._emit_status("Scanning stories folder for .txt and .csv files...")
            cache = self._load_cache()

            story_records = list(self._iter_story_records())
            ids = [record['story_id'] for record in story_records]
            if len(set(ids)) != len(ids):
                raise ValueError('Duplicate story IDs in the collection')
            self._emit_status(f"Found {len(story_records)} story entries")

            if self.benchmark_encoding:
                self.load_model()
                benchmark_texts = [
                    self._build_embedding_text(record)
                    for record in story_records[: self.benchmark_sample_size]
                ]
                self._benchmark_encoding(benchmark_texts)

            updated_cache = {}
            story_entries = []
            new_embeddings_count = 0
            cached_embeddings_count = 0

            for record in story_records:
                cache_key = hashlib.sha256(json.dumps([
                    self.embedding_provider, self.model_name, self._build_embedding_text(record)
                ], ensure_ascii=False).encode('utf-8')).hexdigest()
                embedding = cache.get(cache_key)
                if embedding is None and self._build_embedding_text(record) == self._build_book_text(record):
                    embedding = cache.get(record.get("legacy_cache_key"))
                cached_embeddings_count += embedding is not None
                story_entries.append(
                    {"record": record, "cache_key": cache_key, "embedding": embedding}
                )

            uncached_entries = [entry for entry in story_entries if entry["embedding"] is None]

            if uncached_entries:
                self.load_model()
                embeddings = self._encode_texts(
                    [self._build_embedding_text(entry["record"]) for entry in uncached_entries]
                )
                if isinstance(embeddings, np.ndarray) and embeddings.ndim == 1:
                    embeddings = np.expand_dims(embeddings, axis=0)

                for entry, embedding in zip(uncached_entries, embeddings):
                    entry["embedding"] = np.array(embedding)
                new_embeddings_count = len(uncached_entries)

            self._emit_status(
                f"Embedding reuse summary: {cached_embeddings_count} cached, {new_embeddings_count} new"
            )

            for entry in story_entries:
                if entry["embedding"] is None:
                    continue
                updated_cache[entry["cache_key"]] = entry["embedding"]
                story_data = {
                    "title": entry["record"]["title"],
                    "content": (
                        entry["record"]["content"][:500] + "..."
                        if len(entry["record"]["content"]) > 500
                        else entry["record"]["content"]
                    ),
                    "embedding": entry["embedding"],
                    "filename": entry["record"]["source_filename"],
                    "author": entry["record"]["author"],
                }
                for key in (
                    "calibre_id",
                    "summary",
                    "cover",
                    "series",
                    "series_index",
                    "genre",
                    "tags",
                    "year",
                ):
                    if entry["record"].get(key):
                        story_data[key] = entry["record"][key]
                self.stories[entry["record"]["story_id"]] = story_data

            if new_embeddings_count > 0:
                self._emit_status(f"Generated {new_embeddings_count} new embeddings")

            # Rewriting ~100 MB of vectors on every boot is most of a warm start; skip it.
            if new_embeddings_count or set(updated_cache) != set(cache) or not os.path.exists(self.cache_file):
                self._save_cache(updated_cache)
            self._build_index()

            self.dataset_fingerprint = hashlib.sha256(json.dumps([
                self.embedding_provider, self.model_name, bool(HAS_UMAP),
                [(entry['record']['story_id'], entry['cache_key']) for entry in story_entries]
            ]).encode('utf-8')).hexdigest()

            # Load or compute 2D projections
            projection_cache = self._load_projection_cache()
            if (len(self.story_keys) > 1 and isinstance(projection_cache, dict)
                    and projection_cache.get('fingerprint') == self.dataset_fingerprint
                    and set(projection_cache.get('positions', {})) == set(self.story_keys)):
                self._emit_status("Using cached 2D projections")
                self.projections_2d = np.array(
                    [projection_cache['positions'][key] for key in self.story_keys]
                )
            else:
                self._compute_2d_projections()

            self._emit_status(
                f"Loaded {len(self.stories)} stories with embeddings and 2D projections"
            )
            self.is_ready = True
            return self.stories
        except Exception as e:
            self.load_error = str(e)
            self._emit_status(f"Failed to load stories: {e}")
            raise
        finally:
            self.is_loading = False

    def _get_story_position(self, story_id: str) -> dict:
        """Get the 2D projection position for a story."""
        idx = self.story_index[story_id]
        return {
            "x": float(self.projections_2d[idx][0]),
            "y": float(self.projections_2d[idx][1]),
        }

    def _story_metadata(self, key: str) -> Dict:
        """Extract optional metadata fields from a story."""
        story = self.stories[key]
        meta = {}
        for field in (
            "calibre_id",
            "author",
            "summary",
            "cover",
            "series",
            "series_index",
            "genre",
            "tags",
            "year",
        ):
            if field in story:
                meta[field] = story[field]
        if story.get("cover"):
            meta["cover_url"] = self._cover_url_for_story(key)
        return meta

    @staticmethod
    def _excerpt(text: str, limit: int = 280) -> str:
        """Plain-text opening of a (possibly Markdown) summary, without its heading lines."""
        lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        plain = re.sub(r"[*_`>]+", "", " ".join(lines)).strip()
        return plain if len(plain) <= limit else plain[:limit].rsplit(" ", 1)[0] + "…"

    def get_all_stories(self) -> List[Dict]:
        """Get all stories with their fixed 2D positions; summaries are cut to an excerpt
        (the full one is served by /api/story/<id>), or a big library ships tens of MB."""
        if not self.is_ready or self.projections_2d is None:
            return []
        results = []
        for key in self.story_keys:
            story = self.stories[key]
            entry = {
                "id": key,
                "title": story["title"],
                "position": self._get_story_position(key),
            }
            entry.update(self._story_metadata(key))
            entry["excerpt"] = self._excerpt(entry.pop("summary", "") or story["content"])
            results.append(entry)
        return results

    def search(self, query: str) -> List[Dict]:
        """Rank every story against the query, best first, with its radial layout position."""
        with self._lock:
            query_embedding = np.asarray(self._encode_query(query), dtype=np.float32)
        norm = np.linalg.norm(query_embedding)
        similarities = self.normalized_matrix @ (query_embedding / (norm or 1))
        order = np.argsort(-similarities, kind="stable")
        ranked = similarities[order]
        radial = radial_layout(ranked)
        return [
            {
                "id": self.story_keys[index],
                "similarity": round(float(similarity), 4),
                "rank": rank + 1,
                "radialPosition": {"x": round(float(x), 4), "y": round(float(y), 4)},
            }
            for rank, (index, similarity, (x, y)) in enumerate(zip(order, ranked, radial))
        ]
