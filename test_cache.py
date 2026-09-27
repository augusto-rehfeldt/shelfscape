"""Exercise actual loading with a tiny offline encoder; no model downloads."""
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
with patch.dict(sys.modules, {'umap': None}):
    from backend import embeddings as module


class SharedSuiteTests(unittest.TestCase):
    """LM Studio embeddings run on the shared ai-suite AIService, not a client of our own."""

    def manager(self, folder):
        return module.EmbeddingsManager(stories_folder=folder, embedding_provider="lm_studio",
                                        model_name="embed-model", max_batch_size=2,
                                        lm_studio_base_url="http://127.0.0.1:1234/v1", lm_studio_api_key="lm-key")

    def test_lm_studio_uses_the_shared_service(self):
        service = type("S", (), {})()
        service.calls = []
        service.embed = lambda texts, model=None: (service.calls.append((list(texts), model)),
                                                   [[float(len(t)), 1.0] for t in texts])[1]
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(module, "shared_embedding_service", return_value=service) as factory, \
                patch.object(module.EmbeddingsManager, "_ensure_lm_studio"):
            manager = self.manager(folder)
            manager.load_model()
            vectors = manager._encode_texts(["a\nb", "cc", "ddd"])
            query = manager._encode_query("find")
        self.assertEqual(factory.call_args.args, ("http://127.0.0.1:1234/v1", "lm-key", "embed-model"))
        self.assertEqual([c[0] for c in service.calls[:2]], [["a b", "cc"], ["ddd"]])  # batched, newlines flattened
        self.assertTrue(all(model == "embed-model" for _texts, model in service.calls))
        self.assertEqual(np.asarray(vectors).shape, (3, 2))
        self.assertEqual(query.dtype, np.float32)

    def test_shared_embedding_service_is_book_writers(self):
        built = []
        fake = type("M", (), {"AIService": lambda *a, **k: built.append((a, k)) or "svc"})
        with patch.object(module, "_suite_service_module", return_value=fake):
            self.assertEqual(module.shared_embedding_service("http://h/v1", "k", "m"), "svc")
        args, kwargs = built[0]
        self.assertIsNone(args[0])
        overrides = kwargs["config_overrides"]
        self.assertEqual((overrides["provider"], overrides["base_url"], overrides["api_key"], overrides["writing_model"]),
                         ("openrouter", "http://h/v1", "k", "m"))
        self.assertFalse(kwargs["allow_auth_prompt"])

    def test_no_provider_client_of_its_own(self):
        source = Path(module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from openai import", source)


class CacheTests(unittest.TestCase):
    def test_edit_replace_and_reorder_reuse_only_unchanged_embeddings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stories = root / 'stories'
            stories.mkdir()
            calls = []
            def encode(texts):
                calls.extend(texts)
                return np.array([list(hashlib.sha256(t.encode()).digest()[:4]) for t in texts], dtype=float)
            class Reducer:
                def fit_transform(self, matrix): return matrix[:, :2]
                def transform(self, matrix): return matrix[:, :2]
            def load(rows):
                with (stories / 'library.csv').open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['id', 'title', 'summary'])
                    writer.writeheader()
                    writer.writerows(rows)
                manager = module.EmbeddingsManager(str(stories), str(root / 'embeddings.json'),
                                                    str(root / 'projection.json'))
                with patch.object(manager, 'load_model'), patch.object(manager, '_encode_texts', encode), \
                     patch.object(manager, '_create_reducer', return_value=Reducer()), \
                     patch.object(manager, '_emit_status'):
                    manager.load_stories()
                return manager
            rows = [{'id': 'library-1', 'title': 'One', 'summary': 'First'},
                    {'id': 'library-2', 'title': 'Two', 'summary': 'Second'}]
            manager = load(rows)
            self.assertEqual(len(calls), 2)
            before = manager.dataset_fingerprint
            self.assertEqual(manager.story_keys, ['library-1', 'library-2'])
            with patch.object(module.EmbeddingsManager, 'load_model') as model:
                load(rows)
                model.assert_not_called()
            self.assertEqual(len(calls), 2)
            rows[0]['summary'] = 'Changed'
            manager = load(rows)
            self.assertEqual(len(calls), 3)
            self.assertNotEqual(manager.dataset_fingerprint, before)
            self.assertEqual(json.loads((root / 'projection.json').read_text())['fingerprint'], manager.dataset_fingerprint)
            rows[0]['id'] = 'replacement'
            manager = load(list(reversed(rows)))
            self.assertEqual(len(calls), 3)
            self.assertEqual(manager.story_keys, ['library-2', 'replacement'])
            self.assertTrue(np.isfinite(manager.projections_2d).all())
            with patch.object(manager, '_encode_query', return_value=manager.embeddings_matrix[1]):
                ranked = manager.search('anything')
            self.assertEqual([r['id'] for r in ranked][0], 'replacement')
            self.assertEqual([r['rank'] for r in ranked], [1, 2])
            cache = root / 'embeddings.npz'
            stamp = cache.stat().st_mtime_ns
            load(list(reversed(rows)))
            self.assertEqual(cache.stat().st_mtime_ns, stamp)  # nothing new: cache not rewritten
            self.assertEqual(load(rows[:1]).projections_2d.tolist(), [[0.0, 0.0]])
            self.assertEqual(load([]).get_all_stories(), [])

    def test_legacy_json_cache_converts_without_reencoding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'stories').mkdir()
            manager = module.EmbeddingsManager(str(root / 'stories'), str(root / 'embeddings.json'), str(root / 'p.json'))
            (root / 'embeddings.json').write_text(json.dumps({'k': {'embedding': [1.0, 2.0], 'title': 'T'}}))
            with patch.object(manager, '_emit_status'):
                cache = manager._load_cache()
                manager._save_cache(cache)
                self.assertFalse((root / 'embeddings.json').exists())
                self.assertEqual(manager._load_cache()['k'].tolist(), [1.0, 2.0])

    def test_radial_layout_puts_every_band_on_its_ring(self):
        sims = np.array([0.95, 0.91, 0.55, 0.05, -0.3])
        radii = np.hypot(*module.radial_layout(sims).T)
        np.testing.assert_allclose(radii, [0.08, 0.08 + 0.03 * np.sin(2.5), 0.48, 0.96, 0.96 + 0.03 * np.sin(2.5)])


# Keep startup regressions in the workspace's existing test_cache entry point.
from test_startup import StartupTests


if __name__ == '__main__':
    unittest.main()
