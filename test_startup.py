"""Offline startup and embedding regressions."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from backend.embeddings import EmbeddingsManager


class StartupTests(unittest.TestCase):
    def test_remote_endpoint_never_starts_local_server(self):
        manager = EmbeddingsManager(embedding_provider='lm_studio', lm_studio_base_url='https://example.com/v1')
        with patch('socket.create_connection', side_effect=OSError('offline')), patch('subprocess.run') as run:
            with self.assertRaisesRegex(RuntimeError, 'example.com'):
                manager._ensure_lm_studio()
            run.assert_not_called()

    def test_nemotron_query_uses_model_prefix(self):
        manager = EmbeddingsManager(embedding_provider='lm_studio', model_name='Nemotron-3-Embed-8B')
        manager.model = MagicMock()
        manager.model.embed.return_value = [[1., 2.]]
        manager._encode_query('political philosophy')
        self.assertEqual(manager.model.embed.call_args.args[0], ['query: political philosophy'])

    def test_empty_collection_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = EmbeddingsManager(directory, str(Path(directory) / 'cache.json'), str(Path(directory) / 'projection.json'))
            with patch.object(manager, 'load_model') as load:
                manager.load_stories()
            load.assert_not_called()

    def test_calibre_refresh_runs_each_startup_in_separate_folder(self):
        import main
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory)
            (library / 'metadata.db').touch()
            args = main.parse_args(['--library', str(library)])
            with patch('subprocess.run') as run:
                first = main.prepare_collection(args)
                second = main.prepare_collection(args)
            self.assertEqual(first, second)
            self.assertEqual(run.call_count, 2)
            self.assertNotEqual(first, main.ROOT / 'stories')
            self.assertIn(str(library), run.call_args.args[0])

    def test_nemotron_documents_include_tags_and_prefix(self):
        manager = EmbeddingsManager(model_name='Nemotron-3-Embed-8B')
        record = dict(title='Book', author='Author', summary='Summary', tags='politics')
        self.assertEqual(manager._build_embedding_text(record),
                         'passage: Book. Author. Summary. Topics: politics')

    def test_health_does_not_retry_failed_load(self):
        from backend import app
        with patch.object(app.embeddings_manager, 'load_error', 'offline'), patch.object(app.threading, 'Thread') as thread:
            response = app.app.test_client().get('/api/health')
            self.assertEqual(response.json['error'], 'offline')
            thread.assert_not_called()

    def test_csv_row_cache_migrates_without_encoding(self):
        import hashlib
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stories = root / 'stories'
            stories.mkdir()
            source = stories / 'books.csv'
            source.write_text('title,summary' + chr(10) + 'Book,Summary' + chr(10))
            key = 'books.csv_' + hashlib.md5(source.read_bytes()).hexdigest() + '_row_1'
            (root / 'cache.json').write_text(json.dumps({key: {'embedding': [1., 2.]}}))
            manager = EmbeddingsManager(str(stories), str(root / 'cache.json'), str(root / 'p.json'))
            with patch.object(manager, 'load_model') as model, patch.object(manager, '_encode_texts') as encode:
                manager.load_stories()
            model.assert_not_called()
            encode.assert_not_called()
            self.assertEqual(len(manager.stories), 1)
            self.assertTrue((root / 'cache.npz').exists())

    def test_cli_sources_are_exclusive(self):
        import main
        with self.assertRaises(SystemExit):
            main.parse_args(['--library', 'a', '--stories', 'b'])

    def test_explicit_missing_library_does_not_fall_back(self):
        import main
        with tempfile.TemporaryDirectory() as directory:
            args = main.parse_args(['--library', directory])
            with self.assertRaisesRegex(ValueError, 'metadata.db'):
                main.prepare_collection(args)


if __name__ == '__main__':
    unittest.main()
