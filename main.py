"""Launch Shelfscape and refresh its read-only Calibre snapshot."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--library', type=Path, help='Calibre folder containing metadata.db')
    source.add_argument('--stories', type=Path, help='Use TXT/CSV collection instead of Calibre')
    parser.add_argument('--model', help='Embedding model identifier loaded by LM Studio')
    parser.add_argument('--no-open', action='store_true', help='Do not open the browser')
    return parser.parse_args(argv)


def prepare_collection(args):
    if args.stories:
        folder = args.stories.expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f'Collection folder does not exist: {folder}')
        return folder
    configured = os.getenv('CALIBRE_LIBRARY')
    library = args.library or (Path(configured) if configured else Path.home() / 'Calibre Library')
    library = library.expanduser().resolve()
    if not (library / 'metadata.db').is_file():
        if args.library or configured:
            raise ValueError(f'Calibre library must contain metadata.db: {library}')
        return ROOT / 'stories'
    helper = ROOT.parent / 'book-watch'
    if not (helper / 'library_exchange.py').is_file():
        raise ValueError('Calibre integration requires the sibling book-watch checkout; use --stories for an exported collection.')
    namespace = hashlib.sha256(str(library).encode()).hexdigest()[:12]
    folder = ROOT / 'data' / namespace
    subprocess.run([sys.executable, '-B', '-c',
        'import sys; from library_exchange import export_atlas; '
        'n,s=export_atlas({"library":{"path":sys.argv[1]}},sys.argv[2]); '
        'print(f"Calibre: {n} books, {s} with summaries; {n-s} metadata-only")',
        str(library), str(folder / 'library.csv')], cwd=helper, check=True, timeout=180)
    return folder


def main(argv=None):
    # Import loads .env but does not start a model or server.
    from backend import app
    args = parse_args(argv)
    try:
        folder = prepare_collection(args)
    except (ValueError, subprocess.SubprocessError) as exc:
        raise SystemExit(str(exc)) from exc
    manager = app.embeddings_manager
    manager.stories_folder = str(folder)
    app.covers_path = str(folder / 'covers')
    if args.model:
        manager.model_name = args.model
        manager.embedding_provider = 'lm_studio'
    namespace = hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:12]
    stem = app.build_cache_stem(manager.embedding_provider, manager.model_name)
    cache = ROOT / 'data' / 'cache' / namespace
    cache.mkdir(parents=True, exist_ok=True)
    # Preserve the original collection's existing cache when no model override is requested.
    if folder.resolve() == (ROOT / 'stories').resolve() and not args.model:
        app.run_server(open_browser=not args.no_open)
        return
    manager.cache_file = str(cache / f'embeddings_cache_{stem}.npz')
    manager.legacy_cache_file = str(cache / f'embeddings_cache_{stem}.json')
    manager.projection_cache_file = str(cache / f'projection_cache_{stem}.json')
    app.run_server(open_browser=not args.no_open)


if __name__ == '__main__':
    main()
