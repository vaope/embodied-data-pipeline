import argparse
import json
from pathlib import Path
from .common import load_config
from .features.feature_store import build_index
from .search.retrieval_engine import RetrievalEngine

def main():
    parser = argparse.ArgumentParser(description='EgoDex Similarity Retrieval V1')
    parser.add_argument('--config', default=None)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build-index')
    build.add_argument('--dataset-root', required=True, help='sources.json or directory containing it')
    build.add_argument('--annotation-root')
    find = commands.add_parser('retrieve')
    find.add_argument('--query', required=True)
    find.add_argument('--top-k', type=int, default=10)
    find.add_argument('--recall-top-n', type=int)
    find.add_argument('--method', choices=['pose_language_active', 'legacy_v1'], default=None)
    find.add_argument('--refine-phase', action='store_true', help='also diagnose Top10 sources at +/-14 frames')
    find.add_argument('--allow-same-episode', action='store_true')
    find.add_argument('--unlimited-episode', action='store_true')
    find.add_argument('--report', action='store_true')
    inspect = commands.add_parser('inspect')
    inspect.add_argument('--clip-id')
    serve = commands.add_parser('serve')
    serve.add_argument('--port', type=int, default=8771)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == 'build-index':
        print(json.dumps(build_index(args.dataset_root, args.annotation_root, cfg), ensure_ascii=False, indent=2))
    elif args.command == 'serve':
        from .server import make_server
        server = make_server(cfg['artifact_root'], args.port)
        print(f'http://127.0.0.1:{server.server_port}/reports/index.html', flush=True)
        server.serve_forever()
    else:
        engine = RetrievalEngine(cfg['artifact_root'], cfg)
        if args.command == 'inspect':
            value = engine.get_clip_features(args.clip_id)['metadata'] if args.clip_id else engine.store.meta
        else:
            options = {'allow_same_episode': args.allow_same_episode,
                       'max_per_episode': None if args.unlimited_episode else 1,
                       'refine_phase': args.refine_phase}
            if args.method:
                options['method'] = args.method
            if args.recall_top_n is not None:
                options['recall_top_n'] = args.recall_top_n
            value = engine.retrieve(args.query, args.top_k, options)
            if args.report:
                directory = Path(cfg['artifact_root'])/'reports'
                if value.get('method') == 'pose_language_active':
                    directory /= 'stable'
                path = directory/f'{args.query}.html'
                engine.generate_report(args.query, value, path)
                if value.get('phase_diagnostic'):
                    diagnostic = dict(value, results=value['phase_diagnostic']['results'],
                                      raw_ranking=value['phase_diagnostic']['results'], method='phase_refined')
                    engine.generate_report(args.query, diagnostic, path.with_name(path.stem+'_phase.html'))
            value = {k: v for k, v in value.items() if k != 'raw_ranking'}
        print(json.dumps(value, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
