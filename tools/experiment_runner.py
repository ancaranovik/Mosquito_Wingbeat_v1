"""Explicit cache-only experiments; smoke is the default and never trains."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
sys.dont_write_bytecode = True

from project_paths import ProjectPaths, REPO_ROOT, require_mutable_output


def runtime():
    import importlib.metadata
    import torch
    return {'python': platform.python_version(), 'executable': sys.executable,
            'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'torch')},
            'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def smoke(device='cuda'):
    import torch
    from cache_consumer import prepare_caches, require
    from model_zoo import FAMILIES, build_model
    info = runtime()
    require(sys.version_info >= (3, 11), 'Python 3.11 or newer required for the cache consumer')
    if device == 'cuda':
        require(info['cuda_available'] and info['cuda'], 'Select a Colab GPU and CUDA-enabled PyTorch')
    caches, evidence = prepare_caches()
    torch.set_num_threads(4)
    checks = []
    for branch, (features, metadata, cache) in caches.items():
        for family in FAMILIES:
            model = build_model(family, features.shape[-1]).to(device).eval()
            with torch.inference_mode():
                logits = model(torch.zeros(2, 1, *features.shape[1:], device=device))
            require(tuple(logits.shape) == (2, 4) and bool(torch.isfinite(logits).all()), 'Forward failed')
            checks.append({'frontend': branch, 'model': family, 'parameters': sum(p.numel() for p in model.parameters())})
    return {'status': 'PASS', 'training_executed': False, 'feature_generation_executed': False,
            'runtime': info, 'models': checks, 'cache_sha256': {b: c[2]['sha256'] for b, c in caches.items()},
            'split_sha256': evidence['sha256']['data/manifests/current/core_single_4species_frozen_split.csv'],
            'future_results': str(ProjectPaths.from_env().EXPERIMENT_ROOT)}


def validate_config(config):
    from model_zoo import FAMILIES
    expected = {'experiment_id', 'parent_baseline_id', 'frontend', 'model', 'seed', 'scientific_overrides', 'description'}
    if set(config) != expected:
        raise ValueError(f'Config must contain exactly {sorted(expected)}')
    if config['parent_baseline_id'] != 'stage04_colab_seed42_20260929':
        raise ValueError('Unknown parent baseline')
    if config['frontend'] not in ('03A', '03B') or config['model'] not in FAMILIES:
        raise ValueError('Unknown frozen frontend/model')
    if config['seed'] != 42 or config['scientific_overrides'] != {}:
        raise ValueError('Scientific changes require a separately reviewed experiment implementation')
    return ProjectPaths.from_env().experiment(config['experiment_id'])


def run(config_path):
    import shutil
    import traceback
    from cache_consumer import read_json, write_json, sha256
    import experiment_training as training
    config = read_json(config_path)
    target = validate_config(config)
    require_mutable_output(target)
    if target.exists():
        raise FileExistsError(f'Experiment ID already exists; preserve it and choose a new ID: {target}')
    # Require committed source so metadata identifies the implementation actually run.
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=REPO_ROOT, text=True)
    if dirty.strip():
        raise RuntimeError('Commit source/config changes before starting an experiment')
    report = smoke('cuda')
    target.mkdir(parents=True, exist_ok=False)
    for name in ('config', 'checkpoints', 'results', 'logs'):
        (target / name).mkdir()
    write_json(target / 'config/experiment.json', config)
    metadata = {'experiment_id': config['experiment_id'], 'parent_baseline_id': config['parent_baseline_id'],
                'git_commit': commit, 'configuration': config, 'frontend': config['frontend'],
                'model': config['model'], 'seed': config['seed'], 'runtime': report['runtime'],
                'cache_sha256': report['cache_sha256'], 'split_sha256': report['split_sha256'],
                'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'RUNNING'}
    write_json(target / 'config/metadata.json', metadata)
    training.OUTPUT = target / 'results'
    training.RUNS = training.OUTPUT / 'runs'
    training.SELECTED_FAMILIES = (config['model'],)
    training.EXPERIMENT_ID = config['experiment_id']
    with (target / 'logs/console.log').open('x', encoding='utf-8') as log:
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                training.run_branch(config['frontend'])
            run_root = training.RUNS / f"{config['frontend']}_{config['model']}_seed42"
            shutil.copyfile(run_root / 'best_model.pt', target / 'checkpoints/best_model.pt')
            result = read_json(run_root / 'result.json')
            write_json(target / 'results/confusion_matrix.json', result['final_test_metrics']['confusion_matrix'])
            metadata.update(status='COMPLETE', checkpoint_sha256=sha256(target / 'checkpoints/best_model.pt'),
                            selected_checkpoint='checkpoints/best_model.pt')
        except Exception:
            log.write(traceback.format_exc())
            metadata['status'] = 'FAILED'
            raise
        finally:
            write_json(target / 'config/metadata.json', metadata)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('smoke', 'train', 'baseline'), nargs='?', default='smoke')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    if args.action == 'smoke':
        print(json.dumps(smoke(args.device), indent=2))
    elif args.action == 'baseline':
        from baseline_review import compare_saved_results
        print(compare_saved_results().to_string(index=False))
    else:
        if args.config is None:
            parser.error('train requires --config and a never-used experiment ID')
        print(run(args.config))


if __name__ == '__main__':
    main()
