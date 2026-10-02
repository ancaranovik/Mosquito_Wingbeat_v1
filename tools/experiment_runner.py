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

VERSIONED_PROTOCOLS = ('stage04_experiment_v2', 'stage04_experiment_v3', 'stage04_experiment_v4', 'stage04_experiment_v5', 'stage04_experiment_v6')


def runtime():
    import importlib.metadata
    import torch
    return {'python': platform.python_version(), 'executable': sys.executable,
            'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'torch')},
            'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def smoke(device='cuda', config=None):
    import torch
    from cache_consumer import prepare_caches, require
    from model_zoo import FAMILIES, build_model
    info = runtime()
    require(sys.version_info >= (3, 11), 'Python 3.11 or newer required for the cache consumer')
    if device == 'cuda':
        require(info['cuda_available'] and info['cuda'], 'Select a Colab GPU and CUDA-enabled PyTorch')
    caches, evidence = prepare_caches()
    plan = run_plan(config) if config is not None else [
        {'frontend': b, 'model': f} for b in caches for f in FAMILIES]
    torch.set_num_threads(4)
    checks = []
    for branch in dict.fromkeys(item['frontend'] for item in plan):
        features, metadata, cache = caches[branch]
        for family in [item['model'] for item in plan if item['frontend'] == branch]:
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
    common = {'experiment_id', 'parent_baseline_id', 'seed', 'scientific_overrides', 'description'}
    suite = 'frontends' in config or 'models' in config
    targeted = config.get('protocol_version') == 'stage04_experiment_v6'
    if 'protocol_version' in config and config['protocol_version'] not in VERSIONED_PROTOCOLS:
        raise ValueError('Unsupported experiment protocol_version; update the repo and restart the kernel')
    versioned = config.get('protocol_version') in VERSIONED_PROTOCOLS
    expected = common | ({'frontends', 'models'} if suite else {'frontend', 'model'})
    if versioned:
        expected |= {'protocol_version', 'training_overrides', 'evaluate_test', 'reference_experiment_id'}
    if set(config) != expected:
        raise ValueError(f'Config must contain exactly {sorted(expected)}')
    if config['parent_baseline_id'] != 'stage04_colab_seed42_20260929':
        raise ValueError('Unknown parent baseline')
    if targeted and (not suite or config['frontends'] != ['03A'] or config['models'] != ['ds_cnn']
                     or config['reference_experiment_id'] != 'exp_007_power075'):
        raise ValueError('v6 requires DS-CNN/03A only, compared with exp_007_power075')
    if suite and not targeted and (config['frontends'] != ['03A', '03B'] or config['models'] != list(FAMILIES)):
        raise ValueError('A suite must contain 03A and 03B and all four frozen models exactly once, in order')
    if not suite and (config['frontend'] not in ('03A', '03B') or config['model'] not in FAMILIES):
        raise ValueError('Unknown frozen frontend/model')
    if config['seed'] != 42 or config['scientific_overrides'] != {}:
        raise ValueError('Scientific changes require a separately reviewed experiment implementation')
    if versioned:
        effective_config = training_module(config).effective_config
        if not suite:
            raise ValueError('Versioned protocols require a complete eight-configuration suite')
        effective_config(config['training_overrides'], config['evaluate_test'])
        ProjectPaths.from_env().experiment(config['reference_experiment_id'])
        if config['reference_experiment_id'] == config['experiment_id']:
            raise ValueError('An experiment cannot be its own reference')
    return ProjectPaths.from_env().experiment(config['experiment_id'])


def training_module(config):
    if config.get('protocol_version') == 'stage04_experiment_v2':
        import experiment_training_v2 as training
    elif config.get('protocol_version') == 'stage04_experiment_v3':
        import experiment_training_v3 as training
    elif config.get('protocol_version') == 'stage04_experiment_v4':
        import experiment_training_v4 as training
    elif config.get('protocol_version') == 'stage04_experiment_v5':
        import experiment_training_v5 as training
    elif config.get('protocol_version') == 'stage04_experiment_v6':
        import experiment_training_v6 as training
    elif 'protocol_version' in config:
        raise ValueError('Unknown experiment protocol_version')
    else:
        import experiment_training as training
    return training


def run_plan(config):
    """Legacy pair, full suite or the reviewed v6 DS-CNN/03A run; seed 42 stays fixed."""
    validate_config(config)
    frontends = config.get('frontends', [config.get('frontend')])
    models = config.get('models', [config.get('model')])
    return [{'run_id': f'{frontend}_{model}_seed42', 'frontend': frontend, 'model': model}
            for frontend in frontends for model in models]


class Tee:
    """Flush each epoch to both the notebook and the persistent experiment log."""
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def run(config_path, experiment_id=None, description=None):
    import shutil
    import traceback
    from cache_consumer import read_json, write_json, sha256
    config = read_json(config_path)
    # Runtime naming does not require changing a tracked config or notebook in Colab.
    if experiment_id is not None:
        config['experiment_id'] = experiment_id
    if description is not None:
        config['description'] = description
    target = validate_config(config)
    training = training_module(config)
    plan = run_plan(config)
    require_mutable_output(target)
    if target.exists():
        raise FileExistsError(f'Experiment ID already exists; preserve it and choose a new ID: {target}')
    reference_evidence = None
    if config.get('protocol_version') == 'stage04_experiment_v6':
        from bounded_aggregation import authenticate_reference
        from cache_consumer import require
        _, reference_evidence, _ = authenticate_reference(ProjectPaths.from_env())
        before = reference_evidence['source_training_configuration']
        after = training.effective_config(config['training_overrides'], config['evaluate_test'])
        changes = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
        require(changes == {'protocol_version', 'augmentation', 'augmentation_config'},
                'Exp010 differs from exp007 beyond the reviewed augmentation')
    # Require committed source so metadata identifies the implementation actually run.
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=REPO_ROOT, text=True)
    if dirty.strip():
        raise RuntimeError('Commit source/config changes before starting an experiment')
    report = smoke('cuda', config) if config.get('protocol_version') == 'stage04_experiment_v6' else smoke('cuda')
    target.mkdir(parents=True, exist_ok=False)
    for name in ('config', 'checkpoints', 'results', 'logs'):
        (target / name).mkdir()
    write_json(target / 'config/experiment.json', config)
    metadata = {'experiment_id': config['experiment_id'], 'parent_baseline_id': config['parent_baseline_id'],
                'git_commit': commit, 'configuration': config,
                'source_config_sha256': sha256(config_path),
                'seed': config['seed'], 'runtime': report['runtime'], 'planned_runs': plan,
                'expected_runs': len(plan), 'completed_runs': 0,
                'runs': {item['run_id']: {'status': 'PENDING'} for item in plan},
                'cache_sha256': report['cache_sha256'], 'split_sha256': report['split_sha256'],
                'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'RUNNING'}
    if reference_evidence is not None:
        metadata['reference_evidence'] = reference_evidence
    write_json(target / 'config/metadata.json', metadata)
    previous = (training.OUTPUT, training.RUNS, training.SELECTED_FAMILIES, training.EXPERIMENT_ID)
    previous_config = training.CONFIG
    with (target / 'logs/console.log').open('x', encoding='utf-8') as log:
        try:
            training.OUTPUT = target / 'results'
            training.RUNS = training.OUTPUT / 'runs'
            training.EXPERIMENT_ID = config['experiment_id']
            if config.get('protocol_version') in VERSIONED_PROTOCOLS:
                training.CONFIG = training.effective_config(config['training_overrides'], config['evaluate_test'])
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                for index, item in enumerate(plan, 1):
                    run_id = item['run_id']
                    metadata['current_run'] = run_id
                    metadata['runs'][run_id] = {'status': 'RUNNING', 'started_at_utc': datetime.now(timezone.utc).isoformat()}
                    write_json(target / 'config/metadata.json', metadata)
                    print(f"[{index}/{len(plan)}] {config['experiment_id']} / {run_id}", flush=True)
                    training.SELECTED_FAMILIES = (item['model'],)
                    training.run_branch(item['frontend'])
                    run_root = training.RUNS / run_id
                    checkpoint = target / 'checkpoints' / run_id / 'best_model.pt'
                    checkpoint.parent.mkdir(exist_ok=False)
                    shutil.copyfile(run_root / 'best_model.pt', checkpoint)
                    result = read_json(run_root / 'result.json')
                    display_metrics = result['final_test_metrics'] or result['selected_validation_metrics']
                    confusion = display_metrics['confusion_matrix']
                    if config.get('protocol_version') in VERSIONED_PROTOCOLS:
                        confusion = {'split': 'test' if result['test_evaluated'] else 'validation',
                                     'class_order': result['class_order'], 'matrix': confusion}
                    write_json(run_root / 'confusion_matrix.json', confusion)
                    metadata['runs'][run_id].update(status='COMPLETE',
                        result_sha256=sha256(run_root / 'result.json'), checkpoint_sha256=sha256(checkpoint),
                        selected_checkpoint=checkpoint.relative_to(target).as_posix(),
                        completed_at_utc=datetime.now(timezone.utc).isoformat())
                    metadata['completed_runs'] = index
                    metadata['protocol_sha256'] = sha256(target / 'results/protocol.json')
                    write_json(target / 'config/metadata.json', metadata)
                from experiment_review import verify_experiment, comparison_summary
                records = verify_experiment(config['experiment_id'], require_complete=False)
                summary, table = comparison_summary(records)
                table.to_csv(target / 'results/comparison.csv', index=False)
                write_json(target / 'results/comparison.json', summary)
                metadata.update(status='COMPLETE', current_run=None,
                    completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    comparison_sha256={name: sha256(target / 'results' / name)
                                       for name in ('comparison.csv', 'comparison.json')})
                print(f"COMPLETE: {len(plan)}/{len(plan)} runs; results: {target}", flush=True)
        except BaseException as error:
            log.write(traceback.format_exc())
            log.flush()
            metadata.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'FAILED',
                            failure_type=type(error).__name__, stopped_at_utc=datetime.now(timezone.utc).isoformat())
            current = metadata.get('current_run')
            if current and metadata['runs'][current]['status'] == 'RUNNING':
                metadata['runs'][current]['status'] = metadata['status']
            raise
        finally:
            training.OUTPUT, training.RUNS, training.SELECTED_FAMILIES, training.EXPERIMENT_ID = previous
            training.CONFIG = previous_config
            write_json(target / 'config/metadata.json', metadata)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('smoke', 'train', 'baseline'), nargs='?', default='smoke')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--experiment-id', help='New output folder name; overrides the config template ID')
    parser.add_argument('--description', help='Purpose/change notes saved with this run')
    args = parser.parse_args()
    if args.action == 'smoke':
        from cache_consumer import read_json
        print(json.dumps(smoke(args.device, read_json(args.config) if args.config else None), indent=2))
    elif args.action == 'baseline':
        from baseline_review import compare_saved_results
        print(compare_saved_results().to_string(index=False))
    else:
        if args.config is None:
            parser.error('train requires --config and a never-used experiment ID')
        print(run(args.config, args.experiment_id, args.description))


if __name__ == '__main__':
    main()
