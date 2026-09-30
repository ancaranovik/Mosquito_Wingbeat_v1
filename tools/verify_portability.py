"""Lightweight integrity gates; never executes preprocessing or training."""
from pathlib import Path
import ast
import json
import sys
sys.dont_write_bytecode = True
from project_paths import REPO_ROOT, resolve_path
from cache_consumer import sha256, prepare_caches
from baseline_review import verify_baseline


def verify():
    from cache_consumer import require, read_json
    # Compare parsed source in the SAME interpreter. CPython AST schemas differ
    # across versions; original source bytes remain separately hash-pinned.
    pins = read_json(REPO_ROOT / 'configs/baseline/refactor_source_pins.json')
    for relative, expected in pins.items():
        require(sha256(REPO_ROOT / relative) == expected, f'Archived accepted source changed: {relative}')
    contract = read_json(REPO_ROOT / 'reports/stage04/scientific_lock.json')
    for relative, expected in contract['scientific_gate']['artifact_sha256'].items():
        require(sha256(resolve_path(REPO_ROOT, relative)) == expected, f'Frozen identity changed: {relative}')
    for relative, functions in contract['scientific_gate']['ast_sha256'].items():
        versions = []
        for path in (REPO_ROOT / 'archive/portability_originals' / relative, REPO_ROOT / relative):
            nodes = {}
            for cell in read_json(path)['cells']:
                if cell['cell_type'] == 'code':
                    for node in ast.parse(''.join(cell['source'])).body:
                        if isinstance(node, ast.FunctionDef) and node.name in functions:
                            nodes[node.name] = ast.dump(node, include_attributes=False)
            require(set(nodes) == set(functions), f'Missing scientific function: {relative}')
            versions.append(nodes)
        require(versions[0] == versions[1], f'Scientific functions changed: {relative}')
    execution = read_json(REPO_ROOT / 'reports/stage04/frontend_execution_lock.json')
    for branch, name in [('03A', '03A_logmel_reference.ipynb'), ('03B', '03B_edge_mfe.ipynb')]:
        original = read_json(REPO_ROOT / 'archive/portability_originals/notebooks/current' / name)
        current = read_json(REPO_ROOT / 'notebooks/current' / name)
        for index in execution[branch]:
            before = ast.dump(ast.parse(''.join(original['cells'][int(index)]['source'])))
            after = ast.dump(ast.parse(''.join(current['cells'][int(index)]['source'])))
            require(before == after, f'Executed frontend cell changed: {branch}/{index}')
    require(sha256(REPO_ROOT / 'tools/model_zoo.py') ==
            sha256(REPO_ROOT / 'archive/colab_baseline/tools/model_zoo.py'), 'Model definitions changed')
    training_nodes = []
    stable_training = ('FeatureDataset', 'make_loader', 'classification_metrics', 'source_evaluation',
                       'epoch_pass', 'fit_validation_checkpoint', 'load_fixed_checkpoint', 'comparison_row')
    for relative in ('archive/colab_baseline/tools/training_common.py', 'tools/experiment_training.py'):
        tree = ast.parse((REPO_ROOT / relative).read_text(encoding='utf-8'))
        training_nodes.append({n.name: ast.dump(n, include_attributes=False) for n in tree.body
                               if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in stable_training})
    require(set(training_nodes[0]) == set(stable_training) and training_nodes[0] == training_nodes[1],
            'Accepted training/selection/metric behavior changed')
    import experiment_training
    require(experiment_training.CONFIG == read_json(REPO_ROOT / 'configs/baseline/colab_stage04.json')['protocol']['training_configuration'],
            'Experiment protocol differs from accepted CUDA baseline')
    for path in (REPO_ROOT / 'notebooks/current').glob('*.ipynb'):
        notebook = json.loads(path.read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
        if path.name.startswith(('04A_', '04B_', '04D_')):
            assignments = [n.value for c in notebook['cells'] if c['cell_type'] == 'code'
                           for n in ast.parse(''.join(c['source'])).body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == 'RUN_TRAINING' for t in n.targets)]
            require(len(assignments) == 1 and isinstance(assignments[0], ast.Constant)
                    and assignments[0].value is False, f'Training must default to False: {path.name}')
    from pipeline_contract import authenticate_project
    raw = authenticate_project(REPO_ROOT, audio=True, require_grid_freeze=True)
    caches, evidence = prepare_caches()
    baseline = verify_baseline()
    return {'status': 'PASS', 'scientific_ast_locks': 'PASS', 'executed_frontend_cell_locks': 'PASS',
            'training_function_ast_parity': 'PASS', 'model_source_byte_parity': 'PASS', 'training_defaults_disabled': True,
            'raw_wav_sha256_verified': raw['wav_files_authenticated'], 'core_clips': 1898, 'sources': 1189, 'source_leakage': 0, 'examples': 32645,
            'counts': evidence['counts'], 'label_mapping': evidence['label_mapping'],
            'manifest_sha256': evidence['sha256']['data/manifests/current/core_example_grid_v2_stride15360_ref15600_candidates.csv'],
            'split_sha256': evidence['sha256']['data/manifests/current/core_single_4species_frozen_split.csv'],
            'upstream_mel_sha256': sha256(REPO_ROOT / 'reference/humbug/vggish/mel_features.py'),
            'cache_sha256': {b: c[2]['sha256'] for b, c in caches.items()},
            'accepted_baseline_runs': len(baseline), 'training_executed': False, 'preprocessing_executed': False}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
