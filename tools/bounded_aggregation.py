"""CPU-only validation diagnostic on authenticated saved predictions; no model import."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import platform
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from cache_consumer import CLASSES, META_COLUMNS, digest_json, read_json, require, sha256, write_json
from pipeline_contract import EXPECTED_SHA256, MANIFEST
from project_paths import ProjectPaths, REPO_ROOT, require_mutable_output

VERSION = 'bounded_two_window_v1'
REFERENCE = 'exp_007_power075'
RUN = '03A_ds_cnn_seed42'
DEFAULT_ID = 'exp_009_two_window_agg'
BOOTSTRAP_REPEATS = 4000
BOOTSTRAP_SEED = 20261002


def metrics(labels, predictions):
    cm = np.bincount(4 * np.asarray(labels, dtype=int) + np.asarray(predictions, dtype=int),
                     minlength=16).reshape(4, 4)
    tp = np.diag(cm)
    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    precision = divide(tp, cm.sum(0))
    recall = divide(tp, cm.sum(1))
    f1 = divide(2 * tp, cm.sum(0) + cm.sum(1))
    return {'macro_f1': float(f1.mean()), 'accuracy': float(tp.sum() / cm.sum()),
            'confusion_matrix': cm.tolist(), 'per_class': {
                name: {'precision': float(precision[i]), 'recall': float(recall[i]),
                       'f1': float(f1[i]), 'support': int(cm.sum(1)[i])}
                for i, name in enumerate(CLASSES)}}


def authenticate_reference(paths):
    """Authenticate only the selected run and small manifest; never open cache/raw/TEST."""
    root = paths.experiment(REFERENCE)
    directory = root / 'results/runs' / RUN
    files = {}
    def checked(path, expected=None):
        actual = sha256(path)
        require(expected is None or actual == expected, f'Hash mismatch: {path}')
        files[path] = actual
        return read_json(path) if path.suffix == '.json' else path
    meta = checked(root / 'config/metadata.json')
    config = checked(root / 'config/experiment.json')
    require(meta['status'] == 'COMPLETE' and meta['experiment_id'] == REFERENCE
            and meta['configuration'] == config and config['experiment_id'] == REFERENCE,
            'Reference identity/status mismatch')
    require(meta['completed_runs'] == meta['expected_runs'] == 8, 'Incomplete reference suite')
    require(config['protocol_version'] == 'stage04_experiment_v4' and config['evaluate_test'] is False,
            'Expected validation-only exp007 v4')
    protocol = checked(root / 'results/protocol.json', meta['protocol_sha256'])
    run_meta = meta['runs'][RUN]
    require(run_meta['status'] == 'COMPLETE', 'Incomplete reference run')
    result = checked(directory / 'result.json', run_meta['result_sha256'])
    require(result['status'] == 'COMPLETE' and result['run_id'] == RUN
            and result['class_order'] == CLASSES and result['test_evaluated'] is False
            and result['test_used_for_selection'] is False
            and result['final_test_metrics'] is None
            and result['protocol_sha256'] == digest_json(protocol)
            and result['training_configuration'] == protocol['training_configuration'],
            'Reference run/protocol/TEST policy mismatch')
    selected = checked(directory / 'selection.json', result['artifacts_sha256']['selection.json'])
    require(selected['model_sha256'] == result['model_sha256']
            and selected['best_epoch'] == result['best_epoch']
            and selected['selected_validation_metrics'] == result['selected_validation_metrics'],
            'Checkpoint selection mismatch')
    checked(directory / 'best_model.pt', result['model_sha256'])
    manifest_path = paths.resolve(MANIFEST)
    checked(manifest_path, EXPECTED_SHA256[MANIFEST])
    require(protocol['accepted_input_sha256'][MANIFEST] == EXPECTED_SHA256[MANIFEST],
            'Reference used a different grid')
    prediction_path = checked(directory / 'validation_predictions.csv',
                              result['artifacts_sha256']['validation_predictions.csv'])
    predictions = pd.read_csv(prediction_path)
    grid = pd.read_csv(manifest_path)
    expected = grid.loc[grid.split.eq('validation'), META_COLUMNS].reset_index(drop=True)
    expected['label'] = expected.species.map({name: i for i, name in enumerate(CLASSES)})
    pd.testing.assert_frame_equal(predictions[META_COLUMNS + ['label']], expected,
                                  check_dtype=False, check_exact=True)
    require(len(predictions) == 4937 and predictions.example_uid.is_unique, 'Incomplete validation predictions')
    probability = predictions[[f'p{i}' for i in range(4)]].to_numpy()
    require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
            and np.allclose(probability.sum(1), 1, rtol=0, atol=1e-6), 'Invalid probabilities')
    require(np.array_equal(predictions.prediction, probability.argmax(1)), 'Argmax mismatch')
    measured = metrics(predictions.label, predictions.prediction)
    recorded = result['selected_validation_metrics']
    require(measured['confusion_matrix'] == recorded['confusion_matrix']
            and np.isclose(measured['macro_f1'], recorded['macro_f1'], rtol=0, atol=1e-12),
            'Saved validation metrics mismatch')
    provenance = {'reference_experiment_id': REFERENCE, 'reference_run_id': RUN,
                  'reference_commit': meta['git_commit'], 'selected_epoch': result['best_epoch'],
                  'selected_model_sha256': result['model_sha256'],
                  'source_training_configuration': result['training_configuration'],
                  'input_sha256': {str(p): h for p, h in files.items()}}
    return predictions, provenance, files


def pair_windows(frame):
    require(len(frame) > 0 and frame.example_uid.is_unique, 'Empty/duplicate input')
    require(frame.split.eq('validation').all(), 'Only validation is permitted')
    pairs, tails = [], []
    for clip, group in frame.groupby('id', sort=True):
        group = group.sort_values('example_index').reset_index(drop=True)
        for col in ['name', 'label', 'species', 'target_sample_rate',
                    'example_stride_samples', 'reference_context_samples']:
            require(group[col].nunique() == 1, f'Mixed {col} within clip {clip}')
        require(int(group.target_sample_rate.iloc[0]) == 16000
                and int(group.example_stride_samples.iloc[0]) == 15360
                and int(group.reference_context_samples.iloc[0]) == 15600, 'Unexpected window geometry')
        require(np.array_equal(group.example_index, np.arange(len(group)))
                and np.array_equal(group.start_sample_16k, np.arange(len(group)) * 15360)
                and np.array_equal(group.reference_end_sample_16k, group.start_sample_16k + 15600),
                f'Gap/order/context mismatch: clip {clip}')
        for i in range(0, len(group) - 1, 2):
            a, b = group.iloc[i], group.iloc[i + 1]
            probability = (a[[f'p{j}' for j in range(4)]].to_numpy(dtype=float)
                           + b[[f'p{j}' for j in range(4)]].to_numpy(dtype=float)) / 2
            pairs.append({'pair_id': f'clip{clip}_pair{i//2:04d}', 'id': clip,
                          'name': a['name'], 'species': a.species, 'label': int(a.label),
                          'uid_first': a.example_uid, 'uid_second': b.example_uid,
                          'start_sample_16k': int(a.start_sample_16k),
                          'end_sample_16k': int(b.reference_end_sample_16k),
                          'support_seconds': (b.reference_end_sample_16k - a.start_sample_16k) / 16000,
                          'prediction_first': int(a.prediction), 'prediction_second': int(b.prediction),
                          'prediction': int(probability.argmax()),
                          **{f'p{j}': float(probability[j]) for j in range(4)}})
        if len(group) % 2:
            tail = group.iloc[-1]
            tails.append({'example_uid': tail.example_uid, 'id': clip, 'name': tail['name'],
                          'species': tail.species, 'label': int(tail.label), 'reason': 'odd_clip_tail'})
    require(bool(pairs), 'No eligible pairs')
    pairs = pd.DataFrame(pairs)
    excluded = pd.DataFrame(tails, columns=['example_uid', 'id', 'name', 'species', 'label', 'reason'])
    used = set(pairs.uid_first) | set(pairs.uid_second)
    require(len(used) == 2 * len(pairs) and used.isdisjoint(set(excluded.example_uid))
            and len(used) + len(excluded) == len(frame), 'Pair coverage mismatch')
    matched = frame[frame.example_uid.isin(used)].copy()
    return pairs, excluded, matched


def paired_intervals(pairs):
    """Stratified source-block resampling; same source draw for both endpoints."""
    blocks_a, blocks_b, labels = [], [], []
    for _, g in pairs.groupby('name', sort=True):
        require(g.label.nunique() == 1, 'Source spans classes')
        labels.append(int(g.label.iloc[0]))
        y = np.repeat(g.label.to_numpy(dtype=int), 2)
        pred = g[['prediction_first', 'prediction_second']].to_numpy(dtype=int).ravel()
        blocks_a.append(np.bincount(4*y + pred, minlength=16).reshape(4, 4))
        blocks_b.append(np.bincount(4*g.label + g.prediction, minlength=16).reshape(4, 4))
    a, b = np.array(blocks_a), np.array(blocks_b)
    by_class = [np.flatnonzero(np.array(labels) == i) for i in range(4)]
    require(all(len(g) for g in by_class), 'All four classes need matched sources')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    def f1(c):
        den = c.sum(-1) + c.sum(-2)
        return np.divide(2*np.diagonal(c, axis1=-2, axis2=-1), den,
                         out=np.zeros_like(den, dtype=float), where=den != 0)
    draws = np.concatenate([rng.choice(g, size=(BOOTSTRAP_REPEATS, len(g)), replace=True)
                            for g in by_class], axis=1)
    deltas = np.concatenate([100*(f1(b[ix].sum(1)) - f1(a[ix].sum(1)))
                             for ix in np.array_split(draws, 16)])
    point = 100*(f1(b.sum(0)) - f1(a.sum(0)))
    return {name: {'delta_f1_pp': float(value),
                  'ci95_low_pp': float(np.quantile(sample, .025)),
                  'ci95_high_pp': float(np.quantile(sample, .975))}
            for name, value, sample in [('macro', point.mean(), deltas.mean(1))]
            + [(name, point[i], deltas[:, i]) for i, name in enumerate(CLASSES)]}


def analyze(predictions):
    pairs, excluded, matched = pair_windows(predictions)
    results = {'all_validation_single_window': metrics(predictions.label, predictions.prediction),
               'matched_single_window': metrics(matched.label, matched.prediction),
               'two_window': metrics(pairs.label, pairs.prediction)}
    intervals = paired_intervals(pairs)
    rows = [{'endpoint': name, 'unit': 'pair' if name == 'two_window' else 'window',
             'examples': sum(v['support'] for v in m['per_class'].values()),
             'macro_f1_pct': 100*m['macro_f1'], 'accuracy_pct': 100*m['accuracy']}
            for name, m in results.items()]
    per_class = [{'endpoint': endpoint, 'species': name, **m}
                 for endpoint, record in results.items() for name, m in record['per_class'].items()]
    require(np.allclose(pairs.support_seconds, 1.935), 'Unexpected bounded support')
    report = {'protocol_version': VERSION, 'split': 'validation', 'class_order': CLASSES,
              'training_executed': False, 'inference_executed': False,
              'feature_generation_executed': False, 'test_evaluated': False,
              'pairing': {'group': 'labeled_clip_id', 'order': 'example_index', 'windows': 2,
                          'stride_samples': 15360, 'window_context_samples': 15600,
                          'target_sample_rate': 16000, 'pair_support_seconds': 1.935,
                          'aggregation': 'arithmetic_mean_saved_softmax; argmax; ties lowest class index',
                          'nonoverlapping_pairs': True, 'shared_audio_between_pair_members_seconds': .015,
                          'odd_tail_policy': 'exclude_and_report', 'cross_clip_pairs': False},
              'population': {'original_windows': len(predictions), 'matched_windows': len(matched),
                             'pairs': len(pairs), 'excluded_windows': len(excluded),
                             'original_sources': predictions.name.nunique(), 'matched_sources': pairs.name.nunique(),
                             'clips': predictions.id.nunique()},
              'metrics': results, 'paired_source_bootstrap': {
                  'repeats': BOOTSTRAP_REPEATS, 'seed': BOOTSTRAP_SEED,
                  'stratified_by': 'source_label', 'intervals': intervals,
                  'limitations': 'Conditional on fixed predictions and validation selection; excludes seed/search uncertainty; assumes independent sources'},
              'interpretation': 'Two-window classification uses more audio and fewer decisions; compare only matched endpoints. Not an improvement of the one-window model or a TEST result.'}
    return report, pairs, excluded, pd.DataFrame(rows), pd.DataFrame(per_class)


def run(diagnostic_id=DEFAULT_ID, save=False):
    paths = ProjectPaths.from_env()
    target = paths.experiment(diagnostic_id)
    require(diagnostic_id != REFERENCE, 'Diagnostic cannot replace its reference')
    if save:
        require(not target.exists(), f'Output exists; read it or choose a new diagnostic ID: {target}')
        require_mutable_output(target)
    predictions, provenance, inputs = authenticate_reference(paths)
    report, pairs, excluded, summary, per_class = analyze(predictions)
    require(all(sha256(p) == h for p, h in inputs.items()), 'Input changed during analysis')
    report.update(diagnostic_id=diagnostic_id, provenance=provenance)
    if save:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(exist_ok=False)
        metadata = {'status': 'RUNNING', 'diagnostic_id': diagnostic_id, 'kind': 'saved_prediction_diagnostic',
                    'protocol_version': VERSION, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                    'implementation_sha256': sha256(Path(__file__)),
                    'python': platform.python_version(),
                    'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas')},
                    'training_executed': False, 'test_evaluated': False}
        try:
            metadata['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
            metadata['git_status'] = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO_ROOT, text=True)
            write_json(target / 'metadata.json', metadata)
            write_json(target / 'result.json', report)
            for name, frame in [('pairs.csv', pairs), ('excluded_windows.csv', excluded),
                                ('summary.csv', summary), ('per_class.csv', per_class)]:
                frame.to_csv(target / name, index=False)
            metadata.update(status='COMPLETE', completed_at_utc=datetime.now(timezone.utc).isoformat(),
                            artifacts_sha256={n: sha256(target / n) for n in
                                ['result.json', 'pairs.csv', 'excluded_windows.csv', 'summary.csv', 'per_class.csv']})
            write_json(target / 'metadata.json', metadata)
        except Exception as error:
            metadata.update(status='FAILED', error_type=type(error).__name__, error=str(error))
            write_json(target / 'metadata.json', metadata)
            raise
    return report, summary, per_class, target


def load_saved(diagnostic_id=DEFAULT_ID):
    target = ProjectPaths.from_env().experiment(diagnostic_id)
    meta = read_json(target / 'metadata.json')
    require(meta['status'] == 'COMPLETE' and meta['diagnostic_id'] == diagnostic_id
            and meta['protocol_version'] == VERSION and not meta['training_executed'], 'Incomplete/unknown diagnostic')
    expected = {'result.json', 'pairs.csv', 'excluded_windows.csv', 'summary.csv', 'per_class.csv'}
    require(set(meta['artifacts_sha256']) == expected, 'Diagnostic artifact inventory mismatch')
    for name, value in meta['artifacts_sha256'].items():
        require(sha256(target / name) == value, f'Diagnostic artifact changed: {name}')
    report = read_json(target / 'result.json')
    require(report['diagnostic_id'] == diagnostic_id and report['protocol_version'] == VERSION
            and report['test_evaluated'] is False and report['training_executed'] is False, 'Diagnostic identity mismatch')
    return report, pd.read_csv(target / 'summary.csv'), pd.read_csv(target / 'per_class.csv'), target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-id', default=DEFAULT_ID)
    parser.add_argument('--save', action='store_true', help='Reserve a new output folder; default is read-only preview')
    parser.add_argument('--read-saved', action='store_true')
    args = parser.parse_args()
    require(not (args.save and args.read_saved), 'Choose save or read-saved')
    report, summary, per_class, target = (load_saved(args.diagnostic_id) if args.read_saved
                                        else run(args.diagnostic_id, args.save))
    print(summary.to_string(index=False))
    print('Paired F1 delta (pp):', report['paired_source_bootstrap']['intervals']['macro'])
    print('Population:', report['population'])
    print('Saved at:' if args.save or args.read_saved else 'Preview; future output:', target)
