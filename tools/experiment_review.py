"""Read and authenticate a named Drive experiment; never train or rewrite results."""
from cache_consumer import read_json, sha256, require
from project_paths import ProjectPaths
from experiment_runner import run_plan, validate_config, training_module, VERSIONED_PROTOCOLS


def comparison_summary(records):
    import pandas as pd
    training = training_module(records[0])
    table = pd.DataFrame([training.comparison_row(record) for record in records])
    shortlist = sorted(records, key=lambda r: (-r['selected_validation_metrics']['macro_f1'],
                                               r['parameter_count'], r['run_id']))[:3]
    summary = {'interpretation': 'controlled comparison of frozen acoustic frontend/model configurations',
            'experiments': table.to_dict(orient='records'),
            'protocol_sha256': records[0]['protocol_sha256'],
            'shortlist_rule': records[0]['training_configuration']['shortlist'],
            'stage05_shortlist': [r['run_id'] for r in shortlist],
            'edge_winner_declared': False, 'test_used_for_shortlist': False,
            'limitation': 'Correlated windows are not independent mosquitoes; source grouping does not prove biological or domain independence'}
    if records[0].get('protocol_version') in VERSIONED_PROTOCOLS:
        summary.update(protocol_version=records[0]['protocol_version'],
                       test_evaluated=records[0]['test_evaluated'],
                       evaluation_policy=records[0]['training_configuration']['evaluation_policy'])
    if records[0].get('protocol_version') == 'stage04_experiment_v6':
        summary.update(interpretation='single DS-CNN/03A time-mask ablation; compare with the matching exp007 run',
                       run_scope='targeted_single_run', shortlist_is_cross_model_comparison=False)
    return summary, table


def verify_experiment(experiment_id, require_complete=True):
    import pandas as pd
    root = ProjectPaths.from_env().experiment(experiment_id)
    config = read_json(root / 'config/experiment.json')
    require(validate_config(config) == root, 'Experiment folder/config identity mismatch')
    training = training_module(config)
    plan = run_plan(config)
    metadata = read_json(root / 'config/metadata.json')
    require(metadata['experiment_id'] == experiment_id and metadata['configuration'] == config,
            'Experiment metadata/config mismatch')
    if require_complete:
        require(metadata['status'] == 'COMPLETE',
                f"Experiment is {metadata['status']}: {metadata['completed_runs']}/{len(plan)} completed; see logs/console.log")
    require(metadata['planned_runs'] == plan and metadata['expected_runs'] == len(plan)
            and metadata['completed_runs'] == len(plan), 'Incomplete experiment plan')
    expected_ids = {item['run_id'] for item in plan}
    require(set(metadata['runs']) == expected_ids, 'Run manifest differs from experiment plan')
    output = root / 'results'
    require({p.name for p in (output / 'runs').iterdir() if p.is_dir()} == expected_ids,
            'Missing or unexpected run directories')
    require(sha256(output / 'protocol.json') == metadata['protocol_sha256'], 'Experiment protocol changed')
    protocol = read_json(output / 'protocol.json')
    if config.get('protocol_version') in VERSIONED_PROTOCOLS:
        require(protocol.get('protocol_version') == config['protocol_version']
                and protocol['training_configuration'] == training.effective_config(
                    config['training_overrides'], config['evaluate_test']), 'Config/effective protocol mismatch')
    previous = training.OUTPUT
    try:
        training.OUTPUT = output
        records = []
        for item in plan:
            run_id = item['run_id']
            directory = output / 'runs' / run_id
            run_metadata = metadata['runs'][run_id]
            require(run_metadata['status'] == 'COMPLETE', f'Incomplete run: {run_id}')
            require(sha256(directory / 'result.json') == run_metadata['result_sha256'], f'Result changed: {run_id}')
            record = read_json(directory / 'result.json')
            training.validate_saved_run(record, directory, protocol)
            checkpoint = root / 'checkpoints' / run_id / 'best_model.pt'
            require(run_metadata['selected_checkpoint'] == checkpoint.relative_to(root).as_posix(),
                    f'Checkpoint path mismatch: {run_id}')
            require(sha256(checkpoint) == run_metadata['checkpoint_sha256'] == record['model_sha256'],
                    f'Checkpoint copy changed: {run_id}')
            display_metrics = record['final_test_metrics'] or record['selected_validation_metrics']
            confusion = display_metrics['confusion_matrix']
            if config.get('protocol_version') in VERSIONED_PROTOCOLS:
                confusion = {'split': 'test' if record['test_evaluated'] else 'validation',
                             'class_order': record['class_order'], 'matrix': confusion}
            require(read_json(directory / 'confusion_matrix.json') == confusion,
                    f'Confusion matrix changed: {run_id}')
            records.append(record)
    finally:
        training.OUTPUT = previous
    if require_complete:
        require(set(metadata['comparison_sha256']) == {'comparison.csv', 'comparison.json'}, 'Missing comparison hashes')
        for name, expected in metadata['comparison_sha256'].items():
            require(sha256(output / name) == expected, f'Comparison changed: {name}')
        summary, table = comparison_summary(records)
        require(read_json(output / 'comparison.json') == summary, 'Comparison does not match authenticated runs')
        pd.testing.assert_frame_equal(pd.read_csv(output / 'comparison.csv'), table, check_dtype=False)
    return records


def compare_saved_results(experiment_id):
    return comparison_summary(verify_experiment(experiment_id))[1]


def compare_validation_to_reference(experiment_id, reference_id=None):
    """Compare authenticated, paired validation scores; never infer or consult TEST."""
    import pandas as pd
    records = verify_experiment(experiment_id)
    metadata = read_json(ProjectPaths.from_env().experiment(experiment_id) / 'config/metadata.json')
    reference_id = reference_id or metadata['configuration'].get('reference_experiment_id')
    require(reference_id is not None and reference_id != experiment_id, 'Choose a distinct reference experiment')
    reference = verify_experiment(reference_id)
    original = {r['run_id']: r for r in reference}
    current_ids = {r['run_id'] for r in records}
    if records[0].get('protocol_version') == 'stage04_experiment_v6':
        require(current_ids == {'03A_ds_cnn_seed42'} and current_ids <= set(original),
                'Missing matching targeted reference run')
    else:
        require(set(original) == current_ids, 'Reference has a different run plan')
    rows = []
    for record in records:
        old = original[record['run_id']]
        require(record['cache_sha256'] == old['cache_sha256'] and record['counts'] == old['counts']
                and record['class_order'] == old['class_order'] and record['architecture'] == old['architecture'],
                'Reference differs in data, split, labels or architecture')
        before = old['selected_validation_metrics']['macro_f1']
        after = record['selected_validation_metrics']['macro_f1']
        rows.append({'Run': record['run_id'], 'Reference': reference_id, 'Experiment': experiment_id,
                     'Reference val Macro-F1': before, 'Current val Macro-F1': after,
                     'Delta val F1 (percentage points)': 100 * (after - before),
                     'Reference selected epoch': old['best_epoch'], 'Current selected epoch': record['best_epoch'],
                     'Same runtime': old['software'] == record['software'],
                     'Reference criterion': old['criterion'], 'Current criterion': record['criterion']})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment_id')
    print(compare_saved_results(parser.parse_args().experiment_id).to_string(index=False))
