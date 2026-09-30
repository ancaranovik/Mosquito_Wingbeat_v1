"""Read and authenticate a named Drive experiment; never train or rewrite results."""
from cache_consumer import read_json, sha256, require
from project_paths import ProjectPaths
from experiment_runner import run_plan, validate_config


def comparison_summary(records):
    import pandas as pd
    import experiment_training as training
    table = pd.DataFrame([training.comparison_row(record) for record in records])
    shortlist = sorted(records, key=lambda r: (-r['selected_validation_metrics']['macro_f1'],
                                               r['parameter_count'], r['run_id']))[:3]
    return {'interpretation': 'controlled comparison of frozen acoustic frontend/model configurations',
            'experiments': table.to_dict(orient='records'),
            'protocol_sha256': records[0]['protocol_sha256'],
            'shortlist_rule': training.CONFIG['shortlist'],
            'stage05_shortlist': [r['run_id'] for r in shortlist],
            'edge_winner_declared': False, 'test_used_for_shortlist': False,
            'limitation': 'Correlated windows are not independent mosquitoes; source grouping does not prove biological or domain independence'}, table


def verify_experiment(experiment_id, require_complete=True):
    import pandas as pd
    import experiment_training as training
    root = ProjectPaths.from_env().experiment(experiment_id)
    config = read_json(root / 'config/experiment.json')
    require(validate_config(config) == root, 'Experiment folder/config identity mismatch')
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
            require(read_json(directory / 'confusion_matrix.json') == record['final_test_metrics']['confusion_matrix'],
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment_id')
    print(compare_saved_results(parser.parse_args().experiment_id).to_string(index=False))
