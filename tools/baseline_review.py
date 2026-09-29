"""Verify and display saved baseline evidence without training or rewriting it."""
import json
import pandas as pd
from project_paths import ProjectPaths, REPO_ROOT
from cache_consumer import sha256, read_json, require


def verify_baseline():
    root = ProjectPaths.from_env().BASELINE_ROOT
    pins = read_json(REPO_ROOT / 'configs/baseline/accepted_artifacts.json')
    for relative, expected in pins.items():
        require(sha256(root / relative) == expected, f'Baseline artifact changed: {relative}')
    import experiment_training as training
    previous = training.OUTPUT
    try:
        training.OUTPUT = root
        protocol = read_json(root / 'protocol.json')
        records = []
        for branch in training.BRANCHES:
            for family in training.FAMILIES:
                directory = root / 'runs' / f'{branch}_{family}_seed42'
                record = read_json(directory / 'result.json')
                training.validate_saved_run(record, directory, protocol)
                records.append(record)
        require(len(records) == 8, 'Expected all eight accepted runs')
        return records
    finally:
        training.OUTPUT = previous


def compare_saved_results():
    import experiment_training as training
    return pd.DataFrame([training.comparison_row(r) for r in verify_baseline()])


if __name__ == '__main__':
    print(compare_saved_results().to_string(index=False))
