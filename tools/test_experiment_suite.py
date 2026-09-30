"""Exercise orchestration with COPIES of saved baseline fixtures, never training."""
import contextlib
import io
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from project_paths import ProjectPaths, REPO_ROOT
from cache_consumer import MANIFEST, read_json, sha256
from experiment_runner import run, run_plan, validate_config
from experiment_review import verify_experiment
import experiment_training as training


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.baseline = ProjectPaths(repo=REPO_ROOT).BASELINE_ROOT
        self.template = REPO_ROOT / 'configs/experiments/baseline_suite.json'
        self.config = read_json(self.template)
        self.temp = tempfile.TemporaryDirectory(prefix='suite_test_', dir=REPO_ROOT / '_local_only')
        self.addCleanup(self.temp.cleanup)
        self.storage = Path(self.temp.name)
        self.env = patch.dict(os.environ, {'EDGEAI_DATA_ROOT': str(self.storage)})
        self.env.start()
        self.addCleanup(self.env.stop)
        manifest = ProjectPaths.from_env().resolve(MANIFEST)
        manifest.parent.mkdir(parents=True)
        shutil.copyfile(REPO_ROOT / MANIFEST, manifest)
        self.calls = []
        self.original_globals = (training.OUTPUT, training.RUNS, training.SELECTED_FAMILIES, training.EXPERIMENT_ID)

    def copy_saved_fixture(self, branch):
        # Replace ONLY the training entry. All result/artifact validation remains real.
        self.calls.append((branch, training.SELECTED_FAMILIES))
        training.OUTPUT.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.baseline / 'protocol.json', training.OUTPUT / 'protocol.json')
        for model in training.SELECTED_FAMILIES:
            run_id = f'{branch}_{model}_seed42'
            shutil.copytree(self.baseline / 'runs' / run_id, training.RUNS / run_id)

    def execute(self, name='exp_test_suite', branch=None):
        with patch('experiment_runner.smoke', return_value={
            'runtime': {'gpu': 'MOCK: no GPU/training executed'}, 'cache_sha256': {}, 'split_sha256': 'fixture'}), \
             patch('experiment_runner.subprocess.check_output', side_effect=['test-commit', '']), \
             patch.object(training, 'run_branch', side_effect=branch or self.copy_saved_fixture), \
             patch.object(training, 'fit_validation_checkpoint', side_effect=AssertionError('Training forbidden in tests')), \
             contextlib.redirect_stdout(io.StringIO()):
            return run(self.template, experiment_id=name, description='Mock fixture test; no training')

    def test_plan_is_all_eight_unique_pairs_and_rejects_incomplete_or_duplicate_suites(self):
        plan = run_plan(self.config)
        self.assertEqual(len(plan), 8)
        self.assertEqual(len({item['run_id'] for item in plan}), 8)
        self.assertEqual([(p['frontend'], p['model']) for p in plan],
                         [(b, f) for b in ('03A', '03B') for f in training.FAMILIES])
        for field, values in [('frontends', ['03A']), ('frontends', ['03A', '03A']),
                              ('models', ['ds_cnn']), ('models', ['ds_cnn'] * 4)]:
            with self.subTest(field=field, values=values), self.assertRaises(ValueError):
                validate_config({**self.config, field: values})
        with self.assertRaises(ValueError):
            validate_config({**self.config, 'experiment_id': 'exp_../fixed'})

    def test_eight_run_round_trip_saved_provenance_read_only_review_and_tampering(self):
        template_hash = sha256(self.template)
        target = self.execute()
        self.assertEqual(target, self.storage / 'experiments/exp_test_suite')
        self.assertEqual(self.calls, [(b, (f,)) for b in ('03A', '03B') for f in training.FAMILIES])
        self.assertEqual(self.original_globals,
                         (training.OUTPUT, training.RUNS, training.SELECTED_FAMILIES, training.EXPERIMENT_ID))
        metadata = read_json(target / 'config/metadata.json')
        self.assertEqual((metadata['status'], metadata['expected_runs'], metadata['completed_runs']), ('COMPLETE', 8, 8))
        self.assertEqual(metadata['git_commit'], 'test-commit')
        self.assertEqual(metadata['configuration']['description'], 'Mock fixture test; no training')
        self.assertEqual(sha256(self.template), template_hash)
        before = {p.relative_to(target).as_posix(): sha256(p) for p in target.rglob('*') if p.is_file()}
        records = verify_experiment('exp_test_suite')
        self.assertEqual(len(records), 8)
        # Execute the actual 04C selection/display cell against the named group.
        import pandas as pd
        notebook = read_json(REPO_ROOT / 'notebooks/current/04C_compare_models.ipynb')
        selection = next(''.join(c['source']) for c in notebook['cells']
                         if c['cell_type'] == 'code' and 'EXPERIMENT_NAME = ""' in ''.join(c['source']))
        scope = {'PATHS': ProjectPaths.from_env(), 'BRANCHES': training.BRANCHES,
                 'FAMILIES': training.FAMILIES, 'read_json': read_json, 'pd': pd,
                 'display': lambda _: None}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(selection.replace('EXPERIMENT_NAME = ""', 'EXPERIMENT_NAME = "exp_test_suite"'), scope)
        self.assertTrue(scope['all_complete'])
        self.assertEqual(scope['RUNS'], target / 'results/runs')
        self.assertEqual(len(scope['comparison']), 8)
        after = {p.relative_to(target).as_posix(): sha256(p) for p in target.rglob('*') if p.is_file()}
        self.assertEqual(before, after)  # Reading results must not rewrite any file.
        self.assertFalse(read_json(target / 'results/comparison.json')['test_used_for_shortlist'])
        self.assertEqual(len(read_json(target / 'results/comparison.json')['experiments']), 8)
        self.assertEqual(len(list((target / 'checkpoints').glob('*/best_model.pt'))), 8)
        self.assertIn('[8/8]', (target / 'logs/console.log').read_text(encoding='utf-8'))
        with patch('experiment_runner.smoke') as smoke, self.assertRaises(FileExistsError):
            run(self.template, experiment_id='exp_test_suite')
        smoke.assert_not_called()
        # Tamper only a disposable copy: verifier must catch altered metrics/checkpoints.
        checkpoint = target / metadata['runs'][records[0]['run_id']]['selected_checkpoint']
        with checkpoint.open('ab') as stream:
            stream.write(b'fixture tampering')
        with self.assertRaisesRegex(RuntimeError, 'Checkpoint copy changed'):
            verify_experiment('exp_test_suite')

    def test_failure_preserves_completed_runs_and_restores_output_globals(self):
        def fail_third(branch):
            if len(self.calls) == 2:
                raise RuntimeError('mock failure before run three')
            self.copy_saved_fixture(branch)
        with self.assertRaisesRegex(RuntimeError, 'mock failure'):
            self.execute(branch=fail_third)
        target = ProjectPaths.from_env().experiment('exp_test_suite')
        metadata = read_json(target / 'config/metadata.json')
        self.assertEqual((metadata['status'], metadata['completed_runs']), ('FAILED', 2))
        self.assertEqual(metadata['runs']['03A_tc_resnet8_seed42']['status'], 'FAILED')
        self.assertEqual(metadata['runs']['03B_small_cnn_seed42']['status'], 'PENDING')
        self.assertTrue((target / 'results/runs/03A_ds_cnn_seed42/result.json').is_file())
        self.assertFalse((target / 'results/comparison.csv').exists())
        self.assertEqual(self.original_globals,
                         (training.OUTPUT, training.RUNS, training.SELECTED_FAMILIES, training.EXPERIMENT_ID))
        with self.assertRaisesRegex(RuntimeError, 'Experiment is FAILED: 2/8'):
            verify_experiment('exp_test_suite')

    def test_interrupt_is_recorded_and_old_experiment_stays_reserved(self):
        with self.assertRaises(KeyboardInterrupt):
            self.execute(branch=lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
        metadata = read_json(ProjectPaths.from_env().experiment('exp_test_suite') / 'config/metadata.json')
        self.assertEqual(metadata['status'], 'INTERRUPTED')
        self.assertEqual(metadata['completed_runs'], 0)
        self.assertEqual(self.original_globals,
                         (training.OUTPUT, training.RUNS, training.SELECTED_FAMILIES, training.EXPERIMENT_ID))

    def test_dirty_checkout_is_rejected_before_smoke_or_any_training(self):
        with patch('experiment_runner.subprocess.check_output', side_effect=['commit', ' M tools/model_zoo.py']), \
             patch('experiment_runner.smoke') as smoke, patch.object(training, 'run_branch') as branch:
            with self.assertRaisesRegex(RuntimeError, 'Commit source/config changes'):
                run(self.template, experiment_id='exp_dirty')
            smoke.assert_not_called()
            branch.assert_not_called()
        self.assertFalse(ProjectPaths.from_env().experiment('exp_dirty').exists())

    def test_notebook_training_control_routes_named_suite_and_default_is_disabled(self):
        notebook = read_json(REPO_ROOT / 'notebooks/current/04D_experiments.ipynb')
        source = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and 'RUN_TRAINING = False' in ''.join(c['source']))
        scope = {'paths': ProjectPaths.from_env(), 'repo': REPO_ROOT,
                 'sys': __import__('sys'), 'subprocess': __import__('subprocess')}
        with patch('subprocess.run') as command, contextlib.redirect_stdout(io.StringIO()):
            exec(source, scope)
            command.assert_not_called()
            exec(source.replace('RUN_TRAINING = False', 'RUN_TRAINING = True'), scope)
            args = command.call_args.args[0]
            self.assertIn('configs/experiments/baseline_suite.json', args)
            self.assertEqual(args[args.index('--experiment-id') + 1], scope['EXPERIMENT_NAME'])
            self.assertEqual(args[args.index('--description') + 1], scope['EXPERIMENT_DESCRIPTION'])


if __name__ == '__main__':
    unittest.main()
