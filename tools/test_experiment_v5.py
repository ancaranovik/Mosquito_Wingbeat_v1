"""Exp008 routing, Adam argument and artifact checks; scientific training is mocked."""
import ast
import contextlib
from copy import deepcopy
import io
import unittest
from unittest.mock import patch

import test_experiment_v2 as fixture
import experiment_training_v4 as v4
import experiment_training_v5 as v5
from cache_consumer import CLASSES, COUNTS, TRAIN_COUNTS, digest_json, read_json, sha256, write_json
from experiment_runner import run, run_plan, training_module, validate_config
from experiment_review import verify_experiment, compare_validation_to_reference
from project_paths import ProjectPaths, REPO_ROOT

TEMPLATE = REPO_ROOT / 'configs/experiments/power075_wd1e4_v5.json'


class V5Tests(fixture.V2Tests):
    def setUp(self):
        binding = patch.object(fixture, 'v2', v5)
        binding.start()
        self.addCleanup(binding.stop)
        super().setUp()
        actual = read_json(TEMPLATE)
        actual['training_overrides']['input_normalization'] = 'none'  # Synthetic artifact fixtures.
        config = patch.object(v5, 'CONFIG', v5.effective_config(actual['training_overrides']))
        config.start()
        self.addCleanup(config.stop)
        self.template = self.storage / 'v5_template.json'
        write_json(self.template, actual)

    def write_synthetic_run(self, branch):
        super().write_synthetic_run(branch)
        protocol = read_json(v5.OUTPUT / 'protocol.json')
        accepted = deepcopy(protocol['class_weights'])
        protocol['accepted_class_weights'] = accepted
        protocol['class_weights'] = v5.effective_class_weights({'class_weights': accepted})
        for family in v5.SELECTED_FAMILIES:
            directory = v5.RUNS / f'{branch}_{family}_seed42'
            record = read_json(directory / 'result.json')
            record.update(class_weights=protocol['class_weights'], accepted_class_weights=accepted,
                          protocol_sha256=digest_json(protocol))
            write_json(directory / 'run_spec.json', {'protocol': protocol})
            write_json(directory / 'class_weights.json', v5.class_weight_record(v5.CONFIG, accepted))
            files = set(record['artifacts_sha256']) | {'class_weights.json'}
            record['artifacts_sha256'] = {name: sha256(directory / name) for name in files}
            write_json(directory / 'result.json', record)
        write_json(v5.OUTPUT / 'protocol.json', protocol)

    def test_only_reviewed_overrides_and_no_test_by_default(self):
        old = read_json(REPO_ROOT / 'configs/experiments/power075_v4.json')
        new = read_json(TEMPLATE)
        before = v4.effective_config(old['training_overrides'], old['evaluate_test'])
        after = v5.effective_config(new['training_overrides'], new['evaluate_test'])
        changes = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
        self.assertEqual(changes, {'protocol_version', 'weight_decay'})
        self.assertEqual((before['weight_decay'], after['weight_decay']), (0, 1e-4))
        self.assertEqual(new['reference_experiment_id'], old['experiment_id'])
        self.assertEqual(run_plan(old), run_plan(new))
        self.assertEqual(len(run_plan(new)), 8)
        self.assertFalse(after['evaluate_test'])
        self.assertIs(training_module(new), v5)
        self.assertIs(training_module(old), v4)
        accepted = {s: COUNTS['train']/(4*n) for s, n in zip(CLASSES, TRAIN_COUNTS)}
        self.assertEqual(v5.class_weight_record(after, accepted), v4.class_weight_record(before, accepted))
        for value in (True, False, None, '0.0001', [], float('nan'), float('inf'), -1e-4, .0101):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_config({**new, 'training_overrides': {**new['training_overrides'], 'weight_decay': value}})
        for changes in ({'seed': 123}, {'dropout': .5}, {'optimizer': 'AdamW'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                v5.effective_config(changes)
        with self.assertRaises(ValueError):
            v5.effective_config({}, evaluate_test='False')
        with self.assertRaises(ValueError):
            v4.effective_config({'weight_decay': 1e-4})

    def test_shared_scientific_functions_unchanged_from_v4(self):
        def nodes(name):
            return {n.name: ast.dump(n, include_attributes=False) for n in
                    ast.parse((REPO_ROOT / 'tools' / name).read_text(encoding='utf-8')).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        old, new = nodes('experiment_training_v4.py'), nodes('experiment_training_v5.py')
        self.assertEqual(set(old), set(new))
        for name in set(old) - {'effective_config', 'protocol_record', 'validate_saved_run'}:
            self.assertEqual(old[name], new[name], name)

    def test_dual_checkpoint_selection_ties_and_unchanged_loss_stopping_without_training(self):
        from unittest.mock import Mock
        import torch
        directory = self.storage / 'optimizer_probe'
        directory.mkdir()
        config = {**v5.CONFIG, 'max_epochs': 1}
        metrics = {'loss': 1.0, 'macro_f1': .5}
        with patch.object(v5, 'CONFIG', config), patch('torch.optim.Adam', return_value=Mock()) as optimizer, \
             patch.object(v5.baseline, 'epoch_pass', return_value=(metrics, None)), \
             contextlib.redirect_stdout(io.StringIO()):
            v5.fit_validation_checkpoint(torch.nn.Linear(1, 1), 'TRAIN', 'VAL', None, directory)
        self.assertEqual(optimizer.call_args.kwargs['weight_decay'], 1e-4)
        self.assertEqual(optimizer.call_args.kwargs['lr'], .001)
        optimizer.return_value.step.assert_not_called()
        super().test_dual_checkpoint_selection_ties_and_unchanged_loss_stopping_without_training()

    def test_actual_notebook_preview_routes_v5_and_defaults_disabled(self):
        notebook = read_json(REPO_ROOT / 'notebooks/current/04D_experiments.ipynb')
        source = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and 'EXPERIMENT_CONFIG =' in ''.join(c['source']))
        scope = {'paths': ProjectPaths.from_env(), 'repo': REPO_ROOT,
                 'sys': __import__('sys'), 'subprocess': __import__('subprocess')}
        with patch('subprocess.run') as command, contextlib.redirect_stdout(io.StringIO()):
            exec(source, scope)
            command.assert_not_called()
            self.assertFalse(scope['RUN_TRAINING'])
            self.assertIs(scope['engine'], v5)
            self.assertEqual(scope['effective']['weight_decay'], 1e-4)
            self.assertEqual(scope['effective']['class_weight_exponent'], .75)
            self.assertEqual(scope['config_preview']['reference_experiment_id'], 'exp_007_power075')
            self.assertEqual(scope['EXPERIMENT_NAME'], 'exp_008_power075_wd1e4')
            exec(source.replace('RUN_TRAINING = False', 'RUN_TRAINING = True'), scope)
            args = command.call_args.args[0]
            self.assertEqual(args[args.index('--config') + 1], 'configs/experiments/power075_wd1e4_v5.json')
        self.assertFalse(ProjectPaths.from_env().experiment('exp_008_power075_wd1e4').exists())

    def test_stale_kernel_blocks_preview_even_with_training_true(self):
        import experiment_runner
        notebook = read_json(REPO_ROOT / 'notebooks/current/04D_experiments.ipynb')
        source = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and 'EXPERIMENT_CONFIG =' in ''.join(c['source']))
        scope = {'paths': ProjectPaths.from_env(), 'repo': REPO_ROOT,
                 'sys': __import__('sys'), 'subprocess': __import__('subprocess')}
        with patch.object(experiment_runner, 'VERSIONED_PROTOCOLS', ('stage04_experiment_v2', 'stage04_experiment_v3', 'stage04_experiment_v4')), \
             patch('subprocess.run') as command, self.assertRaisesRegex(RuntimeError, 'restart the kernel'):
            exec(source.replace('RUN_TRAINING = False', 'RUN_TRAINING = True'), scope)
        command.assert_not_called()

    def test_v5_compares_to_frozen_v4_and_saved_decay_not_ambient_default(self):
        target = self.execute_suite('exp_v5_cross')
        old = read_json(REPO_ROOT / 'configs/experiments/power075_v4.json')
        old['training_overrides']['input_normalization'] = 'none'
        template = self.storage / 'v4_template.json'
        write_json(template, old)
        with patch.object(fixture, 'v2', v4), \
             patch.object(v4, 'CONFIG', v4.effective_config(old['training_overrides'])), \
             patch('experiment_runner.smoke', return_value={'runtime': {'gpu': 'MOCK'}, 'cache_sha256': {}, 'split_sha256': 'MOCK'}), \
             patch('experiment_runner.subprocess.check_output', side_effect=['mock-commit', '']), \
             patch.object(v4, 'run_branch', side_effect=self.write_v4_synthetic), \
             patch.object(v4, 'fit_validation_checkpoint', side_effect=AssertionError('Training forbidden')), \
             patch.object(v4, 'save_evaluation', side_effect=AssertionError('Inference forbidden')), \
             contextlib.redirect_stdout(io.StringIO()):
            run(template, experiment_id='exp_v4_cross')
        with patch.object(v5, 'CONFIG', v5.effective_config({})):
            records = verify_experiment('exp_v5_cross')
        self.assertTrue(all(r['training_configuration']['weight_decay'] == 1e-4 for r in records))
        table = compare_validation_to_reference('exp_v5_cross', 'exp_v4_cross')
        self.assertEqual(len(table), 8)
        self.assertTrue(table['Delta val F1 (percentage points)'].eq(0).all())
        protocol = read_json(target / 'results/protocol.json')
        directory = target / 'results/runs/03A_ds_cnn_seed42'
        record = read_json(directory / 'result.json')
        record['training_configuration']['weight_decay'] = 0
        with patch.object(v5, 'OUTPUT', target / 'results'), self.assertRaisesRegex(RuntimeError, 'Run protocol mismatch'):
            v5.validate_saved_run(record, directory, protocol)

    def write_v4_synthetic(self, branch):
        # Same synthetic writer with the old engine bound, no execution of a model.
        with patch(__name__ + '.v5', v4):
            self.write_synthetic_run(branch)


if __name__ == '__main__':
    unittest.main()
