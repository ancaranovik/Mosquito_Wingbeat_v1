"""Protocol v4 power weights, routing and artifact checks; no scientific training or inference.

Reuse v2 synthetic fixtures with scoped module bindings; baseline/v2/v3 engines stay intact.
"""
import contextlib
from copy import deepcopy
import io
import re
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_experiment_v2 as fixture
import experiment_training_v2 as v2
import experiment_training_v4 as v4
from cache_consumer import CLASSES, COUNTS, TRAIN_COUNTS, digest_json, read_json, sha256, write_json
from experiment_runner import run, run_plan, training_module, validate_config
from experiment_review import compare_validation_to_reference
from project_paths import ProjectPaths, REPO_ROOT


class V4Tests(fixture.V2Tests):
    def setUp(self):
        binding = patch.object(fixture, 'v2', v4)
        binding.start()
        self.addCleanup(binding.stop)
        super().setUp()
        config = patch.object(v4, 'CONFIG', v4.effective_config({'class_weight_policy': 'power_inverse_frequency', 'class_weight_exponent': 0.75}))
        config.start()
        self.addCleanup(config.stop)
        self.template = self.storage / 'v4_template.json'
        template = read_json(REPO_ROOT / 'configs/experiments/power075_v4.json')
        template['training_overrides']['input_normalization'] = 'none'  # Synthetic fixtures have no real feature tensor.
        write_json(self.template, template)

    def write_synthetic_run(self, branch):
        super().write_synthetic_run(branch)
        protocol = read_json(v4.OUTPUT / 'protocol.json')
        accepted = deepcopy(protocol['class_weights'])
        protocol['accepted_class_weights'] = accepted
        protocol['class_weights'] = v4.effective_class_weights({'class_weights': accepted})
        protocol['implementation_sha256'].update({name: sha256(REPO_ROOT / name) for name in
            ('tools/experiment_training_v4.py', 'tools/experiment_training_v3.py', 'tools/experiment_training_v2.py', 'tools/experiment_runner.py')})
        write_json(v4.OUTPUT / 'protocol.json', protocol)
        for family in v4.SELECTED_FAMILIES:
            directory = v4.RUNS / f'{branch}_{family}_seed42'
            record = read_json(directory / 'result.json')
            record.update(class_weights=protocol['class_weights'], accepted_class_weights=accepted,
                          protocol_sha256=digest_json(protocol))
            write_json(directory / 'run_spec.json', {'protocol': protocol})
            write_json(directory / 'class_weights.json', v4.class_weight_record(v4.CONFIG, accepted))
            files = set(record['artifacts_sha256']) | {'class_weights.json'}
            record['artifacts_sha256'] = {name: sha256(directory / name) for name in files}
            write_json(directory / 'result.json', record)

    def test_power_weights_use_frozen_train_only_and_leave_evidence_unchanged(self):
        accepted = {s: COUNTS['train']/(4*n) for s,n in zip(CLASSES, TRAIN_COUNTS)}
        evidence = {'class_weights': deepcopy(accepted), 'other_evidence': {'keep': True}}
        before = deepcopy(evidence)
        weights = v4.effective_class_weights(evidence, 'power_inverse_frequency', 0.75)
        self.assertEqual(weights, {s: value**0.75 for s,value in accepted.items()})
        self.assertEqual(evidence, before)
        self.assertEqual(v4.effective_class_weights(evidence, exponent=1.0), accepted)
        invalid = deepcopy(evidence)
        invalid['class_weights'][CLASSES[2]] += 0.01
        with self.assertRaisesRegex(RuntimeError, 'Accepted TRAIN'):
            v4.effective_class_weights(invalid)
        original = v2.effective_config({'input_normalization': 'train_global_zscore'})
        current = v4.effective_config(read_json(REPO_ROOT / 'configs/experiments/power075_v4.json')['training_overrides'])
        differences = {k for k in original.keys() | current.keys() if original.get(k) != current.get(k)}
        self.assertEqual(differences, {'protocol_version', 'class_weight_policy', 'class_weight_formula', 'class_weight_exponent'})
        self.assertFalse(current['evaluate_test'])
        for policy in ('uniform', True, None, []):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                v4.effective_config({'class_weight_policy': policy})
        with self.assertRaises(ValueError):
            v2.effective_config({'class_weight_policy': 'power_inverse_frequency'})
        with self.assertRaises(ValueError):
            training_module({'protocol_version': 'unknown'})

    def test_real_branch_wiring_has_no_test_loader_and_authenticates_outputs(self):
        # The inherited branch test mocks CUDA transport, checkpoint fitting and all evaluation.
        # Capture the actual tensor construction to prove power weights reach the branch.
        with patch('torch.tensor', wraps=torch.tensor) as tensor:
            super().test_real_branch_wiring_has_no_test_loader_and_authenticates_outputs()
        accepted = {s: COUNTS['train']/(4*n) for s,n in zip(CLASSES, TRAIN_COUNTS)}
        expected = [value**0.75 for value in accepted.values()]
        calls = [call for call in tensor.call_args_list if call.args and isinstance(call.args[0], list)
                 and len(call.args[0]) == 4]
        self.assertTrue(any(np.allclose(call.args[0], expected, rtol=0, atol=1e-12) for call in calls))
        directory = ProjectPaths.from_env().experiment('exp_branch_probe') / 'results/runs/03A_ds_cnn_seed42'
        artifact = read_json(directory / 'class_weights.json')
        self.assertEqual(artifact['effective_weights'], dict(zip(CLASSES, expected)))
        self.assertEqual(artifact['fit_split'], 'train')
        self.assertEqual(artifact['exponent'], 0.75)

    def test_v4_can_compare_to_v2_v3_and_reject_synchronized_weight_tampering(self):
        target = self.execute_suite('exp_v4_cross')
        with patch.object(fixture, 'v2', v2), \
             patch('experiment_runner.smoke', return_value={'runtime': {'gpu': 'MOCK'}, 'cache_sha256': {}, 'split_sha256': 'MOCK'}), \
             patch('experiment_runner.subprocess.check_output', side_effect=['mock-commit', '']), \
             patch.object(v2, 'run_branch', side_effect=lambda branch: fixture.V2Tests.write_synthetic_run(self, branch)), \
             patch.object(v2, 'fit_validation_checkpoint', side_effect=AssertionError('Training forbidden')), \
             patch.object(v2, 'save_evaluation', side_effect=AssertionError('Inference forbidden')), \
             contextlib.redirect_stdout(io.StringIO()):
            run(REPO_ROOT / 'configs/experiments/control_v2.json', experiment_id='exp_v2_cross')
        table = compare_validation_to_reference('exp_v4_cross', 'exp_v2_cross')
        self.assertEqual(len(table), 8)
        self.assertTrue(table['Delta val F1 (percentage points)'].eq(0).all())
        legacy = __import__('experiment_training_v3')
        legacy_config = read_json(REPO_ROOT / 'configs/experiments/sqrt_weights_v3.json')
        legacy_config['training_overrides']['input_normalization'] = 'none'
        legacy_template = self.storage / 'legacy_v3_template.json'
        write_json(legacy_template, legacy_config)
        with patch.object(fixture, 'v2', legacy), \
             patch(__name__ + '.v4', legacy), \
             patch.object(legacy, 'CONFIG', legacy.effective_config(legacy_config['training_overrides'])), \
             patch('experiment_runner.smoke', return_value={'runtime': {'gpu': 'MOCK'}, 'cache_sha256': {}, 'split_sha256': 'MOCK'}), \
             patch('experiment_runner.subprocess.check_output', side_effect=['mock-commit', '']), \
             patch.object(legacy, 'run_branch', side_effect=self.write_synthetic_run), \
             patch.object(legacy, 'fit_validation_checkpoint', side_effect=AssertionError('Training forbidden')), \
             patch.object(legacy, 'save_evaluation', side_effect=AssertionError('Inference forbidden')), \
             contextlib.redirect_stdout(io.StringIO()):
            run(legacy_template, experiment_id='exp_v3_cross')
        table = compare_validation_to_reference('exp_v4_cross', 'exp_v3_cross')
        self.assertEqual(len(table), 8)
        self.assertTrue(table['Delta val F1 (percentage points)'].eq(0).all())
        directory = target / 'results/runs/03A_ds_cnn_seed42'
        protocol = read_json(target / 'results/protocol.json')
        record = read_json(directory / 'result.json')
        protocol['class_weights'][CLASSES[2]] += 1
        record['class_weights'] = deepcopy(protocol['class_weights'])
        record['protocol_sha256'] = digest_json(protocol)
        with patch.object(v4, 'OUTPUT', target / 'results'), self.assertRaisesRegex(RuntimeError, 'Effective class weights'):
            v4.validate_saved_run(record, directory, protocol)

    def test_stale_kernel_runner_stops_preview_before_training(self):
        import experiment_runner
        notebook = read_json(REPO_ROOT / 'notebooks/current/04D_experiments.ipynb')
        source = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and 'EXPERIMENT_CONFIG =' in ''.join(c['source']))
        scope = {'paths': ProjectPaths.from_env(), 'repo': REPO_ROOT,
                 'sys': __import__('sys'), 'subprocess': __import__('subprocess')}
        with patch.object(experiment_runner, 'VERSIONED_PROTOCOLS', ('stage04_experiment_v2', 'stage04_experiment_v3')), \
             patch('subprocess.run') as command, \
             self.assertRaisesRegex(RuntimeError, 'restart the kernel'):
            exec(source.replace('RUN_TRAINING = False', 'RUN_TRAINING = True'), scope)
        command.assert_not_called()
        self.assertFalse(ProjectPaths.from_env().experiment('exp_007_power075').exists())

    def test_unknown_protocol_reports_update_before_legacy_schema_error(self):
        config = read_json(REPO_ROOT / 'configs/experiments/power075_v4.json')
        config['protocol_version'] = 'stage04_experiment_unknown'
        with self.assertRaisesRegex(ValueError, 'Unsupported experiment protocol_version'):
            run_plan(config)

    def test_actual_notebook_preview_never_trains_and_routes_v4(self):
        notebook = read_json(REPO_ROOT / 'notebooks/current/04D_experiments.ipynb')
        source = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and 'EXPERIMENT_CONFIG =' in ''.join(c['source']))
        scope = {'paths': ProjectPaths.from_env(), 'repo': REPO_ROOT,
                 'sys': __import__('sys'), 'subprocess': __import__('subprocess')}
        with patch('subprocess.run') as command, contextlib.redirect_stdout(io.StringIO()):
            exec(source, scope)
            command.assert_not_called()
            self.assertEqual(scope['effective']['class_weight_policy'], 'power_inverse_frequency')
            self.assertEqual(scope['effective']['learning_rate'], 0.001)
            self.assertEqual(scope['effective']['class_weight_exponent'], 0.75)
            self.assertEqual(scope['config_preview']['reference_experiment_id'], 'exp_004_train_norm')
            self.assertIs(scope['engine'], v4)
            self.assertEqual(len(scope['plan']), 8)
            exec(source.replace('RUN_TRAINING = False', 'RUN_TRAINING = True'), scope)
            args = command.call_args.args[0]
            self.assertEqual(args[args.index('--config') + 1], 'configs/experiments/power075_v4.json')
            self.assertEqual(args[args.index('--experiment-id') + 1], 'exp_007_power075')
        self.assertFalse(ProjectPaths.from_env().experiment('exp_007_power075').exists())


    def test_exponent_validation_and_endpoint_controls(self):
        accepted = {s: COUNTS['train']/(4*n) for s,n in zip(CLASSES, TRAIN_COUNTS)}
        for exponent in (True, False, None, '0.75', [], float('nan'), float('inf'), -0.01, 1.01):
            with self.subTest(exponent=exponent), self.assertRaises(ValueError):
                v4.effective_config({'class_weight_exponent': exponent})
            if exponent is not None:  # None is the helper's documented ambient-default sentinel.
                with self.subTest(weights_exponent=exponent), self.assertRaises(ValueError):
                    v4.effective_class_weights({'class_weights': accepted}, exponent=exponent)
        self.assertEqual(v4.effective_class_weights({'class_weights': accepted}, exponent=None),
                         v4.effective_class_weights({'class_weights': accepted}, exponent=0.75))
        self.assertEqual(v4.effective_class_weights({'class_weights': accepted}, exponent=0),
                         dict.fromkeys(CLASSES, 1.0))
        self.assertEqual(v4.effective_class_weights({'class_weights': accepted}, exponent=1), accepted)
        with self.assertRaises(ValueError):
            __import__('experiment_training_v3').effective_config({'class_weight_exponent': 0.75})
        with self.assertRaises(ValueError):
            v4.effective_config({'weight_decay': 0.01})

    def test_saved_exponent_not_ambient_default_and_artifact_tamper_rejected(self):
        from experiment_review import verify_experiment
        target = self.execute_suite('exp_v4_saved')
        with patch.object(v4, 'CONFIG', v4.effective_config({})):
            self.assertEqual(len(verify_experiment('exp_v4_saved')), 8)
        directory = target / 'results/runs/03A_ds_cnn_seed42'
        artifact = read_json(directory / 'class_weights.json')
        artifact['exponent'] = 0.5
        write_json(directory / 'class_weights.json', artifact)
        record = read_json(directory / 'result.json')
        record['artifacts_sha256']['class_weights.json'] = sha256(directory / 'class_weights.json')
        protocol = read_json(target / 'results/protocol.json')
        with patch.object(v4, 'OUTPUT', target / 'results'), self.assertRaisesRegex(RuntimeError, 'Class-weight provenance'):
            v4.validate_saved_run(record, directory, protocol)

    def test_04c_two_references_and_no_self_comparison(self):
        notebook = read_json(REPO_ROOT / 'notebooks/current/04C_compare_models.ipynb')
        source = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and 'ADDITIONAL_REFERENCE_EXPERIMENT_NAME =' in ''.join(c['source']))
        for name in ('exp_004_train_norm', 'exp_006_sqrt_weights'):
            path = ProjectPaths.from_env().experiment(name) / 'config/metadata.json'
            write_json(path, {})
        scope = {'PATHS': ProjectPaths.from_env(), 'all_complete': True,
                 'EXPERIMENT_NAME': 'exp_007_power075',
                 'metadata': {'configuration': {'reference_experiment_id': 'exp_004_train_norm'}},
                 'display': lambda value: None}
        with patch('experiment_review.compare_validation_to_reference') as compare, contextlib.redirect_stdout(io.StringIO()):
            exec(source, scope)
            self.assertEqual([c.args for c in compare.call_args_list],
                [('exp_007_power075', 'exp_004_train_norm'), ('exp_007_power075', 'exp_006_sqrt_weights')])
            compare.reset_mock()
            scope['EXPERIMENT_NAME'] = 'exp_006_sqrt_weights'
            exec(source, scope)
            compare.assert_called_once_with('exp_006_sqrt_weights', 'exp_004_train_norm')

if __name__ == '__main__':
    unittest.main()
