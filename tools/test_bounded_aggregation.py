"""Pair boundaries, matched metrics, source bootstrap and immutable output checks."""
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import bounded_aggregation as agg


def fixture():
    rows = []
    for label, species in enumerate(agg.CLASSES):
        for i in range(3):
            prob = np.full(4, .01)
            # First window wrong, second strong/correct: averaging recovers label.
            if i == 0:
                prob[label] = .45
                prob[(label+1) % 4] = .53
            else:
                prob[label] = .97
            rows.append({'example_uid': f'{label}_{i}', 'id': label, 'name': f'source{label}',
                         'species': species, 'split': 'validation', 'label': label,
                         'example_index': i, 'start_sample_16k': i*15360,
                         'reference_end_sample_16k': i*15360+15600,
                         'example_stride_samples': 15360, 'reference_context_samples': 15600,
                         'target_sample_rate': 16000, 'prediction': int(prob.argmax()),
                         **{f'p{j}': float(prob[j]) for j in range(4)}})
    return pd.DataFrame(rows)


class AggregationTests(unittest.TestCase):
    def test_matched_population_and_known_f1(self):
        report, pairs, excluded, summary, per_class = agg.analyze(fixture())
        self.assertEqual((len(pairs), len(excluded)), (4, 4))
        self.assertEqual(set(excluded.example_uid), {f'{i}_2' for i in range(4)})
        self.assertTrue(np.allclose(pairs.support_seconds, 1.935))
        self.assertEqual(report['metrics']['matched_single_window']['macro_f1'], .5)
        self.assertEqual(report['metrics']['two_window']['macro_f1'], 1.)
        self.assertEqual(report['paired_source_bootstrap']['intervals']['macro'],
                         {'delta_f1_pp': 50., 'ci95_low_pp': 50., 'ci95_high_pp': 50.})
        self.assertEqual([r['support'] for r in report['metrics']['matched_single_window']['per_class'].values()], [2]*4)
        self.assertEqual([r['support'] for r in report['metrics']['two_window']['per_class'].values()], [1]*4)

    def test_no_cross_clip_pair_even_same_source(self):
        a = fixture()
        b = a.copy()
        b['id'] += 10
        b.example_uid = 'other_' + b.example_uid
        pairs, tails, matched = agg.pair_windows(pd.concat([a,b], ignore_index=True))
        self.assertEqual((len(pairs), len(tails), len(matched)), (8,8,16))
        self.assertTrue(all((row.uid_first.startswith('other_')) == (row.uid_second.startswith('other_'))
                            for row in pairs.itertuples()))

    def test_sorting_is_deterministic(self):
        expected = agg.pair_windows(fixture())[0]
        actual = agg.pair_windows(fixture().sample(frac=1, random_state=5))[0]
        pd.testing.assert_frame_equal(expected, actual)

    def test_gap_wrong_context_and_mixed_labels_fail(self):
        for col, value in [('start_sample_16k', 40000), ('example_index', 4),
                           ('reference_end_sample_16k', 2), ('label', 2),
                           ('target_sample_rate', 8000), ('name', 'different')]:
            with self.subTest(col=col):
                f = fixture(); f.loc[1, col] = value
                with self.assertRaises(RuntimeError):agg.pair_windows(f)

    def test_test_split_and_duplicate_uid_fail(self):
        f = fixture(); f.loc[0,'split'] = 'test'
        with self.assertRaises(RuntimeError):agg.pair_windows(f)
        f = fixture(); f.loc[1,'example_uid'] = f.loc[0,'example_uid']
        with self.assertRaises(RuntimeError):agg.pair_windows(f)

    def test_fixed_four_class_metric_including_missing_prediction(self):
        m = agg.metrics([0,1,2,3], [0,0,0,0])
        self.assertEqual(m['macro_f1'], .1)
        self.assertEqual(m['per_class'][agg.CLASSES[0]]['precision'], .25)
        self.assertEqual(m['per_class'][agg.CLASSES[1]]['recall'], 0)

    def test_preview_writes_nothing_and_save_preserves_inputs(self):
        report, pairs, tails, summary, per_class = agg.analyze(fixture())
        with tempfile.TemporaryDirectory(dir=agg.REPO_ROOT/'_local_only') as tmp:
            with patch.dict(os.environ, {'EDGEAI_DATA_ROOT': tmp}):
                source = Path(tmp)/'reference.txt'; source.write_text('immutable')
                h = agg.sha256(source)
                with patch.object(agg, 'authenticate_reference', return_value=(fixture(), {}, {source: h})):
                    agg.run('exp_test_two', save=False)
                    self.assertFalse((Path(tmp)/'experiments').exists())
                    agg.run('exp_test_two', save=True)
                    loaded = agg.load_saved('exp_test_two')
                    self.assertEqual(loaded[0]['metrics'], report['metrics'])
                    self.assertEqual(agg.sha256(source), h)
                    with self.assertRaisesRegex(RuntimeError, 'Output exists'):agg.run('exp_test_two', save=True)
                    out = Path(tmp)/'experiments/exp_test_two/summary.csv'
                    out.write_text('tampered')
                    with self.assertRaisesRegex(RuntimeError, 'artifact changed'):agg.load_saved('exp_test_two')

    def test_input_changes_fail_before_output_reservation(self):
        with tempfile.TemporaryDirectory(dir=agg.REPO_ROOT/'_local_only') as tmp:
            source=Path(tmp)/'input.txt'; source.write_text('changed')
            with patch.dict(os.environ, {'EDGEAI_DATA_ROOT': tmp}), patch.object(
                    agg, 'authenticate_reference', return_value=(fixture(), {}, {source: '0'*64})):
                with self.assertRaisesRegex(RuntimeError, 'Input changed'):agg.run('exp_test_changed', save=True)
                self.assertFalse((Path(tmp)/'experiments').exists())


if __name__ == '__main__':
    unittest.main()
