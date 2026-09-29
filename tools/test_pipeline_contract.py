"""Failure-path tests for identity/geometry and audit persistence safeguards."""
from pathlib import Path
from project_paths import resolve_path
import json
import tempfile
import unittest

import pandas as pd
from pipeline_contract import validate_grid, write_gated_audit, require_passed_gates

ROOT = Path(__file__).resolve().parents[1]


class ContractFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = pd.read_csv(resolve_path(ROOT, "data/manifests/current/core_example_grid_v2_stride15360_ref15600_candidates.csv"))
        cls.split = pd.read_csv(resolve_path(ROOT, "data/manifests/current/core_single_4species_frozen_split.csv"))
        cls.report = json.loads((resolve_path(ROOT, "data/audits/current/core_example_grid_v2_stride15360_ref15600_audit.json")).read_text())

    def test_coordinate_change_with_same_count_is_rejected(self):
        grid = self.grid.copy()
        grid.loc[0, "start_sample_16k"] += 1
        with self.assertRaises(AssertionError):
            validate_grid(grid, self.split, self.report)

    def test_parent_label_change_is_rejected(self):
        grid = self.grid.copy()
        grid.loc[0, "species"] = "incorrect parent label"
        with self.assertRaises(AssertionError):
            validate_grid(grid, self.split, self.report)

    def test_missing_source_is_rejected(self):
        split = self.split.copy()
        split.loc[0, "name"] = pd.NA
        with self.assertRaises(AssertionError):
            validate_grid(self.grid, split, self.report)

    def test_failed_gate_preserves_existing_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            old = b'{"previous_evidence":true}\n'
            path.write_bytes(old)
            with self.assertRaises(AssertionError):
                write_gated_audit(path, {"final_gates": {"full_train_confirmation": False}})
            self.assertEqual(path.read_bytes(), old)

    def test_empty_gate_set_cannot_pass(self):
        with self.assertRaises(AssertionError):
            require_passed_gates({})

    def test_missing_audio_authentication_cannot_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            with self.assertRaises(AssertionError):
                write_gated_audit(path, {"final_gates": {"geometry": True},
                                        "authenticated_inputs": {"wav_files_authenticated": 0}})
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
