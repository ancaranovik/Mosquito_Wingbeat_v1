"""Engineering checks only. No mosquito classifier is trained by this suite."""
from pathlib import Path
import inspect
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import json

import stage04_data as data
# These pre-portability unit tests target the preserved historical CPU protocol.
# Current storage/runner safety is covered separately by test_portability.py.
import importlib.util
_legacy = Path(__file__).resolve().parents[1] / "archive/portability_originals/tools/training_common.py"
_spec = importlib.util.spec_from_file_location("accepted_cpu_training_tests", _legacy)
training = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(training)
from model_zoo import FAMILIES, build_model, architecture_config


class Stage04Tests(unittest.TestCase):
    def test_notebook_serialization_does_not_change_scientific_ast(self):
        path = data.ROOT / "notebooks/current/00_prepare_dataset.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        mutated = json.loads(json.dumps(notebook))
        for i, cell in enumerate(mutated["cells"]):
            cell["id"] = f"mutated-{i}"
            cell.setdefault("metadata", {})["serialization_test"] = True
            if cell["cell_type"] == "code":
                cell["execution_count"] = 999
                cell["outputs"] = [{"output_type": "stream", "text": ["irrelevant\n"]}]
        def fingerprint(nb):
            result = {}
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    for node in __import__("ast").parse("".join(cell["source"])).body:
                        if isinstance(node, (__import__("ast").FunctionDef, __import__("ast").AsyncFunctionDef)):
                            result[node.name] = __import__("hashlib").sha256(
                                __import__("ast").dump(node, include_attributes=False).encode()).hexdigest()
            return result
        self.assertEqual(fingerprint(notebook), fingerprint(mutated))

    def test_real_gates_ignore_notebook_outputs_and_serialization(self):
        original = data.read_json
        def changed_serialization(path):
            value = original(path)
            if Path(path).suffix == ".ipynb":
                for i, cell in enumerate(value["cells"]):
                    cell["id"] = f"audit-{i}"
                    cell["metadata"] = {"audit_only": True}
                    if cell["cell_type"] == "code":
                        cell["execution_count"] = 999
                        cell["outputs"] = []
                        cell["source"] = ["# Serialization-only audit fixture\n", *cell["source"]]
            return value
        with patch.object(data, "read_json", side_effect=changed_serialization):
            data.check_frozen_lock()

    def test_all_eight_shapes_gradients_and_checkpoint_roundtrip(self):
        for family in FAMILIES:
            for width in (64, 40):
                with self.subTest(family=family, width=width):
                    training.set_seeds()
                    model = build_model(family, width)
                    x = torch.linspace(-1, 1, 4 * 96 * width).reshape(4, 1, 96, width)
                    logits = model(x)
                    self.assertEqual(tuple(logits.shape), (4, 4))
                    loss = torch.nn.functional.cross_entropy(logits, torch.arange(4))
                    loss.backward()  # no parameter update / no real data
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                    model.eval()
                    with torch.no_grad():
                        expected = model(x)
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / "model.pt"
                        torch.save(model.state_dict(), path)
                        restored = build_model(family, width)
                        restored.load_state_dict(torch.load(path, weights_only=True)); restored.eval()
                        with torch.no_grad():
                            self.assertTrue(torch.equal(expected, restored(x)))

    def test_same_design_and_only_tc_parameters_depend_on_width(self):
        for family in FAMILIES:
            a, b = (architecture_config(family, width) for width in (64, 40))
            a.pop("input_shape"); b.pop("input_shape")
            self.assertEqual(a, b)
            sizes = [sum(p.numel() for p in build_model(family, w).parameters()) for w in (64, 40)]
            self.assertEqual(sizes[0] - sizes[1], 1152 if family == "tc_resnet8" else 0)

    def test_repeated_initialization_and_shuffling(self):
        metadata = pd.DataFrame({"split": ["train"] * 17, "label": np.arange(17) % 4})
        features = np.zeros((17, 96, 40), dtype=np.float32)
        for i in range(17):
            features[i] = i
        orders = []
        states = []
        for _ in range(2):
            training.set_seeds()
            states.append(build_model("cnn_lstm", 40).state_dict())
            loader = training.make_loader(features, metadata, "train")
            orders.append([x[:, 0, 0, 0].tolist() for x, y in loader])
        self.assertEqual(orders[0], orders[1])
        self.assertTrue(all(torch.equal(states[0][k], states[1][k]) for k in states[0]))

    def test_metrics_hand_calculated_and_absent_prediction_class(self):
        metrics = training.classification_metrics([0, 1, 2, 3], [0, 0, 2, 2])
        self.assertEqual(metrics["accuracy"], .5)
        self.assertAlmostEqual(metrics["macro_precision"], .25)
        self.assertAlmostEqual(metrics["macro_recall"], .5)
        self.assertAlmostEqual(metrics["macro_f1"], 1/3)
        self.assertEqual(metrics["per_class"][data.CLASSES[1]]["precision"], 0)

    def test_source_mean_probability_not_majority_vote(self):
        metadata = pd.DataFrame({"name": ["one"] * 3, "species": [data.CLASSES[1]] * 3,
                                 "label": [1] * 3, "split": ["test"] * 3})
        probs = np.array([[.51, .49, 0, 0], [.51, .49, 0, 0], [0, 1., 0, 0]])
        metrics, sources = training.source_evaluation(metadata, probs)
        self.assertEqual(sources.prediction.tolist(), [1])
        self.assertEqual(metrics["accuracy"], 1)
        metadata.loc[0, "label"] = 0
        with self.assertRaisesRegex(RuntimeError, "conflicting"):
            training.source_evaluation(metadata, probs)

    def test_epoch_loss_global_weight_denominator(self):
        class Logits(torch.nn.Module):
            def forward(self, x):
                return x
        x = torch.tensor([[2., 0, 0, 0], [0, 0, 0, 0], [0, 0, 1., 0]])
        y = torch.tensor([0, 1, 2]); w = torch.tensor([.5, 2., 3., 1.])
        batches = [(x[:2], y[:2]), (x[2:], y[2:])]
        actual, _ = training.epoch_pass(Logits(), batches, w)
        expected = torch.nn.functional.cross_entropy(x, y, weight=w)
        self.assertAlmostEqual(actual["loss"], float(expected), places=6)

    def test_earliest_tie_patience_and_fixed_checkpoint_gate(self):
        train_loader, val_loader = object(), object()
        seen = []
        def fake_epoch(model, loader, weights, optimizer=None):
            self.assertIn(loader, (train_loader, val_loader))
            seen.append(loader)
            return {"loss": 1.0, "macro_f1": .25}, None
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            model = torch.nn.Linear(1, 4)
            with self.assertRaises(FileNotFoundError):
                training.load_fixed_checkpoint(model, directory)
            with patch.object(training, "epoch_pass", side_effect=fake_epoch):
                selection = training.fit_validation_checkpoint(model, train_loader, val_loader, torch.ones(4), directory)
            self.assertEqual(selection["best_epoch"], 1)
            self.assertEqual(selection["epochs_run"], 9)
            self.assertEqual(len(seen), 18)
            training.load_fixed_checkpoint(model, directory)
            with (directory / "best_model.pt").open("ab") as file:
                file.write(b"corruption")
            with self.assertRaisesRegex(RuntimeError, "model changed"):
                training.load_fixed_checkpoint(model, directory)
        self.assertNotIn("test_loader", inspect.signature(training.fit_validation_checkpoint).parameters)

    def test_corrupt_cache_and_reordered_metadata_fail(self):
        grid = pd.DataFrame({"example_uid": ["a", "b"], "id": [1, 2], "name": ["a", "b"],
                             "species": data.CLASSES[:2], "split": ["train", "test"]})
        for column in data.META_COLUMNS[5:]:
            grid[column] = [0, 1]
        identity = {"sha256": "test-only"}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            meta = data.cache_metadata(grid, "03A", identity).iloc[::-1]
            meta.to_csv(directory / "examples.csv", index=False)
            np.save(directory / "features.npy", np.zeros((2, 96, 64), dtype=np.float32))
            info = {"frontend_identity": identity, "frozen_lock_sha256": data.LOCK_SHA256,
                    "schema_version": 2, "scientific_lock_sha256": data.SCIENTIFIC_LOCK_SHA256,
                    "rows": 2, "dtype": "float32", "normalization": "none",
                    "software": {p: data.importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy")},
                    "sha256": {f: data.sha256(directory / f) for f in ("features.npy", "examples.csv")}}
            data.write_json(directory / "cache.json", info)
            with self.assertRaises(AssertionError):
                data.validate_cache(directory, grid, "03A", identity)
            with (directory / "examples.csv").open("a") as stream:
                stream.write("corrupt")
            with self.assertRaisesRegex(RuntimeError, "Cache bytes changed"):
                data.validate_cache(directory, grid, "03A", identity)

    def test_comparison_refuses_partial_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data.write_json(root / "reports/stage04/protocol.json", {})
            with patch.object(training, "ROOT", root), patch.object(training, "RUNS", root / "runs"):
                with self.assertRaisesRegex(RuntimeError, "requires all 8"):
                    training.compare_saved_results()
            self.assertFalse((root / "reports/stage04/comparison.csv").exists())

    def test_missing_protocol_clear_failure(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(training, "ROOT", Path(tmp)):
            with self.assertRaisesRegex(RuntimeError, "Missing Stage-04 protocol"):
                training.compare_saved_results()

    def test_stale_protocol_rejected_before_reading_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data.write_json(root / "reports/stage04/protocol.json", {"old": True})
            for b in data.BRANCHES:
                for f in FAMILIES:
                    data.write_json(root / "runs" / f"{b}_{f}_seed42/result.json", {})
            with patch.object(training, "ROOT", root), patch.object(training, "RUNS", root / "runs"), \
                 patch.object(training, "authenticate", return_value=(None, {})), \
                 patch.object(training, "protocol_record", return_value={"current": True}):
                with self.assertRaisesRegex(RuntimeError, "Stale protocol"):
                    training.compare_saved_results()

    def test_cache_permission_and_partial_errors(self):
        with patch.object(data, "read_json", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(RuntimeError, "Windows ACL or file lock"):
                data.validate_cache(Path("unused"), None, "03A", {})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "Incomplete Stage-04 cache"):
                data.validate_cache(tmp, None, "03A", {})
            with data.cache_staging(tmp, "03A") as folder:
                self.assertTrue(folder.is_dir())
                self.assertFalse((Path(tmp) / "03A").exists())
            self.assertTrue(folder.is_dir())  # failed/unpublished build preserved

    def test_executed_constants_and_loaders_are_gated(self):
        original = data.read_json
        notebook = data.BRANCHES["03A"]["notebook"]
        for old, new in [("LOG_OFFSET = 0.01", "LOG_OFFSET = 0.02"),
                         ("float(2**31)", "float(2**30)")]:
            mutated = original(data.ROOT / notebook)
            for cell in mutated["cells"]:
                cell["source"] = "".join(cell["source"]).replace(old, new).splitlines(keepends=True)
            def fake_read(path):
                return mutated if Path(path) == data.ROOT / notebook else original(path)
            with patch.object(data, "read_json", side_effect=fake_read):
                with self.assertRaisesRegex(RuntimeError, "executed definitions/constants changed"):
                    data.check_frozen_lock()

    def test_expected_parameter_counts(self):
        expected = {"small_cnn": [23668, 23668], "ds_cnn": [6244, 6244],
                    "tc_resnet8": [65940, 64788], "cnn_lstm": [13428, 13428]}
        for family, counts in expected.items():
            self.assertEqual([sum(p.numel() for p in build_model(family, w).parameters())
                              for w in (64, 40)], counts)

    def test_complete_synthetic_comparison_and_tampered_summary(self):
        # Fabricated results live only in a temporary fixture. No fit/optimizer.
        from pipeline_contract import MANIFEST
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grid = pd.DataFrame({"example_uid": [f"synthetic-{i}" for i in range(4)],
                                 "id": range(4), "name": [f"source-{i}" for i in range(4)],
                                 "species": data.CLASSES, "split": ["test"] * 4})
            for column in data.META_COLUMNS[5:]:
                grid[column] = 0
            (root / MANIFEST).parent.mkdir(parents=True)
            grid.to_csv(root / MANIFEST, index=False)
            meta = grid.copy(); meta["label"] = range(4)
            probs = np.full((4, 4), .25)
            metrics = dict(training.classification_metrics(meta.label, probs.argmax(axis=1)), loss=1.0)
            source_metrics, sources = training.source_evaluation(meta, probs)
            protocol = {"training_configuration": training.CONFIG, "class_weights": {},
                        "software": {}, "frontends": {b: {} for b in data.BRANCHES}}
            data.write_json(root / "reports/stage04/protocol.json", protocol)
            records = []
            for b in data.BRANCHES:
                for f in FAMILIES:
                    run_id = f"{b}_{f}_seed42"
                    directory = root / "runs" / run_id
                    directory.mkdir(parents=True)
                    model = build_model(f, data.BRANCHES[b]["shape"][1])
                    torch.save(model.state_dict(), directory / "best_model.pt")
                    history = [{"epoch": 1, "train": metrics, "validation": metrics}]
                    data.write_json(directory / "history.json", history)
                    selection = {"status": "VALIDATION_CHECKPOINT_FIXED", "test_used_for_selection": False,
                                 "best_epoch": 1, "epochs_run": 1, "selected_validation_metrics": metrics,
                                 "best_validation_loss": 1.0, "model_sha256": data.sha256(directory / "best_model.pt"),
                                 "history_sha256": data.sha256(directory / "history.json")}
                    data.write_json(directory / "selection.json", selection)
                    data.write_json(directory / "run_spec.json", {"protocol": protocol, "run_id": run_id})
                    predictions = meta.copy(); predictions["prediction"] = 0
                    for i in range(4): predictions[f"p{i}"] = probs[:, i]
                    predictions.to_csv(directory / "test_predictions.csv", index=False)
                    sources.to_csv(directory / "source_predictions.csv", index=False)
                    record = dict(selection, status="COMPLETE", run_id=run_id, frontend=b,
                                  model_family=f, seed=42, protocol_sha256=data.digest_json(protocol),
                                  training_configuration=training.CONFIG, class_weights={}, class_order=data.CLASSES,
                                  counts=data.COUNTS, software={}, frontend_identity={},
                                  architecture=architecture_config(f, data.BRANCHES[b]["shape"][1]),
                                  parameter_count=sum(p.numel() for p in model.parameters()),
                                  model_file_path=(directory / "best_model.pt").relative_to(root).as_posix(),
                                  serialized_model_bytes=(directory / "best_model.pt").stat().st_size,
                                  final_test_metrics=metrics, source_test_metrics=source_metrics,
                                  artifacts_sha256={p.name: data.sha256(p) for p in directory.iterdir()})
                    data.write_json(directory / "result.json", record)
                    records.append((record, directory))
            with patch.object(training, "ROOT", root), patch.object(training, "RUNS", root / "runs"), \
                 patch.object(training, "authenticate", return_value=(grid, {})), \
                 patch.object(training, "protocol_record", return_value=protocol):
                self.assertEqual(len(training.compare_saved_results()), 8)
                record, directory = records[0]
                record = json.loads(json.dumps(record))
                record["final_test_metrics"]["accuracy"] = .99
                with self.assertRaisesRegex(RuntimeError, "Saved TEST metrics mismatch"):
                    training.validate_saved_run(record, directory, protocol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
