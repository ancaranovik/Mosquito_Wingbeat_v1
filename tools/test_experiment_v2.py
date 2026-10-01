"""Selection, storage, artifact and notebook checks with synthetic predictions; never train.

Optimizer construction and epoch passes are mocked in the selection test. The
eight-run test writes synthetic predictions over real frozen metadata and uses
fresh untrained state dictionaries; it never runs forward/backward or TEST.
"""
import contextlib
from copy import deepcopy
import io
import re
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

from cache_consumer import (CLASSES, COUNTS, MANIFEST, META_COLUMNS, SPLIT,
                            digest_json, read_json, sha256, write_json)
from experiment_runner import run, run_plan, validate_config
from experiment_review import verify_experiment, compare_validation_to_reference
from model_zoo import FAMILIES, architecture_config, build_model
from project_paths import ProjectPaths, REPO_ROOT
import experiment_training as baseline
import experiment_training_v2 as v2


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v2_test_", dir=REPO_ROOT / "_local_only")
        self.addCleanup(self.temp.cleanup)
        self.storage = Path(self.temp.name)
        env = patch.dict(os.environ, {"EDGEAI_DATA_ROOT": str(self.storage)})
        env.start()
        self.addCleanup(env.stop)
        config = patch.object(v2, "CONFIG", v2.effective_config({}))
        config.start()
        self.addCleanup(config.stop)
        self.template = REPO_ROOT / "configs/experiments/control_v2.json"
        for relative in (MANIFEST, SPLIT):
            target = ProjectPaths.from_env().resolve(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / relative, target)

    def test_only_reviewed_overrides_and_no_test_by_default(self):
        for filename in ("control_v2.json", "lr3e4_v2.json", "train_norm_v2.json"):
            config = read_json(REPO_ROOT / "configs/experiments" / filename)
            self.assertEqual(len(run_plan(config)), 8)
            self.assertFalse(config["evaluate_test"])
            effective = v2.effective_config(config["training_overrides"], config["evaluate_test"])
            self.assertEqual(effective["selection"], v2.SELECTION)
            self.assertEqual(effective["early_stopping_monitor"], baseline.CONFIG["selection"])
        control = v2.effective_config({})
        for key, value in baseline.CONFIG.items():
            if key != "selection":
                self.assertEqual(control[key], value, key)
        config = read_json(self.template)
        for changes in ({"learning_rate": float("nan")}, {"learning_rate": True},
                        {"learning_rate": 0}, {"input_normalization": "test_statistics"},
                        {"seed": 123}, {"weight_decay": 0.01}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config({**config, "training_overrides": changes})
        with self.assertRaises(ValueError):
            validate_config({**config, "evaluate_test": "False"})
        with self.assertRaises(ValueError):
            validate_config({**config, "protocol_version": "unknown"})

    def test_normalization_fits_train_only_and_owns_tensor(self):
        metadata = pd.DataFrame({"split": ["train", "train", "validation", "test"], "label": [0, 1, 2, 3]})
        features = np.array([[[1, 3]], [[5, 7]], [[1000, 2000]], [[-2000, -1000]]], dtype=np.float32)
        features.setflags(write=False)
        original = features.copy()
        with patch.object(v2, "CONFIG", v2.effective_config({"input_normalization": "train_global_zscore"})), \
             patch.dict(v2.COUNTS, {"train": 2}):
            stats = v2.fit_normalization(features, metadata)
            self.assertEqual(stats["mean"], 4.0)
            self.assertAlmostEqual(stats["std"], np.sqrt(5))
            x, _ = v2.FeatureDataset(features, metadata, "train", stats)[0]
            self.assertTrue(np.allclose(x.numpy(), (features[0][None] - 4) / np.sqrt(5)))
            x.zero_()
        np.testing.assert_array_equal(features, original)
        self.assertFalse(features.flags.writeable)

    def test_dual_checkpoint_selection_ties_and_unchanged_loss_stopping_without_training(self):
        directory = self.storage / "selection"
        directory.mkdir()
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.zero_()
        val = [{"loss": 0.9, "macro_f1": 0.4}, {"loss": 0.7, "macro_f1": 0.7},
               {"loss": 0.5, "macro_f1": 0.6}, {"loss": 0.8, "macro_f1": 0.7},
               {"loss": 1.0, "macro_f1": 0.5}]
        calls = []
        def mocked_pass(model, loader, weights, optimizer=None):
            calls.append(loader)
            if optimizer is not None:
                # Identify the mocked epoch in the serialized checkpoint, not an optimizer step.
                model.weight.data.fill_(len(calls) // 2 + 1)
                return {"loss": 1.0, "macro_f1": 0.2}, None
            return val[len(calls) // 2 - 1], None
        config = {**v2.CONFIG, "max_epochs": 10, "early_stopping_patience": 2}
        with patch.object(v2, "CONFIG", config), patch("torch.optim.Adam", return_value=Mock()) as optimizer, \
             patch.object(baseline, "epoch_pass", side_effect=mocked_pass), \
             contextlib.redirect_stdout(io.StringIO()):
            selection = v2.fit_validation_checkpoint(model, "TRAIN", "VALIDATION", None, directory)
        self.assertEqual(selection["best_epoch"], 2)  # F1 tie retains earliest epoch.
        self.assertEqual(selection["checkpoint_candidates"]["minimum_loss"]["epoch"], 3)
        self.assertEqual(selection["epochs_run"], 5)  # Still loss epoch + patience, not F1 epoch + patience.
        self.assertFalse(selection["test_used_for_selection"])
        self.assertEqual(calls, ["TRAIN", "VALIDATION"] * 5)
        self.assertEqual(float(torch.load(directory / "best_model.pt", weights_only=True)["weight"][0, 0]), 2)
        self.assertEqual(float(torch.load(directory / "best_val_loss.pt", weights_only=True)["weight"][0, 0]), 3)
        optimizer.return_value.step.assert_not_called()
        optimizer.return_value.zero_grad.assert_not_called()

    def test_eval_does_not_update_batchnorm(self):
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(2, 4), torch.nn.BatchNorm1d(4))
        loader = [(torch.ones(4, 1, 1, 2), torch.arange(4))]
        before = {k: t.clone() for k, t in model.state_dict().items()}
        metrics, _ = baseline.epoch_pass(model, loader, torch.ones(4))
        self.assertTrue(np.isfinite(metrics["loss"]))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[key], value), key)

    def write_synthetic_run(self, branch):
        v2.OUTPUT.mkdir(parents=True, exist_ok=True)
        original = read_json(ProjectPaths(repo=REPO_ROOT).BASELINE_ROOT / "protocol.json")
        protocol = {**original, "protocol_version": v2.VERSION, "training_configuration": deepcopy(v2.CONFIG)}
        write_json(v2.OUTPUT / "protocol.json", protocol)
        grid = pd.read_csv(REPO_ROOT / MANIFEST)
        for family in v2.SELECTED_FAMILIES:
            run_id = f"{branch}_{family}_seed42"
            directory = v2.RUNS / run_id
            directory.mkdir(parents=True)
            model = build_model(family, v2.BRANCHES[branch]["shape"][1])
            for filename in ("best_model.pt", "best_val_loss.pt", "best_val_macro_f1.pt"):
                torch.save(model.state_dict(), directory / filename)
            shutil.copyfile(directory / "best_val_macro_f1.pt", directory / "best_model.pt")
            measured = {}
            for name, split in (("train_eval", "train"), ("validation", "validation")):
                meta = grid[grid.split.eq(split)][META_COLUMNS].reset_index(drop=True)
                meta["label"] = meta.species.map({s: i for i, s in enumerate(CLASSES)})
                probabilities = np.full((len(meta), 4), 0.1)
                probabilities[np.arange(len(meta)), meta.label] = 0.7
                predictions = meta.copy()
                predictions["prediction"] = probabilities.argmax(axis=1)
                for i in range(4): predictions[f"p{i}"] = probabilities[:, i]
                predictions.to_csv(directory / f"{name}_predictions.csv", index=False)
                measured[name] = {**baseline.classification_metrics(meta.label, predictions.prediction), "loss": 0.8}
                measured[name + "_source"], sources = baseline.source_evaluation(meta, probabilities)
                sources.to_csv(directory / f"{name}_source_predictions.csv", index=False)
            metrics = measured["validation"]
            history = [{"epoch": 1, "learning_rate": v2.CONFIG["learning_rate"],
                        "train": measured["train_eval"], "validation": metrics}]
            write_json(directory / "history.json", history)
            candidates = {key: {"epoch": 1, "metrics": metrics, "file": filename,
                               "model_sha256": sha256(directory / filename)}
                          for key, filename in (("minimum_loss", "best_val_loss.pt"),
                                                ("maximum_macro_f1", "best_val_macro_f1.pt"))}
            normalization = {"kind": "none", "fit_split": None, "mean": 0.0, "std": 1.0}
            write_json(directory / "normalization.json", normalization)
            write_json(directory / "diagnostics.json", {"validation_subgroups": v2.subgroup_diagnostics(directory, "validation", "validation"),
                       "train_eval_subgroups": [], "batchnorm_selected_checkpoint": {}})
            write_json(directory / "run_spec.json", {"protocol": protocol})
            selection = {"status": "VALIDATION_CHECKPOINT_FIXED", "criterion": v2.SELECTION,
                "best_epoch": 1, "epochs_run": 1, "selected_validation_metrics": metrics,
                "best_validation_loss": 0.8, "best_validation_macro_f1_observed": 1.0,
                "checkpoint_candidates": candidates, "selected_checkpoint": "best_val_macro_f1.pt",
                "early_stopping_monitor": v2.CONFIG["early_stopping_monitor"],
                "model_sha256": sha256(directory / "best_model.pt"),
                "history_sha256": sha256(directory / "history.json"), "fixed_at_utc": "MOCK", "test_used_for_selection": False}
            write_json(directory / "selection.json", selection)
            files = [p.name for p in directory.iterdir() if p.is_file()]
            record = {**selection, "status": "COMPLETE", "protocol_version": v2.VERSION, "run_id": run_id,
                "frontend": branch, "frontend_identity": protocol["frontends"][branch], "model_family": family,
                "architecture": architecture_config(family, v2.BRANCHES[branch]["shape"][1]),
                "seed": 42, "training_configuration": deepcopy(v2.CONFIG), "class_weights": protocol["class_weights"],
                "class_order": CLASSES, "counts": COUNTS, "software": protocol["software"],
                "protocol_sha256": digest_json(protocol), "normalization": normalization,
                "train_eval_metrics": measured["train_eval"], "train_eval_source_metrics": measured["train_eval_source"],
                "validation_source_metrics": measured["validation_source"], "test_evaluated": False,
                "final_test_metrics": None, "source_test_metrics": None,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "serialized_model_bytes": (directory / "best_model.pt").stat().st_size,
                "model_file_path": (directory / "best_model.pt").relative_to(v2.OUTPUT).as_posix(),
                "cache_sha256": original["accepted_input_sha256"],
                "artifacts_sha256": {f: sha256(directory / f) for f in files}}
            write_json(directory / "result.json", record)

    def execute_suite(self, name):
        with patch("experiment_runner.smoke", return_value={"runtime": {"gpu": "MOCK"}, "cache_sha256": {}, "split_sha256": "MOCK"}), \
             patch("experiment_runner.subprocess.check_output", side_effect=["mock-commit", ""]), \
             patch.object(v2, "run_branch", side_effect=self.write_synthetic_run), \
             patch.object(v2, "fit_validation_checkpoint", side_effect=AssertionError("Training forbidden")), \
             patch.object(v2, "save_evaluation", side_effect=AssertionError("Model evaluation forbidden")), \
             contextlib.redirect_stdout(io.StringIO()):
            return run(self.template, experiment_id=name)

    def test_real_branch_wiring_has_no_test_loader_and_authenticates_outputs(self):
        """Exercise run_branch itself, blocking all optimization and real model evaluation."""
        target = ProjectPaths.from_env().experiment("exp_branch_probe") / "results"
        original = read_json(ProjectPaths(repo=REPO_ROOT).BASELINE_ROOT / "protocol.json")
        contract = read_json(REPO_ROOT / "archive/colab_baseline/provenance/export_contract.json")
        evidence = contract["authenticated_evidence"]
        meta = pd.read_csv(REPO_ROOT / MANIFEST)
        meta["label"] = meta.species.map({s: i for i, s in enumerate(CLASSES)})
        features = np.broadcast_to(np.zeros((1, 96, 64), dtype=np.float32), (len(meta), 96, 64))
        cache = contract["cache_records"]["03A"]
        model = build_model("ds_cnn", 64)
        model.to = lambda device: model  # Simulate transport only, on this CPU host.
        real_tensor = torch.tensor
        def cpu_tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return real_tensor(*args, **kwargs)
        def synthetic_evaluation(model, loader, weights, metadata, split, directory):
            frame = metadata[metadata.split.eq(split)][META_COLUMNS + ["label"]].reset_index(drop=True)
            probabilities = np.full((len(frame), 4), 0.1)
            probabilities[np.arange(len(frame)), frame.label] = 0.7
            frame["prediction"] = probabilities.argmax(axis=1)
            for i in range(4): frame[f"p{i}"] = probabilities[:, i]
            frame.to_csv(directory / f"{loader.report_name}_predictions.csv", index=False)
            source_metrics, sources = baseline.source_evaluation(frame, probabilities)
            sources.to_csv(directory / f"{loader.report_name}_source_predictions.csv", index=False)
            metrics = {**baseline.classification_metrics(frame.label, frame.prediction), "loss": 0.8}
            return metrics, source_metrics
        def fixed_untrained_checkpoint(model, train, validation, weights, directory):
            # Produce the same schema as the synthetic suite without a forward/backward pass.
            frame = meta[meta.split.eq("validation")]
            metrics = {**baseline.classification_metrics(frame.label, frame.label), "loss": 0.8}
            train_frame = meta[meta.split.eq("train")]
            train_metrics = {**baseline.classification_metrics(train_frame.label, train_frame.label), "loss": 0.8}
            torch.save(model.state_dict(), directory / "best_val_macro_f1.pt")
            for filename in ("best_model.pt", "best_val_loss.pt"):
                shutil.copyfile(directory / "best_val_macro_f1.pt", directory / filename)
            write_json(directory / "history.json", [{"epoch": 1, "learning_rate": v2.CONFIG["learning_rate"],
                       "train": train_metrics, "validation": metrics}])
            selection = {"status": "VALIDATION_CHECKPOINT_FIXED", "criterion": v2.SELECTION,
                "best_epoch": 1, "epochs_run": 1, "selected_validation_metrics": metrics,
                "best_validation_loss": 0.8, "best_validation_macro_f1_observed": 1.0,
                "selected_checkpoint": "best_val_macro_f1.pt", "early_stopping_monitor": v2.CONFIG["early_stopping_monitor"],
                "checkpoint_candidates": {k: {"epoch": 1, "metrics": metrics, "file": f,
                    "model_sha256": sha256(directory / f)} for k, f in
                    (("minimum_loss", "best_val_loss.pt"), ("maximum_macro_f1", "best_val_macro_f1.pt"))},
                "model_sha256": sha256(directory / "best_model.pt"), "history_sha256": sha256(directory / "history.json"),
                "fixed_at_utc": "MOCK", "test_used_for_selection": False}
            write_json(directory / "selection.json", selection)
            return selection
        with patch.object(v2, "OUTPUT", target), patch.object(v2, "RUNS", target / "runs"), \
             patch.object(v2, "EXPERIMENT_ID", "exp_branch_probe"), patch.object(v2, "SELECTED_FAMILIES", ("ds_cnn",)), \
             patch.object(v2, "prepare_caches", return_value=({"03A": (features, meta, cache)}, evidence)), \
             patch.object(baseline, "set_seeds", return_value={"framework": 42}), \
             patch.object(baseline, "software_versions", return_value=original["software"]), \
             patch("torch.cuda.is_available", return_value=True), patch("torch.version.cuda", "MOCK"), \
             patch("torch.tensor", side_effect=cpu_tensor), patch.object(v2, "build_model", return_value=model), \
             patch.object(v2, "fit_validation_checkpoint", side_effect=fixed_untrained_checkpoint), \
             patch.object(v2, "save_evaluation", side_effect=synthetic_evaluation), \
             patch.object(v2, "make_loader", wraps=v2.make_loader) as loaders, \
             patch("torch.optim.Adam", side_effect=AssertionError("Training forbidden")), \
             patch.object(baseline, "epoch_pass", side_effect=AssertionError("Real model evaluation forbidden")):
            table = v2.run_branch("03A")
            self.assertEqual([call.args[2] for call in loaders.call_args_list], ["train", "validation", "train", "validation"])
            self.assertEqual(len(table), 1)
            directory = target / "runs/03A_ds_cnn_seed42"
            v2.validate_saved_run(read_json(directory / "result.json"), directory, read_json(target / "protocol.json"))
            self.assertFalse(list(directory.glob("test*predictions.csv")))

    def test_eight_run_validation_only_roundtrip_notebooks_reference_and_tamper(self):
        previous = (v2.OUTPUT, v2.RUNS, v2.SELECTED_FAMILIES, v2.EXPERIMENT_ID, v2.CONFIG)
        target = self.execute_suite("exp_v2_mock")
        self.assertEqual(previous, (v2.OUTPUT, v2.RUNS, v2.SELECTED_FAMILIES, v2.EXPERIMENT_ID, v2.CONFIG))
        metadata = read_json(target / "config/metadata.json")
        self.assertEqual((metadata["status"], metadata["completed_runs"]), ("COMPLETE", 8))
        self.assertEqual(len(verify_experiment("exp_v2_mock")), 8)
        self.assertEqual(len(list(target.glob("results/runs/*/best_val_loss.pt"))), 8)
        self.assertFalse(list(target.glob("results/runs/*/test*predictions.csv")))
        notebook = read_json(REPO_ROOT / "notebooks/current/04C_compare_models.ipynb")
        selection = next("".join(c["source"]) for c in notebook["cells"] if "all_complete = False" in "".join(c["source"]))
        scope = {"PATHS": ProjectPaths.from_env(), "BRANCHES": v2.BRANCHES, "FAMILIES": FAMILIES,
                 "pd": pd, "read_json": read_json, "display": lambda _: None}
        before = {p.relative_to(target).as_posix(): sha256(p) for p in target.rglob("*") if p.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(re.sub(r'^EXPERIMENT_NAME = .*$', 'EXPERIMENT_NAME = "exp_v2_mock"', selection, count=1, flags=re.MULTILINE), scope)
            self.assertTrue(scope["all_complete"])
            self.assertEqual(len(scope["comparison"]), 8)
            plot = next("".join(c["source"]) for c in notebook["cells"] if "labels_short =" in "".join(c["source"]))
            import matplotlib
            matplotlib.use("Agg")
            with patch("matplotlib.pyplot.show"):
                exec(plot, scope)
            import matplotlib.pyplot as plt
            plt.close("all")
        after = {p.relative_to(target).as_posix(): sha256(p) for p in target.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.execute_suite("exp_v2_reference")
        paired = compare_validation_to_reference("exp_v2_mock", "exp_v2_reference")
        self.assertTrue(paired["Delta val F1 (percentage points)"].eq(0).all())
        with patch("experiment_runner.smoke") as smoke, self.assertRaises(FileExistsError):
            run(self.template, experiment_id="exp_v2_mock")
        smoke.assert_not_called()
        artifact = target / "results/runs/03A_ds_cnn_seed42/best_val_loss.pt"
        with artifact.open("ab") as stream: stream.write(b"synthetic tampering")
        with self.assertRaisesRegex(RuntimeError, "Run artifact changed"):
            verify_experiment("exp_v2_mock")


if __name__ == "__main__":
    unittest.main()
