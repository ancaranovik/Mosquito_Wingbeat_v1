"""Stage-04 CUDA export: unchanged learning protocol; explicit device transport only."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from datetime import datetime, timezone
import importlib.metadata
import os
import platform
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model_zoo import FAMILIES, architecture_config, build_model
from stage04_data import (ROOT, CLASSES, BRANCHES, COUNTS, LOCK_SHA256, META_COLUMNS,
                          prepare_caches, sha256, read_json, write_json, require, digest_json,
                          authenticate, SCIENTIFIC_LOCK_SHA256, EXECUTION_LOCK_SHA256, OUTPUT, MANIFEST, CONTRACT_SHA256)

CONFIG = {
    "seed": 42, "device": "cuda", "dtype": "float32", "cpu_threads": 4,
    "batch_size": 64, "max_epochs": 50, "early_stopping_patience": 8,
    "selection": "minimum_validation_weighted_cross_entropy", "min_delta": 0.0,
    "tie_rule": "earliest_epoch", "optimizer": "Adam", "learning_rate": 0.001,
    "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "amsgrad": False,
    "lr_policy": "constant", "loss": "weighted_cross_entropy",
    "loss_reduction": "sum(weight[label]*NLL)/sum(weight[label])",
    "class_weight_formula": "N_train/(4*n_train_class)", "label_smoothing": 0.0,
    "input_normalization": "none", "augmentation": "none", "oversampling": False,
    "shuffle_train": True, "shuffle_validation_test": False, "num_workers": 0,
    "drop_last": False, "mixed_precision": False, "pretrained": False,
    "deterministic_algorithms": True, "mkldnn_enabled": False,
    "source_aggregation": "arithmetic mean softmax over all windows sharing source name; argmax",
    "undefined_precision_recall_f1": 0.0,
    "shortlist": "top 3 by selected-validation macro-F1 descending, params ascending, run_id ascending",
}
RUNS = OUTPUT / "runs"


def software_versions():
    packages = ("torch", "numpy", "pandas")
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{p: importlib.metadata.version(p) for p in packages},
            "processor": platform.processor(), "torch_build": torch.__config__.show(),
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "tf32": False, "mixed_precision": False}


def set_seeds():
    require(torch.__version__.split("+")[0] == "2.8.0", "Install pinned requirements.txt")
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(CONFIG["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {"python": 42, "numpy": 42, "framework": 42, "data_loader": 42,
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "unset; no hash-order-dependent operations"),
            "limitations": "CUDA benchmark; identical software/GPU required; CPU-versus-CUDA bitwise identity not promised"}


class FeatureDataset(Dataset):
    def __init__(self, features, metadata, split):
        self.features = features
        self.indices = np.flatnonzero(metadata.split.to_numpy() == split)
        self.labels = metadata.label.to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        # Own the row buffer: never expose writable tensors backed by the frozen cache.
        x = np.array(self.features[row], dtype=np.float32, copy=True)[None, :, :]
        return torch.from_numpy(x), int(self.labels[row])


def make_loader(features, metadata, split):
    generator = torch.Generator().manual_seed(CONFIG["seed"])
    return DataLoader(FeatureDataset(features, metadata, split), batch_size=CONFIG["batch_size"],
                      shuffle=(split == "train"), generator=generator, num_workers=0, drop_last=False)


def classification_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    require(labels.shape == predictions.shape and labels.ndim == 1 and len(labels) > 0, "Invalid predictions")
    require(((labels >= 0) & (labels < 4) & (predictions >= 0) & (predictions < 4)).all(), "Unknown label")
    cm = np.bincount(4 * labels + predictions, minlength=16).reshape(4, 4)
    true = cm.sum(axis=1); predicted = cm.sum(axis=0); tp = cm.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(4, dtype=float), where=predicted != 0)
    recall = np.divide(tp, true, out=np.zeros(4, dtype=float), where=true != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(4, dtype=float), where=(precision + recall) != 0)
    return {"accuracy": float(tp.sum() / len(labels)), "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()),
            "weighted_f1": float(np.sum(f1 * true) / len(labels)), "examples": len(labels),
            "confusion_matrix": cm.tolist(), "class_order": CLASSES,
            "per_class": {s: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(true[i])} for i, s in enumerate(CLASSES)}}


def source_evaluation(metadata, probabilities):
    require(len(metadata) == len(probabilities), "Source aggregation alignment mismatch")
    require(metadata.groupby("name").label.nunique().eq(1).all(), "Source has conflicting labels")
    require(metadata.groupby("name").split.nunique().eq(1).all(), "Source crosses partitions")
    frame = metadata[["name", "species", "label"]].reset_index(drop=True).copy()
    for i in range(4):
        frame[f"p{i}"] = probabilities[:, i]
    agg = frame.groupby("name", sort=True).agg(
        species=("species", "first"), label=("label", "first"), windows=("label", "size"),
        **{f"p{i}": (f"p{i}", "mean") for i in range(4)}).reset_index()
    agg["prediction"] = agg[[f"p{i}" for i in range(4)]].to_numpy().argmax(axis=1)
    metrics = classification_metrics(agg.label, agg.prediction)
    metrics["sources"] = metrics.pop("examples")
    metrics["aggregation"] = CONFIG["source_aggregation"]
    return metrics, agg


def epoch_pass(model, loader, weights, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_numerator = total_denominator = 0.0
    truths, probabilities = [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            device = weights.device
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            numerator = nn.functional.cross_entropy(logits, y, weight=weights, reduction="sum")
            denominator = weights[y].sum()
            loss = numerator / denominator
            require(bool(torch.isfinite(loss)), "Nonfinite loss; stop without altering the frozen pipeline")
            if training:
                loss.backward()
                optimizer.step()
            total_numerator += float(numerator.detach())
            total_denominator += float(denominator)
            truths.append(y.cpu().numpy())
            probabilities.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())
    y = np.concatenate(truths); probs = np.concatenate(probabilities)
    require(np.isfinite(probs).all(), "Nonfinite model probabilities")
    metrics = classification_metrics(y, probs.argmax(axis=1))
    metrics["loss"] = total_numerator / total_denominator
    return metrics, probs


def fit_validation_checkpoint(model, train_loader, val_loader, weights, directory):
    """Deliberately accepts no TEST loader or labels."""
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"],
                                  betas=tuple(CONFIG["betas"]), eps=CONFIG["eps"],
                                  weight_decay=0, amsgrad=False, foreach=False)
    history = []; best_loss = float("inf"); best_epoch = 0; best_metrics = None
    checkpoint = directory / "best_model.pt"
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        train, _ = epoch_pass(model, train_loader, weights, optimizer)
        val, _ = epoch_pass(model, val_loader, weights)
        history.append({"epoch": epoch, "learning_rate": CONFIG["learning_rate"],
                        "train": train, "validation": val})
        write_json(directory / "history.json", history)
        if val["loss"] < best_loss:  # exact ties retain the earliest checkpoint
            best_loss, best_epoch, best_metrics = val["loss"], epoch, val
            temporary = directory / "best_model.tmp"
            torch.save(model.state_dict(), temporary)
            os.replace(temporary, checkpoint)
        print(f"epoch {epoch:02d}: train loss={train['loss']:.6f}, val loss={val['loss']:.6f}, "
              f"val macro-F1={val['macro_f1']:.6f}; best={best_epoch}", flush=True)
        if epoch - best_epoch >= CONFIG["early_stopping_patience"]:
            break
    selection = {"status": "VALIDATION_CHECKPOINT_FIXED", "best_epoch": best_epoch,
                 "epochs_run": len(history), "criterion": CONFIG["selection"],
                 "selected_validation_metrics": best_metrics,
                 "best_validation_loss": best_loss,
                 "best_validation_macro_f1_observed": max(h["validation"]["macro_f1"] for h in history),
                 "model_sha256": sha256(checkpoint), "history_sha256": sha256(directory / "history.json"),
                 "fixed_at_utc": datetime.now(timezone.utc).isoformat(), "test_used_for_selection": False}
    # Persist the decision before a TEST loader is even constructed.
    write_json(directory / "selection.json", selection)
    return selection


def load_fixed_checkpoint(model, directory):
    selection = read_json(directory / "selection.json")
    require(selection["status"] == "VALIDATION_CHECKPOINT_FIXED" and not selection["test_used_for_selection"],
            "TEST evaluation requires a fixed validation-only checkpoint")
    require(sha256(directory / "best_model.pt") == selection["model_sha256"], "Selected model changed")
    require(sha256(directory / "history.json") == selection["history_sha256"], "Training history changed")
    model.load_state_dict(torch.load(directory / "best_model.pt", map_location="cpu", weights_only=True))
    return selection


def protocol_record(evidence):
    return {"training_configuration": CONFIG, "frozen_lock_sha256": LOCK_SHA256,
            "scientific_lock_sha256": SCIENTIFIC_LOCK_SHA256,
            "execution_lock_sha256": EXECUTION_LOCK_SHA256, "export_contract_sha256": CONTRACT_SHA256,
            "accepted_input_sha256": evidence["sha256"],
            "class_order": CLASSES, "class_weights": evidence["class_weights"],
            "counts": COUNTS, "software": software_versions(),
            "frontends": evidence["frontends"],
            "implementation_sha256": {f: sha256(ROOT / f) for f in
                ("tools/stage04_data.py", "tools/model_zoo.py", "tools/training_common.py", "requirements.txt")}}


def run_branch(branch):
    require(branch in BRANCHES, "Unknown frontend")
    require(torch.cuda.is_available(), "CUDA GPU required: select a Colab GPU runtime; no CPU fallback")
    require(torch.version.cuda is not None, "Install the bundled CUDA PyTorch requirements")
    caches, evidence = prepare_caches()  # BOTH branches authenticated before any training
    seed_record = set_seeds()
    protocol = protocol_record(evidence)
    protocol_path = OUTPUT / "protocol.json"
    if protocol_path.exists():
        require(read_json(protocol_path) == protocol, "Protocol/runtime changed: do not mix benchmark runs")
    else:
        write_json(protocol_path, protocol)
    features, metadata, cache = caches[branch]
    weights = torch.tensor([evidence["class_weights"][s] for s in CLASSES], dtype=torch.float32, device="cuda")
    records = []
    for family in FAMILIES:
        run_id = f"{branch}_{family}_seed42"
        directory = RUNS / run_id
        if (directory / "result.json").exists():
            saved = read_json(directory / "result.json")
            validate_saved_run(saved, directory, protocol)
            records.append(saved)
            continue
        require(not directory.exists(), f"Incomplete run at {directory}; preserve it and investigate before rerunning")
        directory.mkdir(parents=True)
        seed_record = set_seeds()  # reset independently for every model/frontend
        model = build_model(family, BRANCHES[branch]["shape"][1]).to("cuda")
        architecture = architecture_config(family, BRANCHES[branch]["shape"][1])
        write_json(directory / "run_spec.json", {"run_id": run_id, "architecture": architecture,
                   "module_repr": str(model), "protocol": protocol, "seeds": seed_record,
                   "frontend_cache": cache})
        train_loader = make_loader(features, metadata, "train")
        val_loader = make_loader(features, metadata, "validation")
        selection = fit_validation_checkpoint(model, train_loader, val_loader, weights, directory)
        load_fixed_checkpoint(model, directory)
        # TEST is used exactly once after selection; results cannot enter fit().
        test_loader = make_loader(features, metadata, "test")
        test_metrics, probs = epoch_pass(model, test_loader, weights)
        test_meta = metadata[metadata.split.eq("test")].reset_index(drop=True)
        predictions = test_meta[META_COLUMNS + ["label"]].copy()
        predictions["prediction"] = probs.argmax(axis=1)
        for i in range(4):
            predictions[f"p{i}"] = probs[:, i]
        predictions.to_csv(directory / "test_predictions.csv", index=False)
        source_metrics, sources = source_evaluation(test_meta, probs)
        sources.to_csv(directory / "source_predictions.csv", index=False)
        result = {"status": "COMPLETE", "run_id": run_id, "frontend": branch,
                  "frontend_identity": evidence["frontends"][branch], "model_family": family,
                  "architecture": architecture, "module_repr": str(model), "seed": 42, "seeds": seed_record,
                  "training_configuration": CONFIG, "class_weights": evidence["class_weights"],
                  "class_order": CLASSES, "counts": COUNTS, "protocol_sha256": digest_json(protocol),
                  **selection, "status": "COMPLETE", "final_test_metrics": test_metrics,
                  "source_test_metrics": source_metrics,
                  "parameter_count": sum(p.numel() for p in model.parameters()),
                  "serialized_model_bytes": (directory / "best_model.pt").stat().st_size,
                  "model_file_path": (directory / "best_model.pt").relative_to(OUTPUT).as_posix(),
                  "software": protocol["software"], "cache_sha256": cache["sha256"],
                  "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                  "artifacts_sha256": {f: sha256(directory / f) for f in
                    ("best_model.pt", "history.json", "selection.json", "run_spec.json",
                     "test_predictions.csv", "source_predictions.csv")}}
        write_json(directory / "result.json", result)
        records.append(result)
    return pd.DataFrame([comparison_row(r) for r in records])


def validate_saved_run(record, directory, protocol):
    require(record["status"] == "COMPLETE" and record["protocol_sha256"] == digest_json(protocol),
            "Incomplete or incomparable saved run")
    require(record["training_configuration"] == protocol["training_configuration"], "Run protocol mismatch")
    require(record["class_weights"] == protocol["class_weights"] and record["class_order"] == CLASSES,
            "Run weights/labels mismatch")
    require(record["counts"] == COUNTS and record["software"] == protocol["software"], "Run population/runtime mismatch")
    require(record["frontend_identity"] == protocol["frontends"][record["frontend"]], "Run frontend mismatch")
    expected_id = f"{record['frontend']}_{record['model_family']}_seed42"
    require(record["run_id"] == directory.name == expected_id and record["seed"] == 42, "Run identity mismatch")
    require(record["architecture"] == architecture_config(record["model_family"], BRANCHES[record["frontend"]]["shape"][1]),
            "Run architecture mismatch")
    require(set(record["artifacts_sha256"]) == {"best_model.pt", "history.json", "selection.json", "run_spec.json",
                                               "test_predictions.csv", "source_predictions.csv"},
            "Incomplete run artifact hash record")
    for file, expected in record["artifacts_sha256"].items():
        require(sha256(directory / file) == expected, f"Run artifact changed: {directory.name}/{file}")
    require(record["model_sha256"] == sha256(directory / "best_model.pt"), "Model hash mismatch")
    require(record["serialized_model_bytes"] == (directory / "best_model.pt").stat().st_size, "Model size mismatch")
    selection = read_json(directory / "selection.json")
    require(selection["status"] == "VALIDATION_CHECKPOINT_FIXED" and selection["test_used_for_selection"] is False,
            "Saved run lacks validation-only selection evidence")
    for key in ("model_sha256", "best_epoch", "epochs_run", "selected_validation_metrics", "best_validation_loss"):
        require(record[key] == selection[key], f"Selection differs: {key}")
    require(selection["history_sha256"] == sha256(directory / "history.json"), "Selection history mismatch")
    spec = read_json(directory / "run_spec.json")
    require(spec["protocol"] == protocol and spec["run_id"] == expected_id, "Run specification mismatch")
    history = read_json(directory / "history.json")
    require(len(history) == record["epochs_run"] and 1 <= len(history) <= CONFIG["max_epochs"],
            "Invalid epoch history")
    require([h["epoch"] for h in history] == list(range(1, len(history) + 1)), "Nonsequential history")
    best = min(history, key=lambda h: h["validation"]["loss"])
    require(best["epoch"] == record["best_epoch"] and best["validation"] == record["selected_validation_metrics"],
            "Checkpoint is not earliest minimum validation loss")
    require(record["best_validation_loss"] == best["validation"]["loss"], "Selected loss mismatch")
    expected_parameters = sum(p.numel() for p in build_model(record["model_family"],
                                BRANCHES[record["frontend"]]["shape"][1]).parameters())
    require(record["parameter_count"] == expected_parameters, "Parameter count mismatch")
    require(record["model_file_path"] == (directory / "best_model.pt").relative_to(OUTPUT).as_posix(),
            "Checkpoint path mismatch")
    for key in ("final_test_metrics", "source_test_metrics", "selected_validation_metrics"):
        require(record[key]["class_order"] == CLASSES, f"Metric class order mismatch: {key}")
    predictions = pd.read_csv(directory / "test_predictions.csv")
    grid = pd.read_csv(ROOT / MANIFEST)
    expected = grid[grid.split.eq("test")][META_COLUMNS].reset_index(drop=True)
    expected["label"] = expected.species.map({s: i for i, s in enumerate(CLASSES)})
    pd.testing.assert_frame_equal(predictions[META_COLUMNS + ["label"]], expected, check_dtype=False, check_exact=True)
    probs = predictions[[f"p{i}" for i in range(4)]].to_numpy()
    require(np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
            and np.allclose(probs.sum(axis=1), 1, atol=1e-6, rtol=0), "Invalid saved probabilities")
    require(np.array_equal(predictions.prediction, probs.argmax(axis=1)), "Saved predictions mismatch")
    metrics = classification_metrics(predictions.label, predictions.prediction)
    require(all(record["final_test_metrics"][k] == v for k, v in metrics.items()), "Saved TEST metrics mismatch")
    source_metrics, sources = source_evaluation(expected, probs)
    require(record["source_test_metrics"] == source_metrics, "Saved source metrics mismatch")
    pd.testing.assert_frame_equal(pd.read_csv(directory / "source_predictions.csv"), sources,
                                  check_dtype=False, rtol=1e-6, atol=1e-8)


def comparison_row(record):
    metrics = record["final_test_metrics"]
    return {"Frontend": record["frontend"], "Model": record["model_family"],
            "Params": record["parameter_count"], "Model Size (bytes)": record["serialized_model_bytes"],
            "Accuracy": metrics["accuracy"], "Macro-F1": metrics["macro_f1"],
            "Macro-Recall": metrics["macro_recall"], "Macro-Precision": metrics["macro_precision"],
            "Weighted-F1": metrics["weighted_f1"], "Best epoch": record["best_epoch"],
            "Selected val loss": record["best_validation_loss"],
            "Selected val Macro-F1": record["selected_validation_metrics"]["macro_f1"]}


def compare_saved_results():
    """04C reads saved Stage-04 artifacts only. Never generates features or trains."""
    protocol_path = OUTPUT / "protocol.json"
    require(protocol_path.is_file(), "Missing Stage-04 protocol; comparison requires all 8 completed authenticated runs")
    missing = [f"{b}_{f}_seed42" for b in BRANCHES for f in FAMILIES
               if not (RUNS / f"{b}_{f}_seed42" / "result.json").is_file()]
    require(not missing, f"Missing completed runs: {missing}; comparison requires all 8")
    protocol = read_json(protocol_path)
    _, evidence = authenticate()  # identity checks only; no feature generation or evaluation
    require(protocol == protocol_record(evidence), "Stale protocol/frontend/implementation/runtime; comparison refused")
    records = []
    for branch in BRANCHES:
        for family in FAMILIES:
            directory = RUNS / f"{branch}_{family}_seed42"
            require((directory / "result.json").exists(), f"Missing completed run: {directory.name}; comparison requires all 8")
            record = read_json(directory / "result.json")
            validate_saved_run(record, directory, protocol)
            records.append(record)
    table = pd.DataFrame([comparison_row(r) for r in records])
    table.to_csv(OUTPUT / "comparison.csv", index=False)
    shortlist = sorted(records, key=lambda r: (-r["selected_validation_metrics"]["macro_f1"],
                                               r["parameter_count"], r["run_id"]))[:3]
    write_json(OUTPUT / "comparison.json", {
        "interpretation": "controlled comparison of two frozen acoustic frontend configurations",
        "experiments": table.to_dict(orient="records"), "protocol_sha256": digest_json(protocol),
        "shortlist_rule": CONFIG["shortlist"], "stage05_shortlist": [r["run_id"] for r in shortlist],
        "edge_winner_declared": False, "test_used_for_shortlist": False,
        "limitation": "Correlated windows are not independent mosquitoes; source grouping does not prove biological or domain independence"})
    return table


