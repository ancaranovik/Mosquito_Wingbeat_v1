"""Adam L2 weight-decay ablation on authenticated caches; v4 stays frozen.

Derived from v4; loaders, models, class weights, selection and stopping are
unchanged. Only the reviewed Adam weight_decay override is added.
TEST is disabled by default; no cache writes or feature generation.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from copy import deepcopy
from datetime import datetime, timezone
import math
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import experiment_training as baseline
from cache_consumer import (BRANCHES, CLASSES, COUNTS, MANIFEST, META_COLUMNS, ROOT,
                            SPLIT, TRAIN_COUNTS, digest_json, prepare_caches, read_json, require,
                            sha256, write_json)
from model_zoo import FAMILIES, architecture_config, build_model
from project_paths import ProjectPaths, require_mutable_output, resolve_path

VERSION = "stage04_experiment_v5"
SELECTION = "maximum_validation_macro_f1"
OVERRIDES = {"learning_rate", "input_normalization", "class_weight_policy", "class_weight_exponent", "weight_decay"}
POLICY = "power_inverse_frequency"


def checked_exponent(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("class_weight_exponent must be a finite number between 0 and 1")
    return float(value)


def effective_config(overrides, evaluate_test=False):
    if not isinstance(overrides, dict) or not set(overrides) <= OVERRIDES:
        raise ValueError(f"v5 supports only these overrides: {sorted(OVERRIDES)}")
    decay = overrides.get("weight_decay", 0.0)
    if type(decay) not in (int, float) or not math.isfinite(decay) or not 0 <= decay <= 0.01:
        raise ValueError("weight_decay must be a finite number between 0 and 0.01")
    from experiment_training_v4 import effective_config as v4_config
    config = v4_config({k: v for k, v in overrides.items() if k != "weight_decay"}, evaluate_test)
    config.update(protocol_version=VERSION, weight_decay=float(decay))
    return config


def effective_class_weights(evidence, policy=None, exponent=None):
    """Transform authenticated TRAIN weights without changing cache evidence."""
    policy = CONFIG["class_weight_policy"] if policy is None else policy
    require(policy == POLICY, "Unknown class-weight policy")
    exponent = checked_exponent(CONFIG["class_weight_exponent"] if exponent is None else exponent)
    accepted = {s: COUNTS["train"] / (4 * n) for s, n in zip(CLASSES, TRAIN_COUNTS)}
    require(evidence["class_weights"] == accepted, "Accepted TRAIN class weights changed")
    return {s: value ** exponent for s, value in accepted.items()}


def class_weight_record(config, accepted):
    effective = effective_class_weights({"class_weights": accepted},
                                       config["class_weight_policy"], config["class_weight_exponent"])
    return {"policy": config["class_weight_policy"], "exponent": config["class_weight_exponent"],
            "fit_split": "train", "formula": config["class_weight_formula"],
            "train_counts": dict(zip(CLASSES, TRAIN_COUNTS)),
            "accepted_inverse_frequency": deepcopy(accepted),
            "effective_weights": effective,
            "effective_weights_float32": {s: float(np.float32(v)) for s, v in effective.items()},
            "loss_reduction": config["loss_reduction"]}


CONFIG = effective_config({})
OUTPUT = baseline.OUTPUT
RUNS = OUTPUT / "runs"
SELECTED_FAMILIES = FAMILIES
EXPERIMENT_ID = None


def fit_normalization(features, metadata):
    """Fit one scalar mean/std on TRAIN only, without writing to the cache."""
    if CONFIG["input_normalization"] == "none":
        return {"kind": "none", "fit_split": None, "mean": 0.0, "std": 1.0}
    indices = np.flatnonzero(metadata.split.to_numpy() == "train")
    require(len(indices) == COUNTS["train"], "Normalization TRAIN population changed")
    total = squared = 0.0
    count = 0
    for start in range(0, len(indices), 256):
        block = np.asarray(features[indices[start:start + 256]], dtype=np.float64)
        total += float(block.sum())
        squared += float(np.square(block).sum())
        count += block.size
    mean = total / count
    std = math.sqrt(max(0.0, squared / count - mean * mean))
    require(math.isfinite(mean) and math.isfinite(std) and std > 1e-8,
            "Invalid TRAIN normalization statistics")
    return {"kind": "train_global_zscore", "fit_split": "train", "mean": mean,
            "std": std, "train_examples": len(indices), "scalar_values": count,
            "calculation_dtype": "float64", "output_dtype": "float32"}


class FeatureDataset(baseline.FeatureDataset):
    def __init__(self, features, metadata, split, normalization):
        super().__init__(features, metadata, split)
        self.normalization = normalization

    def __getitem__(self, index):
        x, y = super().__getitem__(index)  # Own buffer; frozen mmap is never writable.
        if self.normalization["kind"] != "none":
            x.sub_(self.normalization["mean"]).div_(self.normalization["std"])
        return x, y


def make_loader(features, metadata, split, normalization, shuffle=False):
    generator = torch.Generator().manual_seed(CONFIG["seed"])
    return DataLoader(FeatureDataset(features, metadata, split, normalization),
                      batch_size=CONFIG["batch_size"], shuffle=shuffle,
                      generator=generator, num_workers=0, drop_last=False)


def fit_validation_checkpoint(model, train_loader, val_loader, weights, directory):
    """Accepts TRAIN and validation only; loss stopping is unchanged from baseline."""
    require_mutable_output(directory)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"],
        betas=tuple(CONFIG["betas"]), eps=CONFIG["eps"],
        weight_decay=CONFIG["weight_decay"], amsgrad=CONFIG["amsgrad"], foreach=False)
    history, candidates = [], {}
    best_loss, best_f1 = float("inf"), -float("inf")
    loss_epoch = 0
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        train, _ = baseline.epoch_pass(model, train_loader, weights, optimizer)
        val, _ = baseline.epoch_pass(model, val_loader, weights)
        history.append({"epoch": epoch, "learning_rate": CONFIG["learning_rate"],
                        "train": train, "validation": val})
        write_json(directory / "history.json", history)
        for key, improved, filename in (
                ("minimum_loss", val["loss"] < best_loss, "best_val_loss.pt"),
                ("maximum_macro_f1", val["macro_f1"] > best_f1, "best_val_macro_f1.pt")):
            if improved:
                temporary = directory / (filename + ".tmp")
                torch.save(model.state_dict(), temporary)
                os.replace(temporary, directory / filename)
                candidates[key] = {"epoch": epoch, "metrics": val, "file": filename,
                                   "model_sha256": sha256(directory / filename)}
        if val["loss"] < best_loss:
            best_loss, loss_epoch = val["loss"], epoch
        best_f1 = max(best_f1, val["macro_f1"])
        print(f"epoch {epoch:02d}: train loss={train['loss']:.6f}, val loss={val['loss']:.6f}, "
              f"val macro-F1={val['macro_f1']:.6f}; best loss={loss_epoch}, "
              f"best F1={candidates['maximum_macro_f1']['epoch']}", flush=True)
        if epoch - loss_epoch >= CONFIG["early_stopping_patience"]:
            break
    selected = candidates["maximum_macro_f1"]
    shutil.copyfile(directory / selected["file"], directory / "best_model.pt")
    selection = {"status": "VALIDATION_CHECKPOINT_FIXED", "criterion": CONFIG["selection"],
        "best_epoch": selected["epoch"], "epochs_run": len(history),
        "selected_validation_metrics": selected["metrics"], "best_validation_loss": best_loss,
        "best_validation_macro_f1_observed": best_f1, "checkpoint_candidates": candidates,
        "selected_checkpoint": selected["file"], "early_stopping_monitor": CONFIG["early_stopping_monitor"],
        "model_sha256": sha256(directory / "best_model.pt"),
        "history_sha256": sha256(directory / "history.json"),
        "fixed_at_utc": datetime.now(timezone.utc).isoformat(), "test_used_for_selection": False}
    write_json(directory / "selection.json", selection)  # Fixed before any TEST loader.
    return selection


def save_evaluation(model, loader, weights, metadata, split, directory):
    metrics, probabilities = baseline.epoch_pass(model, loader, weights)
    selected = metadata[metadata.split.eq(split)].reset_index(drop=True)
    predictions = selected[META_COLUMNS + ["label"]].copy()
    predictions["prediction"] = probabilities.argmax(axis=1)
    for i in range(4):
        predictions[f"p{i}"] = probabilities[:, i]
    sources_metrics, sources = baseline.source_evaluation(selected, probabilities)
    predictions.to_csv(directory / f"{loader.report_name}_predictions.csv", index=False)
    sources.to_csv(directory / f"{loader.report_name}_source_predictions.csv", index=False)
    return metrics, sources_metrics


def subgroup_diagnostics(directory, report_name, split):
    """Descriptive class/method recalls, including supports; never a selection score."""
    frozen = pd.read_csv(resolve_path(ROOT, SPLIT))
    frozen = frozen[frozen.split.eq(split)]
    predictions = pd.read_csv(directory / f"{report_name}_predictions.csv")
    joined = predictions.merge(frozen[["id", "method"]], on="id", how="left", validate="many_to_one")
    require(len(joined) == len(predictions) and joined.method.notna().all(), "Missing method metadata")
    # A source spanning methods is explicitly marked mixed; no arbitrary first-row choice.
    source_methods = frozen.groupby("name").method.agg(lambda x: "|".join(sorted(set(x))))
    sources = pd.read_csv(directory / f"{report_name}_source_predictions.csv").merge(
        source_methods.rename("method"), on="name", how="left", validate="one_to_one")
    rows = []
    for unit, frame in (("window", joined), ("source", sources)):
        require(frame.method.notna().all(), "Source method metadata mismatch")
        for (species, method), group in frame.groupby(["species", "method"], sort=True):
            rows.append({"unit": unit, "species": species, "method": method,
                         "support": len(group), "recall": float(group.prediction.eq(group.label).mean())})
    return rows


def batchnorm_diagnostics(model):
    return {name: {"running_mean_abs_mean": float(layer.running_mean.abs().mean()),
                   "running_var_min": float(layer.running_var.min()),
                   "running_var_max": float(layer.running_var.max()),
                   "num_batches_tracked": int(layer.num_batches_tracked)}
            for name, layer in model.named_modules()
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm) and layer.track_running_stats}


def protocol_record(evidence):
    record = baseline.protocol_record(evidence)
    record["training_configuration"] = deepcopy(CONFIG)
    record["protocol_version"] = VERSION
    record["accepted_class_weights"] = deepcopy(evidence["class_weights"])
    record["class_weights"] = effective_class_weights(evidence)
    record["implementation_sha256"].update({
        name: sha256(ROOT / name) for name in ("tools/experiment_training_v5.py", "tools/experiment_training_v4.py", "tools/experiment_training_v3.py", "tools/experiment_training_v2.py", "tools/experiment_runner.py")})
    return record


def run_branch(branch):
    require(EXPERIMENT_ID is not None, "Use experiment_runner.py to reserve a unique experiment ID")
    require(OUTPUT == ProjectPaths.from_env().experiment(EXPERIMENT_ID) / "results"
            and RUNS == OUTPUT / "runs", "Output must stay inside its reserved experiment")
    require_mutable_output(OUTPUT)
    require(branch in BRANCHES and torch.cuda.is_available() and torch.version.cuda is not None,
            "A known frontend and CUDA GPU are required; no CPU fallback")
    caches, evidence = prepare_caches()
    baseline.set_seeds()
    protocol = protocol_record(evidence)
    path = OUTPUT / "protocol.json"
    if path.exists():
        require(read_json(path) == protocol, "Protocol/runtime changed within experiment")
    else:
        write_json(path, protocol)
    features, metadata, cache = caches[branch]
    normalization = fit_normalization(features, metadata)
    weights = torch.tensor([protocol["class_weights"][s] for s in CLASSES], dtype=torch.float32, device="cuda")
    records = []
    for family in SELECTED_FAMILIES:
        run_id = f"{branch}_{family}_seed42"
        directory = RUNS / run_id
        require(not directory.exists(), f"Run already exists; preserve it: {directory}")
        directory.mkdir(parents=True)
        seeds = baseline.set_seeds()
        model = build_model(family, BRANCHES[branch]["shape"][1]).to("cuda")
        architecture = architecture_config(family, BRANCHES[branch]["shape"][1])
        write_json(directory / "normalization.json", normalization)
        write_json(directory / "class_weights.json", class_weight_record(CONFIG, evidence["class_weights"]))
        write_json(directory / "run_spec.json", {"run_id": run_id, "architecture": architecture,
            "module_repr": str(model), "protocol": protocol, "seeds": seeds, "frontend_cache": cache})
        train = make_loader(features, metadata, "train", normalization, shuffle=True)
        validation = make_loader(features, metadata, "validation", normalization)
        selection = fit_validation_checkpoint(model, train, validation, weights, directory)
        baseline.load_fixed_checkpoint(model, directory)
        measured = {}
        for name, split in (("train_eval", "train"), ("validation", "validation")):
            loader = make_loader(features, metadata, split, normalization)
            loader.report_name = name
            measured[name], measured[name + "_source"] = save_evaluation(
                model, loader, weights, metadata, split, directory)
        require(np.isclose(measured["validation"]["loss"], selection["selected_validation_metrics"]["loss"],
                           rtol=1e-6, atol=1e-7)
                and measured["validation"]["confusion_matrix"] == selection["selected_validation_metrics"]["confusion_matrix"],
                "Selected checkpoint does not reproduce validation metrics")
        write_json(directory / "diagnostics.json", {
            "batchnorm_selected_checkpoint": batchnorm_diagnostics(model),
            "train_eval_subgroups": subgroup_diagnostics(directory, "train_eval", "train"),
            "validation_subgroups": subgroup_diagnostics(directory, "validation", "validation"),
            "interpretation": "Descriptive supports/recalls; acquisition association does not establish causation"})
        test_metrics = source_test_metrics = None
        if CONFIG["evaluate_test"]:
            loader = make_loader(features, metadata, "test", normalization)
            loader.report_name = "test"
            test_metrics, source_test_metrics = save_evaluation(model, loader, weights, metadata, "test", directory)
        files = ["best_model.pt", "best_val_loss.pt", "best_val_macro_f1.pt", "history.json",
                 "selection.json", "run_spec.json", "normalization.json", "class_weights.json", "diagnostics.json",
                 "train_eval_predictions.csv", "train_eval_source_predictions.csv",
                 "validation_predictions.csv", "validation_source_predictions.csv"]
        if CONFIG["evaluate_test"]:
            files += ["test_predictions.csv", "test_source_predictions.csv"]
        result = {**selection, "status": "COMPLETE", "run_id": run_id, "protocol_version": VERSION,
            "frontend": branch, "frontend_identity": evidence["frontends"][branch], "model_family": family,
            "architecture": architecture, "module_repr": str(model), "seed": 42, "seeds": seeds,
            "training_configuration": deepcopy(CONFIG), "class_weights": protocol["class_weights"],
            "accepted_class_weights": protocol["accepted_class_weights"],
            "class_order": CLASSES, "counts": COUNTS, "protocol_sha256": digest_json(protocol),
            "train_eval_metrics": measured["train_eval"], "train_eval_source_metrics": measured["train_eval_source"],
            "validation_source_metrics": measured["validation_source"],
            "test_evaluated": CONFIG["evaluate_test"], "final_test_metrics": test_metrics,
            "source_test_metrics": source_test_metrics, "normalization": normalization,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "serialized_model_bytes": (directory / "best_model.pt").stat().st_size,
            "model_file_path": (directory / "best_model.pt").relative_to(OUTPUT).as_posix(),
            "software": protocol["software"], "cache_sha256": cache["sha256"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts_sha256": {f: sha256(directory / f) for f in files}}
        write_json(directory / "result.json", result)
        records.append(result)
    return pd.DataFrame([comparison_row(r) for r in records])


def verify_predictions(directory, report_name, split, metrics, source_metrics):
    predictions = pd.read_csv(directory / f"{report_name}_predictions.csv")
    grid = pd.read_csv(resolve_path(ROOT, MANIFEST))
    expected = grid[grid.split.eq(split)][META_COLUMNS].reset_index(drop=True)
    expected["label"] = expected.species.map({s: i for i, s in enumerate(CLASSES)})
    pd.testing.assert_frame_equal(predictions[META_COLUMNS + ["label"]], expected, check_dtype=False, check_exact=True)
    probabilities = predictions[[f"p{i}" for i in range(4)]].to_numpy()
    require(np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
            and np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6), "Invalid saved probabilities")
    require(np.array_equal(predictions.prediction, probabilities.argmax(axis=1)), "Saved argmax mismatch")
    actual = baseline.classification_metrics(predictions.label, predictions.prediction)
    require(all(metrics[k] == v for k, v in actual.items()), f"Saved {report_name} metrics mismatch")
    aggregated, sources = baseline.source_evaluation(expected, probabilities)
    require(aggregated == source_metrics, f"Saved {report_name} source metrics mismatch")
    pd.testing.assert_frame_equal(pd.read_csv(directory / f"{report_name}_source_predictions.csv"), sources,
                                  check_dtype=False, rtol=1e-6, atol=1e-8)


def validate_saved_run(record, directory, protocol):
    require(record["status"] == "COMPLETE" and record["protocol_version"] == protocol["protocol_version"] == VERSION,
            "Incomplete or unknown v5 run")
    require(record["protocol_sha256"] == digest_json(protocol)
            and record["training_configuration"] == protocol["training_configuration"], "Run protocol mismatch")
    config = record["training_configuration"]
    require(config == effective_config({k: config[k] for k in OVERRIDES}, config["evaluate_test"]), "Invalid v5 configuration")
    require(record["class_weights"] == protocol["class_weights"] and record["class_order"] == CLASSES
            and record["counts"] == COUNTS and record["software"] == protocol["software"], "Population/runtime mismatch")
    accepted = {s: COUNTS["train"] / (4 * n) for s, n in zip(CLASSES, TRAIN_COUNTS)}
    require(record["accepted_class_weights"] == protocol["accepted_class_weights"] == accepted,
            "Accepted class weights changed")
    require(record["class_weights"] == effective_class_weights({"class_weights": accepted}, config["class_weight_policy"], config["class_weight_exponent"]),
            "Effective class weights mismatch")
    require(read_json(directory / "class_weights.json") == class_weight_record(config, accepted),
            "Class-weight provenance mismatch")
    require(record["frontend_identity"] == protocol["frontends"][record["frontend"]], "Frontend mismatch")
    require(record["run_id"] == directory.name == f"{record['frontend']}_{record['model_family']}_seed42"
            and record["seed"] == 42, "Run identity mismatch")
    spec = architecture_config(record["model_family"], BRANCHES[record["frontend"]]["shape"][1])
    require(record["architecture"] == spec, "Architecture mismatch")
    # Counts are architectural, not CUDA-dependent; never run inference in this verifier.
    with torch.device("meta"):
        parameters = sum(p.numel() for p in build_model(record["model_family"], BRANCHES[record["frontend"]]["shape"][1]).parameters())
    require(record["parameter_count"] == parameters, "Parameter count mismatch")
    expected_files = {"best_model.pt", "best_val_loss.pt", "best_val_macro_f1.pt", "history.json",
        "selection.json", "run_spec.json", "normalization.json", "class_weights.json", "diagnostics.json",
        "train_eval_predictions.csv", "train_eval_source_predictions.csv",
        "validation_predictions.csv", "validation_source_predictions.csv"}
    if config["evaluate_test"]:
        expected_files |= {"test_predictions.csv", "test_source_predictions.csv"}
    require(set(record["artifacts_sha256"]) == expected_files, "Incomplete artifact hash record")
    for name, expected in record["artifacts_sha256"].items():
        require(sha256(directory / name) == expected, f"Run artifact changed: {directory.name}/{name}")
    require(record["serialized_model_bytes"] == (directory / "best_model.pt").stat().st_size
            and record["model_file_path"] == (directory / "best_model.pt").relative_to(OUTPUT).as_posix(), "Checkpoint path/size mismatch")
    selection = read_json(directory / "selection.json")
    require(selection["status"] == "VALIDATION_CHECKPOINT_FIXED" and not selection["test_used_for_selection"]
            and all(record[k] == v for k, v in selection.items() if k != "status"), "Selection/result mismatch")
    require(selection["criterion"] == SELECTION and selection["early_stopping_monitor"] == config["early_stopping_monitor"],
            "Selection/stopping criterion mismatch")
    history = read_json(directory / "history.json")
    require(len(history) == record["epochs_run"] and [h["epoch"] for h in history] == list(range(1, len(history) + 1)),
            "Invalid epoch history")
    for key, chosen, filename in (
        ("minimum_loss", min(history, key=lambda h: h["validation"]["loss"]), "best_val_loss.pt"),
        ("maximum_macro_f1", max(history, key=lambda h: h["validation"]["macro_f1"]), "best_val_macro_f1.pt")):
        candidate = selection["checkpoint_candidates"][key]
        require(candidate["epoch"] == chosen["epoch"] and candidate["metrics"] == chosen["validation"]
                and candidate["file"] == filename and candidate["model_sha256"] == sha256(directory / filename),
                "Checkpoint candidate does not match history")
    chosen = selection["checkpoint_candidates"]["maximum_macro_f1"]
    require(selection["best_epoch"] == chosen["epoch"] and selection["selected_validation_metrics"] == chosen["metrics"]
            and selection["selected_checkpoint"] == chosen["file"]
            and selection["model_sha256"] == chosen["model_sha256"] == sha256(directory / "best_model.pt")
            and selection["history_sha256"] == sha256(directory / "history.json"), "Selected checkpoint/hash mismatch")
    require(record["normalization"] == read_json(directory / "normalization.json")
            and record["normalization"]["kind"] == config["input_normalization"], "Normalization mismatch")
    verify_predictions(directory, "train_eval", "train", record["train_eval_metrics"], record["train_eval_source_metrics"])
    verify_predictions(directory, "validation", "validation", record["selected_validation_metrics"], record["validation_source_metrics"])
    require(record["test_evaluated"] == config["evaluate_test"], "TEST policy mismatch")
    if config["evaluate_test"]:
        verify_predictions(directory, "test", "test", record["final_test_metrics"], record["source_test_metrics"])
    else:
        require(record["final_test_metrics"] is None and record["source_test_metrics"] is None,
                "TEST metrics must be absent for validation-only tuning")


def comparison_row(record):
    metrics = record["final_test_metrics"]
    row = {"Frontend": record["frontend"], "Model": record["model_family"],
        "Params": record["parameter_count"], "Model Size (bytes)": record["serialized_model_bytes"],
        "Best epoch": record["best_epoch"], "Selected val loss": record["selected_validation_metrics"]["loss"],
        "Selected val Macro-F1": record["selected_validation_metrics"]["macro_f1"],
        "Validation source Macro-F1": record["validation_source_metrics"]["macro_f1"],
        "Train eval Macro-F1": record["train_eval_metrics"]["macro_f1"],
        "TEST evaluated": record["test_evaluated"]}
    if metrics is not None:
        row.update({"Accuracy": metrics["accuracy"], "Macro-F1": metrics["macro_f1"],
                    "Macro-Recall": metrics["macro_recall"], "Macro-Precision": metrics["macro_precision"],
                    "Weighted-F1": metrics["weighted_f1"]})
    return row
