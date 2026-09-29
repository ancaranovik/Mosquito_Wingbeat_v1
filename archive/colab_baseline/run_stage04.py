"""Colab entry point. Preflight never constructs an optimizer or calls fit."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
import torch
from stage04_data import ROOT, OUTPUT, BRANCHES, CLASSES, COUNTS, prepare_caches, write_json, read_json, require
from model_zoo import FAMILIES, build_model
from training_common import set_seeds, run_branch, compare_saved_results, software_versions, CONFIG, RUNS


def preflight(device):
    if device == "cuda":
        require(torch.cuda.is_available(), "No CUDA GPU: Runtime > Change runtime type > GPU")
        require(torch.version.cuda is not None, "CUDA PyTorch wheel required")
    set_seeds()
    caches, evidence = prepare_caches()
    del caches
    expected = {"small_cnn": {64: 23668, 40: 23668}, "ds_cnn": {64: 6244, 40: 6244},
                "tc_resnet8": {64: 65940, 40: 64788}, "cnn_lstm": {64: 13428, 40: 13428}}
    models = []
    for branch, spec in BRANCHES.items():
        width = spec["shape"][1]
        for family in FAMILIES:
            set_seeds()
            model = build_model(family, width).to(device).eval()
            with torch.no_grad():
                logits = model(torch.zeros(2, 1, 96, width, device=device))
            count = sum(p.numel() for p in model.parameters())
            require(logits.shape == (2, 4) and bool(torch.isfinite(logits).all()), "Synthetic forward failed")
            require(count == expected[family][width], "Parameter count changed")
            models.append({"branch": branch, "family": family, "parameters": count,
                           "output_shape": list(logits.shape), "device": device})
            del model, logits
    report = {"status": "PASS", "models": models, "counts": COUNTS, "class_order": CLASSES,
              "software": software_versions(), "training_configuration": CONFIG,
              "class_weights": evidence["class_weights"], "training_executed": False,
              "feature_generation_executed": False}
    write_json(OUTPUT / "runtime_validation.json", report)
    print(f"PASS: both caches; 32645 examples each; {COUNTS}; all eight synthetic forwards on {device}.", flush=True)
    for row in models:
        print(row, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "03A", "03B", "compare"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                        help="CPU is permitted only for non-training preflight")
    args = parser.parse_args()
    if args.action == "preflight":
        preflight(args.device)
    elif args.action == "compare":
        print(compare_saved_results().to_string(index=False), flush=True)
        matrices = {}
        for branch in BRANCHES:
            for family in FAMILIES:
                name = f"{branch}_{family}_seed42"
                result = read_json(RUNS / name / "result.json")
                matrices[name] = {"class_order": CLASSES,
                                  "example": result["final_test_metrics"]["confusion_matrix"],
                                  "source": result["source_test_metrics"]["confusion_matrix"]}
        write_json(OUTPUT / "confusion_matrices.json", matrices)
    else:
        require(args.device == "cuda", "Training is CUDA-only in this export")
        print(run_branch(args.action).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
