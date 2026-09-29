from pathlib import Path
import ast
import hashlib
import json
import shutil
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(__file__).resolve().parent
PACKAGE = BUILD / "stage04_colab_bundle"
sys.path.insert(0, str(ROOT / "tools"))
import stage04_data as original_data
import training_common as original_training
import torch
import pandas as pd
import nbformat


def write(path, text):
    path = PACKAGE / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path, value):
    write(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def copy(relative):
    destination = PACKAGE / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / relative, destination)


print("Authenticate original inputs and validate existing caches (read-only)", flush=True)
grid, evidence = original_data.authenticate()
cache_records = {}
paired = {}
for branch in original_data.BRANCHES:
    features, metadata, cache = original_data.validate_cache(
        ROOT / "data/stage04_cache" / branch, grid, branch, evidence["frontends"][branch])
    cache_records[branch] = cache
    paired[branch] = metadata
    del features
pd.testing.assert_frame_equal(paired["03A"][original_data.META_COLUMNS + ["label"]],
                              paired["03B"][original_data.META_COLUMNS + ["label"]], check_exact=True)

PACKAGE.mkdir(exist_ok=True)
for branch in original_data.BRANCHES:
    for filename in ("features.npy", "examples.csv", "cache.json"):
        copy(f"data/stage04_cache/{branch}/{filename}")
for path in ("data/manifests/current/core_example_grid_v2_stride15360_ref15600_candidates.csv",
             "data/manifests/current/core_single_4species_frozen_split.csv", "tools/model_zoo.py"):
    copy(path)
for filename in ("scientific_lock.json", "frozen_input_lock.json", "frontend_execution_lock.json"):
    write(f"provenance/{filename}", (ROOT / "reports/stage04" / filename).read_text(encoding="utf-8"))

source = (ROOT / "tools/training_common.py").read_text(encoding="utf-8")


def replace(old, new, count=1):
    global source
    assert source.count(old) == count, (old, source.count(old))
    source = source.replace(old, new)


replace('"""One immutable local CPU training protocol for all eight Stage-04 experiments."""',
        '"""Stage-04 CUDA export: unchanged learning protocol; explicit device transport only."""\n'
        'import os\nos.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")')
replace('authenticate, SCIENTIFIC_LOCK_SHA256, EXECUTION_LOCK_SHA256)',
        'authenticate, SCIENTIFIC_LOCK_SHA256, EXECUTION_LOCK_SHA256, OUTPUT, MANIFEST, CONTRACT_SHA256)')
replace('"device": "cpu"', '"device": "cuda"')
replace('RUNS = ROOT / "reports/stage04/runs"', 'RUNS = OUTPUT / "runs"')
replace('packages = ("torch", "numpy", "scipy", "pandas", "nbformat", "ipykernel")',
        'packages = ("torch", "numpy", "pandas")')
replace('"processor": platform.processor(), "torch_build": torch.__config__.show()',
        '"processor": platform.processor(), "torch_build": torch.__config__.show(),\n'
        '            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),\n'
        '            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,\n'
        '            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),\n'
        '            "tf32": False, "mixed_precision": False')
replace('torch.manual_seed(CONFIG["seed"])', 'torch.manual_seed(CONFIG["seed"])\n'
        '    if torch.cuda.is_available():\n        torch.cuda.manual_seed_all(CONFIG["seed"])\n'
        '    torch.backends.cuda.matmul.allow_tf32 = False\n    torch.backends.cudnn.allow_tf32 = False')
replace('CPU only; identical software/hardware required; cross-platform bitwise identity not promised',
        'CUDA benchmark; identical software/GPU required; CPU-versus-CUDA bitwise identity not promised')
replace('for x, y in loader:\n            if training:',
        'for x, y in loader:\n            device = weights.device\n'
        '            x, y = x.to(device), y.to(device)\n            if training:')
replace('truths.append(y.numpy())', 'truths.append(y.cpu().numpy())')
replace('torch.softmax(logits.detach(), dim=1).numpy()', 'torch.softmax(logits.detach(), dim=1).cpu().numpy()')
replace('"execution_lock_sha256": EXECUTION_LOCK_SHA256,',
        '"execution_lock_sha256": EXECUTION_LOCK_SHA256, "export_contract_sha256": CONTRACT_SHA256,')
replace('"requirements-stage04.txt"', '"requirements.txt"', count=1)
replace('require(branch in BRANCHES, "Unknown frontend")',
        'require(branch in BRANCHES, "Unknown frontend")\n'
        '    require(torch.cuda.is_available(), "CUDA GPU required: select a Colab GPU runtime; no CPU fallback")\n'
        '    require(torch.version.cuda is not None, "Install the bundled CUDA PyTorch requirements")')
replace('ROOT / "reports/stage04/protocol.json"', 'OUTPUT / "protocol.json"', count=2)
replace('weights = torch.tensor([evidence["class_weights"][s] for s in CLASSES], dtype=torch.float32)',
        'weights = torch.tensor([evidence["class_weights"][s] for s in CLASSES], dtype=torch.float32, device="cuda")')
replace('model = build_model(family, BRANCHES[branch]["shape"][1])',
        'model = build_model(family, BRANCHES[branch]["shape"][1]).to("cuda")')
replace('(directory / "best_model.pt").relative_to(ROOT)', '(directory / "best_model.pt").relative_to(OUTPUT)', count=2)
replace('    from pipeline_contract import MANIFEST\n', '')
replace('ROOT / "reports/stage04/comparison.csv"', 'OUTPUT / "comparison.csv"')
replace('ROOT / "reports/stage04/comparison.json"', 'OUTPUT / "comparison.json"')
source = source.split('\nif __name__ == "__main__":')[0] + '\n'
write("tools/training_common.py", source)
write("requirements.txt", "--extra-index-url https://download.pytorch.org/whl/cu126\n"
      "torch==2.8.0+cu126\nnumpy==2.4.6\npandas==3.0.6\n")

contract = {
    "format_version": 1,
    "scope": "Verified existing caches only; no feature generation or upstream notebook execution",
    "authenticated_evidence": evidence,
    "cache_records": cache_records,
    "consumer_versions": {"torch": "2.8.0", "numpy": "2.4.6", "pandas": "3.0.6"},
    "original_training_configuration": original_training.CONFIG,
    "original_source_sha256": {p: original_data.sha256(ROOT / p) for p in
        ("tools/model_zoo.py", "tools/training_common.py", "tools/stage04_data.py",
         "notebooks/current/04A_train_logmel.ipynb", "notebooks/current/04B_train_edge_mfe.ipynb")},
    "packaged_input_sha256": {p.relative_to(PACKAGE).as_posix(): original_data.sha256(p)
        for p in PACKAGE.rglob("*") if p.is_file()},
    "adaptations": ["device=CUDA, tensor transfers and CPU conversion of saved predictions",
                    "CUDA seed/determinism and TF32 disabled; no AMP",
                    "cache-only authentication avoids Python-version AST differences and raw-audio dependencies",
                    "separate portable output root and consumer-only package dependencies"],
    "existing_cpu_results_included": False,
}
write_json("provenance/export_contract.json", contract)
contract_hash = original_data.sha256(PACKAGE / "provenance/export_contract.json")
adapter = (BUILD / "cache_adapter.py").read_text(encoding="utf-8").replace("__CONTRACT_SHA256__", contract_hash)
write("tools/stage04_data.py", adapter)

# Scientific training functions remain AST-identical to their canonical versions.
old_ast = ast.parse((ROOT / "tools/training_common.py").read_text(encoding="utf-8"))
new_ast = ast.parse(source)
unchanged = ["make_loader", "FeatureDataset", "classification_metrics", "source_evaluation",
             "fit_validation_checkpoint", "load_fixed_checkpoint", "comparison_row"]
for name in unchanged:
    a = next(n for n in old_ast.body if getattr(n, "name", None) == name)
    b = next(n for n in new_ast.body if getattr(n, "name", None) == name)
    assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False), name
assert original_data.sha256(PACKAGE / "tools/model_zoo.py") == original_data.sha256(ROOT / "tools/model_zoo.py")
write_json("provenance/build_validation.json", {
    "status": "PASS", "original_caches_validated_read_only": True,
    "examples_per_branch": 32645, "counts": original_data.COUNTS,
    "shapes": {b: s["shape"] for b, s in original_data.BRANCHES.items()},
    "exact_paired_metadata": True, "source_name_leakage": False,
    "models_byte_identical": True, "ast_identical_training_components": unchanged,
    "training_executed": False, "feature_generation_executed": False,
    "cuda_execution_tested_locally": False,
})
(PACKAGE / "results/runs").mkdir(parents=True, exist_ok=True)
print("Bundle sources and verified caches prepared", flush=True)
