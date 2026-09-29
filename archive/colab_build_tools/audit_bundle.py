from pathlib import Path
import ast
import difflib
import hashlib
import importlib
import json
import os
import sys

BUILD = Path(__file__).resolve().parent
ROOT = BUILD.parent
PACKAGE = BUILD / "stage04_colab_bundle"
os.environ["STAGE04_RESULTS"] = str(BUILD / "validation_results")
sys.dont_write_bytecode = True
sys.path.insert(0, str(PACKAGE / "tools"))
sys.path.insert(0, str(PACKAGE))
import stage04_data as data
import training_common as training
import model_zoo
import run_stage04
import torch

for module in (data, training, model_zoo, run_stage04):
    assert Path(module.__file__).resolve().is_relative_to(PACKAGE.resolve())

original = (ROOT / "tools/training_common.py").read_text(encoding="utf-8")
bundled = (PACKAGE / "tools/training_common.py").read_text(encoding="utf-8")
original_ast = ast.parse(original)
original_config = next(ast.literal_eval(n.value) for n in original_ast.body
                       if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets))
assert {k: v for k, v in training.CONFIG.items() if k != "device"} == {
    k: v for k, v in original_config.items() if k != "device"}
assert training.CONFIG["device"] == "cuda"
assert original_config["device"] == "cpu"

imports = set()
for path in list((PACKAGE / "tools").glob("*.py")) + [PACKAGE / "run_stage04.py"]:
    source = path.read_text(encoding="utf-8")
    assert "D:\\" not in source and "C:\\" not in source and ".venv-stage04" not in source
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(n.name.split('.')[0] for n in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split('.')[0])
for name in sorted(imports):
    importlib.import_module(name)
assert imports - set(sys.stdlib_module_names) == {"numpy", "pandas", "torch", "stage04_data", "model_zoo", "training_common"}

# Evaluation-only weighted loss regression; no optimizer and no model fitting.
class SyntheticLogits(torch.nn.Module):
    def forward(self, x):
        return x

x = torch.tensor([[2., 0, 0, 0], [0, 0, 0, 0], [0, 0, 1., 0]])
y = torch.tensor([0, 1, 2])
w = torch.tensor([.5, 2., 3., 1.])
metrics, probs = training.epoch_pass(SyntheticLogits(), [(x[:2], y[:2]), (x[2:], y[2:])], w)
assert abs(metrics["loss"] - float(torch.nn.functional.cross_entropy(x, y, weight=w))) < 1e-6
assert probs.shape == (3, 4)

# Confirm training fails closed without CUDA, before cache access or output creation.
from unittest.mock import patch
with patch.object(torch.cuda, "is_available", return_value=False), \
     patch.object(training, "prepare_caches", side_effect=AssertionError("Should not be reached")):
    try:
        training.run_branch("03A")
    except RuntimeError as exc:
        assert "CUDA GPU required" in str(exc)
    else:
        raise AssertionError("CPU training fallback must not exist")

contract = data.read_json(PACKAGE / "provenance/export_contract.json")
assert data.sha256(PACKAGE / "provenance/export_contract.json") == data.CONTRACT_SHA256
for relative, expected in contract["packaged_input_sha256"].items():
    assert data.sha256(PACKAGE / relative) == expected, relative

validation = data.read_json(PACKAGE / "provenance/build_validation.json")
validation.update(import_closure_verified=True, external_runtime_dependencies=["numpy", "pandas", "torch"],
                  local_module_imports_from_bundle_only=True, learning_config_identical_except_device=True,
                  cpu_training_refused=True, synthetic_weighted_loss_passed=True)
data.write_json(PACKAGE / "provenance/build_validation.json", validation)
(PACKAGE / "provenance/cuda_adaptation.diff").write_text(''.join(difflib.unified_diff(
    original.splitlines(keepends=True), bundled.splitlines(keepends=True),
    fromfile='canonical/tools/training_common.py', tofile='bundle/tools/training_common.py')), encoding='utf-8')
print("PASS: isolated bundle imports, configuration parity, weighted loss, CUDA-only gate and input hashes")
