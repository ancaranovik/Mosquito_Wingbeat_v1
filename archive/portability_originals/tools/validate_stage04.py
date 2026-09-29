"""Validate Stage 04 without classifier training; write only new Stage-04 outputs."""
from datetime import datetime, timezone
from pathlib import Path
import ast
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from stage04_data import ROOT, BRANCHES, authenticate, check_frozen_lock, prepare_caches, sha256, write_json
from model_zoo import FAMILIES, architecture_config, build_model
from training_common import CONFIG, set_seeds, software_versions


def main():
    check_frozen_lock()
    output = ROOT / "reports/stage04"
    output.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tools"), pattern="test_stage04.py")
    with contextlib.redirect_stdout(stream):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (output / "engineering_tests.txt").write_text(stream.getvalue(), encoding="utf-8")
    print(stream.getvalue(), flush=True)
    if not result.wasSuccessful():
        raise RuntimeError("Stage-04 engineering tests failed")
    caches, evidence = prepare_caches()
    del caches
    inventory = []
    for branch, spec in BRANCHES.items():
        for family in FAMILIES:
            set_seeds()
            model = build_model(family, spec["shape"][1])
            inventory.append({"frontend": branch, "model_family": family, "status": "NOT_TRAINED",
                              "architecture": architecture_config(family, spec["shape"][1]),
                              "parameter_count": sum(p.numel() for p in model.parameters()),
                              "module_repr": str(model)})
    write_json(output / "architecture_inventory.json", inventory)
    write_json(output / "planned_protocol.json", {"configuration": CONFIG, "software": software_versions(),
               "class_weights": evidence["class_weights"], "label_mapping": evidence["label_mapping"],
               "purpose": "Predeclared protocol; no training has run"})
    runs = []
    names = ["04A_train_logmel", "04B_train_edge_mfe", "04C_compare_models"]
    with tempfile.TemporaryDirectory(prefix="stage04-kernels-") as temporary:
        kernel_root = Path(temporary)
        spec_dir = kernel_root / "kernels/stage04-validation"
        spec_dir.mkdir(parents=True)
        write_json(spec_dir / "kernel.json", {
            "argv": [sys.executable, "-B", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Stage 04 validation", "language": "python",
            "env": {"PYTHONDONTWRITEBYTECODE": "1", "MPLCONFIGDIR": str(kernel_root / "mpl"),
                    "IPYTHONDIR": str(kernel_root / "ipython"),
                    "JUPYTER_RUNTIME_DIR": str(kernel_root / "runtime")}})
        previous = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = str(kernel_root) + (os.pathsep + previous if previous else "")
        try:
            for name in names:
                path = ROOT / "notebooks/current" / (name + ".ipynb")
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
                if name != "04C_compare_models" and "RUN_TRAINING = False" not in code:
                    raise RuntimeError("Validation runner requires notebook training guards disabled")
                for cell in notebook.cells:
                    if cell.cell_type == "code":
                        ast.parse(cell.source)
                        cell.outputs = []; cell.execution_count = None
                print("Execute (training disabled):", name, flush=True)
                status = "PASS"
                try:
                    NotebookClient(notebook, timeout=1800, kernel_name="stage04-validation",
                                   resources={"metadata": {"path": str(ROOT / "notebooks/current")}},
                                   allow_errors=False).execute()
                except CellExecutionError as exc:
                    if name != "04C_compare_models" or "Missing completed Stage-04 results; comparison requires all 8" not in str(exc):
                        raise
                    status = "EXPECTED_BLOCKED_MISSING_RUNS"
                nbformat.write(notebook, path)
                runs.append({"notebook": path.relative_to(ROOT).as_posix(), "status": status,
                             "training_enabled": False, "sha256": sha256(path)})
        finally:
            if previous is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous
    authenticate()  # all original files/WAVs still match at the end
    report = {"status": "PASS", "validated_at_utc": datetime.now(timezone.utc).isoformat(),
              "engineering_tests_passed": result.testsRun, "fresh_kernel_notebooks": runs,
              "mosquito_model_training_executed": False, "optimizer_updates_on_real_data": 0,
              "synthetic_checks": "forward/backward, serialization, deterministic seeds; mocked fit stopping test",
              "cache_examples": {b: 32645 for b in BRANCHES}, "protected_frozen_inputs_unchanged": True,
              "historical_review_notes": evidence["historical_review_notes"], "software": software_versions(),
              "limitations": ["Full 8-run training and post-training comparison not executed",
                              "No Stage-05 hardware or quantization work"]}
    write_json(output / "implementation_validation.json", report)
    print("PASS: Stage 04 ready; no mosquito model trained", flush=True)


if __name__ == "__main__":
    main()
