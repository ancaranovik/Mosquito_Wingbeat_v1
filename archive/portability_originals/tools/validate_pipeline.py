"""Run 00–03B in fresh kernels; preserve accepted scientific inputs and old reports.

Install requirements-pipeline.txt into an isolated environment, then run this script
from anywhere. No model is trained. Optional PDF export stays disabled. Successful
notebook runs save fresh execution evidence; failed runs do not replace notebooks.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import json
import os
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from pipeline_contract import authenticate_project, sha256_file, software_versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", nargs="*", help="Explicit notebook stems, otherwise all five in order")
    args = parser.parse_args()
    import nbformat
    from nbclient import NotebookClient

    names = args.notebooks or ["00_prepare_dataset", "01_dataset_audit", "02_segmentation",
                              "03A_logmel_reference", "03B_edge_mfe"]
    protected = {}
    # Record even historical files: scientific inputs, vendor/reference bytes, old
    # reproduction copies, exports and archives must stay byte-identical.
    for directory in ["data/audio", "data/metadata", "data/manifests", "reference", "third_party",
                      "archive", "reports/reproduced_00", "reports/pdf", "docs/reorganization"]:
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts and path != ROOT / "reference/README.md":
                protected[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    for relative in ["data/audits/current/core_audio_audit.csv",
                     "data/audits/current/core_example_grid_v2_stride15360_ref15600_audit.json"]:
        protected[relative] = sha256_file(ROOT / relative)
    authenticate_project(ROOT, audio=True)
    runs = []
    with tempfile.TemporaryDirectory(prefix="edge-ai-kernel-") as temporary:
        kernel_root = Path(temporary)
        spec_dir = kernel_root / "kernels" / "edge-ai-review"
        spec_dir.mkdir(parents=True)
        (spec_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-B", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Edge AI validation", "language": "python",
            "env": {"PYTHONDONTWRITEBYTECODE": "1", "MPLCONFIGDIR": str(kernel_root / "mpl")},
        }), encoding="utf-8")
        old_jupyter_path = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = str(kernel_root) + (os.pathsep + old_jupyter_path if old_jupyter_path else "")
        try:
            for name in names:
                path = ROOT / "notebooks/current" / (name + ".ipynb")
                notebook = nbformat.read(path, as_version=4)
                for cell in notebook.cells:
                    if cell.cell_type == "code":
                        ast.parse(cell.source)
                        cell.outputs = []
                        cell.execution_count = None
                start = time.monotonic()
                print(f"START {name}", flush=True)
                client = NotebookClient(notebook, timeout=3600, kernel_name="edge-ai-review",
                                        resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
                client.execute()
                assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == "code")
                nbformat.write(notebook, path)
                entry = {"notebook": path.relative_to(ROOT).as_posix(), "status": "PASS",
                         "seconds": round(time.monotonic() - start, 3), "sha256": sha256_file(path),
                         "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
                         "optional_pdf_export": "disabled"}
                runs.append(entry)
                print(f"PASS {name}: {entry['seconds']} seconds", flush=True)
        finally:
            if old_jupyter_path is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = old_jupyter_path
    for relative, expected in protected.items():
        assert sha256_file(ROOT / relative) == expected, f"Protected artifact changed: {relative}; STOP"
    identities = authenticate_project(ROOT, audio=True, require_grid_freeze=True)
    report = {"validated_at_utc": datetime.now(timezone.utc).isoformat(),
              "model_training_performed": False, "notebook_04_created": False,
              "software": software_versions(), "runs": runs,
              "protected_files_unchanged": len(protected), "protected_file_sha256": protected,
              "authenticated_inputs": identities}
    out = ROOT / "reports/validation/pretraining_review.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Protected artifacts unchanged: {len(protected)}; report: {out}", flush=True)


if __name__ == "__main__":
    main()
