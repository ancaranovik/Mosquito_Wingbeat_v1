"""Read-only frozen frontend adapter and separately authenticated Stage-04 caches.

Only explicitly selected definition/import cells execute. Scientific source remains
in the frozen notebooks. No audit-writing, diagnostic selection or tuning cells run.
"""
from pathlib import Path
from project_paths import resolve_path, require_mutable_output
import contextlib
import ast
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import sys
import tempfile
import wave
import uuid

import numpy as np
import pandas as pd
import scipy
from scipy import signal
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = "reports/stage04/frozen_input_lock.json"
LOCK_SHA256 = "042d383587cc8e393c1f5c14bfde49b280c464881e04c48d05c68671f7cd8ab4"
SCIENTIFIC_LOCK_PATH = "reports/stage04/scientific_lock.json"
SCIENTIFIC_LOCK_SHA256 = "315687fc77549f546be08ef124e021933f64809d39bbb4f462c7f4ebce27986e"
EXECUTION_LOCK_PATH = "reports/stage04/frontend_execution_lock.json"
EXECUTION_LOCK_SHA256 = "a16a6fb0c88e68bec1b3e76295059ca022d7a136d67d59f5a6f33541833147cb"
CLASSES = ["an arabiensis", "an funestus ss", "an squamosus", "culex pipiens complex"]
COUNTS = {"train": 22613, "validation": 4937, "test": 5095}
TRAIN_COUNTS = [10457, 5128, 1426, 5602]
BRANCHES = {
    "03A": {"notebook": "notebooks/current/03A_logmel_reference.ipynb",
            "audit": "data/audits/current/03A_logmel_reference_frontend_audit.json",
            "cells": [8, 10, 12], "shape": [96, 64], "function": "logmel_reference",
            "extract": "extract_reference_context", "contract_key": "frontend"},
    "03B": {"notebook": "notebooks/current/03B_edge_mfe.ipynb",
            "audit": "data/audits/current/03B_edge_mfe_implementation_audit.json",
            "cells": [8, 10, 12, 13, 15], "shape": [96, 40], "function": "edge_mfe_v4",
            "extract": "extract_edge_context", "contract_key": "frozen_contract"},
}
META_COLUMNS = ["example_uid", "id", "name", "species", "split", "example_index",
                "start_sample_16k", "start_second", "example_stride_samples",
                "reference_context_samples", "reference_end_sample_16k",
                "reference_end_second", "target_sample_rate"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    """Atomic write; caller supplies a Stage-04 destination."""
    path = require_mutable_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def check_frozen_lock():
    require(__debug__, "Stage 04 must not run with python -O: frozen functions use assertions")
    require(sha256(ROOT / LOCK_PATH) == LOCK_SHA256, "Stage-04 frozen-input lock changed")
    require(sha256(ROOT / SCIENTIFIC_LOCK_PATH) == SCIENTIFIC_LOCK_SHA256, "Scientific lock changed; STOP")
    require(sha256(ROOT / EXECUTION_LOCK_PATH) == EXECUTION_LOCK_SHA256, "Frontend execution lock changed; STOP")
    execution = read_json(ROOT / EXECUTION_LOCK_PATH)
    for branch, spec in BRANCHES.items():
        cells = read_json(ROOT / spec["notebook"])["cells"]
        actual = {str(i): hashlib.sha256(ast.dump(ast.parse("".join(cells[i]["source"])),
                   include_attributes=False).encode()).hexdigest() for i in spec["cells"]}
        require(actual == execution[branch], f"{branch} executed definitions/constants changed; STOP")
    lock = read_json(ROOT / LOCK_PATH)
    # Whole-notebook hashes are provenance only. Scientific gates below use
    # canonical ASTs/artifact hashes, so notebook serialization is irrelevant.
    for relative, expected in lock.items():
        if relative == "tools/pipeline_contract.py":
            # The original accepted contract stays byte-authenticated. Only path
            # routing and audit persistence changed in the portable adapter.
            original = ROOT / "archive/portability_originals" / relative
            require(sha256(original) == expected, "Original pipeline contract changed")
            old = ast.parse(original.read_text(encoding="utf-8"))
            new = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            names = ("validate_grid", "feature_audio_contract", "require_passed_gates",
                     "load_pinned_ei_wrapper", "sha256_file", "software_versions")
            for name in names:
                before = next(n for n in old.body if getattr(n, "name", None) == name)
                after = next(n for n in new.body if getattr(n, "name", None) == name)
                require(ast.dump(before) == ast.dump(after), f"Scientific contract changed: {name}")
            continue
        if relative.startswith("notebooks/current/") or relative in {
            "data/audits/current/02_example_grid_v2_freeze_audit.json",
            "data/audits/current/03A_logmel_reference_frontend_audit.json",
            "data/audits/current/03B_edge_mfe_implementation_audit.json",
        }:
            continue
        require(sha256(resolve_path(ROOT, relative)) == expected, f"Frozen input changed: {relative}; STOP")
    scientific = read_json(ROOT / SCIENTIFIC_LOCK_PATH)
    for relative, expected in scientific["scientific_gate"]["artifact_sha256"].items():
        require(sha256(resolve_path(ROOT, relative)) == expected, f"Scientific artifact changed: {relative}; STOP")
    for notebook, functions in scientific["scientific_gate"]["ast_sha256"].items():
        actual = {}
        for cell in read_json(ROOT / notebook)["cells"]:
            if cell["cell_type"] != "code":
                continue
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.FunctionDef) and node.name in functions:
                    actual[node.name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        require(actual == functions, f"Scientific function changed: {notebook}; STOP")
    return lock


def authenticate():
    lock = check_frozen_lock()
    from pipeline_contract import authenticate_project, validate_grid, MANIFEST, SPLIT, GRID_AUDIT
    evidence = authenticate_project(ROOT, audio=True, require_grid_freeze=True)
    review = read_json(ROOT / "reports/validation/pretraining_review.json")
    require(len(review["runs"]) == 5, "Incomplete accepted notebook review")
    review_notes = []
    for run in review["runs"]:
        require(run["status"] == "PASS", f"Failed historical notebook review: {run['notebook']}")
        current = sha256(ROOT / run["notebook"])
        if current != run["sha256"]:
            review_notes.append({"notebook": run["notebook"], "historical_sha256": run["sha256"],
                                 "current_frozen_sha256": current,
                                 "note": "Historical whole-notebook hash differs; handled as provenance only. Scientific AST/artifact gates are checked separately."})
    # Independently check all recorded frozen scientific function ASTs, including 02.
    for name, expected in review["supplementary_checks"]["frozen_function_ast_sha256_unchanged"].items():
        actual = {}
        for cell in read_json(ROOT / "notebooks/current" / name)["cells"]:
            if cell["cell_type"] == "code":
                for node in ast.parse("".join(cell["source"])).body:
                    if isinstance(node, ast.FunctionDef) and node.name in expected:
                        actual[node.name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        require(actual == expected, f"Frozen scientific function changed: {name}; STOP")
    # DSP binaries matter as well as source bytes. Do not silently switch versions.
    for package in ("numpy", "pandas", "scipy"):
        require(importlib.metadata.version(package) == review["software"][package],
                f"Install requirements-stage04.txt: frozen {package} version required")
    grid = pd.read_csv(resolve_path(ROOT, MANIFEST))
    split = pd.read_csv(resolve_path(ROOT, SPLIT))
    geometry = validate_grid(grid, split, read_json(resolve_path(ROOT, GRID_AUDIT)))
    require(sorted(grid.species.unique()) == CLASSES, "Frozen label mapping changed")
    require(grid.groupby("split").size().to_dict() == COUNTS, "Split counts changed")
    counts = grid[grid.split.eq("train")].species.value_counts().reindex(CLASSES).tolist()
    require(counts == TRAIN_COUNTS, "TRAIN class distribution changed")
    identities = {}
    for branch, spec in BRANCHES.items():
        audit = read_json(resolve_path(ROOT, spec["audit"]))
        require(audit["status"] == "VERIFIED_FROZEN", f"{branch} not frozen")
        require(audit["final_gates"] and all(v is True for v in audit["final_gates"].values()),
                f"{branch} has a failed gate")
        require(audit["authenticated_inputs"]["sha256"] == evidence["sha256"], "Upstream identity mismatch")
        contract = audit[spec["contract_key"]]
        require(contract["output_shape"] == spec["shape"], "Frozen shape mismatch")
        if branch == "03B":
            require(contract["noise_floor_db"] == -70, "Frozen noise floor mismatch")
        ast_identity = read_json(ROOT / SCIENTIFIC_LOCK_PATH)["scientific_gate"]["ast_sha256"][spec["notebook"]]
        identity = {"notebook_sha256_provenance": lock[spec["notebook"]], "frontend_ast_sha256": ast_identity,
                    "execution_ast_sha256": read_json(ROOT / EXECUTION_LOCK_PATH)[branch],
                    "audit_sha256": sha256(resolve_path(ROOT, spec["audit"])),
                    "contract": contract, "definition_cells": spec["cells"], "frontend": branch}
        identities[branch] = dict(identity, sha256=digest_json(identity))
    evidence.update(frozen_lock_sha256=LOCK_SHA256, historical_review_notes=review_notes,
                    geometry=geometry, counts=COUNTS,
                    label_mapping={s: i for i, s in enumerate(CLASSES)}, train_counts=counts,
                    class_weights={s: COUNTS["train"] / (4 * n) for s, n in zip(CLASSES, counts)},
                    frontends=identities)
    return grid, evidence


def load_frontend(branch):
    """Call after authenticate(); recheck notebook/source identities before exec."""
    check_frozen_lock()
    sys.dont_write_bytecode = True  # no __pycache__ writes into frozen source trees
    from pipeline_contract import EXPECTED_SHA256, load_pinned_ei_wrapper
    for rel, expected in EXPECTED_SHA256.items():
        require(sha256(resolve_path(ROOT, rel)) == expected, f"Accepted source changed: {rel}")
    spec = BRANCHES[branch]
    scope = {"np": np, "pd": pd, "scipy": scipy, "signal": signal, "wavfile": wavfile,
             "Path": Path, "wave": wave, "sys": sys, "importlib": importlib,
             "PROJECT": ROOT, "AUDIO_DIR": resolve_path(ROOT, "data/audio"),
             "EI_MFE_DIR": ROOT / "third_party/edgeimpulse-processing-blocks/mfe",
             "EI_SPEECHPY_DIR": ROOT / "third_party/edgeimpulse-processing-blocks/mfe/third_party/speechpy",
             "sha256_file": sha256, "EXPECTED_SHA256": EXPECTED_SHA256,
             "load_pinned_ei_wrapper": load_pinned_ei_wrapper}
    notebook = read_json(ROOT / spec["notebook"])
    with contextlib.redirect_stdout(io.StringIO()):
        for index in spec["cells"]:
            cell = notebook["cells"][index]
            require(cell["cell_type"] == "code", "Definition cell changed")
            exec(compile("".join(cell["source"]), f"{spec['notebook']}:cell{index}", "exec"), scope)
    return scope


def verify_frontend_parity(grid):
    """Synthetic/boundary and one TRAIN context per class; never select by signal quality."""
    scopes = {branch: load_frontend(branch) for branch in BRANCHES}
    spec = importlib.util.spec_from_file_location("stage04_reference_mel", ROOT / "reference/humbug/vggish/mel_features.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    contexts = {"zeros": np.zeros(15600), "noise": np.random.default_rng(42).normal(0, .05, 15600)}
    for index in (255, 319, 15200, 15599):
        x = np.zeros(15600); x[index] = .5
        contexts[f"impulse_{index}"] = x
    for _, row in grid[grid.split.eq("train")].sort_values("example_uid").groupby("species").head(1).iterrows():
        a = scopes["03A"]["load_core_wav_16k"](row.id)
        b = scopes["03B"]["load_core_wav_16k"](row.id)
        require(np.array_equal(a, b), "Frozen waveform loaders disagree; STOP")
        contexts[row.example_uid] = scopes["03A"]["extract_reference_context"](row, a)
    rows = []
    for name, x in contexts.items():
        a = scopes["03A"]["logmel_reference"](x)
        expected = upstream.log_mel_spectrogram(x, audio_sample_rate=16000,
            log_offset=.01, window_length_secs=.025, hop_length_secs=.010,
            num_mel_bins=64, lower_edge_hertz=125., upper_edge_hertz=7500.)
        require(a.shape == (96, 64) and a.dtype == np.float64, "03A output contract")
        require(np.allclose(a, expected, atol=1e-10, rtol=0), "03A numerical parity failed; STOP")
        b = scopes["03B"]["edge_mfe_v4"](x)
        ref = scopes["03B"]["pinned_generate_features"](
            4, False, x * 2**15, ["audio"], 16000, .020, .010, 40, 256, 0, 0, 101, -70)
        expected_b = np.asarray(ref["features"], dtype=np.float32).reshape(96, 40)
        require(b.dtype == np.float32 and np.array_equal(b, expected_b), "03B numerical parity failed; STOP")
        rows.append({"case": name, "03A_max_abs_diff": float(np.max(np.abs(a-expected))), "03B_exact": True})
    return rows


def cache_metadata(grid, branch, identity):
    frame = grid[META_COLUMNS].copy()
    frame["label"] = frame.species.map({s: i for i, s in enumerate(CLASSES)})
    frame["feature_shape"] = "x".join(map(str, BRANCHES[branch]["shape"]))
    frame["frontend"] = branch
    frame["frontend_identity"] = identity["sha256"]
    return frame


def validate_cache(directory, grid, branch, identity):
    directory = Path(directory)
    try:
        info = read_json(directory / "cache.json")
    except PermissionError as exc:
        raise RuntimeError(f"Cannot read Stage-04 cache {directory / 'cache.json'}: Windows ACL or file lock. "
                           "Preserve the derived cache, close kernels holding it, and rebuild in an accessible "
                           "Stage-04 directory. Do not change frozen inputs or pins.") from exc
    except (FileNotFoundError, IsADirectoryError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Incomplete Stage-04 cache at {directory}; preserve it and rebuild derived data") from exc
    require(info.get("schema_version") == 2, "Stale cache schema; rebuild derived Stage-04 cache")
    require(info.get("rows") == len(grid) and info.get("dtype") == "float32"
            and info.get("normalization") == "none", "Cache contract mismatch")
    require(info.get("scientific_lock_sha256") == SCIENTIFIC_LOCK_SHA256, "Cache scientific lock mismatch")
    require(info["frontend_identity"] == identity, "Cache frontend identity mismatch")
    require(info["frozen_lock_sha256"] == LOCK_SHA256, "Cache frozen-input lock mismatch")
    require(info["software"] == {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy")},
            "Cache numerical runtime differs")
    require(set(info["sha256"]) == {"features.npy", "examples.csv"}, "Incomplete cache hash record")
    for file, expected in info["sha256"].items():
        require(sha256(directory / file) == expected, f"Cache bytes changed: {file}")
    actual = pd.read_csv(directory / "examples.csv")
    pd.testing.assert_frame_equal(actual, cache_metadata(grid, branch, identity), check_dtype=False, check_exact=True)
    features = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
    require(features.shape == (32645, *BRANCHES[branch]["shape"]) and features.dtype == np.float32,
            "Cache shape/dtype mismatch")
    for start in range(0, len(features), 1024):
        require(np.isfinite(features[start:start+1024]).all(), "Nonfinite cache features")
    return features, actual, info


@contextlib.contextmanager
def cache_staging(base, branch):
    # Windows tempfile.mkdtemp uses mode 0700 (owner-only ACL). Renaming that
    # directory preserves its ACL and can deny access to subsequent kernels.
    # Ordinary mkdir inherits the destination parent's ACL; keep failed builds
    # clearly separate for diagnosis rather than publishing or deleting them.
    folder = Path(base) / f"{branch}-building-{uuid.uuid4().hex}"
    folder.mkdir()
    yield folder


def prepare_caches():
    """Read-only cache consumer. Missing caches fail; generation is never implicit."""
    from cache_consumer import prepare_caches as consume
    return consume()
