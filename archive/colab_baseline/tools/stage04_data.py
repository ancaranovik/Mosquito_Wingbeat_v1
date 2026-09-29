"""Portable, read-only Stage-04 cache consumer. Feature extraction is not included."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import tempfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("STAGE04_RESULTS", str(ROOT / "results"))).resolve()
CONTRACT_SHA256 = "d438c5901fe7f78e79d6e18fa09731f01d69f121b9e0035de57eaebb61fd7bd4"
MANIFEST = "data/manifests/current/core_example_grid_v2_stride15360_ref15600_candidates.csv"
SPLIT = "data/manifests/current/core_single_4species_frozen_split.csv"
CLASSES = ["an arabiensis", "an funestus ss", "an squamosus", "culex pipiens complex"]
COUNTS = {"train": 22613, "validation": 4937, "test": 5095}
TRAIN_COUNTS = [10457, 5128, 1426, 5602]
BRANCHES = {"03A": {"shape": [96, 64]}, "03B": {"shape": [96, 40]}}
META_COLUMNS = ["example_uid", "id", "name", "species", "split", "example_index",
                "start_sample_16k", "start_second", "example_stride_samples",
                "reference_context_samples", "reference_end_sample_16k",
                "reference_end_second", "target_sample_rate"]
LOCK_SHA256 = "042d383587cc8e393c1f5c14bfde49b280c464881e04c48d05c68671f7cd8ab4"
SCIENTIFIC_LOCK_SHA256 = "315687fc77549f546be08ef124e021933f64809d39bbb4f462c7f4ebce27986e"
EXECUTION_LOCK_SHA256 = "a16a6fb0c88e68bec1b3e76295059ca022d7a136d67d59f5a6f33541833147cb"


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def authenticate():
    """Authenticate exported bytes, not ASTs in an unrelated Python runtime."""
    require(sha256(ROOT / "provenance/export_contract.json") == CONTRACT_SHA256,
            "Export contract changed; restore the original bundle")
    contract = read_json(ROOT / "provenance/export_contract.json")
    for path, expected in contract["packaged_input_sha256"].items():
        require(sha256(ROOT / path) == expected, f"Packaged input changed: {path}")
    for package, version in contract["consumer_versions"].items():
        require(importlib.metadata.version(package).split("+")[0] == version,
                f"Wrong {package} version; run the notebook's isolated environment setup")
    grid = pd.read_csv(ROOT / MANIFEST)
    split = pd.read_csv(ROOT / SPLIT)
    require(len(grid) == 32645 and grid.example_uid.is_unique, "Example population changed")
    require(grid.groupby("split").size().to_dict() == COUNTS, "Split counts changed")
    require(sorted(grid.species.unique()) == CLASSES, "Class order changed")
    require(grid[META_COLUMNS].notna().all().all(), "Missing metadata")
    require(len(split) == 1898 and split.id.is_unique and split.name.nunique() == 1189,
            "Frozen source population changed")
    require(grid.groupby("name").split.nunique().eq(1).all()
            and split.groupby("name").split.nunique().eq(1).all(), "Source-name leakage")
    parent = grid[["id", "name", "species", "split"]].drop_duplicates().sort_values("id").reset_index(drop=True)
    pd.testing.assert_frame_equal(parent, split[parent.columns].sort_values("id").reset_index(drop=True),
                                  check_dtype=False, check_exact=True)
    counts = grid[grid.split.eq("train")].species.value_counts().reindex(CLASSES).tolist()
    require(counts == TRAIN_COUNTS, "TRAIN class counts changed")
    evidence = contract["authenticated_evidence"]
    weights = {s: COUNTS["train"] / (4 * n) for s, n in zip(CLASSES, counts)}
    require(evidence["class_weights"] == weights, "TRAIN weights changed")
    evidence["cache_consumption_only"] = True
    evidence["export_contract_sha256"] = CONTRACT_SHA256
    return grid, evidence


def prepare_caches():
    """Verify both immutable caches. Never create, replace or regenerate features."""
    grid, evidence = authenticate()
    contract = read_json(ROOT / "provenance/export_contract.json")
    caches = {}
    for branch, spec in BRANCHES.items():
        directory = ROOT / "data/stage04_cache" / branch
        info = read_json(directory / "cache.json")
        require(info == contract["cache_records"][branch], f"{branch} cache contract changed")
        require(info["schema_version"] == 2 and info["normalization"] == "none"
                and info["dtype"] == "float32" and info["rows"] == 32645, "Wrong cache contract")
        require(info["frontend_identity"] == evidence["frontends"][branch], "Frontend identity mismatch")
        require(info["scientific_lock_sha256"] == SCIENTIFIC_LOCK_SHA256
                and info["frozen_lock_sha256"] == LOCK_SHA256, "Scientific lock mismatch")
        meta = pd.read_csv(directory / "examples.csv")
        expected = grid[META_COLUMNS].copy()
        expected["label"] = expected.species.map({s: i for i, s in enumerate(CLASSES)})
        expected["feature_shape"] = "x".join(map(str, spec["shape"]))
        expected["frontend"] = branch
        expected["frontend_identity"] = evidence["frontends"][branch]["sha256"]
        pd.testing.assert_frame_equal(meta, expected, check_dtype=False, check_exact=True)
        features = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
        require(features.shape == (32645, *spec["shape"]) and features.dtype == np.float32,
                f"{branch} shape/dtype mismatch")
        for start in range(0, len(features), 1024):
            require(np.isfinite(features[start:start + 1024]).all(), f"{branch} nonfinite features")
        caches[branch] = features, meta, info
    pd.testing.assert_frame_equal(caches["03A"][1][META_COLUMNS + ["label"]],
                                  caches["03B"][1][META_COLUMNS + ["label"]], check_exact=True)
    evidence["identical_branch_example_uids_labels_splits"] = True
    evidence["cache_records"] = {b: caches[b][2] for b in BRANCHES}
    write_json(OUTPUT / "preflight.json", evidence)
    return caches, evidence
