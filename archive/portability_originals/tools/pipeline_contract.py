"""Identity and audit gates for the accepted 00–03B experiment; no DSP changes."""
from pathlib import Path
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile

EXPECTED_SHA256 = {'archive/legacy_data/core_096s_candidate_windows.csv': '08df667e694466d23bd8c0a4ecb215f8505ea2e72c71cc131574bce4152628f9',
 'data/audits/current/core_audio_audit.csv': '8551ada9c64998e69922779b42054ec17245c8923457877d24b633813a1f2965',
 'data/audits/current/core_example_grid_v2_stride15360_ref15600_audit.json': 'e94cb9c65e4ce3e63266ac34b05d03bbe372a2f51935bfc3f8c7af20a36ee75b',
 'data/manifests/current/core_example_grid_v2_stride15360_ref15600_candidates.csv': '36881d79d33cd42521aa24004236022dc25b4927f65ae0246a76832835206234',
 'data/manifests/current/core_single_4species_frozen_split.csv': '3e40928622dcea26e723540cfb493a35fd1d78393bf5531810583eae7966e0c4',
 'data/manifests/current/core_single_4species_metadata.csv': 'e72974da9a4e1a419bb459363be37ff91fdfdd6e65ff9a32582e9de13e786614',
 'data/metadata/neurips_2021_zenodo_0_0_1.csv': 'baa4de3da5965dfea41dbd505023d392fab4f01db1e6e137cf50442423f79727',
 'reference/humbug/vggish/mel_features.py': '68803c00743cb43139db12836b5e745a977cdb81443257fa716cd520d7e5e948',
 'reference/humbug/vggish/vggish.py': 'fd227c0ff5e6646c5b8097d54fa2c1d1a3921bab7dda037f58cee4a8bfaa6ee8',
 'reference/humbug/vggish/vggish_input.py': 'a2a1bb264f1bd1a276c2c73964a27b64fa9dd65e4ace350d93614cb721501d13',
 'reference/humbug/vggish/vggish_params.py': '61698102149a94a6ce8f567f6abfad8b1b04760825396f33f9d4f31e66e79c5f',
 'third_party/edgeimpulse-processing-blocks/mfe/__init__.py': 'a72e578358398436068d455d3cd4efb2e336b5b23bb373b44ece7faf427dfba9',
 'third_party/edgeimpulse-processing-blocks/mfe/common/dataset.py': '2e999e679d126619f2e7b72f263637bb78140fcacb3469628348b9e0e1cdee81',
 'third_party/edgeimpulse-processing-blocks/mfe/common/errors.py': '3a1b74b873961bbd5012a71ef57ac06f1a3d05c9ae22aea195fb9b4de2f8121c',
 'third_party/edgeimpulse-processing-blocks/mfe/common/graphing.py': 'bd98e8d8d39444e4b10043159ea5d8d80ba859e80c43776e479c9ffdf52b551e',
 'third_party/edgeimpulse-processing-blocks/mfe/common/sampling.py': '4a3532ceeb51c9af9a0dc1df90151888554469937abf85daac111a6c43b22b52',
 'third_party/edgeimpulse-processing-blocks/mfe/common/spectrum.py': '09f6ba0335905f193ab8b5dbd15a8cf5b88e059201f913aa4e05c6662e43dd29',
 'third_party/edgeimpulse-processing-blocks/mfe/common/wavelet.py': 'f2f5c47a387cd2a879b486f6af6a12be2a52f9f3f2e42ade6289d13247c0a7c5',
 'third_party/edgeimpulse-processing-blocks/mfe/dsp-server.py': 'f3afe7717f8612a2d925aac5088037a793900b711b7818966466fa8ecf46ff16',
 'third_party/edgeimpulse-processing-blocks/mfe/dsp.py': '68e9ac2873c822791d6e4651ae4ed16aa9e0aaf4b61ffda8a20a4e2366f16f4e',
 'third_party/edgeimpulse-processing-blocks/mfe/parameters.json': '9aa802f189079f552018a2a45151ae7bab6a5eb20e0ccd04f9865b006a2c5b8e',
 'third_party/edgeimpulse-processing-blocks/mfe/third_party/speechpy/__init__.py': 'ccfd42a6cc1ee35d88ee4771e8a493a987575d1972cb0d49f79664a27f62a187',
 'third_party/edgeimpulse-processing-blocks/mfe/third_party/speechpy/feature.py': '85f7d5c72470a37bd2d88e26b02ccae5b293d63ec881247d7b1481911dbd6d3f',
 'third_party/edgeimpulse-processing-blocks/mfe/third_party/speechpy/functions.py': '12b921981eb3b0cda1de223cce445e5ee6941c2d8516df16d9f19dc292a8296c',
 'third_party/edgeimpulse-processing-blocks/mfe/third_party/speechpy/processing.py': '223e652894b20d47580d96bd198a25bbf48627f8b97e0b3828d7a023d8493619'}
MANIFEST = "data/manifests/current/core_example_grid_v2_stride15360_ref15600_candidates.csv"
SPLIT = "data/manifests/current/core_single_4species_frozen_split.csv"
GRID_AUDIT = "data/audits/current/core_example_grid_v2_stride15360_ref15600_audit.json"
GRID_FREEZE = "data/audits/current/02_example_grid_v2_freeze_audit.json"
EI_REVISION = "9b0ceb2b5d22658d3193a5b77dc773cfc8d4cab0"
HUMBUG_TREE = "50656758594982480f568598874f79c222432e01"


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def authenticate_project(project, *, audio=True, require_grid_freeze=False):
    """Fail closed on every accepted data/source identity, including WAV bytes."""
    project = Path(project)
    for relative, expected in EXPECTED_SHA256.items():
        assert sha256_file(project / relative) == expected, f"Accepted identity changed: {relative}"
    report = json.loads((project / GRID_AUDIT).read_text(encoding="utf-8"))
    assert report["manifest_sha256"] == EXPECTED_SHA256[MANIFEST]
    assert report["checks"]["all_assertions_passed"] is True
    assert report["checks"]["physically_checked_contexts"] == 32645
    assert len(report["clips"]) == 1898
    if audio:
        for clip in report["clips"]:
            assert sha256_file(project / "data/audio" / f"{clip['id']}.wav") == clip["wav_sha256"], f"WAV identity changed: {clip['id']}"
    result = {"sha256": dict(EXPECTED_SHA256), "wav_identity_record": GRID_AUDIT,
              "wav_files_authenticated": 1898 if audio else 0}
    if require_grid_freeze:
        freeze_path = project / GRID_FREEZE
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        assert freeze["status"] == "ACCEPTED_FROZEN"
        assert freeze["manifest_sha256"] == EXPECTED_SHA256[MANIFEST]
        assert freeze["generation_audit_sha256"] == EXPECTED_SHA256[GRID_AUDIT]
        require_passed_gates(freeze["final_gates"])
        result["grid_freeze_audit"] = {"path": GRID_FREEZE, "sha256": sha256_file(freeze_path)}
    return result


def validate_grid(examples, split, report):
    """Complete parent, identity, coverage and eligibility checks; no regeneration."""
    import pandas as pd
    assert len(split) == 1898 and split.id.is_unique
    assert split[["id", "name", "species", "split", "length"]].notna().all().all()
    assert split.name.nunique() == 1189
    assert split.split.value_counts().to_dict() == {"train": 1335, "validation": 269, "test": 294}
    assert split.groupby("name").split.nunique().eq(1).all()
    assert split.groupby("name").species.nunique().eq(1).all()
    assert len(examples) == 32645 and examples.example_uid.is_unique
    assert examples.notna().all().all()
    assert not examples.duplicated(["id", "start_sample_16k"]).any()
    parent = examples[["id", "name", "species", "split"]].drop_duplicates().sort_values("id").reset_index(drop=True)
    pd.testing.assert_frame_equal(parent, split[parent.columns].sort_values("id").reset_index(drop=True), check_dtype=False)
    assert examples.example_stride_samples.eq(15360).all()
    assert examples.reference_context_samples.eq(15600).all()
    assert examples.target_sample_rate.eq(16000).all()
    lengths = {r["id"]: r["resampled_samples"] for r in report["clips"]}
    assert set(lengths) == set(split.id) == set(examples.id)
    for cid, group in examples.groupby("id", sort=False):
        group = group.sort_values("example_index")
        starts = list(range(0, max(0, lengths[cid] - 15600 + 1), 15360))
        assert group.start_sample_16k.tolist() == starts
        assert group.example_index.tolist() == list(range(len(starts)))
        assert group.example_uid.tolist() == [f"gridv2_{cid}_e{i:04d}" for i in range(len(starts))]
    assert (examples.reference_end_sample_16k == examples.start_sample_16k + 15600).all()
    assert (examples.start_second == examples.start_sample_16k / 16000).all()
    assert (examples.reference_end_second == examples.reference_end_sample_16k / 16000).all()
    assert examples.groupby("split").size().to_dict() == {"train":22613,"validation":4937,"test":5095}
    return {"exact_parent_metadata": True, "complete_grid_geometry": True,
            "accepted_membership": True, "zero_source_leakage": True}


def software_versions():
    packages = ["numpy", "pandas", "scipy", "matplotlib", "nbformat", "nbclient", "ipykernel"]
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed (optional execution tooling)"
    return versions


def feature_audio_contract():
    return {"native_rate_hz":44100, "channels":1, "pcm_bits":24,
            "decode":"scipy int32 left-justified", "scale_divisor":2**31,
            "conversion_dtype":"float64", "resampled_dtype":"float64",
            "resampler":"scipy.signal.resample_poly", "up":160, "down":441,
            "scope":"whole_clip", "window":["kaiser",5.0], "padtype":"constant", "cval":0,
            "output_length":"ceil(native_samples*160/441)", "waveform_completion_padding":False,
            "activity_filtering":False, "denoising":False, "amplitude_normalization":False}


def require_passed_gates(gates):
    assert gates and all(value is True for value in gates.values()), f"Final gate failed: {gates}"


def write_gated_audit(path, report):
    """Only write successful reports; atomic replacement avoids partial JSON."""
    require_passed_gates(report["final_gates"])
    assert report["authenticated_inputs"]["wav_files_authenticated"] == 1898
    report = dict(report, validated_at_utc=datetime.now(timezone.utc).isoformat())
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    assert json.loads(path.read_text(encoding="utf-8")) == report
    return report


def load_pinned_ei_wrapper(project, speechpy):
    """Execute the original generate_features body, without dsp.py import side effects.

    The authenticated wrapper uses the actual vendored SpeechPy module; only
    graph/module startup code is excluded. Always call with draw_graphs=False.
    This is Python numerical parity, not a C++ or streaming deployment test.
    """
    import numpy as np
    from common.errors import ConfigurationError
    relative = "third_party/edgeimpulse-processing-blocks/mfe/dsp.py"
    path = Path(project) / relative
    assert sha256_file(path) == EXPECTED_SHA256[relative]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate_features"]
    assert len(functions) == 1
    module = ast.Module(body=functions, type_ignores=[])
    scope = {"np":np, "math":math, "speechpy":speechpy, "ConfigurationError":ConfigurationError}
    exec(compile(module, str(path), "exec"), scope)
    return scope["generate_features"]
