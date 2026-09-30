# Edge AI Mosquito Acoustic Research — portable workflow

Current workflow: **local PC / VS Code / Codex → GitHub → Colab GPU → Google Drive**.
Eight accepted Stage-04 Colab baseline runs are complete and immutable. No experiment
was trained and no 00–03B output was regenerated during this portability refactor.

GitHub holds code, notebooks, configs, small scientific manifests, contracts,
environment definitions and documentation. Drive holds raw data, validated feature
caches, baseline evidence and future results. `_local_only/` holds retained local
build/runtime material excluded from Git. Original local data and `colab_result/`
remain available. Nothing was deleted.

Hướng dẫn thao tác từng lần chạy bằng tiếng Việt: [Hướng dẫn chạy thử nghiệm](docs/HUONG_DAN_CHAY_THU_NGHIEM.md).
Lần tiếp theo **exp_002_control_v2**: [Hướng dẫn experiment v2](docs/EXPERIMENT_V2_GUIDE.md).
04D đã đặt sẵn control v2 và training tắt; 04C đọc validation diagnostics theo tên.

## Start here

- [04D_experiments.ipynb](notebooks/current/04D_experiments.ipynb): mount Drive,
  clone/pull, install the pinned cache-consumer environment, verify hashes, check
  CUDA/GPU and run a **non-training** smoke test. Training defaults to disabled.
- [Storage and workflow](docs/PORTABLE_WORKFLOW.md): exact upload mappings,
  environment requirements, immutable boundaries and first-run instructions.
- [Refactor report](docs/portability/REPORT.md): changes, verification, local moves
  and limitations. [Migration manifest](docs/portability/drive_migration.csv) lists
  every assigned file with destination, size, SHA256 and staging status.

Set `EDGEAI_DATA_ROOT` to the **EdgeAI root**, e.g.
`/content/drive/MyDrive/EdgeAI`, before importing project tools. When unset on the
local PC, accepted inputs use the original repository data layout and new outputs
use `_runtime/`. `tools/project_paths.py` is the single storage mapping layer.
An explicitly configured external root never silently falls back to local caches.

Use any appropriate Python environment; the folder name `.venv-stage04` is not
required. Install `requirements-colab.txt` for cache-only CUDA consumption.
`requirements-pipeline.txt` and `requirements-stage04.txt` document the historical
CPython 3.12 numerical/CPU environment. The accepted Colab baseline used Python
3.13.15, torch 2.8.0+cu126, numpy 2.4.6, pandas 3.0.6 and a Tesla T4; its full
environment and protocol are preserved in `configs/baseline/colab_stage04.json`.

```sh
python -B tools/verify_portability.py
python -B tools/experiment_runner.py smoke --device cpu
python -B tools/experiment_runner.py baseline
python -B -m unittest discover -s tools -p test_portability.py -v
```

The CPU smoke check is suitable for local validation. Use `--device cuda` in Colab;
missing CUDA fails explicitly. Neither smoke command constructs an optimizer.

The scientific pipeline remains **00 → 01 → 02 → 03A / 03B → 04A / 04B → 04C**.
00–03B preserve the accepted scientific function and executed frontend-cell ASTs.
04A/04B retain the historical CPU protocol for inspection, read existing caches,
and cannot launch baseline retraining. 04C verifies and displays either the saved accepted
CUDA baseline or a named completed eight-run experiment without rewriting it. 04D supplies the new portable execution path.

For a full future run, open 04D, set `EXPERIMENT_NAME` once and explicitly enable
`RUN_TRAINING` for that execution. The suite trains all four existing models with
03A, then all four with 03B, sequentially in one named experiment. The equivalent CLI is:

```sh
python -u -B tools/experiment_runner.py train --config configs/experiments/control_v2.json --experiment-id exp_002_control_v2
```

Commit/push the implementation, then pull it into a clean Colab checkout before
training. Names and optional descriptions are runtime overrides; no new tracked
config is needed just to name a run. Keep the committed notebook default at False.
The existing `baseline_reuse.json` single-pair CLI remains supported.

The original `baseline_suite.json` engine preserves the accepted CUDA protocol.
The reviewed v2 engine keeps seed 42, architectures and validated caches, selects
by validation Macro-F1 while retaining loss-based early stopping, and saves both
checkpoint candidates plus TRAIN-eval / validation diagnostics. TEST defaults to
disabled during tuning. Only learning-rate and TRAIN-global normalization overrides
are supported in v2; other scientific options require separate implementation.
Each experiment saves
effective config, commit, runtime, cache/split identities and eight separate histories,
checkpoints, predictions, metrics and confusion matrices under
`EdgeAI/experiments/<EXPERIMENT_NAME>/`. A completed group includes `comparison.csv`
and `comparison.json`; 04C selects the group by the same name. Existing names fail
rather than overwrite. Partial/failed groups retain their artifacts and status.
Waveform, segmentation or feature changes require a new versioned `derived/` path.

Standard cycle: **edit locally → commit → push → Colab pull → mount Drive →
verify data/cache → explicitly train → results saved to Drive**. A notebook opened
in local VS Code can execute on a remote kernel. Its code dependencies must come
from the cloned repository in that runtime, never a local Windows drive.

The original README follows as historical pre-Stage-04 documentation. Its statements
that no models were trained are superseded by the completed baseline above.

---

# Historical pre-Stage-04 overview

The current pipeline is **00 → 01 → 02 → 03A / 03B**. No classifier has been trained and notebook 04 has not been created.

| Notebook | Role and successful-gate status |
|---|---|
| [00_prepare_dataset](notebooks/current/00_prepare_dataset.ipynb) | Reproduce and verify the accepted Core and source split; canonical promotion disabled |
| [01_dataset_audit](notebooks/current/01_dataset_audit.ipynb) | Independent metadata, exact parent, source-split and WAV audit |
| [02_segmentation](notebooks/current/02_segmentation.ipynb) | Example Grid v2 — **ACCEPTED / FROZEN** after identity, physical-context and exact reconstruction gates |
| [03A_logmel_reference](notebooks/current/03A_logmel_reference.ipynb) | Log-Mel frontend — **VERIFIED / FROZEN** after upstream parity and final gates |
| [03B_edge_mfe](notebooks/current/03B_edge_mfe.ipynb) | Edge MFE frontend — **VERIFIED / FROZEN** after direct pinned-wrapper regression and full-TRAIN confirmation |

These statuses are established by the current successful audits, not by this table alone. See `data/audits/current/02_example_grid_v2_freeze_audit.json`, the two frontend audits, and `reports/validation/pretraining_review.json` for execution evidence. A missing or failing gate blocks downstream use.

Frozen Core: **1,898 clips / 1,189 source recordings**; train/validation/test **1,335 / 269 / 294 clips**; zero source-name leakage. The accepted grid contains **32,645 examples** (22,613 / 4,937 / 5,095), with **15,360-sample starts** and **15,600-sample contexts** at 16 kHz. Its existing `_candidates.csv` filename is retained as an artifact identifier; the separate acceptance record freezes its exact hash.

**03B upstream UI default: −52 dB. Project-frozen baseline: −70 dB.** The override was selected using TRAIN-only anti-collapse diagnostics; it is not a claim of global optimality.

Run notebooks from the repository root or their own directory. Install [requirements-pipeline.txt](requirements-pipeline.txt) in an isolated Python environment; run `python -B tools/validate_pipeline.py` for fresh-kernel validation. The validated Python/package versions are recorded in the audits. Optional PDF exports are disabled by default and are not scientific gates.

| Folder | Purpose |
|---|---|
| `notebooks/current/` | Current pipeline |
| `data/audio/`, `data/metadata/` | Preserved accepted WAVs and release metadata |
| `data/manifests/current/` | Frozen Core/split and accepted Example Grid v2 |
| `data/audits/current/` | Original generation evidence plus gated freeze/frontend audits |
| `reports/` | Validation evidence, unchanged reproduction copies and historical PDF exports |
| `reference/`, `third_party/` | Authenticated upstream source bytes |
| `tools/` | Identity/gate helpers, validation runner and historical review utilities |
| `docs/` | Current methodology and provenance; historical reorganization records |
| `archive/` | Historical artifacts; never promoted into current inputs |

See [pipeline and comparison limits](docs/PIPELINE.md), [data provenance](docs/DATA_PROVENANCE.md), and [reference identities](reference/README.md). The [reorganization report](docs/REORGANIZATION_REPORT.md) describes an earlier state and is preserved unchanged; its missing-reference statements are historical, not current status.
