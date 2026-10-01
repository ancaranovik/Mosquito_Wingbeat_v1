# Portable storage and execution

The scientific pipeline remains 00 → 01 → 02 → 03A/03B → 04A/04B → 04C.
The new 04D notebook provides Colab setup, cache-only smoke testing and explicitly
requested future experiments. No preprocessing or training is part of this refactor.

## Final repository layout

```text
Edge AI Project/
├── notebooks/current/          00–04C plus 04D_experiments.ipynb
├── tools/                      paths, cache consumer, experiment runner, checks
├── configs/{baseline,experiments}/
├── data/{manifests,audits,metadata}/  small frozen reproducibility material
├── reports/                    preserved scientific evidence
├── reference/, third_party/    authenticated upstream sources
├── archive/                    scientific history and pre-refactor originals
├── docs/                       methodology, workflow, migration, verification
├── requirements-{pipeline,stage04,colab}.txt
├── README.md, .gitignore, .gitattributes
├── data/{audio,stage04_cache}/  local originals, excluded from Git
├── colab_result/               original baseline, excluded from Git
├── _local_only/                retained builds/runtime checks, excluded from Git
└── _drive_upload/EdgeAI/        complete manual-upload staging tree, excluded from Git
```

GitHub stores code, notebooks, configs, small manifests/contracts, documentation
and dependencies. Drive stores audio, feature arrays, baseline evidence and new
experiment outputs. `_local_only` stores retained local artifacts, never active
scientific inputs. The untouched external backup was not accessed or used.

## Final Drive layout

```text
MyDrive/EdgeAI/
├── fixed/
│   ├── raw/HumBugDB/             all 2,694 local WAVs
│   ├── data/
│   │   ├── manifests/           Core metadata and accepted example grid
│   │   ├── splits/              frozen source split
│   │   ├── metadata/            original release metadata
│   │   ├── audits/              frozen scientific acceptance evidence
│   │   └── stage04_cache/{03A,03B}/
│   │       └── features.npy, examples.csv, cache.json
│   ├── baseline/stage04/        62 accepted Colab result artifacts
│   └── history/archive/baseline_readiness/
│       ├── raw/                 historical audio; never current inputs
│       └── sources/historical_archives/  four historical ZIPs
├── experiments/<experiment_id>/{config,checkpoints,results,logs}/
├── derived/
├── reports/
└── temp/
```

The main Drive structure was created in the previous session at
[EdgeAI](https://drive.google.com/drive/folders/10Sxiula2ZlFd_TYJtEDe4kBQUX-lVSo6).
No project data was uploaded. Automatic approval review rejected the upload because
it could not establish destination ownership/authorization for private project
data. This task uses the explicitly authorized manual staging fallback. Historical
`fixed/history` folders are included in staging and can be created during upload.

## Exact manual upload mappings

All local paths below are relative to **D:\VGU-27\Edge AI Project**. Copy contents
into the existing corresponding Drive folders; avoid same-named duplicate roots.

| Local source | Drive destination |
|---|---|
| `_drive_upload\EdgeAI\fixed\data\manifests\` | `MyDrive/EdgeAI/fixed/data/manifests/` |
| `_drive_upload\EdgeAI\fixed\data\splits\` | `MyDrive/EdgeAI/fixed/data/splits/` |
| `_drive_upload\EdgeAI\fixed\data\metadata\` | `MyDrive/EdgeAI/fixed/data/metadata/` |
| `_drive_upload\EdgeAI\fixed\data\audits\` | `MyDrive/EdgeAI/fixed/data/audits/` |
| `_drive_upload\EdgeAI\fixed\data\stage04_cache\03A\` | `MyDrive/EdgeAI/fixed/data/stage04_cache/03A/` |
| `_drive_upload\EdgeAI\fixed\data\stage04_cache\03B\` | `MyDrive/EdgeAI/fixed/data/stage04_cache/03B/` |
| `_drive_upload\EdgeAI\fixed\baseline\stage04\` | `MyDrive/EdgeAI/fixed/baseline/stage04/` |
| `data\audio\` | `MyDrive/EdgeAI/fixed/raw/HumBugDB/` |
| `data\stage04_cache\03A\features.npy` | `MyDrive/EdgeAI/fixed/data/stage04_cache/03A/features.npy` |
| `data\stage04_cache\03B\features.npy` | `MyDrive/EdgeAI/fixed/data/stage04_cache/03B/features.npy` |
| `archive\baseline_readiness\raw\` | `MyDrive/EdgeAI/fixed/history/archive/baseline_readiness/raw/` |
| `archive\baseline_readiness\sources\historical_archives\` | `MyDrive/EdgeAI/fixed/history/archive/baseline_readiness/sources/historical_archives/` |

The completed staging tree contains all 5,412 manifest files, including audio,
feature arrays and historical ZIPs (10,311,854,467 bytes). Upload its contents
without introducing an extra EdgeAI directory. The original manifest staging flags
predate the complete copy; source/destination, size and SHA256 remain authoritative in
[drive_migration.csv](portability/drive_migration.csv). Historical payloads are
provenance, not required for the first cache-only test. The two unreadable historical
cache directories under `data/stage04_cache/preserved-before-audit-20260929/` are
untouched locally and are not accepted inputs; their contents could not be inventoried.

## Paths and immutable boundaries

`tools/project_paths.py` owns RAW_ROOT, MANIFEST_ROOT, SPLIT_ROOT, CACHE_ROOT,
BASELINE_ROOT, EXPERIMENT_ROOT, DERIVED_ROOT and REPORT_ROOT. Set EDGEAI_DATA_ROOT
to the **EdgeAI root**, not its `fixed` child, before importing tools.

- Local, environment variable unset: read original local data and `colab_result`;
  new output goes under `_runtime/`.
- Colab: `/content/drive/MyDrive/EdgeAI`; code comes from the cloned repository.
- Explicit external root: never silently fall back to repository data.

The fixed boundary is enforced by application checks, not changed Drive ACLs.
Manual changes can still happen; hash validation rejects them. Both accepted local
data and external fixed data remain protected when an external root is configured.
The test Drive layout under `_local_only/portability_test_drive` uses hardlinks for
immutable audio/arrays to avoid duplication. Do not edit those test payloads.

## Environment and first Colab GPU smoke test

1. Commit and push the final local portability fixes to the configured repository,
   `https://github.com/ancaranovik/Mosquito_Wingbeat_v1.git`. The audit does not
   commit or push. Configure private-repository authentication interactively;
   never embed tokens in URLs, source or notebooks.
2. Open `notebooks/current/04D_experiments.ipynb` in VS Code and connect to a Colab
   GPU kernel. The local notebook file and remote execution filesystem differ.
3. Set REPO_URL to your GitHub HTTPS clone URL. Run the first cell to mount Drive,
   clone or fast-forward pull, check origin and require a clean runtime checkout.
4. Run the configuration cell. In Colab, `configure(install=True)` sets
   EDGEAI_REPO_ROOT, requires `/content/drive/MyDrive/EdgeAI`, installs
   `requirements-colab.txt`, and prints
   code/data/baseline/future-output paths. No local Windows path is used remotely.
5. Run `python -B tools/verify_portability.py` through the supplied notebook cell.
   It verifies scientific source/locks, all 1,898 Core WAV hashes, both cache
   hashes and metadata, source
   separation/class order, and all eight baseline records without training.
6. Run the GPU cell. It prints `torch.cuda.is_available()` and
   `torch.cuda.get_device_name(0)`, then invokes
   `python -B tools/experiment_runner.py smoke --device cuda` in a fresh process.
   Require PASS, `training_executed: false`, `feature_generation_executed: false`.
7. Leave RUN_TRAINING=False. Future results will be written only under
   `/content/drive/MyDrive/EdgeAI/experiments/<experiment_id>/`.

Smoke performs synthetic forward passes only: no optimizer, feature extraction,
training or repeated trained-model TEST evaluation. Missing caches fail; do not
regenerate 00–03B to bypass a failed check. The local test uses `--device cpu`;
CUDA remains to be verified on the actual Colab GPU.

The accepted CUDA environment was Python 3.13.15, torch 2.8.0+cu126, numpy 2.4.6,
pandas 3.0.6 and Tesla T4. Full original runtime/protocol is retained in
`configs/baseline/colab_stage04.json`. Future cache consumers require Python 3.11+
and the pinned package versions, recording actual runtime/hardware. Python-version
AST schemas are handled by comparing current and authenticated original source
under the same interpreter; no scientific pin is rewritten.

Historical DSP/CPU execution used Python 3.12.14 and requirements-pipeline.txt /
requirements-stage04.txt. Any suitable environment directory works; `.venv-stage04`
is not required. 04D also runs locally: it uses the current checkout, keeps
EDGEAI_DATA_ROOT configurable, skips package installation and selects CPU smoke. 04A/04B inspect that CPU protocol with training disabled. 04C reads
the accepted CUDA results. 04D uses the cache-consumer environment. Full 00–03B
notebook reruns are unnecessary for normal experiments.

## Future experiments

The reviewed v2 workflow is documented in [EXPERIMENT_V2_GUIDE.md](EXPERIMENT_V2_GUIDE.md).
04D now defaults to `configs/experiments/train_norm_v2.json`, `exp_004_train_norm`
and `RUN_TRAINING=False`. This separately versioned engine selects by validation
Macro-F1, retains loss-based early stopping, saves both checkpoints and TRAIN-eval /
validation diagnostics, and defaults to validation-only tuning with TEST disabled.
This trial changes only input normalization to TRAIN-only global z-score versus
exp_002_control_v2, retaining Adam learning rate 0.001 and immutable cache bytes.
See [EXP_004_TRAIN_NORM.md](EXP_004_TRAIN_NORM.md) for execution and review steps.
The separate LR trial is documented in [EXP_003_LR3E4.md](EXP_003_LR3E4.md).
The original CUDA engine and baseline suite below remain available unchanged.

Edit locally → commit → push → Colab pull → mount Drive → verify cache/data →
explicitly train → results saved to Drive. In 04D, set `EXPERIMENT_NAME` once and
turn on `RUN_TRAINING` only for the execution you intend. Keep False in the committed
notebook. The named suite trains all four frozen models with 03A, then with 03B:

```sh
python -u -B tools/experiment_runner.py train --config configs/experiments/baseline_suite.json --experiment-id exp_001_full_suite
```

`--experiment-id` overrides the template ID without changing a tracked config.
`--description` optionally records purpose/change notes. The runtime checkout must
be clean and committed. A VS Code local notebook connected to Colab can change these
controls without modifying the remote checkout; a notebook hosted inside that
checkout should be copied outside it for interactive control.

The accepted CUDA protocol and seed 42 remain fixed in the original engine.
Scientific overrides fail explicitly there. The legacy `baseline_reuse.json`
single-pair CLI still works. The v2 engine accepts only the reviewed learning-rate
and TRAIN-global normalization overrides; loss/sampler/other transforms need separately reviewed implementations;
waveform, segmentation or feature changes need versioned `derived/` identities.

The runner reserves a new output directory and refuses reused names. It persists
per-run progress, COMPLETE/FAILED/INTERRUPTED group status, ID, parent baseline,
commit, effective config, seed, runtime/GPU/CUDA and cache/split identities. All eight
runs get their own history, validation selection, checkpoint, predictions, metrics
and confusion matrix. A group comparison/validation-only shortlist is written only
after every planned run is complete and authenticated. Logs are shown live and saved
to Drive. Completed/partial earlier results remain in their named directories;
there is no implicit resume/restart or overwrite. TEST remains isolated from fitting
and checkpoint selection.

04C is read-only: leave `EXPERIMENT_NAME` empty to inspect the accepted baseline,
or enter a completed suite name to inspect its eight results and provenance.
CLI review: `python -B tools/experiment_review.py exp_001_full_suite`.

Current validators are read-only. Original validators, CPU modules/notebooks and
CUDA export remain in `archive/portability_originals` and `archive/colab_baseline`.
Do not run archived builders as the normal workflow. Original methodology documents
and contracts are unchanged; their older operational instructions are superseded
by this document. Git line-ending conversion is disabled to preserve exact bytes.

## Windows Git ownership note

Git was initialized by the sandbox account. Windows denied changing the new
`.git` owner, so this task used a command-scoped `safe.directory` for this exact
project. No global settings were changed. If your normal terminal reports
“dubious ownership”, you can authorize this known local repository once:

```powershell
git config --global --add safe.directory "D:/VGU-27/Edge AI Project"
```

Or use `git -c safe.directory="D:/VGU-27/Edge AI Project" status` without changing
global configuration. This concerns Git metadata ownership, not scientific files.
