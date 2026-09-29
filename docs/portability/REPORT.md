# Portability refactor report

Completed across the interrupted September 29 session and its September 30 continuation.
No training, 00–03B regeneration, accepted-data replacement or baseline overwrite.
Original local data, all eight Colab results, and meaningful historical material remain.

## Changes

- All eight 00–04C notebook setup/path sections use the central storage layer.
  04A no longer requires a particular environment directory; all current training
  guards default to False. Locked scientific functions/frontend definition cells
  remain identical. Accepted audit persistence is disabled in those notebooks.
- `project_paths.py` maps local/Drive inputs, separates fixed and writable storage,
  rejects traversal and protects both the local data tree and external `fixed/`.
- `cache_consumer.py` authenticates and memory-maps both existing caches read-only.
  Missing/corrupt inputs fail; it contains no feature regeneration path.
- `experiment_runner.py`, `experiment_training.py` and experiment configs reserve
  unique IDs, require committed code, record provenance, preserve the accepted CUDA
  protocol, and write only new experiment directories. Scientific overrides are
  intentionally unsupported. Direct training without a reserved ID is rejected.
- `baseline_review.py` validates saved checkpoint/history/prediction hashes and
  metric/selection consistency, without requiring the historical GPU environment
  to display saved results and without rewriting baseline evidence.
- `04D_experiments.ipynb` and `colab_bootstrap.py` provide mount/clone/pull/setup,
  identity checks and GPU smoke steps. Actual Colab GPU execution is still manual.
- Validators now run read-only checks. Original code/notebooks/CUDA export and
  build provenance are retained under `archive/`.
- `.gitignore` excludes raw audio, arrays, checkpoints, ZIPs, runtime environments,
  local staging/builds, bytecode, logs and credentials. Small manifests, source,
  notebooks, contracts, configs and docs remain repository material.
- `.gitattributes` disables checkout line-ending conversion to retain byte hashes.

Exact modified/new primary files: [changed_files.json](changed_files.json).
Full tracked-file inventory: [git_files.txt](git_files.txt).
Trees and exact upload/first-GPU instructions: [PORTABLE_WORKFLOW.md](../PORTABLE_WORKFLOW.md).

## Preserved moves

| Original | Retained destination | Reason |
|---|---|---|
| `.stage04_colab_build/` | `_local_only/.stage04_colab_build/` | Obsolete self-contained export build; source/provenance separately archived |
| `stage04_colab_bundle.zip` | `_local_only/stage04_colab_bundle.zip` | Duplicate old export package |
| `tools/__pycache__/` | `_local_only/tools/__pycache__/` | Runtime bytecode |
| `third_party/edgeimpulse-processing-blocks/.git/` | `_local_only/vendor_git/edgeimpulse-processing-blocks.git/` | Preserve vendor history while tracking source normally, avoiding a broken gitlink |

The vendor checkout was clean at pinned commit
`9b0ceb2b5d22658d3193a5b77dc773cfc8d4cab0`. Its source and Git metadata bytes are
preserved. The moved build's code, original contracts and notebooks have matching
archived copies. No accepted input depends on `_local_only`. That directory also
contains a newly created external-layout test fixture and plotting runtime files.
Test audio/array hardlinks share original bytes and must not be edited.
Exact file moves: [moves.json](moves.json); vendor evidence:
[vendor_git_preservation.json](vendor_git_preservation.json).

## Drive classification and completeness

The main EdgeAI folder structure was created through Drive before the interruption.
The project-data upload was rejected by automatic approval review, and no project
files were uploaded. Manual staging is ready under `_drive_upload/EdgeAI/`.

The complete mapping contains **5,412 files / 10,311,854,467 bytes**, including
active audio, manifests/split, metadata/audits, both caches, 62 baseline artifacts,
historical audio and four historical ZIPs. **75 small files** are copied and
hash-verified in staging. Large payloads remain at their exact original sources,
listed with hashes in [drive_migration.csv](drive_migration.csv). Uploading only
the staging tree will not upload those large files.

All `colab_result/stage04_colab_results` files are retained as baseline evidence
or reproducibility metadata. No file there was moved or treated as disposable.
Historical scientific material remains in `archive/`; bundled local runtimes are
ignored in place where moving them could break historical scripts.

Two obsolete pre-audit cache directories denied enumeration. They remain untouched
at `data/stage04_cache/preserved-before-audit-20260929/{03A,03B}` and are excluded
from active inputs and migration. Their contents cannot be claimed verified.

## Verification

- Manifest: `36881d79d33cd42521aa24004236022dc25b4927f65ae0246a76832835206234`.
- Split: `3e40928622dcea26e723540cfb493a35fd1d78393bf5531810583eae7966e0c4`.
- Upstream mel: `68803c00743cb43139db12836b5e745a977cdb81443257fa716cd520d7e5e948`.
- Core 1,898 clips / 1,189 sources, zero source leakage, 32,645 examples.
  TRAIN/VAL/TEST remain 22,613 / 4,937 / 5,095, with the same four-class order.
- 03A features: `577c730ae2bc2748c09aba0b89c20ace39f5e5e008e01667bc0f6ede546384e0`.
- 03B features: `24bac6eeff33531b40672c99016cc303e586a4b8a247416266e383a635b8d656`.
- Both complete cache metadata/order/labels/shape/dtype/finiteness checks pass.
- All eight saved baseline runs pass byte, prediction, metric and selection checks.
- Locked frontend/function ASTs, model source bytes and accepted CUDA training,
  loss, selection and metric function ASTs remain identical.
- 2,953 protected original files were hash-checked; their bytes remain unchanged
  at original or explicitly recorded retained destinations.
- 20 selected non-training engineering/contract tests pass. The full historical
  suite was not run; selected checks avoid synthetic training/checkpoint writes.
- Local CPU smoke passes all eight synthetic forwards, with no optimizer or
  feature generation. Explicit external Drive-layout verification also passes.
- 00–03B setup cells and all 04A–04C code cells pass against the external layout.
  00–03B preprocessing cells were not run; notebook files were not re-executed/saved.

Evidence: [verification.json](verification.json),
[external_layout_verification.json](external_layout_verification.json),
[nontraining_tests.txt](nontraining_tests.txt).

## Git and remaining actions

Git initialization/staging is performed only after the preceding checks pass.
The initial local commit contains repository material only. No remote is configured
or pushed. Inspect `git log -1 --oneline`, `git status --short` and
[git_files.txt](git_files.txt) for the delivered commit/file inventory.

Manual actions: upload the mapped Drive contents, later create/push your GitHub
repository, set REPO_URL in 04D, connect the remote Colab GPU, and run its default
non-training smoke cells. Local torch is CPU-only; CUDA testing cannot be claimed.

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
