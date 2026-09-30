# Final portability verification — 2026-09-30

**Local source: PASS. Deployment prerequisite: commit/push these fixes and complete the manual Drive upload before the first Colab GPU smoke test.** No actual Colab session or Drive contents were inspected; CUDA availability and remote installation remain to be verified there.

| Notebook | Local layout | Repository-only + external Drive layout |
|---|---|---|
| 00 | PASS | PASS |
| 01 | PASS | PASS |
| 02 | PASS | PASS |
| 03A | PASS | PASS |
| 03B | PASS | PASS |
| 04A | PASS | PASS |
| 04B | PASS | PASS |
| 04C | PASS | PASS |
| 04D | PASS | PASS |

## Scope and evidence

- Executed only setup/input-reading cells of 00–03B; did not execute segmentation, feature extraction, full-TRAIN diagnostics or data regeneration. PASS here means portability and integrity, not a fresh complete scientific notebook rerun.
- Executed all code cells of 04A–04C with training disabled. 04C authenticated and displayed existing baseline results.
- Executed 04D local setup and its disabled-training cell; ran its integrity and CPU smoke commands separately. Colab mount, clone/pull, CUDA routing and disabled training were tested with mocked external operations.
- 14 focused storage/runtime tests passed, including canonical-write rejection, fixed-cache staging rejection, missing-CUDA failure, and existing-experiment preservation.
- Local and external-layout integrity checks passed: 1,898 Core WAV SHA256 identities, both complete cache hashes/metadata, 32,645 example identities and eight baseline runs.
- Eight CPU model/frontend forward checks passed in each layout; no optimizer or training was run.
- External-layout validation used copies of tracked repository files with the entire repo-local data tree omitted, no colab_result directory, and EDGEAI_DATA_ROOT pointing at the completed staging tree. This verifies no fallback to original local data is needed.
- Scientific function AST locks, executed frontend cell locks, training function AST parity and model source byte parity all passed.
- Existing pinned local environment was reused solely as the test interpreter. Active notebook source and kernelspecs do not require the directory name .venv-stage04. The generic system interpreter has an incompatible torch version and is not the validated runtime.

## Targeted fixes

- Routed 03B raw/manifest/split/audit paths through project_paths; removed redundant 03A/03B project discovery.
- Kept the 03B fixed audit read-only; guarded 00 canonical promotion even if its flag is manually enabled.
- Routed optional PDF outputs through REPORT_ROOT; stopped creating an input-evidence directory in 00.
- Made 04D select local checkout/CPU or Colab mount + clone/pull/CUDA. Colab storage is enforced as /content/drive/MyDrive/EdgeAI; local EDGEAI_DATA_ROOT stays configurable.
- Added SciPy/matplotlib to the Colab requirements for notebook imports; no dependencies were installed during this audit.
- Added raw WAV authentication to verify_portability and fixed-write guards to legacy cache staging and experiment output entry points.
- Removed the stale environment-specific 04C kernel display label and corrected obsolete workflow instructions.

## Storage contract

Code, notebooks, configurations, tools and small tracked provenance records come from the repository. In Colab all active raw data, manifests, caches and baseline results resolve beneath /content/drive/MyDrive/EdgeAI/fixed; future experiment results resolve beneath /content/drive/MyDrive/EdgeAI/experiments/<experiment_id>.

Fixed storage is protected by application guards and hash checks, not operating-system or Drive ACLs. Existing experiment IDs are refused. RUN_TRAINING remains False in 04A, 04B and 04D, and the legacy baseline training entry point refuses retraining.

No executable Windows data-path literals or local-only environment dependencies remain in current notebook cells. Historical archives, saved outputs, staging inventories and maintenance scripts can retain historical Windows paths; current notebooks do not import code from those paths.

## Readiness

The corrected local project is ready for the first Colab GPU smoke test **after these uncommitted fixes are published and the complete staging tree is manually uploaded**. The GitHub HEAD inspected during the audit matched the pre-fix local HEAD; these working-tree fixes are not on GitHub yet. Nothing was committed, pushed, uploaded, trained or regenerated in this audit.
