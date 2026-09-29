# Stage-04 pre-training audit — 2026-09-29

## A. Status

**PASS for pre-training readiness.** Final run: 17/17 engineering tests passed; prepare_caches passed; 04A and 04B passed fresh-kernel execution. 04C returned the expected missing-results block and its synthetic completed-run path passed. No required code fixes remain before training.

Final validation results are recorded in `implementation_validation.json`. This report covers source inspection, full cache regeneration/validation, CPU synthetic checks and fresh-kernel notebook execution. No classifier training, real-data optimizer update, TEST model evaluation, or Stage-05 work was performed.

## B–D. Bugs, exact affected files, and scientific risk

| Finding before repair | Risk | Stage-04 repair / affected files |
|---|---|---|
| Published cache directories retained protected owner-only Windows ACLs from `TemporaryDirectory`; subsequent sandbox/kernel access failed at `03A/cache.json`. | Low scientific risk (fail closed); blocking operational issue. | `tools/stage04_data.py`: staging now uses ordinary mkdir with inherited parent ACLs; validates before publication; clearer permission/partial-cache errors. Old derived directories preserved, new caches rebuilt. |
| Scientific lock contents were read without pinning the lock itself. Named-function AST gates omitted executed constants, waveform loaders and the 03A context extractor. | High: modified extraction behavior could evade the named-function gate. | `tools/stage04_data.py` pins the unchanged scientific lock and new `reports/stage04/frontend_execution_lock.json`, covering canonical ASTs of every executed frontend cell. Existing scientific identities are unchanged. |
| Cache metadata omitted recorded sample/second coordinates and geometry; some declared cache contract fields were not checked. | Medium: weaker reproducibility and corruption detection. | `tools/stage04_data.py`: schema 2 stores all 13 manifest columns, checks exact row/coordinate identity, schema, row count, dtype, normalization and scientific lock. Full file hashes, finite-value and shape checks retained. |
| 04C trusted the stored protocol without checking it against current scientific identities, implementation or runtime. | High: a complete stale benchmark could be accepted. | `tools/training_common.py`: authenticate accepted inputs and compare the entire current protocol; explicit manifest/split/source pins and scientific/execution locks enter the protocol. |
| Saved metric summaries, parameter counts, checkpoint paths and minimum-loss selection were incompletely cross-checked against artifacts. | Medium: edited/inconsistent summaries could be reported. | `tools/training_common.py`: validate history, earliest minimum-loss epoch, selected loss, run specification, parameters/path, class order, exact TEST metadata, saved probabilities/predictions, example/source classification metrics and source predictions. No model inference occurs during comparison. |
| 04C silently printed a pending message; direct comparison with no protocol raised a generic file error. | Low: ambiguous readiness. | `notebooks/current/04C_compare_models.ipynb`, `tools/training_common.py`: explicit missing-results/protocol errors; no comparison is published. |
| Saved Stage-04 notebook outputs were stale; imports could retain old module implementations when rerunning setup in an existing kernel. | Medium operational/reproducibility risk. | All three `notebooks/current/04*.ipynb`: clear old outputs, reload Stage-04 modules in dependency order, then execute from fresh kernels. Training flags remain false. |
| Validator assumed 04C must finish normally before runs exist; default IPython profile attempted to write an inaccessible history database. | Low: misleading validation or environment errors. | `tools/validate_stage04.py`: accepts only the precise expected missing-results error and isolates IPython/Jupyter runtime paths. |

Regression coverage is in `tools/test_stage04.py`. `docs/STAGE04.md` documents the corrections; the historical `reports/stage04/IMPLEMENTATION_REPORT.md` is marked superseded. New audit evidence is confined to `reports/stage04/`. `tools/model_zoo.py`, requirements, notebooks 00–03B, `scientific_lock.json`, the frozen lock, manifests, splits, audits, WAVs and frontend sources were not modified.

## E. Data, caches and scientific identities

`prepare_caches()` is exercised by the validation runner and by each training notebook's fresh-kernel preflight. Both caches contain 32,645 float32 examples: TRAIN 22,613; validation 4,937; TEST 5,095. Shapes are 03A `[32645,96,64]` and 03B `[32645,96,40]`. Exact UID, parent ID, species, source name, split, sample and second coordinates, stride/context sizes and target rate match the accepted manifest and each other. Authentication verifies all 1,898 WAVs and all 1,189 source names; zero source-name leakage.

Unchanged accepted SHA256 identities:

- Manifest: `36881d79d33cd42521aa24004236022dc25b4927f65ae0246a76832835206234`
- Frozen split: `3e40928622dcea26e723540cfb493a35fd1d78393bf5531810583eae7966e0c4`
- `mel_features.py`: `68803c00743cb43139db12836b5e745a977cdb81443257fa716cd520d7e5e948`

The accepted audit hashes/final gates, Stage-02 geometry/function identities, upstream source pins and vendored Edge Impulse implementation remain hard gates. Whole-notebook byte hashes remain provenance only. Tests confirm output/metadata/comment serialization changes pass while scientific constants or loader changes fail.

03A uses the frozen float64 waveform loader/resampler and 15,600-sample context; 96×64 extraction completes before float32 caching. 03B uses its exact accepted wrapper and 15,600-sample context, 96×40 output, and −70 dB project floor. The −52 dB upstream default is not substituted. MFE's internal scaling/quantization is part of that frozen wrapper; Stage 04 adds no normalization, augmentation or preprocessing. Ten synthetic/boundary/fixed-TRAIN parity cases passed; 03A maximum absolute discrepancy was about 2.31e-14 and 03B was bitwise exact against the pinned wrapper.

Authentication precedes generation and repeats before publication. The fully written staged cache is validated and its memmap released before atomic directory rename. Partial `*-building-*` directories cannot be mistaken for final caches and are preserved for diagnosis. Existing invalid caches fail closed instead of silently rebuilding.

Both regenerated feature files exactly match the original files' hashes (see `cache_rebuild_byte_parity.json`):

- 03A: `577c730ae2bc2748c09aba0b89c20ace39f5e5e008e01667bc0f6ede546384e0`
- 03B: `24bac6eeff33531b40672c99016cc303e586a4b8a247416266e383a635b8d656`

## F–G. Notebook and training readiness

04A/04B cell-by-cell review: setup/imports (cell 1), configuration/architecture inspection (3), mandatory paired-cache preflight (5), guarded training (7); all intervening cells are explanatory Markdown. No missing imports, undefined variables, duplicate definitions, hidden-state dependency or ordering bug remains. Project discovery supports the root and `notebooks/current/` working directories. Setup reloads shared modules, and fresh-kernel execution verifies the real preflight. `RUN_TRAINING=False` never calls `run_branch` or a fitting routine.

The fixed protocol is PyTorch/CPU, seed 42, Adam at 0.001, batch 64, maximum 50 epochs, patience 8, earliest minimum weighted validation loss, no augmentation or additional input normalization. TRAIN counts `[10457,5128,1426,5602]` alone determine weights `22613/(4*n_class)`. Weighted CE divides the summed weighted NLL by summed sample weights; epoch loss aggregates numerator/denominator globally. No softmax precedes the loss.

The fit function receives only TRAIN and validation loaders. It fixes/hashes the checkpoint decision and reloads that checkpoint before constructing the TEST loader. Completed runs are reused; incomplete run directories block reruns, avoiding automatic repeated TEST evaluation. Cache preparation necessarily reads TEST inputs/metadata for model-independent feature generation and integrity checks; no TEST predictions, metrics or model selection occur during preflight. Model BatchNorm learns only during TRAIN. Class order is shared by both branches and all metrics: an arabiensis, an funestus ss, an squamosus, culex pipiens complex.

| Model | 03A parameters | 03B parameters | Shape review |
|---|---:|---:|---|
| Small 2D CNN | 23,668 | 23,668 | N×1×96×F; global time/frequency pooling |
| DS-CNN | 6,244 | 6,244 | 32 depthwise groups and 1×1 pointwise convolutions |
| TC-ResNet8 | 65,940 | 64,788 | F channels, 96 temporal steps; residual lengths 48/24/12 |
| CNN+LSTM | 13,428 | 13,428 | Frequency mean; N×24×32 temporal sequence |

All outputs are four logits. All expected counts match exactly; TC-ResNet's 1,152-parameter difference is `(64-40)*16*3` in the stem. CPU forward/backward and checkpoint roundtrip checks passed for all eight combinations, without optimizer updates.

## H. Comparison and reproducibility

04C's setup is cell 1; cell 2 displays all eight run statuses and requires complete authenticated results; cell 4 plots only after successful validation. It is ready for later comparison and deliberately blocked now because no real training results exist. A complete eight-run temporary synthetic fixture tests the successful comparison path; stale protocols, incomplete runs and tampered metrics are rejected.

Run records retain branch/model/seed, software/build, implementation and frontend identities, accepted input identities, class order/weights/counts, architecture/parameter count, checkpoint path/hash, epoch/selection details, history and predictions. TRAIN/validation metrics and confusion matrices are saved per epoch in `history.json`; selected validation and final example/source TEST metrics are in `result.json`. TRAIN history metrics are online with dropout, not a fresh evaluation of the selected checkpoint. Comparison performs identity checks and reads saved results; it neither regenerates caches nor trains/evaluates models.

## I. Permission diagnosis, recovery and remaining requirements

Observed cause: protected directory ACLs granting FullControl only to OWNER RIGHTS, SYSTEM and Administrators. Read/list access failed in the sandbox and succeeded in the owning-user context. `cache.json` was a regular Archive file, not a directory or read-only file. The original files were complete/readable in that context. The runtime's `tempfile.mkdtemp` calls mkdir with mode 0700; publishing that directory by rename preserved its restrictive ACL. This is consistent with the observed ACL and explains the reproduced access failure. Several Python 3.11 Jupyter kernels were running; their presence alone is not evidence of a file lock. No persistent file lock was observed, and the rename succeeded without terminating processes.

No ACLs were reset and no processes were killed. Both old branch directories were moved into `data/stage04_cache/preserved-before-audit-20260929/`; rebuilt directories and files inherit the cache parent's permissions and are readable in the sandbox. Deleting and rebuilding **only** `data/stage04_cache/` is scientifically safe because it contains derived Stage-04 features, provided active readers/writers are closed and the authenticated upstream inputs remain intact. This audit preserved the old data instead of deleting it.

Use `.venv-stage04/Scripts/python.exe` (CPython 3.12.14 with pinned dependencies) and a fresh kernel for future training. No scientific/code fixes remain if the final validation record is PASS. Training still requires the user's deliberate opt-in; this audit leaves `RUN_TRAINING=False`. Comparison must wait for all eight real completed runs. Full training behavior on real data is inspected and covered by engineering tests, not claimed empirically validated by this pre-training audit.

Evidence: `implementation_validation.json`, `engineering_tests.txt`, `audit_validation_console.txt`, `preflight.json`, `architecture_inventory.json`, `planned_protocol.json`, `audit_protected_before.json`, `audit_protected_verification.json`, `cache_rebuild_byte_parity.json`, `audit_artifact_inventory.json`.
