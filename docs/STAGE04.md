# Stage 04: controlled local architecture benchmark

This stage implements a **controlled comparison of two frozen acoustic frontend configurations**. It does not isolate only Log-Mel versus MFE, reproduce a published classifier experiment, or establish an Edge winner. All existing 00–03B files are preserved. Stage 05 is outside this implementation.

## Run the notebooks

Use a dedicated CPython 3.12 environment with `requirements-stage04.txt`. It retains the accepted numerical versions and adds PyTorch 2.8.0. Select that environment as the notebook kernel. Do not upgrade numerical dependencies to resolve an installation conflict.

```powershell
python -m venv .venv-stage04
.\.venv-stage04\Scripts\python.exe -m pip install -r requirements-stage04.txt
.\.venv-stage04\Scripts\python.exe -m ipykernel install --user --name edge-ai-stage04
```

A persistent `.venv-stage04` environment is now configured in this workspace. In the notebook kernel picker, select **Python (Edge AI Stage 04)** (or the interpreter `.venv-stage04/Scripts/python.exe`), then restart the kernel and rerun from the first cell. An already-open notebook can retain its previous kernel despite updated metadata or workspace defaults.

Run from the repository root or `notebooks/current/`:

1. `04A_train_logmel.ipynb`: authenticate both caches, then train/evaluate four 03A models.
2. `04B_train_edge_mfe.ipynb`: same shared implementation for four 03B models.
3. `04C_compare_models.ipynb`: saved Stage-04 records only; no frontend execution or training.

Training cells default to `RUN_TRAINING = False`. Set it to `True` to launch the full CPU benchmark. This guard permits inspection and preflight without starting long jobs. Both complete caches are required before either branch trains. There is no partial-population benchmark option.

CLI equivalents (training commands have no notebook guard):

```powershell
python -B tools/training_common.py prepare
python -B tools/training_common.py 03A
python -B tools/training_common.py 03B
python -B tools/training_common.py compare
python -B tools/validate_stage04.py
```

Run branches sequentially in the same environment. The first training invocation locks the protocol, implementation hashes and software/build identity. Completed runs are authenticated and reused without retraining or repeated TEST evaluation. Incomplete runs fail: preserve their directory and investigate before explicitly restarting. Optimizer resume is not implemented. TEST results must never justify a restart, tuning or a changed benchmark protocol.

## Frozen inputs and frontend reuse

Exactly 32,645 examples: TRAIN 22,613, validation 4,937, TEST 5,095; 1,898 clips, 1,189 source names, four Core species. There is no activity/RMS gate, silence removal, weak-example removal, augmentation, denoising, peak normalization, resplitting or new frontend processing.

`stage04_data.py` authenticates the source/data pins in `pipeline_contract.py`, all 1,898 WAV hashes, the 02 freeze audit, both frontend audits and their final gates, recorded scientific function AST fingerprints, and canonical ASTs of every executed frontend definition cell (including constants, imports and waveform/context loaders). Whole-notebook byte hashes are provenance only. It reuses `validate_grid` to verify exact parent metadata, full geometry, UIDs, split membership and source-name isolation. It checks labels and TRAIN class counts explicitly. Python `-O` is rejected because frozen functions use assertions.

`reports/stage04/frozen_input_lock.json` records the current accepted notebook/audit bytes without modifying upstream artifacts; the adapter also pins this lock's hash. A provenance discrepancy remains visible: historical execution hashes in `pretraining_review.json` for 00, 01 and 02 differ from their current files. This does not establish why they differ. The user accepted the current foundation; Stage 04 records those bytes as provenance, never executes 00–02, independently authenticates their output artifacts and verifies Stage-02's recorded scientific function ASTs. The unchanged scientific lock is itself hash-pinned. The supplementary `frontend_execution_lock.json` gates all executed frontend cells without depending on notebook outputs, metadata or serialization. Stage 04 preserves the historical report and does not claim a fresh execution of 00–02.

Only the following zero-based definition/import cells execute, directly from the hash-authenticated notebook JSON:

| Branch | Cells | Functions called |
|---|---|---|
| 03A | 8, 10, 12 | `load_core_wav_16k`, `extract_reference_context`, `logmel_reference` |
| 03B | 8, 10, 12, 13, 15 | `load_core_wav_16k`, `extract_edge_context`, `edge_mfe_v4` |

Scientific code stays in the frozen notebooks. No audit-writing or noise-floor-selection cells execute. The existing pinned local SpeechPy implementation is reused by 03B; there are no Edge Impulse service, Studio, SDK, deployment or profiling calls.

Both branches retain float64 PCM24 conversion, whole-clip resampling, 15,600-sample contexts and the frozen starts. 03A completes its float64 96×64 computation before Stage 04 casts to float32. 03B retains its frozen float32 96×40 output and −70 dB floor. **No model-input normalization** is used: this avoids introducing an unnecessary fitted transform in the initial controlled benchmark. Model BatchNorm statistics learn on TRAIN only and are fixed in evaluation mode for validation/TEST.

Synthetic/boundary and fixed TRAIN cases are checked against the authenticated upstream 03A function and pinned 03B wrapper before caching. The two waveform loaders are checked for bitwise equality on those TRAIN clips. All full-cache features receive shape/finiteness checks without value-based filtering. Upstream artifacts are reauthenticated after generation.

Caches are separate under `data/stage04_cache/03A/` and `03B/`. Each contains `features.npy`, `examples.csv`, `cache.json`. Every metadata row retains UID, parent `id`, source `name`, species, split, all manifest coordinates/geometry, label, feature shape, frontend and frontend identity. Loading verifies SHA256 of both data files, exact metadata order/content, numerical runtime, full tensor shape/dtype/finiteness and identical UID/label/split ordering across branches. Memory-mapped float32 payloads total 1,303,710,720 bytes, plus NPY headers/metadata. Generation validates the staged cache and reauthenticates upstream inputs before publishing by directory rename. Staging uses ordinary mkdir to inherit the parent Windows ACL; Python temporary directories can retain owner-only ACLs after rename. Failed staging directories remain visibly named `*-building-*`; incomplete or corrupt final caches fail rather than silently regenerate. Old caches from the September 29 audit are preserved under `preserved-before-audit-20260929/`. Only derived Stage-04 caches may be rebuilt; no upstream artifact is modified.

## Exact architectures

Input layout is `[batch,1,96,F]`, F=64 or 40. Convolutions have no bias. BatchNorm uses epsilon 1e-5, momentum 0.1 and affine parameters. ReLU follows convolution/BN except the second residual convolution, which is activated after the sum. Every head uses dropout 0.2 and a biased linear layer producing four logits. Softmax is used only for predictions.

| Family | Architecture |
|---|---|
| Small CNN | 3×3 conv 1→16, BN/ReLU, max-pool 2×2; 3×3 conv 16→32, BN/ReLU, max-pool 2×2; 3×3 conv 32→64, BN/ReLU; global time/frequency mean; head |
| DS-CNN | 3×3 conv 1→32, stride 2×2, BN/ReLU; four blocks of 3×3 depthwise conv (32 groups), BN/ReLU, 1×1 pointwise conv 32→32, BN/ReLU; global time/frequency mean; head |
| TC-ResNet8 adaptation | reshape to `[batch,F,96]`; temporal conv F→16, kernel 3, stride 1, BN/ReLU; three residual blocks 16→24→32→48, each kernel-9 conv stride 2, BN/ReLU, kernel-9 conv stride 1, BN, plus kernel-1 stride-2 projection/BN; ReLU after sum; global time mean; head |
| CNN+LSTM | 3×3 conv 1→16, BN/ReLU, max-pool 2×2; 3×3 conv 16→32, BN/ReLU, max-pool 2×2; frequency mean produces 24 time steps ×32 channels; one unidirectional 32-hidden-unit LSTM with biases and no recurrent dropout; last hidden state; head |

All other 2D strides are one. Convolution padding is symmetric kernel//2 (zero for 1×1); pooling uses kernel-size stride and floor sizing. Temporal block outputs have 48, 24, 12 steps. TC depth counts seven main-path convolutions plus classifier, excluding shortcut projections.

TC-ResNet8 preserves frequency-as-channels temporal convolution and compact residual blocks from the [authors' implementation](https://github.com/hyperconnect/TC-ResNet). Adaptations are 96 frozen frames, four labels, PyTorch Conv1d, symmetric padding, PyTorch BatchNorm/default initialization and the common training protocol. This is not an exact reproduction of the original TensorFlow pipeline/results. DS-CNN is a compact project instance of the family, not a reproduction claim. CNN+LSTM pools frequency after learned convolutions to preserve temporal modeling with a fixed LSTM input width.

Each family's code/design is identical across branches. Frequency dimensions change activation sizes and compute. Only TC-ResNet8 changes parameter count: F→16 in the stem adds 1,152 weights for 03A. Other models retain equal parameter counts through global/frequency pooling; equal parameter counts do not imply equal compute/hardware cost.

## Training, imbalance and selection

PyTorch 2.8.0, CPU, float32, four threads, deterministic algorithms, MKLDNN disabled, no mixed precision, from-scratch default initialization. CPU execution may be slow; this first protocol does not silently switch backends.

Adam: learning rate 0.001 throughout, betas (0.9,0.999), epsilon 1e-8, zero weight decay, `amsgrad=False`, `foreach=False`. Batch size 64 with final partial batch retained. TRAIN shuffles; validation/TEST do not. Zero loader workers. No weighted sampling, oversampling, label smoothing, augmentation, input normalization or gradient clipping.

TRAIN-only weights use `w_c = 22613/(4*n_c)`. Each class contributes equal total TRAIN weight without duplicating correlated examples. Exact computed weights are saved in preflight/protocol/run records and cast to float32 for loss computation.

| Label | Species | TRAIN count | Weight |
|---|---|---:|---:|
| 0 | an arabiensis | 10,457 | 0.5406187242995123 |
| 1 | an funestus ss | 5,128 | 1.1024278471138846 |
| 2 | an squamosus | 1,426 | 3.9644109396914446 |
| 3 | culex pipiens complex | 5,602 | 1.0091485183862907 |

Loss is sum(weight × negative log probability)/sum(sample weights). Epoch loss accumulates numerators/denominators across the full partition rather than averaging batch means. The same TRAIN-derived weights measure validation/TEST loss; classification metrics themselves are unweighted.

Maximum 50 epochs. Select strictly lowest weighted validation loss; `min_delta=0`, exact ties retain earliest epoch. Stop after eight consecutive epochs without improvement. Selected validation macro-F1 and best macro-F1 observed across history are recorded separately; observed-best F1 does not choose the checkpoint.

The fitting function accepts only TRAIN/validation loaders. It saves `best_model.pt`, history and `selection.json` (including checkpoint SHA256 and timestamp) before constructing the TEST loader. TEST runs once on the reloaded fixed checkpoint and cannot affect training or selection. The provisional shortlist uses the top three selected-validation macro-F1 scores, then lower parameter count and lexical run ID; TEST never determines it.

Python, NumPy, PyTorch and independent loader generators use **42**, reset for every run. No seed averaging. The loader generator gives identical example order across families/branches. `PYTHONHASHSEED` is recorded; explicit lists/manifest ordering avoid hash-order dependence. Unsupported nondeterministic operations fail. [PyTorch notes reproducibility limitations](https://docs.pytorch.org/docs/2.14/notes/randomness.html): cross-version/platform/hardware bitwise identity is not promised. Records include software versions, processor, PyTorch build and implementation hashes.

## Results and scientific limits

Each `reports/stage04/runs/{frontend}_{family}_seed42/` saves:

- `run_spec.json`: exact architecture/module representation, protocol, seeds, cache identity.
- `best_model.pt`: validation-selected state_dict including BatchNorm buffers.
- `history.json`: every epoch's TRAIN/validation loss, accuracy, macro/weighted F1, precision/recall, per-class metrics and confusion matrix. TRAIN metrics are online with dropout; validation metrics use evaluation mode.
- `selection.json`: checkpoint decision established before TEST.
- `test_predictions.csv`: all example identities, true/predicted labels and four probabilities.
- `source_predictions.csv`: source label, window count, mean probabilities and prediction.
- `result.json`: configurations, weights, epochs, validation metrics, example/source TEST metrics, parameter count, serialized model bytes/path/SHA256, software and artifact hashes.

Primary metrics are example-level accuracy, macro precision/recall/F1, weighted F1, per-class precision/recall/F1 and confusion matrix (rows true, columns predicted). Undefined metrics are zero; macro averages always include four classes. Secondary source evaluation averages probabilities over **all** windows with the same source name, then argmax in fixed class order, with one metric observation per source. Long clips contribute more windows within a source. This is not majority voting or individual-mosquito evaluation.

04C first requires all eight matching completed runs and reauthenticates the saved protocol against current accepted scientific identities, code and runtime. It validates artifacts/protocol/population/seeds/weights/architecture/frontend identities before creating `comparison.csv` and `comparison.json`. It displays saved histories/confusion matrices and a provisional Stage-05 shortlist. Before completion, it shows pending-run status and raises an explicit missing-results error. The non-training validator treats that precise error as an expected block. Notebook setup reloads the Stage-04 modules in dependency order; use a fresh Stage-04 kernel for each benchmark session.

Serialized PyTorch size includes buffers/container overhead; it is not MCU Flash, RAM, INT8 size, latency, energy or hardware compatibility. Stage 05 must evaluate those, LSTM operator support, and 03B Python-versus-device parity separately. No Edge winner is declared. Windows are correlated and are not independent mosquitoes. Source-name isolation does not establish biological/session/domain independence; species/acquisition-method confounding remains. The first single-seed experiment has no uncertainty intervals or claims of superiority to prior literature.
