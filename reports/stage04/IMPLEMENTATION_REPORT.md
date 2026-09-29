# Stage 04 implementation report

> Historical implementation record. Its original nine-test status, notebook-output claims and artifact hashes are superseded by [the September 29 pre-training audit](PRETRAINING_AUDIT.md), `implementation_validation.json`, and `audit_artifact_inventory.json`. No frozen scientific artifacts were changed by that audit.

## Outcome

Stage 04 is implemented and ready to run in the documented environment. All three notebooks passed fresh-kernel execution with `RUN_TRAINING=False`. Nine engineering tests passed. Both full feature caches were generated and verified: 32,645 examples per branch. **No mosquito classifier training or TEST model evaluation was executed.** No trained checkpoints, eight-run performance table or actual Stage-05 shortlist exists yet; these are produced by the full training/comparison workflow. No Stage-05 work, remote Edge Impulse use, quantization or deployment was performed.

See [complete methodology and execution instructions](../../docs/STAGE04.md), [machine-readable validation](implementation_validation.json), [preflight evidence](preflight.json), [engineering test log](engineering_tests.txt), and [exact architecture inventory](architecture_inventory.json).

## 1. Exact files created

All paths below are relative to the project root. `artifact_inventory.json` records their sizes and SHA256 identities (excluding its own self-hash).

```text
notebooks/current/04A_train_logmel.ipynb
notebooks/current/04B_train_edge_mfe.ipynb
notebooks/current/04C_compare_models.ipynb
tools/model_zoo.py
tools/stage04_data.py
tools/training_common.py
tools/test_stage04.py
tools/validate_stage04.py
requirements-stage04.txt
docs/STAGE04.md
data/stage04_cache/03A/features.npy
data/stage04_cache/03A/examples.csv
data/stage04_cache/03A/cache.json
data/stage04_cache/03B/features.npy
data/stage04_cache/03B/examples.csv
data/stage04_cache/03B/cache.json
reports/stage04/frozen_input_lock.json
reports/stage04/preflight.json
reports/stage04/engineering_tests.txt
reports/stage04/architecture_inventory.json
reports/stage04/planned_protocol.json
reports/stage04/implementation_validation.json
reports/stage04/IMPLEMENTATION_REPORT.md
reports/stage04/artifact_inventory.json
```

PyTorch 2.8.0 and its dependencies were installed in a separate temporary validation directory. The existing accepted numerical runtime was reused read-only. No project-wide Python environment was changed. For a durable user environment, install `requirements-stage04.txt` as documented.

## 2. Exact existing files changed

**None.** The five 00–03B notebooks, all frozen CSV/WAV/manifest/audit files, `tools/pipeline_contract.py`, the accepted dependency file, pipeline documentation and historical review were preserved. The final validation reauthenticated all frozen pins and WAV identities. This workspace has no Git repository, so preservation evidence is file hashes rather than a Git diff.

## 3. Final architecture definitions

All convolutions are bias-free followed by BatchNorm/ReLU, with the residual second convolution activated after addition. BN epsilon=1e-5 and momentum=0.1. Heads use dropout 0.2 and biased linear four-class logits. Symmetric convolution padding is kernel//2; pooling uses floor sizing.

| Model | Definition | 03A params | 03B params |
|---|---|---:|---:|
| Small CNN | 3×3 conv widths 16/32/64; 2×2 max-pool after first two; global mean; head | 23,668 | 23,668 |
| DS-CNN | 32-channel 3×3 stride-2 stem; four 3×3 depthwise +1×1 pointwise blocks; global mean; head | 6,244 | 6,244 |
| TC-ResNet8 adaptation | Frequency-as-channels; kernel-3 stem width16; kernel-9 two-convolution residual blocks widths24/32/48, first conv stride2, projected shortcuts; temporal global mean; head | 65,940 | 64,788 |
| CNN+LSTM | 3×3 conv widths16/32, each followed by 2×2 max-pool; frequency mean; 24-step, 32-hidden-unit, one-layer unidirectional LSTM; final hidden state; head | 13,428 | 13,428 |

TC depth counts seven main-path convolutions plus classifier, excluding projection shortcuts. The temporal/residual idea is preserved, with documented adaptation to 96 frames, four labels, Conv1d, symmetric padding and the common PyTorch protocol. No literature-reproduction or superiority claim is made. Full exact configurations and module representations are saved in the architecture inventory and every future run.

## 4. Training protocol

PyTorch 2.8.0, CPU, float32, four threads, MKLDNN disabled, deterministic algorithms, from-scratch default initialization. Weighted cross-entropy; Adam (lr 0.001 constant, betas 0.9/0.999, epsilon 1e-8, no weight decay, no AMSGrad, foreach disabled); batch 64; maximum 50 epochs. No augmentation, oversampling, label smoothing, gradient clipping or input normalization. TRAIN shuffles; validation/TEST do not. Zero loader workers, final partial batch retained.

Strictly lowest weighted validation loss selects the checkpoint; exact ties keep the earliest epoch. Patience 8 with min_delta 0. The checkpoint SHA256, selected epoch and validation metrics are persisted before TEST loading/evaluation. The selected-validation macro-F1 and maximum observed validation macro-F1 are distinct fields. The shortlist ranks the top three selected-validation macro-F1 values, breaking ties by parameter count then run ID; TEST never selects it.

## 5. Class imbalance

Weights are calculated from TRAIN only as `22613/(4*n_class)`, saved exactly, then cast to float32 for the loss. Both branches use the identical rule and values.

| Species / label order | TRAIN count | Weight |
|---|---:|---:|
| an arabiensis /0 | 10,457 | 0.5406187242995123 |
| an funestus ss /1 | 5,128 | 1.1024278471138846 |
| an squamosus /2 | 1,426 | 3.9644109396914446 |
| culex pipiens complex /3 | 5,602 | 1.0091485183862907 |

Weighted losses accumulate sample-weighted numerator/denominator over each partition. Classification metrics are unweighted. No validation/TEST counts determine weights.

## 6. Reproducibility

Python, NumPy, PyTorch and independent data-loader generators use seed 42. Seeds reset for each run; there is no averaging across seeds. Software, processor/build details, exact architecture, configuration, source hashes, cache hashes and checkpoint hashes are recorded. Deterministic operations are required, but cross-platform/hardware/version bitwise equality is not guaranteed. Completed runs are authenticated and reused; incomplete runs stop for investigation.

## 7. Authentication and validation performed

- All accepted source/data pins and 1,898 WAV SHA256 identities passed.
- 02 accepted freeze and both 03 frontend audit status, final gates and identities passed.
- Full grid/parent/source isolation, label mapping and counts passed: TRAIN 22,613, validation 4,937, TEST 5,095.
- All 32,645 unique UIDs, labels, splits and row ordering match across cache branches.
- Full tensor shape, float32 dtype, finiteness and cache hashes passed: 03A 96×64, 03B 96×40.
- Ten synthetic/boundary/fixed-TRAIN parity cases passed. Worst 03A difference was approximately 2.31e-14 (tolerance 1e-10); 03B was exactly equal to the pinned wrapper. Both waveform loaders matched on the tested TRAIN clips.
- Nine engineering tests passed, including all eight forward/backward shapes, checkpoint roundtrips, deterministic initialization/shuffle, hand-calculated metrics, source probability aggregation, weighted loss accumulation, validation stopping/ties and corruption/partial-comparison rejection.
- All three notebooks passed in fresh kernels from `notebooks/current/`, with training disabled.
- Final reauthentication confirmed original frozen artifacts unchanged.

**Provenance concern retained:** historical execution hashes for 00–02 in `pretraining_review.json` differ from current notebook bytes. Stage 04 does not claim why they differ or claim their fresh execution. It pins the current user-accepted files and authenticates all consumed output artifacts independently. Stage 02's recorded scientific function AST hashes match, and both actually reused 03A/03B notebook hashes match their historical successful executions exactly. The discrepancy is explicitly saved in preflight/validation records; no historical evidence was rewritten.

## 8. Branch differences

The underlying frozen frontends retain all their accepted differences (frame/window/FFT, spectrum, filterbank/frequency range, compression, pre-emphasis and quantization). This is a comparison of **complete frontend configurations**. Stage 04 adds no branch-specific preprocessing or tuning. Only 03A's completed float64 output is cast to the common float32 model dtype. Input dimensions change activation sizes/compute, and TC's input-channel stem changes parameter count by 1,152. Every family's design/protocol otherwise matches.

## 9–11. Readiness and execution status

The notebooks are runnable after selecting the documented environment. Caches already exist. Set `RUN_TRAINING=True` in 04A/04B to execute the full benchmark, then run 04C. No real-data optimizer step or classifier TEST evaluation occurred during implementation. Synthetic backward tests did not update model parameters; early-stopping tests used mocked metrics. Therefore a complete measured eight-run comparison table cannot yet be reported. Pending states are shown honestly in 04C.

## 12. Limits before Stage 05

The positive full-training/result-export/comparison path has not been exercised with trained models. CPU runtime may be substantial; interrupted fitting does not resume optimizer state. The first benchmark uses one seed with no confidence intervals. Windows are correlated, source names do not establish individual-mosquito/session/domain independence, and the frozen acquisition-method confounding remains. PyTorch checkpoint bytes are not INT8 model size, MCU Flash/RAM or latency. Stage 05 must assess conversion/operator support (especially LSTM), quantization, hardware costs and device parity. No Edge winner is declared.
