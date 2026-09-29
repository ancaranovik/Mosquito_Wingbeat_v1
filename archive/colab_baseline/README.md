# Stage 04 portable Colab bundle

## Run in Google Colab

1. Upload **stage04_colab_bundle.zip** to the root of Google Drive (**My Drive**).
2. On your computer, open the ZIP and extract **stage04_colab.ipynb** only.
3. Open https://colab.research.google.com/ and choose **File → Upload notebook**. Upload that notebook.
4. Choose **Runtime → Change runtime type → GPU**, such as a T4. Save the selection.
5. Select **Runtime → Run all**. Authorize Google Drive when prompted. The default `RUN_TRAINING=True`
   starts all eight experiments after installation, integrity checks and GPU synthetic forward checks.
   Set it to `False` before running if you want preflight only.
6. Read the streamed epoch logs. 03A runs first, then 03B; each runs Small CNN, DS-CNN,
   TC-ResNet8 and CNN+LSTM. Summary runs after all eight finish.
7. Results persist under **My Drive/stage04_colab_results/**. Download
   **My Drive/stage04_colab_results_export.zip** after completion.

The notebook uses Linux `/content/` paths and a new Colab-side Python environment. It never references
your Windows checkout or local `.venv-stage04`. The runtime must provide Python 3.11–3.13 and a CUDA GPU.
Dependencies are downloaded using the included requirements: PyTorch 2.8.0 CUDA 12.6,
NumPy 2.4.6 and pandas 3.0.6, plus their automatically resolved dependencies. Internet access is needed
for installation. Windows wheels/environments are deliberately not packaged.

For direct runtime upload instead of Drive, set `USE_DRIVE=False`, upload the ZIP using Colab's Files
panel to `/content/stage04_colab_bundle.zip`, and download the results before disconnecting. Drive is
the default because Colab's local disk is temporary. Do not extract the caches to Drive for training;
the notebook extracts them to the runtime's local disk and writes only results to Drive.

## Included files

```text
stage04_colab.ipynb                 single combined notebook
run_stage04.py                     preflight / branch training / comparison entry point
requirements.txt                   portable package installation pins
bundle_manifest.json               package integrity hashes
tools/model_zoo.py                 byte-identical canonical model definitions
tools/training_common.py           canonical training logic adapted for CUDA/output paths
tools/stage04_data.py              read-only authenticated cache loader
data/stage04_cache/03A/            features.npy, examples.csv, cache.json
data/stage04_cache/03B/            features.npy, examples.csv, cache.json
data/manifests/current/            accepted example manifest and frozen source split
provenance/                        scientific locks, export contract and verification evidence
results/runs/                      empty portable output structure
```

No raw WAVs, frontend generators, old notebooks, local trained results, partial runs, virtual
environments, historical archives or unrelated research dependencies are included. The two feature
arrays and their existing metadata/cache JSON are copied exactly, never regenerated.

## Preserved learning protocol

- Same four model definitions and class order: an arabiensis, an funestus ss, an squamosus,
  culex pipiens complex. Outputs are four logits.
- Seed 42 reset per model; float32; deterministic algorithms; no AMP, TF32, augmentation or input normalization.
- Weighted cross entropy using only TRAIN counts `[10457, 5128, 1426, 5602]` and weights `22613/(4*n_class)`.
- Adam, learning rate 0.001, betas 0.9/0.999, epsilon 1e-8, no weight decay, no scheduler.
- Batch 64, 50-epoch maximum, early-stopping patience 8, earliest strict minimum validation loss.
- Same data-loader ordering/shuffling and CPU initialization, followed by transfer to CUDA.
- Reload the selected checkpoint before TEST. TEST is not used for early stopping or shortlist selection.

**Execution-platform change:** the canonical implementation is CPU-only. This export intentionally
uses CUDA as requested; it preserves the learning protocol but cannot promise identical CPU/GPU
numerical results. CUDA/driver, cuDNN, GPU model, package/build and implementation identities are
recorded. CPU and CUDA results are separate benchmarks and cannot be mixed by the result validator.

All 32,645 rows are retained per cache (TRAIN 22,613; validation 4,937; TEST 5,095).
03A is 96×64 and 03B is 96×40. Frozen frontend behavior is contained in the original feature bytes;
03B's −70 dB contract is preserved. No upstream AST is reinterpreted using Colab's Python version.
Export-time upstream authentication is recorded; Colab verifies the exported caches, manifests and
identity records rather than claiming it reauthenticated WAV files that are not in the ZIP.

## Outputs and interruptions

Each `runs/{branch}_{family}_seed42/` contains `run_spec.json`, `history.json`, `best_model.pt`,
`selection.json`, `test_predictions.csv`, `source_predictions.csv` and, after success, `result.json`.
Histories save TRAIN/validation losses and metrics each epoch. Result JSON saves the selected
validation metrics and final TEST/source metrics, including confusion matrices and class order.
The optional final comparison writes `comparison.csv`, `comparison.json` and `confusion_matrices.json`.

Matching completed runs are verified and reused without another TEST evaluation. An incomplete run
fails with its artifacts preserved. Optimizer resume is not part of the canonical protocol and has
not been added. A change of GPU/runtime identity also stops reuse to avoid mixing environments.
Do not delete successful runs or choose reruns based on TEST performance. Drive storage does not
guarantee that a Colab session will last long enough to finish all eight runs.

## Validation and scope

`provenance/build_validation.json` and `provenance/local_forward_validation.json` record cache,
metadata, class-weight, import and synthetic-forward checks. No training or feature generation ran
during bundle creation. All eight model/shape combinations passed local CPU forward checks; actual
Colab/CUDA execution is checked by the notebook before training and was not tested on this machine.
`provenance/original_files_verification.json` records the original-project preservation check.

The notebook pins the package manifest; the export contract pins the input files and consumer code.
Model definitions are byte-identical, and core training/selection/metric functions were compared by
AST. The notebook itself is editable for paths and run flags and is not included in its own manifest
hash. Bundled provenance documents historical local validation, not a claim of GPU training success.

References: [Colab FAQ](https://research.google.com/colaboratory/faq.html),
[PyTorch CUDA 2.8 wheels](https://pytorch.org/get-started/previous-versions/),
[PyTorch reproducibility](https://docs.pytorch.org/docs/2.8/notes/randomness.html).
