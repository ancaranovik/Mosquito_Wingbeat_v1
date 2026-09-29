# Pipeline and comparison contract

Open 00, 01 and 02 in order, then 03A and 03B. Run from the repository root or `notebooks/current/`. Current paths are mandatory; frontend notebooks do not fall back to obsolete flat-data locations.

## Accepted foundation

- **00:** reconstruct the strict Core and exact recorded split in memory; compare all canonical metadata and existing reproduction copies. Ordinary validation does not rewrite these files. Keep `WRITE_CANONICAL = False`.
- **01:** independently check Core selection, exact split-to-Core parent metadata, non-null required fields, source isolation and every WAV header/duration. Nullable release fields such as recording date are not incorrectly made mandatory.
- **02:** physically check all contexts with the existing float32 geometry-validation path, compare the reconstructed manifest byte-for-byte with the accepted manifest, and authenticate per-clip WAV hashes/lengths. Preserve the manifest and original generation audit; write only a separate gated freeze acceptance audit.

The Core is 1,898 clips / 1,189 sources. Source-level splitting precedes segmentation. The grid is 32,645 examples: TRAIN 22,613, validation 4,937, test 5,095. Each ID retains its parent source, species and split.

At 16 kHz, starts are separated by 15,360 samples (0.96 s). Each reference context has 15,600 samples (0.975 s), producing 96 complete 400-sample frames at hop 160. Adjacent contexts overlap by 240 samples; the groups of reference feature frames do not overlap. Exactly 319 historical terminal examples are excluded because the longer context is incomplete. No activity-based filtering or completion padding is introduced.

The accepted manifest remains `core_example_grid_v2_stride15360_ref15600_candidates.csv`:

`SHA-256 36881d79d33cd42521aa24004236022dc25b4927f65ae0246a76832835206234`

The frozen split remains:

`SHA-256 3e40928622dcea26e723540cfb493a35fd1d78393bf5531810583eae7966e0c4`

All accepted data/source hashes are enforced in `tools/pipeline_contract.py`; both frontends authenticate the grid acceptance audit and all 1,898 WAVs before using them.

## Waveform precision and frontend scope

02's float32 processing verifies geometry, expected resampled lengths and finite support. It does not provide feature-input waveforms or claim bitwise equivalence with the feature notebooks.

Both 03A and 03B retain **float64 PCM24 conversion and float64 whole-clip resampling**: SciPy left-justified int32 divided by 2**31, `resample_poly(160,441,window=('kaiser',5.0),padtype='constant',cval=0)`. Expected output length is `ceil(native_samples*160/441)`. There is no peak/RMS normalization, denoising, silence removal or other activity gate. Resampler boundary extension is not incomplete-example padding.

| Property | 03A Log-Mel | 03B Edge MFE |
|---|---|---|
| Shared context and starts | 15,600 samples; grid starts | Same accepted IDs and starts |
| Frames / hop | 400 / 160 | 320 / 160 |
| Window / FFT | Periodic Hann; pad framed samples to 512 | Rectangular; FFT 256 truncates each 320-sample frame |
| Spectrum | Unnormalized magnitude | Power: magnitude squared / 256 |
| Mel representation | 64 bands, 125–7500 Hz, Mel-domain triangles | 40 bands, 0–8000 Hz, pinned SpeechPy v4 bins |
| Compression | Natural log(mel + 0.01) | Pinned dB/noise-floor transform and quantize/dequantize |
| Output | 96 × 64, float64 | 96 × 40, float32 |

03A verifies `mel_features.py` numerical parity on **identical 16 kHz contexts**. It does not reproduce the full HumBugDB PCM16/resampy loading path or a published classifier experiment. Its output is float64; a complete cache would occupy 1,604,567,040 bytes (1.60 GB / 1.49 GiB) before metadata. Explicit float32 conversion would halve that, but is not performed here.

03B preserves the pinned wrapper's float32 input cast and per-context pre-emphasis (`0.98`, shift 1, `np.roll`). The first sample depends on the last sample of the same context. It also preserves rounding before uint8 conversion and clipping: values rounding to 256 wrap to zero. These are pinned Python behaviors, not validated C++/ESP32/continuous-stream behavior. Equal contexts do not mean equal effective spectral support.

Pinned UI defaults are 20 ms frames, 10 ms stride, 40 filters, FFT 256, 0–Nyquist and **−52 dB**. The pinned CLI instead defaults to 20 ms stride and 32 filters. Project settings are explicit and do not rely on CLI defaults.

**The frozen project noise floor is −70 dB.** A 128-example balanced TRAIN subset explored −52/−55/−60/−65/−70 dB. Full-TRAIN confirmation compared −65 and −70: 11 versus zero entirely zero tensors among 22,613 examples; both had zero near-upper-bound bins. This bounded TRAIN-only anti-collapse rationale does not establish global optimality. No validation/test signal characteristics were used for selection and no examples were removed.

## Gates and evidence

The target statuses are 02 **ACCEPTED / FROZEN**, 03A **VERIFIED / FROZEN**, and 03B **VERIFIED / FROZEN**. The original grid-generation audit remains historical evidence with its original candidate status. Its separate freeze audit identifies the accepted manifest and generation-audit hashes.

Both frontend notebooks enforce input/source identities, validate full grid geometry and require successful numerical/diagnostic gates **before** atomic audit persistence. 03B includes direct comparisons with the original authenticated `generate_features` function body, synthetic/boundary/real TRAIN cases, and the complete −65/−70 TRAIN confirmation. The helper executes that function without the module's unrelated import/graph side effects; it uses the actual vendored SpeechPy implementation.

Audit JSONs contain authenticated inputs and source identities, waveform precision/options, relevant package versions, final gate results and diagnostic evidence. Changed accepted identities or numerical evidence stop execution; do not update pins just to make a failing run pass.

For fresh validation, install `requirements-pipeline.txt`, then run `python -B tools/validate_pipeline.py`. Every notebook gets a fresh kernel. The runner preserves accepted scientific inputs, reference/vendor bytes and historical artifacts, and records successful runs in `reports/validation/pretraining_review.json`. Optional PDF export is disabled; existing PDFs remain historical. The one-time reorganization verifier addresses preservation at reorganization time, not current notebook execution status.

## Fair comparison requirements for the later training plan

These are two complete frontend configurations, not an experiment isolating one mathematical operation. The following requirements carry into notebook 04; this work does not create that notebook or choose a model:

- Use exactly the same accepted example IDs, labels and frozen partitions for both branches. Do not add MFE-only terminal examples or discard low-energy/zero-feature examples.
- Fit learned preprocessing only on TRAIN. If feature standardization or cache/model dtype conversion is used, specify it before training and document it separately from the frozen frontend output.
- Define comparable model capacity, initialization/pretraining policy, seeds, optimization and tuning budgets. Different tensor widths can change parameter count and compute; report those differences. A pretrained-versus-scratch contrast cannot be attributed solely to the frontend.
- Define validation-based selection, stopping and evaluation procedures before training; do not use test performance to tune either branch or reopen the frontend choices.
- Predefine metrics and aggregation at example, clip and/or source level. Windows from the same source are correlated; they are not independent biological replicates. Any uncertainty calculation must respect source grouping.

This review establishes frontend/data readiness, not an already-selected architecture or training/evaluation hyperparameters. Source-name isolation does not prove isolation by mosquito, collection session or all acquisition domains. All Core clips share the recorded device/microphone/location, but acquisition-method distributions differ by species. Preserve the frozen split and disclose these limits rather than claiming cross-domain generalization.
