# Edge Impulse reference configuration â€” pre-training evidence audit

Completed: 26 September 2026. No training, feature extraction on held-out audio, quantization experiment, upload, or hardware implementation has been performed.

## What can be reproduced

Two experiments must not be confused:

1. **Historical reproduction:** Abuzz-derived `aegypti`, `albopictus`, `noise`, `other`, with the original project export and original assignments. Exact reproduction is not currently established: the export, source-safe assignment, and historical DSP version are missing.
2. **Reference workflow reimplementation:** use the recovered Abuzz-derived archives and retain `aegypti / albopictus / other / noise`, with the configuration below. A current DSP version and a source-disjoint split are declared deviations. This is blocked by insufficient recoverable albopictus sources for three partitions.
3. **Project core classifier:** directly classify selected mosquito labels. The current [classification setup reassessment](CLASSIFICATION_SETUP_REASSESSMENT.md) makes the practical Core a four-class HumBugDB track: `ae aegypti`, `an arabiensis`, `culex pipiens complex`, and `ma uniformis`, spanning four genera. It is an exploratory source-group/domain-confounded feasibility benchmark and requires its own audio/taxonomy/overlap audit before training. The completed [HumBugDB audio audit](HUMBUG_CORE_AUDIT.md) remains the secondary controlled two-class check: female, unfed, light-trap-collected *Anopheles arabiensis* versus *Anopheles funestus s.s.* with a frozen 12/4/4 date split. Neither project experiment is an exact reproduction of the historical four-label reference task. This configuration table remains the four-label reference specification, not an implemented Core model. Tiny MosquitoSong remains supplementary, not the main evaluation dataset.

The later proposal's 40-band log-Mel / 512-point Hann frontend is **not used for this reference milestone**. No custom CNN has been developed.

## Evidence and version limits

- Published record: Altayeb, Zennaro, Rovai, *Classifying mosquito wingbeat sound using TinyML*, GoodIT 2022, pp. 132â€“137, [DOI](https://doi.org/10.1145/3524458.3547258). Crossref metadata saved locally. The publisher's PDF returned HTTP 403; its final typeset contents could not be independently compared.
- Full text read: [author repository manuscript](https://github.com/Mjrovai/wingbeat-mosquito-tinyml/blob/27fdf026728c66c3a8448454987aa18b904cfce2/tinyml2022db-paper17.-Mosquito.pdf). Its cover identifies **tinyML Research Symposium, March 2022**, with different author order. All â€œdocumentedâ€ paper settings below refer to this accessible manuscript, not an assertion that every setting is identical in the inaccessible GoodIT version.
- Repository inspected at `27fdf026728c66c3a8448454987aa18b904cfce2`: [preparation notebook](https://github.com/Mjrovai/wingbeat-mosquito-tinyml/blob/27fdf026728c66c3a8448454987aa18b904cfce2/notebooks/10_Dataset_Preparation_SR_16K.ipynb), [classifier notebook](https://github.com/Mjrovai/wingbeat-mosquito-tinyml/blob/27fdf026728c66c3a8448454987aa18b904cfce2/notebooks/20_ei_ictp_mosquito_wingbeat_sound_classification_tinyml_nn_classifier.ipynb). Notebooks were parsed as data, never executed. Embedded credentials were not used; the text extract is redacted.
- Current official Spectrogram code inspected at `9b0ceb2b5d22658d3193a5b77dc773cfc8d4cab0`: [parameters](https://github.com/edgeimpulse/processing-blocks/blob/9b0ceb2b5d22658d3193a5b77dc773cfc8d4cab0/spectrogram/parameters.json), [DSP](https://github.com/edgeimpulse/processing-blocks/blob/9b0ceb2b5d22658d3193a5b77dc773cfc8d4cab0/spectrogram/dsp.py), [FFT/framing](https://github.com/edgeimpulse/processing-blocks/blob/9b0ceb2b5d22658d3193a5b77dc773cfc8d4cab0/spectrogram/third_party/speechpy/processing.py). Current code is corroboration, not proof of the historical implementation.

## Historical dataset and split

The manuscript Â§Â§4.1â€“4.3 names the original Abuzz/Mukundarajan resource and reports:

| Class | Reported original material |
|---|---|
| `aegypti` | 3 files, approximately 11.5 minutes, 44.1 kHz/16-bit |
| `albopictus` | 2 files, approximately 11.1 minutes, 44.1 kHz/16-bit |
| `other` | 16 files, approximately 27 minutes, selected from other mosquito species |
| `noise` | Speech Commands background excerpts (243 files, ~4 min), Abuzz background (9 files, ~3.1 min), authors' laboratory background (1 file, ~4.1 min) |

â€œOtherâ€ pools species examples; it does not define the optional multi-label swarm task. The paper reports approximately 86% training / 14% testing, then holds out 20% of training for validation. It gives class-wise train/test durations, but not a reproducible original-source assignment manifest. Its statement that 297 files are all one minute long conflicts with its roughly 46-minute total and numerous one-second noise files. The actual downloaded archive inventory is reported separately in `historical_archive_summary.json`.

The preparation notebook splits original recordings into roughly minute-sized parts (`np.array_split`), also contains an exploratory within-recording first-fifth test split, and writes 16 kHz PCM16. The classifier notebook calls `train_test_split` on already generated feature windows with `test_size=0.2, random_state=1`; those arrays are then used as training/validation. It does not group by source. Overlapping windows and related recording parts can therefore contaminate validation. The historical protocol must not be copied as a claim of independent-individual generalization. Two albopictus originals cannot populate three disjoint source partitions.

## Studio settings and provenance

Status legend: **D** = documented in accessible original manuscript; **R** = recovered/inferred from repository or official DSP implementation; **P** = proposed for this reimplementation because original setting is unavailable or scientifically unsuitable. â€œUnknownâ€ means it has not been recovered, not that a default is safe.

| Setting | Value to record/use for reference reimplementation after GO | Evidence/status |
|---|---|---|
| Input | Audio, mono PCM16 | D/R: manuscript preparation and notebook |
| Sampling frequency | **16,000 Hz** | D: manuscript Â§Â§4.1â€“4.3 |
| Analysis/window size | **1,000 ms** | D: Â§4.4; selected after exploring 1/2/4 s |
| Window increase | **250 ms** | D: Â§4.4; this is distinct from STFT frame stride |
| Processing block | **Spectrogram** | D: Â§4.4; not MFE, MFCC, or the proposed later log-Mel |
| Frame length | **0.025 seconds** (25 ms) | D: Â§4.4 |
| Frame stride | **0.0125 seconds** (12.5 ms) | D: Â§4.4 |
| FFT length | **128** | R: paper says â€œ128 frequency bandsâ€; its 65 columns and official real FFT mapping support interpreting 128 as FFT length, not 128 output bins |
| Noise floor | **âˆ’52 dB** | D: Â§4.4; numerical normalization also depends on DSP implementation version |
| Frequency range | Entire one-sided FFT, nominal **0â€“8,000 Hz**, including DC/Nyquist | R: no separate lower/upper-band parameter in the inspected Spectrogram block. The paper's discussion of 500â€“700 Hz is not a preprocessing bandpass |
| Window function | No manually added Hann/Hamming window; use the chosen Spectrogram block | R: inspected code passes a rectangular/all-ones frame filter. Historical exported code is unavailable, so historical taper cannot be certified |
| DSP implementation version | **4 for a current-code reimplementation**, if that is the version shown by Studio; explicitly save version/export before training | P: inspected current parameters identify version 4. Original version is UNKNOWN. Do not claim version 4 is the 2021/2022 baseline. If Studio exposes a different version, revise this record before training |
| Expected features | **79 time frames Ã— 65 bins = 5,135 scalars** | D: Â§4.5; R: notebook reshape. Must verify with a development fixture before training; do not silently change FFT to obtain another size |
| Classes â€” historical | `aegypti`, `albopictus`, `noise`, `other` | D/R: manuscript and notebook |
| Classes â€” this reference reimplementation | `aegypti`, `albopictus`, `noise`, `other` | D/R: keep the original task; do not relabel individual species as four reference outputs |
| Learning block | Classification (Keras), original Conv1D layer sequence below | D/R |
| Max epochs/training cycles | **100** | D: Â§4.6 |
| Learning rate | **0.001** | D: Â§4.6 |
| Batch size | **32** | D: Â§4.6 |
| Optimizer | **Adam**, beta1 **0.9**, beta2 **0.999** | R: notebook; epsilon left at framework default, original environment/version UNKNOWN |
| Loss | **Categorical cross-entropy**, one-hot labels | R: notebook |
| Early stopping | `val_loss`, mode `min`, patience **5** | D/R |
| Restore best weights | Notebook does not request it (default false); use **false** for notebook matching | R: code default; Studio default must not be assumed equivalent. Record any deliberate change |
| Training randomness | Python/NumPy/TensorFlow seed **3** in notebook | R: historical feature-level splitter uses seed 1; our source split uses a separately recorded deterministic rule |
| Validation | **Explicit frozen validation membership**; no automatic random feature/window split | P: replaces unsafe historical protocol; do not let Studio take an additional random 20% |
| Learned optimizer / autotuning / auto class weights | **Off** | P: no VeLO/auto weights in inspected notebook; paper used EON Tuner during search but does not provide the complete search. Fixed reference reproduction does not rerun that search |
| Additional data augmentation | **Off** | P: no new noise, pitch, frequency/time masking; the documented 250 ms overlap is retained only within already assigned source groups |
| Training processor | **CPU** initially | P: original processor/environment not recovered; no hardware target needs to be selected for this milestone |
| Quantization | Historical work reports TensorFlow Lite **INT8**, followed by **EON Compiler** | D: Â§5.0.1; calibration set, converter/runtime versions and exact flags UNKNOWN. Record only; no conversion or comparison is authorized now |
| Deployment | Historical export: **Arduino library**, Nano 33 / Portenta / Wio Terminal | D: Â§5.0.2; no ESP32 implementation is implied or performed |

The original layer sequence is a **configuration specification only**, not new model code:

| Order | Layer |
|---|---|
| 1 | Reshape flat 5,135 features to `(79, 65)` |
| 2 | Conv1D: 32 filters, kernel 3, stride 1, `same`, ReLU |
| 3 | MaxPooling1D: pool 2, stride 2, `same` |
| 4 | Conv1D: 64 filters, kernel 3, stride 1, `same`, ReLU |
| 5 | MaxPooling1D: pool 2, stride 2, `same` |
| 6 | Flatten |
| 7 | Dropout 0.5 |
| 8 | Dense 4, softmax; save the actual class-index mapping |

Filter counts/pooling/dropout/head are documented in Â§4.5; kernel size and reshape are recovered from the notebook. Do not select an unrelated Studio architecture preset. Expert mode may be necessary later to express this exact sequence; no expert-mode model was created in this milestone. In the notebook, `tf.data` batches are supplied without an explicit dataset shuffle, and early stopping does not restore best weights. Those details and original TensorFlow/Keras/library versions limit exact numerical reproducibility.

## Important DSP consequence

At 16 kHz, 25 ms contains **400 samples**, while FFT length 128 is only **8 ms**. The inspected official implementation uses `rfft(..., n=128)`, which truncates longer frames; the official documentation explicitly describes clipping/padding to FFT length. The 128-point FFT has **125 Hz bin spacing**. Do not call this a full 25 ms / 40 Hz spectral analysis, or silently change FFT length to 512 to make it more attractive.

Current version 3/4 applies `10*log10(power)` and rescales using the noise floor; version 3 clips the upper normalized value to 1, whereas version 4 does not. Earlier versions have different normalization/framing. A matching feature shape alone does not certify matching DSP values. A pinned project export and numerical reference fixture are needed for faithful DSP reproduction.

The recovered reference archives are already 16 kHz mono PCM16; no resampling is needed. Tiny MosquitoSong acquisition was 96 kHz/24-bit, but its paper explicitly documents downsampling to 8 kHz/16-bit, matching every released mosquito WAV audited. Keep those files at 8 kHz. They are not the reference input dataset. If a later compatibility study deliberately feeds them to a 16 kHz interface, upsampling **cannot restore information above their original 4 kHz Nyquist limit**. No such conversion is prepared or approved here. Zero padding, looping, or concatenating short cuts into one-second examples would change the reference and must not be done silently.

## Remaining unknowns

The final GoodIT PDF's exact correspondence to the author manuscript; historical Edge Impulse project version/export; exact DSP version and historical taper/normalization; original per-source train/test membership; complete EON search; training environment and default epsilon/initializers; exact quantization calibration examples; converter/EON versions and flags. Architecture and principal numerical settings are recoverable, but a fully identical published experiment is not.

Official references: [Spectrogram](https://docs.edgeimpulse.com/studio/projects/processing-blocks/blocks/spectrogram), [learning settings](https://docs.edgeimpulse.com/studio/projects/learning-blocks), [explicit validation](https://docs.edgeimpulse.com/studio/projects/data-acquisition/dataset/advanced-settings), [grouped splits](https://docs.edgeimpulse.com/studio/projects/data-acquisition/dataset/splits).


Current four-class readiness: **GO for four-class training**. See [HUMBUG_FOUR_CLASS_FINAL_AUDIT.md](HUMBUG_FOUR_CLASS_FINAL_AUDIT.md) for the frozen dataset and explicit Core conversion/frontend specification. This does not change the historical reference task or make it an exact reproduction.

