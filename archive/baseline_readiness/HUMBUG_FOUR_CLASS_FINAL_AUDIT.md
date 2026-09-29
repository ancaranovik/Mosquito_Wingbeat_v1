# GO for four-class training

The local WAV audit is complete. The pinned release is HumBugDB 0.0.1, Zenodo record 4904800. Selection remains Tanzania/cup/mosquito, with the same four labels and original `name` grouping. All selected records are from Ifakara. Raw media remains in `../data/audio/`, mapped by `<id>.wav`; it was not reorganized or converted. No model was trained.

## Coverage and integrity

All 1,596 expected clips are present, including all 297 files listed in the previous missing-file report. All decode successfully as **44,100 Hz, mono, PCM24 WAV**. Total decoded duration is **25,949.972449 seconds (7.2083 hours)**. Maximum metadata-duration discrepancy is one sample (22.68 microseconds). All files contain at least one second of audio. No digital-silence files or samples at absolute normalized amplitude >=0.999 were found. These checks do not exclude analog saturation or background interference.

SHA-256 hashes of both file bytes and decoded int32 PCM are recorded. There are no exact decoded-PCM duplicates. A bounded overlap screen searched 4,788 non-silent 100 ms anchors (start, middle and end of each clip) at arbitrary sample-aligned positions in every clip; **zero cross-file matches** were found. This cannot exclude gain-modified copies, different microphone views, overlaps missing the anchors, or undocumented repeat animals. The original-name grouping provides the primary source leakage protection.

The local extracted files were supplied by the user as release 0.0.1. No whole-archive checksum/CRC certification is claimed in this run: this is a decoded-WAV integrity audit, not retrospective verification of ZIP downloads.

## Frozen counts

Each split cell gives **clips / original-name groups**. All files are retained; **zero exclusions**. Usable means decodable, sufficient duration and passing the stated basic checks, not independently confirmed uninterrupted flight.

| Exact label | Total clips / groups | Training | Validation | Testing |
|---|---:|---:|---:|---:|
| ae aegypti | 89 / 49 | 46 / 29 | 22 / 10 | 21 / 10 |
| an arabiensis | 831 / 514 | 489 / 308 | 165 / 103 | 177 / 103 |
| culex pipiens complex | 545 / 336 | 336 / 202 | 116 / 67 | 93 / 67 |
| ma uniformis | 131 / 76 | 81 / 46 | 27 / 15 | 23 / 15 |
| Total | 1,596 / 975 | 952 / 585 | 330 / 195 | 314 / 195 |

The final source manifest was checked row-for-row against the existing candidate: identical source IDs, labels, original names and partition assignments. Group counts are exactly 60/20/20 overall. Every original `name` has one species and one partition. No source groups or exact PCM hashes cross partitions. Dates may cross partitions; this is an original-recording split, not a session/date-disjoint experiment. The source labels remain unchanged: Culex is a species complex and `ma uniformis` denotes Mansonia uniformis in the source classification task; acoustic QC does not re-identify specimens.

Class imbalance remains substantial. Report original-level macro-F1, balanced accuracy, per-class recall, confusion matrices and per-method/date coverage. Do not claim thousands of windows are independent recordings. Class balance methods must use Training only.

## Representative quality inspection

Twelve training-only examples were selected at the 10th/50th/90th percentiles of the spectral-contrast proxy within each class. Any recordings in the secondary benchmark's test set were excluded from this representative selection. The first up to eight seconds were plotted as waveform amplitude envelopes and spectrograms. Both images were visually inspected. No listening-based or entomological validation is claimed.

Examples show intermittent and drifting harmonic ridges amid strong low-frequency backgrounds and transients. Culex 220950 shows bands near 550, 1100 and 1650 Hz; 221058 has changing harmonic structure and a strong initial transient. Mansonia 220659 shows a drifting band near 500–600 Hz with harmonics; 220823 has intermittent weaker activity. These observations are compatible with wingbeats, but stationary tones can also produce high spectral contrast. No tonality threshold was used to exclude hard examples, and the proxy is not an activity percentage or SNR measurement.

See [spectrograms](humbug_four_audit/representative_spectrograms.png), [waveforms](humbug_four_audit/representative_waveforms.png) and [selected examples](humbug_four_audit/representative_audio.csv).

## Interpretation and secondary benchmark

This remains an exploratory genus-spanning, domain-confounded feasibility benchmark. Aedes is LC-only with four dates; other labels have unequal LT/HBN/LC mixtures, sex/feeding compositions and dates. Shared Tascam/Telinga metadata does not establish identical physical device, gain or environmental conditions. Culex is not a single resolved species. A closed-set species score does not detect mosquito presence or unknown species.

**Cross-experiment limitation:** 47 original groups from the secondary two-class benchmark's frozen test set occur in this four-class Training/Validation set. The requested candidate split was preserved. This does not leak within the four-class split, but the secondary result cannot be treated as independent confirmation after four-class development has used those sources. If run, train its model from scratch using only its own training partition, with no four-class weights or data-fitted preprocessing, and predeclare its configuration without adapting it to these shared sources. Report it as a controlled comparison with shared study data, not an independent replication. Exact intersections are saved in `humbug_four_audit/secondary_test_cross_experiment_overlap.csv`.

## First training experiment: fixed preprocessing specification

This is an Altayeb-inspired workflow adaptation, not an exact reproduction of the historical `aegypti/albopictus/other/noise` task. The following conversion choices are proposed engineering settings; reference Spectrogram settings and their evidence remain in BASELINE_CONFIGURATION.md.

1. Preserve native PCM24 WAVs and hashes. Decode to float64 using fixed PCM scaling; no per-clip RMS/peak normalization, denoising, AGC, filtering beyond anti-aliasing, or augmentation.
2. Convert 44,100 to **16,000 Hz mono** with `scipy.signal.resample_poly(x, 160, 441, window=('kaiser', 5.0), padtype='constant')`. Encode PCM16 by `clip(rint(y*32768), -32768, 32767).astype(int16)`. Save dependency versions and converted-file hashes. Conversion has not yet been executed by this audit.
3. Generate **1,000 ms** analysis windows every **250 ms** within each clip. Drop incomplete terminal windows; no padding, looping or concatenation. Every window inherits its clip's frozen original-name partition.
4. Edge Impulse **Spectrogram**, not log-Mel/MFE: frame length **25 ms**, frame stride **12.5 ms**, FFT **128**, noise floor **-52 dB**, all one-sided bins (0–8,000 Hz), expected **79 × 65 = 5,135** features. Use a pinned current implementation version 4 and save the Studio export. A 400-sample frame with FFT 128 is truncated by the inspected implementation; bin spacing is 125 Hz. This peculiarity is retained deliberately for the reference-inspired first run.
5. Before training, verify the pinned frontend's shape and numerical behavior on a Training fixture. If the installed block differs, record the discrepancy rather than silently changing settings. Use explicit manifest validation; disable automatic window/sample repartitioning. Keep Testing local until final evaluation. Fit no settings, thresholds or quantization statistics on Testing.

Use the recovered Conv1D 32/64, kernel 3, same/ReLU, each followed by pool 2; flatten, dropout 0.5, four-output softmax. First baseline: Adam 0.001, batch 32, maximum 100 epochs, validation-loss early stopping patience 5, seed 3, no augmentation and no automatic tuning. Start unweighted to retain the reference training procedure; report imbalance through macro/group metrics. Any later class-weighted comparison is separately declared and selected using Validation only. No model or features were produced here.

## Frozen artifacts and decision

- [Source manifest](humbug_four_audit/four_class_source_manifest.csv)
- [Group manifest](humbug_four_audit/four_class_group_manifest.csv)
- [Training](humbug_four_audit/four_class_training_files.csv), [Validation](humbug_four_audit/four_class_validation_files.csv), [Testing](humbug_four_audit/four_class_testing_files.csv)
- [Freeze record and hashes](humbug_four_audit/four_class_freeze.json)
- [Audit summary](humbug_four_audit/final_audit_summary.json)

**GO for four-class training** on `humbug-four-class-v1`, under the fixed source partitions and limited exploratory claim above. The final test set is now frozen. No training, synthetic mixtures, INT8 conversion or ESP32 implementation was performed.
