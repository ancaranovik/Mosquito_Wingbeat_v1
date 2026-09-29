# HumBugDB Core dataset audit

26 September 2026. **CONDITIONAL GO for the controlled two-species Core experiment specified below.** No training, model inference, augmentation, quantization experiment or hardware work was performed. This decision is separate from the original four-label TinyML reference baseline, which remains blocked by its source-partition problem.

## Decision and exact next dataset

Use **HumBugDB release 0.0.1, Zenodo record 4904800**, Tanzania / Ifakara / cup recordings, restricted to **female, unfed (`fed=f`), individually recorded (`Single`), light-trap-collected (`LT`) mosquitoes**, and only recording dates that contain both eligible species:

- **Anopheles arabiensis** (`an arabiensis` → `anopheles_arabiensis`).
- **Anopheles funestus sensu stricto** (`an funestus ss` → `anopheles_funestus_ss`).

Do **not** include Aedes aegypti in the main Core-v1 experiment. Its label is present and credible, but every selected aegypti source is larval-collected (`LC`) and its 49 originals span only four dates. Capture method perfectly separates aegypti from the two Anopheles species. Audio alone cannot resolve whether a classifier learns species or collection-associated conditions. The restriction is based on metadata, not model results.

The frozen core contains **668 clips, 444 documented original-recording/individual proxies, 20 recording-date groups, approximately 4.03 hours**. Treat recording dates as the partition unit, and original recordings as the individual-level reporting unit.

| Species | Training originals / clips | Validation originals / clips | Test originals / clips | Total originals |
|---|---:|---:|---:|---:|
| Anopheles arabiensis | 179 / 273 | 61 / 92 | 60 / 82 | 300 |
| Anopheles funestus s.s. | 88 / 146 | 28 / 37 | 28 / 38 | 144 |
| Total | 267 / 419 | 89 / 129 | 88 / 120 | 444 |

There are **12 / 4 / 4 disjoint dates**, exactly 60/20/20 by date and approximately 60/20/20 by originals within each species. All selected dates contain both species. Clips and future windows inherit their source and date assignments.

Exact memberships: [Training](humbug_audit/core_training_files.csv), [Validation](humbug_audit/core_validation_files.csv), [Testing](humbug_audit/core_testing_files.csv). The [source manifest](humbug_audit/core_source_manifest.csv) includes all 1,301 audited clips, their metadata, labels, hashes, grouping, membership and exclusions. The [freeze record](humbug_audit/core_freeze.json) identifies the final version and hashes.

## Material actually inspected

The [public release](https://zenodo.org/records/4904800) was inspected at its fixed version, together with the metadata used by the pinned [HumBugDB repository](https://github.com/HumBug-Mosquito/HumBugDB/tree/50656758594982480f568598874f79c222432e01). The earlier cached metadata and the pinned CSV agree row for row. This audit does not silently switch to the newer Zenodo release advertised on that page.

All **1,301** matching mosquito clips for the three proposed species were retrieved from archives 3 and 4 using indexed byte ranges. Archive directories 1 and 2 contained no selected members. Each retrieved member passed its ZIP CRC32/size check and every audio file was decoded; SHA-256 file and decoded-PCM hashes are recorded. Whole multi-gigabyte archives were not downloaded, so full-archive MD5 validation is not claimed. The selected archive index is saved in `sources/humbug_selected_archive_inventory.json`.

This is a complete audio audit of the three proposed species in the Tanzania cup cohort, **not** an audio audit of all 2,582 Tanzania cup mosquito rows or the entire HumBugDB collection. Original uncut field recordings and corresponding negative/background recordings were not retrieved.

| Released species label | Clips audited | Distinct original `name` values | Recording dates | Actual audio seconds | Clip-duration range |
|---|---:|---:|---:|---:|---|
| `ae aegypti` | 89 | 49 | 4 | 1,322.440 | 2.50–61.38 s |
| `an arabiensis` | 831 | 514 | 40 | 14,815.196 | 2.50–117.76 s |
| `an funestus ss` | 381 | 248 | 30 | 7,414.238 | 2.50–81.86 s |

**Every inspected WAV is 44,100 Hz, mono, PCM24.** The maximum duration disagreement with the CSV is one 44.1 kHz sample (approximately 22.68 microseconds), consistent with rounding/truncation of segment endpoints. All clips exceed a one-second analysis context. Preserve these native files; the audit's temporary 8 kHz spectral calculations are QC only and have not replaced the source audio.

The metadata distinguishes `an funestus ss` from `an funestus sl`; only the former is selected. The [paper's collection methods](https://arxiv.org/html/2110.07607#S4.SS1.SSS2) describe PCR resolution for members of the relevant Anopheles complexes and retention of genus/complex labels when species resolution is unavailable. That supports the supplied label interpretation; this audit does not independently re-identify specimens from acoustics or inspect laboratory PCR records.

## Independent provenance and leakage

The [author's species-classification documentation](https://github.com/HumBug-Mosquito/HumBugDB/blob/50656758594982480f568598874f79c222432e01/docs/mosquito_species_classification.md) specifically identifies the Tanzania cup recording subset as unique mosquitoes per original recording. The loader groups on CSV `name` and then reads individual annotation clips by `<id>.wav`. Therefore **`id` is a clip identifier; `name` is the documented individual/original-recording proxy**. There is no separate released mosquito UUID. This provenance is substantially stronger than inferring animal identities from Tiny MosquitoSong cut prefixes.

Across the three-species cohort, species, date, sex, feeding status, capture method and plurality are consistent within each original name. No same-name rows outside the selected three-species metadata cohort were found, and no original names have conflicting nonempty species labels in the full CSV. Terminal original-name tokens are unique among these originals; they are not interpreted as extra undocumented metadata.

The finest supported independent animal grouping is the original name. To guard against shared recording conditions, Core-v1 additionally groups **all selected originals from the same site/cup recording date together**. `record_datetime` is only date-level here: all times read 00:00. It is a conservative session proxy, not proof of one session per date or independent recording apparatus across dates. We do not invent session IDs from `IFA_*` filename fields. Multiple original-name prefixes occur on some dates, so using dates merges rather than splits those possible batches.

The audio comprises pre-extracted annotations. The CSV lacks original cut start/end offsets, so temporal non-overlap of every clip pair cannot be reconstructed. Keeping all clips from each original together prevents that uncertainty from creating cross-partition source leakage.

### Duplicate and overlap checks

- No exact whole-file or decoded-PCM duplicates among the 1,301 clips.
- A multi-pattern exact-PCM screen searched three non-silent 100 ms anchors per clip (start, middle, end) against every complete clip at arbitrary sample-aligned positions: **3,903 anchors, no matching cross-file anchor pairs**. The first 32 samples nominate candidates; all 100 ms must match to count. See [screen results](humbug_audit/overlap_summary.json).
- The final manifests explicitly verify that original names, date groups and PCM hashes are disjoint across training, validation and testing.

The screen is bounded: it cannot rule out overlap that misses the anchors, recordings altered by gain/noise, different microphone views, or undocumented repeat animals. Absence of an exact PCM match is not a biological independence guarantee. The documented individual provenance and conservative date grouping carry the main leakage protection.

## Confounding and cohort restrictions

All three proposed species share recorded site `Ifakara`, `cup`, microphone type `telinga`, device type `tascam`, 44.1 kHz and mono PCM24. Thus there is no device-*type* separation by species in the selected metadata. Serial number, gain settings, microphone distance, temperature/humidity and exact session timing are not present in this CSV; same device type does not guarantee identical conditions.

The three-species cohort has the following substantial imbalances:

- Aegypti: all 89 rows female, unfed and LC, across four dates.
- Arabiensis: 820 female and 11 male rows; 528 LT and 303 HBN rows; both fed and unfed material.
- Funestus s.s.: all female; 305 LT and 76 HBN rows; fed/unfed and one missing feeding-status row.

Core-v1 controls these observed factors by selecting Female / unfed / LT / Single for both Anopheles labels. This leaves 324 arabiensis and 198 funestus originals before date restriction. Keeping only the **20 shared dates** leaves 300 and 144. The large funestus-only date, 23 August 2020, is excluded together with arabiensis-only dates. This reduces species/date confounding without consulting acoustic outcomes.

Exclusions are explicit for all clips: 89 aegypti clips; 421 outside the controlled sex/feeding/capture cohort; 123 from dates with only one eligible species. No clips were excluded because they sounded noisy, had an inconvenient wingbeat frequency, or would be difficult to classify.

Residual confounding remains: class proportions vary by date, season and unmeasured conditions may vary, and all audio comes from a single collection setting. Core-v1 measures performance on held-out recording dates within that setting. It cannot establish broad device/site robustness, male-mosquito performance, mosquito detection against noise, or performance on the proposal's original four species.

## Audio quality and wingbeat activity

Every clip was decoded for integrity and basic QC before the final freeze. No digital-silence files were found; no samples reached the audit's clipping threshold of absolute normalized amplitude ≥0.999. This does not exclude prior analog saturation or environmental interference. RMS ranges are broad: approximately −65.0 to −32.5 dBFS for aegypti, −67.2 to −24.4 for arabiensis, and −70.6 to −28.1 for funestus. These are level measurements, not SNR estimates.

The QC script temporarily resamples to 8 kHz, computes 64 ms spectra and measures peak-to-median spectral contrast in 150–1,200 Hz. Median fractions of frames exceeding 10 dB contrast are about 0.994–0.995 per species. **These fractions are not mosquito-activity percentages**: stationary equipment tones can also pass this unvalidated proxy. No denoising, quality threshold or frequency rule was applied to select the core.

Nine development-only examples were visually inspected using [representative spectrograms](humbug_audit/representative_spectrograms.png): the 10th/50th/90th percentiles of the contrast proxy per species, showing the first up to eight seconds. The six core examples are all assigned to Training; the three aegypti examples were in the earlier candidate's development split and are excluded from Core-v1. Exact IDs and selection rules are in [representative_audio.csv](humbug_audit/representative_audio.csv).

Observed examples contain drifting fundamental/harmonic ridges around several hundred hertz with higher harmonics, compatible with the released wingbeat annotations. For example, arabiensis 221900 shows bands near 400/800 Hz amid broadband noise, and aegypti 220320 shows a band near 550 Hz and a harmonic near 1.1 kHz. Funestus 220485 shows intermittent bands near 500/1,000 Hz. Low-frequency energy and transient vertical streaks occur throughout. A stationary tone near 750 Hz is visible in both arabiensis 221425 and funestus 221539, illustrating why spectral tonality alone is not proof of flight. The images support usable but variable-quality acoustic data; they do not independently verify species or continuous flight over every annotated clip. This was waveform/spectral inspection, **not a claimed human listening or entomological validation exercise**.

## Split construction and freeze

`scripts/prepare_humbug_split.py` applies metadata restrictions, identifies the 20 shared dates, and searches 20,000 deterministic date allocations using seed 20260926. The objective minimizes squared deviation from 60/20/20 **original-count proportions per species**, while fixing the date counts to 12/4/4. No waveform, QC metric, feature vector or model score enters this search. Unequal clip durations are retained rather than forcing equal numbers of correlated windows.

Training dates (2020): 24 March; 17, 21, 22, 23 June; 6, 7, 12, 20, 21, 27 July; 2 August.

Validation dates: 26 March; 16 June; 26 and 28 July.

Test dates: 29 June; 13, 14 and 19 July.

Approximate train/validation/test audio duration is 8,756.08 / 2,707.74 / 3,028.76 seconds. Membership is fixed at source level in [core_source_manifest.csv](humbug_audit/core_source_manifest.csv), with [original-group counts](humbug_audit/core_group_manifest.csv) and [split design](humbug_audit/split_design.json).

The earlier three-species metadata candidate remains preserved for audit history and is explicitly superseded. It was a HOLD proposal, not an approved final test set. Its revision was driven by the newly audited collection-method/sex/date structure before training. Tiny MosquitoSong's frozen candidate is unchanged. From the Core-v1 freeze onward, do not listen to, visualize, tune preprocessing on, or run predictions against test members until final evaluation. Deterministic preprocessing and integrity checks may preserve their assigned membership.

## Comparison with Tiny MosquitoSong

| Criterion | Tiny MosquitoSong | Recommended HumBugDB Core-v1 |
|---|---|---|
| Provenance | Four inferred filename families per species; individual/session identity unknown | 444 original names backed by subset-specific individual provenance; 20 coarser date groups |
| Main split | 2/1/1 inferred families per species | 12/4/4 date groups, both species in every date |
| Species | Original four desired species available | Two controlled Anopheles species; changes project scope |
| Released audio | 8 kHz mono PCM16, many cuts shorter than 1 s | 44.1 kHz mono PCM24; all audited clips ≥2.5 s |
| Known duplication | Two exact duplicate pairs and one additional exact-overlap pair | No whole-file/PCM duplicates or anchor matches detected |
| Observed confounds | Age/sex/microphone families with insufficient source metadata | Known sex, feeding, collection method and date controlled; unmeasured conditions remain |
| Suitable claim | Supplementary feasibility only with current provenance | Limited within-setting species classification on held-out recording dates |

HumBugDB Core-v1 is the more defensible main scientific dataset. Its test set still contains only **four date clusters**, even though it contains 88 originals and 120 clips. Report per-species and per-date performance; do not treat thousands of windows as independent replicates. Aggregate windows to original recording before headline species metrics, and acknowledge that uncertainty over new dates is weakly estimated from four dates. More independent acquisition dates/sites would strengthen a later confirmatory experiment.

## Conditions before the next training run

1. Train only this **two-species Core-v1 cohort**, with the fixed source/date memberships above. A three-species result involving aegypti may be a separately labelled confounded exploratory study, not the main evaluation.
2. Use explicit frozen validation in Edge Impulse; disable automatic sample/window-level repartitioning. Keep Testing local until model selection is locked. Upload labels explicitly and carry both original-name and date-group metadata.
3. For a Core baseline adapted from the reference workflow, document the dataset/label change, a two-output head and a fixed 44.1→16 kHz anti-aliased conversion. Preserve native PCM24 files; conversion to PCM16 is an explicit implementation choice. Retain the reference Spectrogram rather than silently substituting log-Mel. No such conversion, model implementation or training has been performed in this audit. Verify the preprocessing on a Training fixture before training; do not use test clips to choose it.
4. Prespecify original-level balanced accuracy/macro-F1, per-class recall and per-date results; use Validation only for stopping/selection. Do not select another split after seeing results. Restrict conclusions to the defined female/unfed/LT/Ifakara-cup population and record class imbalance.

**CONDITIONAL GO:** the next eligible training experiment is a **two-class Core baseline on frozen HumBugDB Core-v1**, arabiensis versus funestus s.s., using the **12/4/4 date split and 267/89/88 original-recording split** above, subject to these implementation and reporting conditions. The historical four-label reference experiment remains NO-GO under its unchanged three-way source-split requirement. Stop here: no model has been trained.

## Role after the four-class reassessment

This file remains the frozen audit for the **secondary controlled two-class scientific check**. Its `CONDITIONAL GO` applies to the cohort and split described here, but the project-level training order is now governed by [`CLASSIFICATION_SETUP_REASSESSMENT.md`](CLASSIFICATION_SETUP_REASSESSMENT.md): complete and freeze the four-class practical audit first, train that four-class track first, then run this two-class benchmark without changing its test membership.

Current cross-experiment finding: the four-class audit is complete and approved for training. It preserves its existing split, which places 47 original groups from this two-class test set in four-class Training/Validation. See HUMBUG_FOUR_CLASS_FINAL_AUDIT.md and humbug_four_audit/secondary_test_cross_experiment_overlap.csv. This secondary experiment must not be presented as independent confirmation after those sources influence four-class development; do not transfer four-class trained weights or fitted preprocessing into its model. This audit's frozen membership is unchanged.
