# Baseline readiness report

## Current decision: GO for four-class training

The completed local WAV audit supersedes all earlier missing-file/NO-GO statements below. All 1,596 expected clips are present, including all 297 previously missing files. All decode as 44.1 kHz mono PCM24; there are no exact PCM duplicates or matches in the bounded 4,788-anchor overlap screen. Zero files were excluded. The unchanged candidate is now frozen as `humbug-four-class-v1`: Training 952 clips/585 groups, Validation 330/195, Testing 314/195. Group key remains `name` and release remains 0.0.1.

Read [HUMBUG_FOUR_CLASS_FINAL_AUDIT.md](HUMBUG_FOUR_CLASS_FINAL_AUDIT.md) for the final class counts, quality inspection, exact preprocessing specification, manifests and limitations. In particular, 47 secondary-benchmark test groups overlap four-class development sources, so that secondary result is a controlled comparison, not independent confirmation. No model has been trained.

## Preserved earlier audit history

## Current local extraction check

The pinned metadata CSV was found at `sources/humbug_neurips_2021_zenodo_0_0_1.csv`. Metadata clip `id` maps directly to `../data/audio/<id>.wav`; original `name` remains the grouping key, not the released clip filename. Raw files were not reorganized.

The supplied `data/audio` directory contains 1,998 WAVs and no nested directories. Of 1,596 required Tanzania cup four-class clips, 1,299 are present and 297 are missing:

| Source label | Expected | Present | Missing |
|---|---:|---:|---:|
| ae aegypti | 89 | 89 | 0 |
| an arabiensis | 831 | 598 | 233 |
| culex pipiens complex | 545 | 494 | 51 |
| ma uniformis | 131 | 118 | 13 |

The directory's last numeric file is `222020.wav`. The cached official selected-member archive index places all 233 missing arabiensis clips in `humbugdb_neurips_2021_4.zip`. Culex/Mansonia archive membership has not yet been independently indexed. This strongly suggests incomplete extraction of archive 4, but archive integrity is not established from directory names alone. No ZIP archives were found under `data`.

Exact evidence: [mapping](humbug_four_audit/local_mapping.csv), [missing files](humbug_four_audit/missing_local_audio.csv), [counts](humbug_four_audit/local_mapping_summary.csv). The previous cache contains the 233 arabiensis clips but none of the 64 missing Culex/Mansonia clips; no silent pooling or copying was performed.

**NO-GO for four-class training; final freeze deferred.** Complete the official release extraction into `data/audio`, particularly archive 4, then rerun coverage and the WAV-level audit. No audio QC, archive-integrity, overlap or final-freeze claim is made by this mapping check. The existing candidate split is unchanged and no model was trained.

## Earlier audit history (superseded where noted above)

## Current decision â€” classification setup reassessment

**NO-GO for four-class training at this time: the added Culex and Mansonia WAV audit is incomplete because those members are not locally available.** The practical four-class HumBugDB design remains recommended after retrieval and a hash-backed manifest freeze. The two-species HumBugDB Core-v1 is retained as the secondary scientific check; NO-GO also remains for the original four-label reference baseline. Read the [four-class audit status](FOUR_CLASS_AUDIO_AUDIT_STATUS.md), [HumBugDB Core audit](HUMBUG_CORE_AUDIT.md) and [classification setup reassessment](CLASSIFICATION_SETUP_REASSESSMENT.md). No training has been performed.

All 1,301 Tanzania cup clips for the proposed three species were retrieved and decoded: 44.1 kHz mono PCM24, no exact PCM duplicates or matching 100 ms anchors detected. Aegypti is excluded from the **secondary controlled check** because its capture method is completely confounded with species and only four dates are available. The selected controlled cohort is **female, unfed, individually recorded, light-trap-collected Anopheles arabiensis and Anopheles funestus s.s.**, restricted to the 20 dates containing both species. The practical four-class Core recommendation adds metadata-supported `ae aegypti`, `culex pipiens complex`, and `ma uniformis`, but those added classes still require their own audio audit.

The frozen split is **12/4/4 recording dates**, containing **267/89/88 original-recording groups** and **419/129/120 clips**, for Training/Validation/Testing. Arabiensis originals are 179/61/60; funestus s.s. originals are 88/28/28. Use the [final source manifest](humbug_audit/core_source_manifest.csv) and [freeze record](humbug_audit/core_freeze.json), not the superseded three-species metadata-only candidate.

The next eligible experiment remains the **four-class practical HumBugDB track**, but it is currently blocked until its Culex/Mansonia audio, taxonomy, duplicate/overlap and source-group freeze is complete. The metadata-only candidate uses `ae aegypti`, `an arabiensis`, `culex pipiens complex`, and `ma uniformis`; this is an exploratory genus-spanning, domain-confounded feasibility benchmark. The frozen two-class HumBugDB Core-v1 remains the secondary scientific check. The original reference task remains separately blocked; either Core result is not an exact reproduction of that task.

## Previous baseline audit â€” preserved history

Sections 1â€“7 below record the earlier baseline-readiness milestone. Their statements that HumBugDB audio was not downloaded and no Core experiment was eligible describe the earlier state and are superseded by the current audit above. Tiny MosquitoSong's audit and frozen candidate, and the reference-baseline blocker, remain unchanged.

Completed 26 September 2026. **NO-GO for training under the requested source-disjoint train/validation/test protocol.** The data audit and configuration reconstruction are complete; missing provenance and insufficient independent sources cannot be repaired by producing more windows. No model was trained, no synthetic data generated, no cloud upload performed, and no hardware work started.

## 1. Two distinct experiments

**Reference baseline:** retain the original `aegypti / albopictus / other / noise` task using the archives distributed with [the TinyML repository](https://github.com/Mjrovai/wingbeat-mosquito-tinyml). Use the documented Spectrogram/Conv1D workflow in [BASELINE_CONFIGURATION.md](BASELINE_CONFIGURATION.md). A source-disjoint evaluation and a current DSP implementation are declared reimplementation differences. An exact numerical reproduction is not established.

**Project core classifier:** later classify the practical four-class HumBugDB labels `ae aegypti`, `an arabiensis`, `culex pipiens complex`, and `ma uniformis` across four genera. This is an exploratory source-group/domain-confounded benchmark and is not a reproduction of the published TinyML task. The controlled two-class `an arabiensis` versus `an funestus ss` cohort is retained as a secondary scientific check. No core model configuration or training is implemented here.

The revised proposal was read from the preserved source in `sources/Acoustic_Mosquito_Edge_AI_Proposal_Revised.tex`. Its later log-Mel frontend does not replace the reference Spectrogram. Dataset recommendations below follow provenance and available labels, never final-test performance.

## 2. Verified Tiny MosquitoSong inventory

All 1,324 mosquito WAVs and 12 noise WAVs in the linked public Drive release were downloaded and decoded. Source IDs, paths, SHA-256 hashes, headers, durations, labels and exclusions are in [split_manifest.csv](manifests/split_manifest.csv). Counts below are cut files, not independent mosquitoes or original recordings. Durations include related microphone material and duplicates; they are not unique flight time.

| Species | Files | Total seconds | Duration range, seconds | Female / male files | Files at least 1 s | Recoverable families | Verified individuals/sessions |
|---|---:|---:|---|---:|---:|---:|---|
| Aedes aegypti | 362 | 601.181 | 0.562â€“10.319 | 166 / 196 | 269 | 4 | Unknown |
| Aedes albopictus | 400 | 697.945 | 0.441â€“8.336 | 182 / 218 | 303 | 4 | Unknown |
| Anopheles dirus | 330 | 426.148 | 0.405â€“22.827 | 152 / 178 | 172 | 4 | Unknown |
| Culex quinquefasciatus | 232 | 896.724 | 0.320â€“57.444 | 120 / 112 | 173 | 4 | Unknown |

All mosquito files are **8,000 Hz, mono, PCM16 WAV**. All 12 noise files are **8,000 Hz, stereo, PCM16 WAV**. Noise comprises eight environmental files and four trap-fan files; do not count stereo channels or alternate trap microphone positions as independent recordings. They are excluded from the candidate species-only task. Individual file durations are recorded in the manifest.

The released mosquito files are variable-duration cuts of wingbeat-containing periods, not a uniform bank of 300 ms windows. The MosquitoSong+ notebook subsequently constructs overlapping 300 ms epochs with 150 ms overlap. There are 407 cuts shorter than the reference one-second context. They are flagged rather than padded, repeated or concatenated.

### Provenance re-investigation

Inspected the repository tree and prediction notebook at commit `0e9187c454d7ce720cbb4925664554d1d44797c5`, public Drive folder structure, all filenames, and every WAV's RIFF chunks. The repository contains placeholder data/model directories and no individual/session mapping. The downloaded folders contain no metadata sidecars. All 1,336 WAVs have only `fmt ` and `data` chunks: no embedded recording identity, timestamps, BWF or iXML metadata was recovered. The user confirms no additional mapping is available.

The strongest recoverable grouping normalizes species, sex, day token, temperature and apparatus prefix, removes cut numbering and microphone suffix, and merges stray whitespace/case variants. `LowMic` and unmarked recordings stay together. Neither `1F` nor `1M` is treated as a unique mosquito identifier. Day tokens appear to encode age; they are not session dates. Original uninterrupted recordings, cut time offsets, exact microphone synchrony and whether the same mosquito appears at different ages remain unknown.

Each species has F/M crossed with two age tokens: aegypti 5D/21D; albopictus 7D/14D; dirus 5D/14D; quinquefasciatus 6D/14D. Thus there are 16 inferred families total. It is not defensible to turn 1,324 cuts into 1,324 independent samples. The repository's window-based folds do not recover missing provenance and are not adopted.

### Duplicates and overlap

Two exact duplicate pairs were found, both within the dirus male-14D family: Cut56/Cut57 and LowMic Cut17/Cut18. `duplicates.csv` contains eight evidence rows because each pair is listed under both file and PCM hashes; this means **four involved files, two redundant copies**, not eight duplicate files. [duplicate_exclusions.csv](manifests/duplicate_exclusions.csv) chooses one canonical member per pair without changing partitions.

The exact-overlap screen also found an 800-sample (100 ms) matching segment between albopictus male-7D cuts 60 and 64, within one family. The screen used selected active 100 ms PCM anchors; no cross-family match was detected by that bounded screen. This does **not** rule out shorter overlaps, gain changes, different microphone views, or repeated recordings of the same mosquito. Do not interpret the result as proof of independence.

## 3. Frozen candidate partition and scientific adequacy

[group_manifest.csv](manifests/group_manifest.csv) and the source manifest preserve a deterministic family assignment (seed string `20260923`): two training families, one validation family, one testing family per species. A training family of each sex is retained; all cuts/microphone variants inherit their family's membership. No windows or augmentation preceded assignment. SHA-256 checks are in [freeze.json](manifests/freeze.json); its date denotes the original initiation/seed date, while the full audit completed on 26 September. The assignments have not been changed after freezing.

| Species | Train families/files | Validation families/files | Test families/files |
|---|---:|---:|---:|
| Aedes aegypti | 2 / 200 | 1 / 88 | 1 / 74 |
| Aedes albopictus | 2 / 241 | 1 / 70 | 1 / 89 |
| Anopheles dirus | 2 / 170 | 1 / 109 | 1 / 51 |
| Culex quinquefasciatus | 2 / 167 | 1 / 24 | 1 / 41 |

This is **50/25/25 by family**, the closest nonempty three-way integer allocation to 60/20/20 with four families. It is not 60/20/20 by clips or duration. All memberships are **HOLD candidates**, not a certified independent-individual test set.

Even if these families were independent, one validation and one test family per class are inadequate for the main project evaluation: family, age, sex and microphone conditions can dominate the estimate; within-class between-source variability cannot be estimated from one held-out source; hundreds of correlated windows do not increase the independent test sample count. Here independence itself is additionally unverified. Reducing to two or three Tiny MosquitoSong species leaves the same weakness per species. Accordingly, **no reduced Tiny MosquitoSong subset is approved for the main scientific claim**. All four may remain candidates for a clearly labelled supplementary feasibility study only after its limitations and protocol are settled.

The candidate test files remain reserved. No DSP features, model selection, prediction or performance analysis was performed on them after freeze. Future duplicate discoveries must trigger a documented integrity review; never silently move test sources into development or reroll a split for better results.

## 4. Reference dataset and HumBugDB comparison

The historical archives were downloaded, every contained WAV header/hash audited, and 291 WAVs extracted under safe explicit class names in `reference_upload_HOLD/`.

| Reference class | Released files | Released duration, seconds | Recoverable filename ancestors |
|---|---:|---:|---:|
| aegypti | 10 | 692.524 | 3 |
| albopictus | 10 | 666.366 | 2 |
| other | 16 | 687.191 | 16 |
| noise | 255 | 685.745 | 252 |

All are 16 kHz mono PCM16. Ancestors are filename-derived original-source candidates, not verified individual counts; especially the noise count includes speech-background excerpts and must not be treated as 252 independent environmental sessions. The accessible manuscript reports 297 files and roughly 27 minutes of `other`; the release has 291 files and about 11.45 minutes of `other`. These discrepancies preclude pretending the downloaded release exactly matches the paper's training set. Original source-to-partition membership was not recovered. The two albopictus ancestors cannot supply nonempty source-disjoint train, validation and test sets. Recutting them cannot solve this.

**Recommendation for reference reproduction:** use these original archives and labels, not Tiny MosquitoSong or HumBugDB replacements. Recover additional original provenance/data or resolve the evaluation protocol before training. Adding new albopictus data changes the dataset; using a two-way exploratory study changes the requested protocol and is not silently approved here.

**Recommendation for the main core evaluation (superseded by the completed audio audit):** use the provenance-supported, controlled two-species HumBugDB Core-v1 cohort described in [HUMBUG_CORE_AUDIT.md](HUMBUG_CORE_AUDIT.md). The [HumBugDB paper](https://arxiv.org/html/2110.07607) and [species-classification documentation](https://github.com/HumBug-Mosquito/HumBugDB/blob/50656758594982480f568598874f79c222432e01/docs/mosquito_species_classification.md) identify the Tanzania cup subset as individual recordings. Its loader groups by original `name`, then loads annotated clips as `<id>.wav`. Multiple annotation rows therefore belong to one mosquito group. That individual interpretation is specific to this subset and must not be generalized to every HumBugDB recording.

The pinned metadata contains 9,295 rows overall; Tanzania/cup/mosquito has 2,582 annotated rows and 1,575 original names. For a manageable same-context, species-labelled core, the proposed selection is:

| Metadata species | Original-name groups | Annotation files | Candidate train / validation / test groups |
|---|---:|---:|---:|
| ae aegypti | 49 | 89 | 29 / 10 / 10 |
| an arabiensis | 514 | 831 | 308 / 103 / 103 |
| an funestus ss | 248 | 381 | 149 / 50 / 49 |

These are **metadata counts**, not a completed audio audit. Selection is based on source provenance, unambiguous species labels and same collection context, not test performance. Preserve `ss` as sensu stricto; do not merge with `an funestus sl` or call `culex pipiens complex` a single resolved species. Across the entire database, preferred-species names include aegypti 50, albopictus 1, dirus 48 and quinquefasciatus 40; those counts alone do not certify independent animals, and mixing collection sources could confound species with location/device. Therefore simply retaining the proposal's four names is not recommended.

[humbug_metadata_candidate.csv](manifests/humbug_metadata_candidate.csv) records the superseded three-species metadata proposal. The audio audit and final two-species freeze are now in [HUMBUG_CORE_AUDIT.md](HUMBUG_CORE_AUDIT.md) and `humbug_audit/`. The four-class exploratory candidate described in [CLASSIFICATION_SETUP_REASSESSMENT.md](CLASSIFICATION_SETUP_REASSESSMENT.md) still requires its own audio/taxonomy audit before use.

HumBugDB offers far more defensible independent groups than Tiny MosquitoSong for this cohort. The selected metadata lists 44.1 kHz and Tascam for all three species; aegypti/funestus rows are female, while arabiensis includes 11 male annotation rows. This remaining sex imbalance must be addressed in the final cohort specification before approving the split, not by inspecting test performance. It still supports only within-cohort individual generalization, not demonstrated robustness across new sites/devices; report group-level uncertainty and class imbalance in the eventual experiment.

## 5. Sample rates and reference configuration

There is no contradiction between Tiny MosquitoSong's high-rate acquisition and its release. The [MosquitoSong+ methods](https://doi.org/10.1371/journal.pone.0310121) describe 96 kHz/24-bit acquisition and subsequent downsampling to 8 kHz/16-bit. Actual released headers match the latter. Keep these files at 8 kHz; no upsampling was performed. Upsampling to 16 kHz for a separately declared compatibility study cannot restore information above the original 4 kHz Nyquist frequency.

The reference expects **16 kHz**, and its downloaded archives already satisfy that. The proposal's initial sample-rate suggestion is not the reason for this setting. HumBugDB metadata sample rates are not a substitute for checking its released WAV headers; its eventual frontend is separately specified after that audit.

[BASELINE_CONFIGURATION.md](BASELINE_CONFIGURATION.md) provides the complete D/R/P parameter ledger: documented original manuscript, repository-derived/inferred, or proposed. Its reference settings are 1 s context, 250 ms window increase, Spectrogram with 25 ms frame length, 12.5 ms stride, inferred 128-point FFT, âˆ’52 dB floor, expected 79Ã—65 input; two Conv1D layers (32/64 filters, kernel 3, same/ReLU), each followed by pooling, then flatten, dropout 0.5 and four-way softmax. Training is documented/recovered as Adam 0.001, batch 32, at most 100 epochs, early stopping on validation loss with patience 5. No training is run by specifying these values.

Unknowns remain: original exported project and DSP version; historical source assignments; original software versions/defaults; full tuner search; final GoodIT typeset paper's correspondence to the accessible author manuscript; exact quantization calibration/converter/EON settings. Historical deployment used INT8 TFLite/EON and Arduino libraries; this is documentation only. Exact reproduction cannot be promised. A 128-point FFT at 16 kHz has 125 Hz bin spacing; current code truncates a 400-sample frame to 128, so it is not a full 25 ms spectral estimate. Do not silently substitute a 512-point Hann or log-Mel frontend.

## 6. Data and Edge Impulse upload plan

**Files approved for upload/training now: none.** This is a concrete readiness gate, not a suggestion to use Studio's default split.

Prepared reference WAVs and explicit labels are enumerated in [reference_upload_HOLD.csv](manifests/reference_upload_HOLD.csv). All have `HOLD_UNASSIGNED`: assigning ten albopictus pieces independently would conceal the two-source problem. Tiny MosquitoSong's exact candidate memberships are [Training](manifests/training_files_HOLD.csv), [Validation](manifests/validation_files_HOLD.csv), and [Testing](manifests/testing_files_HOLD.csv); apply manifest exclusions and the duplicate overlay before any future approved use. These are supplementary species-task candidates, not reference labels. HumBugDB Core-v1 audio is now prepared and frozen under `humbug_audit/`; the earlier metadata-only candidate paths remain historical and must not replace the final Core manifests.

Once a dataset passes the gate:

1. Assign source groups externally before extracting windows. Use explicit labels; filename-based label inference can collapse dotted `Ae.*` prefixes or misread source IDs. Attach `source_id`, `group_id`, dataset version and frozen partition as metadata. Preserve all manifests and class-index mapping.
2. Upload only frozen Training members to Training and frozen Validation members to explicit Validation, with automatic random splitting disabled. Current [Studio advanced settings](https://docs.edgeimpulse.com/studio/projects/data-acquisition/dataset/advanced-settings) support an explicit validation set. If the uploader cannot express validation directly, stage only development data, move exactly the manifest's validation sources in Studio, and verify membership before feature generation. Do not allow an extra random 20% of windows.
3. If the available Studio interface cannot honor explicit validation membership, stop that workflow. Use an export/external fixed-validation workflow only after documenting the implementation difference; an external holdout is not useful for the reference's early stopping if Studio still trains against a random internal holdout. Group metadata alone is not evidence that assignments were respected.
4. Keep frozen Testing files local and untouched until the chosen model and development decisions are locked. Then upload only those members as Testing and evaluate once under the prespecified protocol. Do not use test data for DSP normalization, thresholds, early stopping, tuner search or quantization calibration.
5. Generate 1 s / 250 ms reference windows only within an approved source/partition, never across clips or group boundaries. Do not augment validation/test. Verify group-set intersections are empty and saved membership counts match the local manifest. Validate the frontend shape/numerical behavior using a development fixture, not held-out audio.

The current [uploader documentation](https://docs.edgeimpulse.com/tools/clis/edge-impulse-cli/uploader) supports explicit labels/metadata through an info file. A runnable upload command is intentionally not issued against these HOLD candidates. All 291 reference WAVs are locally prepared, so the remaining reference obstacle is scientific partitioning rather than file format.

## 7. Scientific issues and decision

The material issues are missing Tiny MosquitoSong animal/session identities; only four inferred families per species; sex/age/microphone confounding; duplicate and overlapping cuts; short-cut exclusions potentially changing class/condition balance; historical window-level validation leakage; only two reference albopictus sources; mismatched paper/archive counts and durations; unresolved historical DSP version; and the unverified four-class HumBugDB expansion. The two-class HumBugDB Core-v1 audio has now been audited and frozen for the secondary check. None is cured by more windows, a random Studio split, or better test accuracy.

The next useful work is to complete the dedicated four-class HumBugDB audit and freeze its source-group manifest, then train that practical four-class track under explicit conditions. The frozen two-class HumBugDB Core-v1 remains available as the secondary controlled check. The reference should retain its four original labels and recovered Spectrogram configuration; the reference, four-class practical Core and two-class check must keep distinct names, manifests and claims.

**NO-GO for four-class training at this time.** The intended next dataset remains the HumBugDB release 0.0.1 Tanzania/Ifakara/cup source labels `ae aegypti`, `an arabiensis`, `culex pipiens complex`, and `ma uniformis`, with the metadata candidate approximately 60/20/20 original-`name` split. Do not train until Culex and Mansonia WAVs have passed the audio integrity/QC/overlap audit and a hash-backed final manifest is frozen. Then reassess as GO or CONDITIONAL GO and train the four-class practical Core first; run the existing two-class Core-v1 Anopheles benchmark afterward as the secondary scientific check. The historical four-label reference task remains blocked by its source-partition problem. This report does not authorize custom CNN development, INT8 comparisons, synthetic swarms or ESP32 work.





