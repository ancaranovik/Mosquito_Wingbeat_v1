# Four-class HumBugDB audit status

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

Completed 26 September 2026. This is a **metadata and partition audit only**. It does not authorize four-class training.

## Metadata verified locally

The pinned HumBugDB release 0.0.1 CSV was filtered to Tanzania, Ifakara, cup, mosquito records for:

- `ae aegypti`: 89 clips, 49 original `name` groups, 4 dates, 1,322.44 metadata seconds.
- `an arabiensis`: 831 clips, 514 groups, 40 dates, 14,815.20 seconds.
- `culex pipiens complex`: 545 clips, 336 groups, 36 dates, 8,157.76 seconds.
- `ma uniformis`: 131 clips, 76 groups, 14 dates, 1,654.58 seconds.

Total: **1,596 metadata rows, 975 original-name groups, approximately 7.21 hours**.

No original `name` occurs under more than one of the four labels. The metadata contains no separate animal UUID, session UUID, microphone serial, gain, temperature or humidity field. The strongest available source grouping remains original `name`; later windows must inherit that membership.

The source labels are retained exactly. `culex pipiens complex` is a species-complex label. `ma uniformis` is the exact HumBugDB source label and is preferred over the smaller, unresolved `ma africanus` candidate.

## Candidate split

A deterministic SHA-256 ordering using seed string `four-core-20260926|species|name` produced this metadata-only candidate:

| Class | Training groups | Validation groups | Testing groups |
|---|---:|---:|---:|
| `ae aegypti` | 29 | 10 | 10 |
| `an arabiensis` | 308 | 103 | 103 |
| `culex pipiens complex` | 202 | 67 | 67 |
| `ma uniformis` | 46 | 15 | 15 |

The candidate memberships are in [four_class_split_candidate_metadata.csv](four_class_split_candidate_metadata.csv) and the group-level summary is in [four_class_group_candidate.csv](four_class_group_candidate.csv). Each `name` is assigned to one partition only. These files are explicitly **candidate/HOLD** artifacts; they are not the final frozen manifest.

## What could and could not be audited

The existing local HumBug audio audit covers the 1,301 Aedes/arabiensis/funestus files, but not the 545 Culex or 131 Mansonia files required by the practical Core. Therefore the following four-class checks remain incomplete:

- WAV header, sample rate, channel count, subtype and actual duration;
- ZIP/member integrity and decoded-PCM hashes;
- exact duplicate and bounded overlap screening;
- representative waveform/spectrogram inspection and wingbeat-activity QC;
- confirmation that every selected metadata row has a matching audio member;
- final exclusion reasons and audio-backed class balance;
- final frozen train/validation/test manifest with hashes.

The metadata-only checks found no duplicate source IDs and no cross-species original-name collisions. These do not substitute for audio-content duplicate/overlap checks.

## Training decision

**NO-GO for four-class training at this time.** The experiment remains scientifically reasonable, and the metadata candidate remains the recommended design, but the required actual-audio audit is incomplete. Once the missing Culex and Mansonia WAVs are retrieved and pass the same integrity, QC, overlap and source-group checks used for Core-v1, replace the candidate files with a hash-backed frozen manifest and reassess as **CONDITIONAL GO** or **GO**.

The historical four-label TinyML reference and the secondary two-class Core-v1 are separate experiments. No model, window extraction, augmentation, synthetic swarm, quantization or ESP32 work was performed here.



