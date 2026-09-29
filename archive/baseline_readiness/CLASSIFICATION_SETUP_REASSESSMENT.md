# Final classification setup reassessment

## Current decision: GO for four-class training

The completed local WAV audit supersedes all earlier missing-file/NO-GO statements below. All 1,596 expected clips are present, including all 297 previously missing files. All decode as 44.1 kHz mono PCM24; there are no exact PCM duplicates or matches in the bounded 4,788-anchor overlap screen. Zero files were excluded. The unchanged candidate is now frozen as `humbug-four-class-v1`: Training 952 clips/585 groups, Validation 330/195, Testing 314/195. Group key remains `name` and release remains 0.0.1.

Read [HUMBUG_FOUR_CLASS_FINAL_AUDIT.md](HUMBUG_FOUR_CLASS_FINAL_AUDIT.md) for the final class counts, quality inspection, exact preprocessing specification, manifests and limitations. In particular, 47 secondary-benchmark test groups overlap four-class development sources, so that secondary result is a controlled comparison, not independent confirmation. No model has been trained.

## Preserved earlier audit history

Completed 26 September 2026. This decision is made before any model training. It uses the audited HumBugDB records, the Tiny MosquitoSong audit, the recovered TinyML reference evidence, and the evaluation practices summarized in [`literature_audit.md`](../literature_audit.md) and [`methodology_review_notes.md`](../methodology_review_notes.md).

## Recommendation

Adopt a two-track project:

1. **Main practical Core:** a four-class HumBugDB proof of concept spanning four genera. It is suitable for the Edge AI course, a later compact ESP32-S3 model, and a later synthetic multi-class swarm exercise, provided it is reported as a source/domain-confounded feasibility benchmark.
2. **Secondary scientific check:** the already audited two-class HumBugDB Core-v1 Anopheles cohort. It is the cleaner control for asking how much performance survives when sex, feeding state, capture method and recording dates are aligned.

The four-class experiment is scientifically recommended but currently **NO-GO for training** because the added Culex and Mansonia WAV members are not locally available for the required integrity, overlap and representative-audio audit. A metadata-only candidate split has been prepared, but it is not frozen. Once those files are retrieved and audited, the four-class experiment may be trained first because it is the project's stated practical Core. The two-class benchmark should then be run with its existing frozen manifests and unchanged test set.

### Exact four classes

Use these HumBugDB Tanzania/Ifakara/cup source labels, preserving the labels in every manifest:

| Project class | Exact release label | Genus / taxon interpretation | Clips | Original `name` groups | Recording dates | Known source mix |
|---|---|---|---:|---:|---:|---|
| Aedes | `ae aegypti` | *Aedes aegypti* | 89 | 49 | 4 | LC only; female; mostly/entirely unfed |
| Anopheles | `an arabiensis` | *Anopheles arabiensis* | 831 | 514 | 40 | LT and HBN; female and a small number of male rows in the full label |
| Culex | `culex pipiens complex` | *Culex pipiens* complex | 545 | 336 | 36 | HBN, LT and a few LC; complex label rather than a single resolved species |
| Mansonia | `ma uniformis` | *Mansonia uniformis* | 131 | 76 | 14 | mostly HBN, some LT; female-dominated |

These are metadata counts from the pinned HumBugDB CSV. They are not a substitute for the required four-class audio audit. The current metadata-only status and candidate files are documented in [FOUR_CLASS_AUDIO_AUDIT_STATUS.md](FOUR_CLASS_AUDIO_AUDIT_STATUS.md). The exact source spelling remains the machine label. Do not silently expand `culex pipiens complex` to a single species or rename a source label.

`ma uniformis` is preferred over `ma africanus` for the fourth class because it has more documented original groups (76 versus 38) and more clips (131 versus 78), and its source spelling is a direct species label rather than the unresolved `ma africanus` mapping. The latest project direction does not require the fourth class to be absent from Vietnam, so sample support and label clarity take priority over sentinel geography. `ma africanus` can remain a later sensitivity analysis after taxonomy verification; it is not the recommended first four-class label.

The fourth output is not an unknown class. A four-way softmax must assign every input to one of its four labels. On later Vietnamese recordings, a large Mansonia score may be logged as a domain-shift warning hypothesis, together with entropy and margin, but it is not an OOD detector and cannot establish absence, presence or correctness.

## What the four-class result may claim

Call the experiment an **exploratory source-group, genus-spanning classification proof of concept** or a **domain-confounded four-class feasibility benchmark**. The labels are species-level for Aedes, Anopheles and Mansonia, but Culex is a species-complex label. Do not call the result clean biological species discrimination or a Vietnam-ready species recognizer.

The practical question is whether a compact audio model can learn useful closed-set labels from a multi-domain mosquito collection under a transparent source split. This is a valid course objective and a reasonable engineering precursor to ESP32 deployment. The scientific limitation is that species, capture method, date distribution and other unmeasured recording conditions are correlated:

- Aedes is LC-only and spans only four dates, so capture method separates it from much of the other data.
- Anopheles and Culex contain different mixtures of LT/HBN/LC and different date ranges.
- Mansonia is smaller and mostly HBN.
- No recording date contains all four labels, so a date-balanced all-class comparison is impossible from this cohort.
- The metadata has no serial-number, gain, distance, temperature/humidity or exact session field. `name` is the documented original-recording/individual proxy for this HumBug subset, not a universal animal UUID.
- Unequal clip counts do not mean unequal independent evidence; the independent reporting unit is the original `name` group, with date as a conservative shared-condition group.

These risks mean a high score can reflect acoustics, collection method, date or device/environment cues. They are disclosed design properties, not grounds to discard the experiment under the project's practical standard. The report must include per-class, per-date and per-method results, macro-F1/balanced accuracy, original-group aggregation, and the gap between the four-class and controlled two-class tracks.

## Defensible split before window extraction

For the four-class audit, use the strongest recoverable group, original `name`, as the mandatory partition key. All clips and all later windows from one `name` stay in one partition. Use a deterministic, stratified approximately 60/20/20 split within each class, with a recorded seed and largest-remainder allocation. With the metadata counts above, the target group counts are approximately:

| Class | Train groups | Validation groups | Test groups | Total |
|---|---:|---:|---:|---:|
| Aedes aegypti | 29 | 10 | 10 | 49 |
| Anopheles arabiensis | 308 | 103 | 103 | 514 |
| Culex pipiens complex | 202 | 67 | 67 | 336 |
| Mansonia uniformis | 46 | 15 | 15 | 76 |

These are candidate counts from metadata; the final audit must recompute them after explicit audio exclusions and save a manifest. Do not extract windows, augment, or allow Edge Impulse's automatic split before the manifest is frozen. Keep test members untouched. If a source date or method has only one class, retain it in the source-group split but report that condition; do not manufacture shared dates by reassigning recordings.

The split is source/recording-group disjoint, not necessarily date-disjoint. This follows the practical literature standard in which clip, epoch, chronological or recording-based splits are used, while still preventing the most direct repeated-recording leakage. Add a diagnostic table showing date and capture-method composition in each partition. A separate date-held-out analysis may be reported only if it can be formed without deleting the small Aedes cohort; it is not the primary four-class split.

For training imbalance, use class-weighted loss or weighted sampling on Training only. Do not duplicate validation or test groups and do not oversample across the partition boundary. Report both clip counts and original-group counts.

## Secondary clean benchmark

Keep HumBugDB Core-v1 exactly as frozen in [`HUMBUG_CORE_AUDIT.md`](HUMBUG_CORE_AUDIT.md): female, unfed, single, LT, Tanzania/Ifakara/cup, only the 20 dates containing both *An. arabiensis* and *An. funestus s.s.*, with 12/4/4 date groups and 267/89/88 original groups (419/129/120 clips). Its four test dates remain untouched.

This is a controlled scientific check, not the project's main label space. It estimates within-setting discrimination under better-matched observed conditions and tests whether the practical four-class result is unusually dependent on obvious domain cues. It still has one site/device family and only four test-date clusters, so it does not prove broad species generalization.

## Alternatives considered

| Option | Scientific validity | Course / ESP32 value | Synthetic-swarm value | Main risk | Recommendation |
|---|---|---|---|---|---|
| Two-class HumBugDB only | Strongest current control, with explicit date and original-group limits | Simple and deployable, but narrower than the stated practical goal | Supports a two-species pilot only | Does not test multi-genus mixtures | Keep as the secondary check |
| HumBugDB four-class | Valid exploratory source-group benchmark under accepted confounding | Best fit for a visible Edge AI project and four-output ESP32-S3 head | Best available local label variety for later synthetic mixtures | Species/method/date/domain cues and severe class imbalance | Use as main practical Core after the four-class audit |
| Tiny MosquitoSong four-class | Public labels and 8 kHz audio, but only four inferred filename families per species and no individual/session mapping; two duplicate pairs and one overlap were found | Easy to upload but weak evidence and short cuts | Useful only as supplementary acoustic material | Pseudoreplication, overlapping cuts, age/sex/microphone confounding | Do not use as main Core |
| BioDCASE 2026 or another public corpus | Strong later domain-shift resource; BioDCASE has 9 species, 5 domains and published seen/unseen-domain metrics, but no equivalent HumBug individual identity and a much larger pipeline | Valuable later extension, too broad for the first course run | Good for transfer testing, not a direct replacement label set | Dataset/domain mismatch and scope explosion | Consider after the HumBug tracks |

The historical TinyML paper's `aegypti / albopictus / other / noise` task remains a separate reference experiment. Its recovered archive cannot support the requested independent three-way source split, and changing it to the HumBug four classes would be a new project experiment, not an exact reproduction. Keep its Spectrogram configuration and limitations in [`BASELINE_CONFIGURATION.md`](BASELINE_CONFIGURATION.md).

## Synthetic swarm and deployment implications

The four real labels are suitable for a later multi-label or mixture experiment because they span four genera and leave a compact four-output head. Synthetic mixtures must use only training-source material for training mixtures; validation and test sources remain clean and untouched. Evaluate both frame-level and source/group-level predictions, and include a background/noise or abstention analysis rather than treating one softmax label as an unknown detector.

The model can be designed for later INT8/ESP32-S3 deployment using the same compact Spectrogram-plus-small-CNN family. Deployment feasibility does not remove the data confounding: a small model can still learn capture signatures efficiently. Record parameter count, RAM/flash and latency only during the later deployment milestone.

## Final decision

**NO-GO for four-class training at this time.** The exact intended dataset remains HumBugDB release 0.0.1, Tanzania/Ifakara/cup, source labels `ae aegypti`, `an arabiensis`, `culex pipiens complex`, and `ma uniformis`, with an approximately 60/20/20 original-`name` split. Training can begin only after the Culex and Mansonia WAVs are retrieved, audited, and represented in a hash-backed frozen manifest.

Then run the frozen two-class Core-v1 Anopheles benchmark as the secondary scientific check. Do not train on Tiny MosquitoSong, do not start the historical four-label reference, do not generate synthetic swarms, and do not perform ESP32 work in this milestone.

The four-class result must be presented as exploratory and domain-confounded. A future paper is reasonable if it reports the provenance, group counts, confounds, per-date/per-method errors, group-level metrics, deployment budget and the comparison with the controlled two-class benchmark; four-class accuracy alone would not support a broad biological claim.



