# Reorganization report

Completed 2026-09-28. Organization and path changes only; no scientific pipeline execution, model training, 03B implementation, feature changes, or new scientific work.

## Final repository tree

```text
Edge AI Project/
├── README.md
├── .vscode/settings.json
├── notebooks/current/
│   ├── 00_prepare_dataset.ipynb
│   ├── 01_dataset_audit.ipynb
│   ├── 02_segmentation.ipynb
│   └── 03A_logmel_reference.ipynb
├── data/
│   ├── audio/                       # 2,694 original WAVs, unchanged
│   ├── metadata/                    # release metadata
│   ├── manifests/current/           # Core, frozen split, Example Grid v2
│   └── audits/current/              # canonical audio and grid audits
├── reports/
│   ├── pdf/                         # existing 00/01/02 exports
│   └── reproduced_00/               # reproduced copies and split recipe
├── docs/
│   ├── PIPELINE.md
│   ├── DATA_PROVENANCE.md
│   ├── REORGANIZATION_REPORT.md
│   ├── proposals/                   # original revised proposal PDF
│   └── reorganization/              # hashes, moves, diff, verification, full tree
├── reference/
│   ├── README.md                    # reference index and missing parity notice
│   └── tinyml/                      # comparison notebooks and paper
├── third_party/edgeimpulse-processing-blocks/  # all original files unchanged
├── tools/
│   ├── maintenance/                 # verifier and guarded one-time cleanup scripts
│   └── review_support/              # self-contained utilities/evidence/runtime bundle
└── archive/
    ├── README.md
    ├── baseline_readiness/          # intact historical experiment/provenance bundle
    ├── legacy_code/code_by_me/      # old step-1 scripts and segmentation notebook
    ├── legacy_data/                 # old windows and train diagnostics
    ├── reviews/                     # historical literature/methodology reviews
    └── reorganization_originals/    # exact originals of edited existing files
```

The complete file-level tree is [final_files.txt](reorganization/final_files.txt). Large audio, vendored-library and historical source subtrees are collapsed above.

## Moves and archives

The exact old-to-new mapping is [moves.json](reorganization/moves.json), including whole-directory moves.

- Root proposal PDF → `docs/proposals/`; TinyML paper → `reference/tinyml/`.
- Root literature and methodology review notes → `archive/reviews/`; runtime-inspection R script → `tools/review_support/`.
- `review_support/` → `tools/review_support/`, keeping sibling dependencies and evidence together.
- `code_by_me/` → `archive/legacy_code/code_by_me/`.
- Original release metadata → `data/metadata/`; current manifests → `data/manifests/current/`; canonical audits → `data/audits/current/`.
- `data/_reproduced_00/` → `reports/reproduced_00/`.
- Old 0.96-second and 1-second window manifests and all five `train_*.csv` diagnostics → `archive/legacy_data/`.
- Current notebooks remain in place. Their edits only change path discovery, locations and the input filename-to-location mapping. Notebook outputs and execution counts were preserved. See [notebook_paths.diff](reorganization/notebook_paths.diff).
- The original notebooks, README and three edited local review utilities are backed up in `archive/reorganization_originals/`. Review utilities now resolve sibling paths and accept external PDF locations as arguments.

## Deleted

Only `build/`: 252 files, 42,399,353 bytes comprising the inspected TeX smoke test (`Build check`), PDF/log and disposable Tectonic cache. No current pipeline dependency referenced this build directory. Empty organizational directories were pruned. No raw audio, scientific artifact, source snapshot, duplicate evidence or third-party file was deleted. Exact deleted file names and hashes are the `build/` entries in [before.json](reorganization/before.json).

## Exact current paths

Pipeline:

- `notebooks/current/00_prepare_dataset.ipynb`
- `notebooks/current/01_dataset_audit.ipynb`
- `notebooks/current/02_segmentation.ipynb`
- `notebooks/current/03A_logmel_reference.ipynb`

Data and artifacts:

- `data/audio/`
- `data/metadata/neurips_2021_zenodo_0_0_1.csv`
- `data/manifests/current/core_single_4species_metadata.csv`
- `data/manifests/current/core_single_4species_frozen_split.csv`
- `data/manifests/current/core_example_grid_v2_stride15360_ref15600_candidates.csv`
- `data/audits/current/core_audio_audit.csv`
- `data/audits/current/core_example_grid_v2_stride15360_ref15600_audit.json`
- `reports/reproduced_00/core_single_4species_metadata.csv`
- `reports/reproduced_00/core_single_4species_frozen_split.csv`
- `reports/reproduced_00/core_audio_audit.csv`
- `reports/reproduced_00/legacy_split_recipe.json`

Notebook 02 still requires `archive/legacy_data/core_096s_candidate_windows.csv` as an authenticated historical comparison. The intended 03A audit path is `data/audits/current/03A_logmel_reference_frontend_audit.json`; this file does not yet exist and was not generated during cleanup.

## References and 03A status

Recursive filename search found **none** of `mel_features.py`, `vggish_input.py`, `vggish_params.py`, or `feat_vggish.py`, including prefixed filename matches. **The pinned HumBugDB/VGGish upstream parity reference is missing.** No replacement was downloaded or invented; 03A numerical parity remains unverified.

Related retained references include:

- `archive/baseline_readiness/sources/humbug_feat_util.py`
- `archive/baseline_readiness/sources/humbug_species_classification.ipynb`
- `archive/baseline_readiness/sources/humbug_species_classification.txt`
- `archive/baseline_readiness/sources/humbug_mosquito_species_classification.md`
- `archive/baseline_readiness/sources/humbug_tree.json`
- `reference/tinyml/tinyml_data_prep_source.ipynb`
- `reference/tinyml/tinyml_classifier_source.ipynb`
- `reference/tinyml/tinyml_primary_source.pdf`
- `third_party/edgeimpulse-processing-blocks/`

See [reference index](../reference/README.md) for context. These related files are not substitutes for the missing parity implementation.

## Verification

[verification.json](reorganization/verification.json) records the results. [before.json](reorganization/before.json) records pre-move SHA-256 hashes; cleanup tooling created during the inventory is excluded from the preservation count.

- 12,837 original retained files verified at their mapped paths; edited original files have byte-identical backups. All third-party/reference file contents, scientific artifacts and all 5,636 audio files across the repository are preserved.
- Core: **1,898 clips**; sources: **1,189**.
- Train / validation / test: **1,335 / 269 / 294**; **zero source leakage**.
- Example Grid v2: **32,645 unique examples**; **15,360-sample stride**; **15,600-sample reference context**. Every grid row retains its parent source, species and split; start/end geometry checks pass.
- Existing grid manifest matches its audit SHA-256. All **1,898 Core WAV hashes** independently match the original grid audit.
- All five pinned notebook-02 input hashes still match. Canonical CSV/JSON bytes, including labels and assignments, are unchanged.
- All code cells parse; all four notebooks resolve their configured data paths from both the root and `notebooks/current/`. Only safe path declarations were evaluated. Notebook outputs and execution counts match originals.
- No complete notebook execution, PDF regeneration, resampling, feature extraction or training was performed. Package availability and full runtime execution are not certified by this cleanup. Missing upstream parity remains a known 03A blocker.

## Absolute paths and deliberately retained uncertainty

[absolute_path_scan.json](reorganization/absolute_path_scan.json) records remaining path-like strings in source/docs/config (not binary files or saved notebook outputs). No operational machine-specific paths remain in current notebook source or the edited local review utilities. Notebook 03A contains only an illustrative Windows-path docstring; the migration script contains the old path as a replacement match.

Historical scripts, original backups, saved outputs/PDFs, vendored sources and historical records retain old machine paths for provenance. Archived scripts are not current runnable entry points; intentional historical reproduction may require adapting paths with the move map. Third-party/reference code was not rewritten.

Kept deliberately: all non-Core audio; historical ZIP archives/raw recordings; the intact baseline-readiness bundle; potentially duplicate source snapshots; legacy scripts and diagnostics; reproduced copies with their different audit schema; bundled review PDF libraries/JAR and extracted evidence; original notebook outputs and PDFs. Their provenance/dependency value outweighs space savings, and no unproven duplicate deletion was attempted.
