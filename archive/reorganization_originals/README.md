# Edge AI Mosquito Acoustic Research

## Current pipeline

1. [Prepare and verify the dataset](notebooks/current/00_prepare_dataset.ipynb)
2. [Audit the frozen dataset](notebooks/current/01_dataset_audit.ipynb)
3. [Generate the shared example grid](notebooks/current/02_segmentation.ipynb)
4. Future: reference Log-Mel (03A) and Edge MFE (03B), followed by models and deployment.

## Current scientific artifacts

- Frozen HumBugDB Core: 1,898 clips, 1,189 source recordings, four species.
- Frozen source grouping and train/validation/test assignments: `data/core_single_4species_frozen_split.csv`.
- Current candidate grid: `data/core_example_grid_v2_stride15360_ref15600_candidates.csv`.
- Current grid audit: `data/core_example_grid_v2_stride15360_ref15600_audit.json`.
- Raw audio remains in `data/audio/`; raw metadata remains in `data/`.

The current grid uses a 15,360-sample start stride and 15,600-sample reference context. The reference frontend is still proposed; this manifest is a verified candidate, not a model input contract.

## Status boundaries

- `data/` contains the current data foundation and generated outputs. Do not overwrite frozen CSVs silently.
- `reference/` contains external or comparison material.
- `code_by_me/step_1/` contains legacy provenance scripts; they are not the current pipeline.
- `archive/` contains earlier experiment definitions and their outputs.
- `third_party/` contains pinned external implementation code.

Do not use files from `archive/` or legacy scripts as current training inputs unless the methodology explicitly says so.

See [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md), [PIPELINE.md](docs/PIPELINE.md), [METHODOLOGY_STATUS.md](docs/METHODOLOGY_STATUS.md), and [DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md).
