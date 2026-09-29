# Data and verification provenance

Raw audio is preserved. `data/audio/` contains 2,694 WAVs; the frozen Core selects 1,898. Release metadata is `data/metadata/neurips_2021_zenodo_0_0_1.csv`. Historical archives and additional recordings remain under `archive/` and are not current training inputs.

## Accepted scientific artifacts (unchanged bytes)

- `data/manifests/current/core_single_4species_metadata.csv`
- `data/manifests/current/core_single_4species_frozen_split.csv`
- `data/manifests/current/core_example_grid_v2_stride15360_ref15600_candidates.csv`
- `data/audits/current/core_audio_audit.csv`
- `data/audits/current/core_example_grid_v2_stride15360_ref15600_audit.json`

The original grid audit contains generation-time software, input hashes, all 1,898 WAV hashes and lengths, geometry and full-context validation evidence. Its original candidate status and software record are preserved. A later validation runtime is recorded separately; historical versions are not overwritten to impersonate a new generation run.

## Current acceptance and frontend evidence

- `data/audits/current/02_example_grid_v2_freeze_audit.json`: successful acceptance gates for the existing grid; **ACCEPTED / FROZEN**. The `_candidates.csv` filename is retained and its exact hash is frozen.
- `data/audits/current/03A_logmel_reference_frontend_audit.json`: authenticated inputs/source, identical-context upstream parity, TRAIN diagnostics and final gates; **VERIFIED / FROZEN** after success.
- `data/audits/current/03B_edge_mfe_implementation_audit.json`: authenticated inputs/source, direct wrapper regression, complete TRAIN −65/−70 confirmation and selection rationale; **VERIFIED / FROZEN** after success. −52 dB is the pinned UI default; −70 dB is the project choice.
- `reports/validation/pretraining_review.json`: fresh-kernel execution evidence and preservation inventory for the approved review edits.

Expected identities are enforced by `tools/pipeline_contract.py`. Audit files are written only after successful final gates. Merely computing a new file hash is not authentication; a mismatch with an accepted pin stops the run.

Core membership, labels, source assignments, example identities/starts/eligibility and frontend mathematics remain unchanged. Both feature notebooks retain float64 waveform conversion/resampling; 02's float32 path verifies geometry. See [PIPELINE.md](PIPELINE.md) for the numerical contracts and limits of parity/generalization claims.

`reports/reproduced_00/` contains accepted reproduction copies and the split recipe. Notebook 00 verifies them without rewriting them. Canonical promotion stays disabled. The legacy `archive/legacy_data/core_096s_candidate_windows.csv` remains authenticated solely for the historical terminal-eligibility comparison in 02.

The pre-reorganization inventory, move mapping and evidence remain in `docs/reorganization/`. Original notebooks and other historical artifacts remain unchanged in `archive/`. `docs/REORGANIZATION_REPORT.md`, its verification tool/results and existing PDF exports describe an earlier state; they are not current frontend readiness claims. Current documentation supersedes their historical statements about a missing 03A reference.
