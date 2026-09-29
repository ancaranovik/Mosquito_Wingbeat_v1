# Edge AI Mosquito Acoustic Research

The current pipeline is **00 → 01 → 02 → 03A / 03B**. No classifier has been trained and notebook 04 has not been created.

| Notebook | Role and successful-gate status |
|---|---|
| [00_prepare_dataset](notebooks/current/00_prepare_dataset.ipynb) | Reproduce and verify the accepted Core and source split; canonical promotion disabled |
| [01_dataset_audit](notebooks/current/01_dataset_audit.ipynb) | Independent metadata, exact parent, source-split and WAV audit |
| [02_segmentation](notebooks/current/02_segmentation.ipynb) | Example Grid v2 — **ACCEPTED / FROZEN** after identity, physical-context and exact reconstruction gates |
| [03A_logmel_reference](notebooks/current/03A_logmel_reference.ipynb) | Log-Mel frontend — **VERIFIED / FROZEN** after upstream parity and final gates |
| [03B_edge_mfe](notebooks/current/03B_edge_mfe.ipynb) | Edge MFE frontend — **VERIFIED / FROZEN** after direct pinned-wrapper regression and full-TRAIN confirmation |

These statuses are established by the current successful audits, not by this table alone. See `data/audits/current/02_example_grid_v2_freeze_audit.json`, the two frontend audits, and `reports/validation/pretraining_review.json` for execution evidence. A missing or failing gate blocks downstream use.

Frozen Core: **1,898 clips / 1,189 source recordings**; train/validation/test **1,335 / 269 / 294 clips**; zero source-name leakage. The accepted grid contains **32,645 examples** (22,613 / 4,937 / 5,095), with **15,360-sample starts** and **15,600-sample contexts** at 16 kHz. Its existing `_candidates.csv` filename is retained as an artifact identifier; the separate acceptance record freezes its exact hash.

**03B upstream UI default: −52 dB. Project-frozen baseline: −70 dB.** The override was selected using TRAIN-only anti-collapse diagnostics; it is not a claim of global optimality.

Run notebooks from the repository root or their own directory. Install [requirements-pipeline.txt](requirements-pipeline.txt) in an isolated Python environment; run `python -B tools/validate_pipeline.py` for fresh-kernel validation. The validated Python/package versions are recorded in the audits. Optional PDF exports are disabled by default and are not scientific gates.

| Folder | Purpose |
|---|---|
| `notebooks/current/` | Current pipeline |
| `data/audio/`, `data/metadata/` | Preserved accepted WAVs and release metadata |
| `data/manifests/current/` | Frozen Core/split and accepted Example Grid v2 |
| `data/audits/current/` | Original generation evidence plus gated freeze/frontend audits |
| `reports/` | Validation evidence, unchanged reproduction copies and historical PDF exports |
| `reference/`, `third_party/` | Authenticated upstream source bytes |
| `tools/` | Identity/gate helpers, validation runner and historical review utilities |
| `docs/` | Current methodology and provenance; historical reorganization records |
| `archive/` | Historical artifacts; never promoted into current inputs |

See [pipeline and comparison limits](docs/PIPELINE.md), [data provenance](docs/DATA_PROVENANCE.md), and [reference identities](reference/README.md). The [reorganization report](docs/REORGANIZATION_REPORT.md) describes an earlier state and is preserved unchanged; its missing-reference statements are historical, not current status.
