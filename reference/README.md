# Authenticated reference implementations

The 03A reference is present under `reference/humbug/vggish/`. These source bytes are preserved unchanged. Their Git blob identities match the archived HumBugDB tree `50656758594982480f568598874f79c222432e01` in `archive/baseline_readiness/sources/humbug_tree.json`.

| File | Git blob SHA-1 |
|---|---|
| `mel_features.py` | `ac58fb5427f772fcced9cbd3cec3373ffbe5908c` |
| `vggish_input.py` | `53cdfc1b5417276e55becae3d4404bef27ee5c7b` |
| `vggish_params.py` | `526784bceaa4c9c8b8dc2b8f82e0f3d395d4bec2` |
| `vggish.py` | `13d9a232dbdfb4a5c677589f7522409a53b4dbc4` |

`mel_features.py` SHA-256 is `68803c00743cb43139db12836b5e745a977cdb81443257fa716cd520d7e5e948`. The notebooks enforce accepted source hashes from `tools/pipeline_contract.py` before use.

03A tests the feature transform on identical 16 kHz contexts. `vggish_input.py` uses a different PCM16/resampy loading path; it is contextual reference, not the code executed by the project. No end-to-end HumBugDB classifier or pretrained-model parity is claimed.

03B uses `third_party/edgeimpulse-processing-blocks/` at revision `9b0ceb2b5d22658d3193a5b77dc773cfc8d4cab0`. Accepted SHA-256 pins enforce the MFE source files, including the SpeechPy fork and imported support files. The notebook reconstructs the wrapper and compares directly with the original authenticated `generate_features` function body, using the actual vendored SpeechPy module. This avoids unrelated module-startup imports without replacing numerical operations.

For that pinned revision, MFE v4 UI defaults are 20 ms frames, 10 ms stride, 40 filters, FFT 256, 0–Nyquist and **−52 dB**. CLI defaults differ (20 ms stride, 32 filters). The project's **−70 dB** is a TRAIN-only selected baseline, not an upstream default. Python uint8 wrap, FFT truncation and per-context `np.roll` pre-emphasis are preserved; deployment/streaming parity remains unverified.

Other preserved comparison material:

- `reference/tinyml/tinyml_data_prep_source.ipynb`
- `reference/tinyml/tinyml_classifier_source.ipynb`
- `reference/tinyml/tinyml_primary_source.pdf`
- `archive/baseline_readiness/sources/humbug_feat_util.py`
- `archive/baseline_readiness/sources/humbug_species_classification.ipynb`
- Related source metadata, pinned notebooks and historical literature evidence in `archive/baseline_readiness/sources/`
- Historical review utilities and their dependencies in `tools/review_support/`

Historical reorganization reports stated that the VGGish reference was missing at that time. They are preserved as historical evidence; this index describes the current files. Neither historical alternative frontend implementations nor paper results replace the current accepted experiment contract.
