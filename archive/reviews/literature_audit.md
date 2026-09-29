# Primary-source audit for the mosquito proposal

Verified 22 September 2026. This is an editorial evidence audit, not a dataset-content audit or a reproduced experiment. Original audio archives were not downloaded. Main LaTeX was not edited.

## Corrections and defensible data choice

**Do not identify Abuzz as certified isolated-individual recordings.** Its reference recordings include phones tracking mosquitoes in densely populated, single-species laboratory cages. The source supports single-species/dominant-source classification; it does not establish one independently identifiable animal per downloaded file. A wingbeat-active segment is also not automatically an isolated individual. Recommend using the original Abuzz paper for attribution and removing the unverified combined claim of 1,285 recordings and 20 species. [Original eLife paper](https://elifesciences.org/articles/27854)

**MosquitoSong+ W-INDOOR is a defensible provisional primary dataset** for *Ae. aegypti*, *Ae. albopictus*, *An. dirus*, and *Cx. quinquefasciatus*: its methods explicitly describe individually recorded mosquitoes. W-INDOOR must be distinguished from W-OUTDOOR and the paper's HumBug subset. Do not claim that the publicly distributed file names establish independent individuals until the source manifest has been examined. It uses overlapping 300 ms epochs, so epoch-level random partitioning is unsuitable. [PLOS methods](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310121)

The author repository describes 1,324 wingbeat recordings, but that is **not** a verified independent-mosquito count. Its data directory contains a placeholder, with the download hosted through Google Drive from the README. The repository therefore supports a Week-1 access/provenance gate, not an assertion that all source metadata have been audited. [Author repository](https://github.com/akaraspt/mosquitosongp)

Suggested proposal decision: select W-INDOOR provisionally; retain the four species only if original recording/individual grouping and enough groups per class support train/validation/test partitions. If only two classes meet this gate, reduce the closed-set task explicitly. If W-INDOOR access or provenance fails, choose a documented HumBugDB cup subset with revised species, or use Abuzz with single-species/dominant-source claims and conservative recording/session grouping. Do not promise the same four species across every dataset.

## Abuzz bibliography

Replace `abuzz2020` as the dataset citation with:

H. Mukundarajan, F. J. H. Hol, E. A. Castillo, C. Newby, and M. Prakash, “Using mobile phones as acoustic sensors for high-throughput mosquito surveillance,” *eLife*, 6:e27854, 2017. DOI [10.7554/eLife.27854](https://doi.org/10.7554/eLife.27854).

The associated data record is [Dryad DOI 10.5061/dryad.98d7s](https://datadryad.org/dataset/doi:10.5061/dryad.98d7s), showing the 2018 repository publication date and 20 named species archives. Some archives include both raw and cleaned audio, so duplicate ancestry must be audited before splitting.

The original proposal's DOI actually identifies M. S. Fernandes, W. Cordeiro, and M. Recamonde-Mendoza, “Detecting Aedes aegypti mosquitoes through audio classification with convolutional neural networks,” *Computers in Biology and Medicine*, 129:104152, 2021 (online 27 November 2020). DOI [10.1016/j.compbiomed.2020.104152](https://doi.org/10.1016/j.compbiomed.2020.104152). This is a later classification study, not Abuzz's originating publication. [Publisher record](https://www.sciencedirect.com/science/article/pii/S0010482520304832)

## HumBugDB

Add a direct reference if discussing HumBugDB:

I. Kiskin et al., “HumBugDB: A Large-scale Acoustic Mosquito Dataset,” *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks*, 1, 2021. [Official proceedings](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/65ded5353c5ee48d0b7d48c591b8f430-Abstract-round2.html); [arXiv:2110.07607](https://arxiv.org/abs/2110.07607).

The paper reports 20 hours of mosquito-labelled audio, 18 hours with species annotations, and 36 species or complexes. Its IHI Tanzania MSC experiment explicitly treats each original `audio_id` as one mosquito and partitions those recordings. This supports an individually grouped fallback much more directly than blanket Abuzz assumptions. Example published train/test individual counts: *An. arabiensis* 385/129, *Cx. pipiens* 252/84, *Ae. aegypti* 36/13, *An. funestus* s.s. 186/62. These are prior-paper counts, **not counts verified in a newly filtered project dataset**. Labels include complexes, and the intended species set must be taxonomically explicit. [Methods and Table 4](https://arxiv.org/html/2110.07607)

Original dataset version: [10.5281/zenodo.4904800](https://zenodo.org/records/4904800), version 0.0.1. The newer version redirects to the ComParE 2022 release [10.5281/zenodo.6478589](https://zenodo.org/records/6478589); pin the chosen version instead of treating the releases interchangeably.

The actual released metadata header includes `id`, `name`, `plurality`, recording date, species, location and device fields. Different `id` rows share the same `name`; `id` is not a trustworthy independent-source grouping key. Recover original source identity from `name` plus corroborating metadata; keep related labels/crops together. Unknown plurality is not evidence of an isolated source. [Original metadata CSV](https://raw.githubusercontent.com/HumBug-Mosquito/HumBugDB/master/data/metadata/neurips_2021_zenodo_0_0_1.csv)

## MosquitoSong+ citation and claims

The existing DOI, year, journal, volume and article number are correct: A. Supratak, P. Haddawy, M. S. Yin, T. Ziemer, W. Siritanakorn, K. Assawavinijkulchai, K. Chiamsakul, T. Chantanalertvilai, W. Suchalermkul, C. Sa-ngamuang, and P. Sriwichai, *PLOS ONE*, 19(10):e0310121, 2024. DOI [10.1371/journal.pone.0310121](https://doi.org/10.1371/journal.pone.0310121).

Safe claim: the paper evaluates noise/level augmentation and performance across several acquisition conditions. Avoid describing its pooled multi-dataset training experiment as a clean zero-shot cross-dataset transfer experiment. Its selected HumBug *Ae. albopictus* material is only 33.2 seconds, versus 1,322.4 seconds for female *Ae. aegypti* and 909.8 seconds for *An. dirus*; species, site and device are confounded. *Cx. quinquefasciatus* plurality required manual selection. These are limitations for a four-species HumBug alternative. W-INDOOR source independence still requires project-level audit. [Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310121)

## Altayeb TinyML baseline

Published citation is correct: M. Altayeb, M. Zennaro, and M. Rovai, “Classifying mosquito wingbeat sound using TinyML,” *GoodIT '22*, pp. 132–137, 2022. DOI [10.1145/3524458.3547258](https://doi.org/10.1145/3524458.3547258). [ACM proceedings metadata](https://doi.org/10.1145/3524458)

The accessible author manuscript is the earlier TinyML Symposium version, with different author order; distinguish it from the published GoodIT record. It specifies 16 kHz PCM16, 1-second windows every 250 ms; spectrogram frames 25 ms, stride 12.5 ms, noise floor -52 dB. It states 128 frequency bands but an actual 79×65 input: **do not silently resolve this inconsistency**. Confirm a pinned DSP export before claiming exact preprocessing reproduction. It used three original aegypti and two albopictus source files, so its split cannot establish independent-individual generalization. [Author manuscript](https://mjrovai.github.io/TinyML4D/papers/TinyML_Research_Symposium_Classifying-mosquito-wingbeat-sound-using-TinyML.pdf)

The author notebook verifies the classifier: Conv1D(32,k=3,same,ReLU), MaxPool(2,2,same), Conv1D(64,k=3,same,ReLU), MaxPool(2,2,same), Flatten, Dropout(0.5), Dense(4,softmax). Classes are aegypti, albopictus, noise, other. Adam learning rate 0.001, batch 32, maximum 100 epochs, early stopping on validation loss with patience 5. For a different grouped dataset/front-end, call it an **Altayeb-inspired reimplementation**, not a faithful numerical replication. [Author code repository](https://github.com/Mjrovai/wingbeat-mosquito-tinyml)

## BioDCASE 2026

The title and authors in the proposal are verified. Cite Y. Hou, V. Zdravkovic, M. Sinka, Y. Li, W. Wang, M. D. Plumbley, K. Willis, and S. Roberts, “BioDCASE 2026 Challenge Baseline for Cross-Domain Mosquito Species Classification,” arXiv:2603.20118, 2026. DOI [10.48550/arXiv.2603.20118](https://doi.org/10.48550/arXiv.2603.20118). It is a preprint, not an established peer-reviewed venue citation. [arXiv record](https://arxiv.org/abs/2603.20118)

The official development set has 271,380 clips, nine species, five domains and 60.66 hours. Species/domain balance is extreme. “Unseen” refers to a species-domain combination absent for that species in training; it does not necessarily mean a globally new domain. The release exposes species and domain IDs, not guaranteed original-individual identity. Its website therefore cannot substantiate source-independent grouping by itself. Treat it as cross-domain context or an optional separate benchmark, not an automatically independent external test for data potentially shared with its source corpora. [Task page](https://biodcase.github.io/challenge2026/task5)

Dataset version v2: [10.5281/zenodo.20478577](https://zenodo.org/records/20478577). Record version and provenance if used.

## ESP32-S3

The hardware claims are correct: dual-core LX7 up to 240 MHz, 512 KB on-chip SRAM, I2S support, and vector instruction extensions for AI/DSP. Cite Espressif Systems, *ESP32-S3 Series Datasheet*, v2.2, accessed 22 September 2026, rather than “2026 documentation” as an invented publication year. [Datasheet](https://www.espressif.com/sites/default/files/documentation/esp32-s3_datasheet_en.pdf)

These specifications do not prove the complete proposed pipeline fits. Budget available internal SRAM after firmware/audio/DMA/task stacks; report external PSRAM separately and identify the exact module, flash/PSRAM configuration, clock and software stack. SIMD capability does not establish that the chosen inference kernels use it.
