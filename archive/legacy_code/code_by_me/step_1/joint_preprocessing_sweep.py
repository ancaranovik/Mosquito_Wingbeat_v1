from pathlib import Path
import sys

import numpy as np
import pandas as pd

from scipy.io import wavfile
from scipy import signal


# ============================================================
# STEP 1C.7E
# JOINT PREPROCESSING SWEEP
#
# low_frequency × noise_floor
#
# TRAIN ONLY
# ============================================================
#
# Goal:
#
# Find a preprocessing configuration that:
#
# 1. Reduces very-low-frequency dominance.
# 2. Does not completely erase too many windows.
# 3. Does not leave the MFE extremely sparse.
#
# We DO NOT choose a winner inside this script.
# We only collect evidence.
# ============================================================


# ============================================================
# PATHS
# ============================================================

manifest_path = Path(
    "D:/VGU-27/Edge AI Project/"
    "data/core_1s_audit_windows.csv"
)

audio_dir = Path(
    "D:/VGU-27/Edge AI Project/"
    "data/audio"
)

output_path = Path(
    "D:/VGU-27/Edge AI Project/"
    "data/train_joint_preprocessing_sweep.csv"
)


# ============================================================
# EDGE IMPULSE SPEECHPY
# ============================================================

edge_impulse_repo = Path(
    "D:/VGU-27/Edge AI Project/"
    "third_party/edgeimpulse-processing-blocks"
)

speechpy_parent = (
    edge_impulse_repo
    / "mfe"
    / "third_party"
)

sys.path.insert(
    0,
    str(speechpy_parent)
)

import speechpy


# ============================================================
# FIXED DSP PARAMETERS
# ============================================================

SOURCE_SAMPLE_RATE = 44100

TARGET_SAMPLE_RATE = 16000

FRAME_LENGTH_SECONDS = 0.020

FRAME_STRIDE_SECONDS = 0.010

FFT_LENGTH = 256

NUM_FILTERS = 40

HIGH_FREQUENCY = 8000


# ============================================================
# PARAMETERS TO TEST
# ============================================================

LOW_FREQUENCIES = [
    0,
    100,
    150,
    200,
]

NOISE_FLOORS = [
    -52.0,
    -60.0,
    -65.0,
    -70.0,
]


# ============================================================
# FFT SIZE
# ============================================================

NUM_FFT_COEFFICIENTS = (
    FFT_LENGTH // 2
    + 1
)


# ============================================================
# PRE-COMPUTE EDGE IMPULSE MEL FILTERBANKS
# ============================================================
#
# One filterbank for every low_frequency.
#
# ============================================================

filterbanks = {}

filter_center_frequencies = {}


for low_frequency in LOW_FREQUENCIES:

    filterbank, centers = (
        speechpy.feature.filterbanks(
            num_filter=NUM_FILTERS,
            coefficients=NUM_FFT_COEFFICIENTS,
            sampling_freq=TARGET_SAMPLE_RATE,
            low_freq=low_frequency,
            high_freq=HIGH_FREQUENCY,
            use_old_mels=False,
        )
    )

    filterbanks[
        low_frequency
    ] = filterbank

    filter_center_frequencies[
        low_frequency
    ] = centers


# ============================================================
# LOAD MANIFEST
# ============================================================

manifest = pd.read_csv(
    manifest_path
)


# ============================================================
# TRAIN ONLY
# ============================================================

train_df = manifest[
    manifest["split"] == "train"
].copy()


print(
    "============================================================"
)

print(
    "STEP 1C.7E - JOINT PREPROCESSING SWEEP"
)

print(
    "============================================================"
)

print(
    "Train windows:",
    len(train_df)
)

print(
    "Low frequencies:",
    LOW_FREQUENCIES
)

print(
    "Noise floors:",
    NOISE_FLOORS
)

print(
    "Total configurations:",
    len(LOW_FREQUENCIES)
    * len(NOISE_FLOORS)
)


# ============================================================
# RESULTS
# ============================================================

records = []


# ============================================================
# PROCESS EACH TRAIN WINDOW
# ============================================================

for count, (_, row) in enumerate(
    train_df.iterrows(),
    start=1
):

    # ========================================================
    # 1. LOAD WAV
    # ========================================================

    audio_path = (
        audio_dir
        / f"{int(row['id'])}.wav"
    )


    sample_rate, audio = wavfile.read(
        audio_path
    )


    if sample_rate != SOURCE_SAMPLE_RATE:

        raise ValueError(
            f"Unexpected sample rate "
            f"{sample_rate} Hz "
            f"for ID {row['id']}"
        )


    # ========================================================
    # 2. PCM24 -> NORMALIZED FLOAT
    # ========================================================

    audio_float = (
        audio.astype(np.float32)
        / 2147483648.0
    )


    # ========================================================
    # 3. EXTRACT SAME 1-SECOND AUDIT WINDOW
    # ========================================================

    start_sample = int(
        row["start_sample"]
    )

    end_sample = int(
        row["end_sample"]
    )


    window = audio_float[
        start_sample:end_sample
    ]


    if len(window) != SOURCE_SAMPLE_RATE:

        raise ValueError(
            f"Window length error "
            f"for ID {row['id']}: "
            f"{len(window)} samples"
        )


    # ========================================================
    # 4. RESAMPLE 44.1 kHz -> 16 kHz
    # ========================================================

    resampled = signal.resample_poly(
        window,
        up=160,
        down=441
    )


    # ========================================================
    # 5. PRE-EMPHASIS
    # ========================================================

    emphasized = (
        speechpy.processing.preemphasis(
            resampled,
            cof=0.98,
            shift=1
        )
    )


    # ========================================================
    # 6. FRAMING
    # ========================================================

    frames = (
        speechpy.processing.stack_frames(
            emphasized,
            sampling_frequency=TARGET_SAMPLE_RATE,
            implementation_version=4,
            frame_length=FRAME_LENGTH_SECONDS,
            frame_stride=FRAME_STRIDE_SECONDS,

            filter=lambda x: np.ones(
                (x,)
            ),

            zero_padding=False,
        )
    )


    # ========================================================
    # 7. POWER SPECTRUM
    #
    # This is common to ALL 16 configurations.
    # ========================================================

    power_spectrum = (
        speechpy.processing.power_spectrum(
            frames,
            FFT_LENGTH
        )
    )


    # ========================================================
    # 8. LOOP THROUGH LOW FREQUENCY SETTINGS
    # ========================================================

    for low_frequency in LOW_FREQUENCIES:

        mel_filterbank = (
            filterbanks[
                low_frequency
            ]
        )

        mel_centers = (
            filter_center_frequencies[
                low_frequency
            ]
        )


        # ====================================================
        # MEL ENERGY
        # ====================================================

        mfe_energy = np.dot(
            power_spectrum,
            mel_filterbank.T
        )


        mfe_energy = np.clip(
            mfe_energy,
            1e-30,
            None
        )


        # ====================================================
        # RAW STRONGEST MEL BAND
        #
        # Calculated BEFORE noise floor / quantization.
        # ====================================================

        mean_energy_per_band = np.mean(
            mfe_energy,
            axis=0
        )


        strongest_band = int(
            np.argmax(
                mean_energy_per_band
            )
        )


        strongest_center_hz = float(
            mel_centers[
                strongest_band
            ]
        )


        # ====================================================
        # ENERGY -> dB
        # ====================================================

        mfe_db = (
            10.0
            * np.log10(
                mfe_energy
            )
        )


        max_mfe_db = float(
            np.max(
                mfe_db
            )
        )


        # ====================================================
        # 9. LOOP THROUGH NOISE FLOORS
        # ====================================================

        for noise_floor_db in NOISE_FLOORS:

            # ================================================
            # % RAW MFE BINS BELOW FLOOR
            # ================================================

            below_floor_percentage = (
                100.0
                * np.mean(
                    mfe_db
                    < noise_floor_db
                )
            )


            # ================================================
            # COMPLETE LOSS
            #
            # Every MFE bin is below this floor.
            # ================================================

            complete_loss = (
                max_mfe_db
                < noise_floor_db
            )


            # ================================================
            # EDGE IMPULSE NORMALIZATION
            # ================================================

            normalized = (
                mfe_db
                - noise_floor_db
            ) / (
                (-noise_floor_db)
                + 12.0
            )


            normalized = np.clip(
                normalized,
                0.0,
                1.0
            )


            # ================================================
            # EDGE IMPULSE QUANTIZATION
            # ================================================

            quantized = np.uint8(
                np.around(
                    normalized
                    * 256.0
                )
            )


            quantized = np.clip(
                quantized,
                0,
                255
            )


            quantized = np.float32(
                quantized
                / 256.0
            )


            # ================================================
            # FINAL FEATURE STATISTICS
            # ================================================

            zero_percentage = (
                100.0
                * np.mean(
                    quantized
                    == 0
                )
            )


            nonzero_values = (
                quantized[
                    quantized > 0
                ]
            )


            mean_feature = float(
                np.mean(
                    quantized
                )
            )


            max_feature = float(
                np.max(
                    quantized
                )
            )


            if len(
                nonzero_values
            ) > 0:

                mean_nonzero = float(
                    np.mean(
                        nonzero_values
                    )
                )

            else:

                mean_nonzero = 0.0


            # ================================================
            # STORE
            # ================================================

            records.append(
                {
                    "id":
                        int(row["id"]),

                    "species":
                        row["species"],

                    "name":
                        row["name"],

                    "low_frequency":
                        int(
                            low_frequency
                        ),

                    "noise_floor_db":
                        float(
                            noise_floor_db
                        ),

                    "below_floor_percentage":
                        float(
                            below_floor_percentage
                        ),

                    "zero_percentage":
                        float(
                            zero_percentage
                        ),

                    "complete_loss":
                        bool(
                            complete_loss
                        ),

                    "max_mfe_db":
                        max_mfe_db,

                    "max_feature":
                        max_feature,

                    "mean_feature":
                        mean_feature,

                    "mean_nonzero_feature":
                        mean_nonzero,

                    "strongest_raw_mel_band":
                        strongest_band,

                    "strongest_raw_center_hz":
                        strongest_center_hz,
                }
            )


    # ========================================================
    # PROGRESS
    # ========================================================

    if (
        count % 100 == 0
        or count == len(train_df)
    ):

        print(
            f"Processed "
            f"{count}/"
            f"{len(train_df)}"
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

result_df = pd.DataFrame(
    records
)


# ============================================================
# SANITY CHECK
# ============================================================

expected_rows = (
    len(train_df)
    * len(LOW_FREQUENCIES)
    * len(NOISE_FLOORS)
)


print(
    "\n============================================================"
)

print(
    "RESULT SIZE CHECK"
)

print(
    "============================================================"
)

print(
    "Expected rows:",
    expected_rows
)

print(
    "Actual rows:",
    len(result_df)
)


# ============================================================
# GLOBAL CONFIGURATION SUMMARY
# ============================================================

config_summary = (
    result_df
    .groupby(
        [
            "low_frequency",
            "noise_floor_db",
        ]
    )
    .agg(

        windows=(
            "id",
            "count"
        ),

        zero_mean=(
            "zero_percentage",
            "mean"
        ),

        zero_median=(
            "zero_percentage",
            "median"
        ),

        zero_25=(
            "zero_percentage",
            lambda x:
                x.quantile(
                    0.25
                )
        ),

        zero_75=(
            "zero_percentage",
            lambda x:
                x.quantile(
                    0.75
                )
        ),

        complete_loss_count=(
            "complete_loss",
            "sum"
        ),

        complete_loss_percent=(
            "complete_loss",
            "mean"
        ),

        mean_feature_median=(
            "mean_feature",
            "median"
        ),

        max_feature_median=(
            "max_feature",
            "median"
        ),

        strongest_center_median=(
            "strongest_raw_center_hz",
            "median"
        ),
    )
    .reset_index()
)


config_summary[
    "complete_loss_percent"
] *= 100.0


# ============================================================
# >=95% ZERO
# >=99% ZERO
# ============================================================

sparsity_summary = (
    result_df
    .groupby(
        [
            "low_frequency",
            "noise_floor_db",
        ]
    )
    .agg(

        windows_95_zero=(
            "zero_percentage",
            lambda x:
                int(
                    np.sum(
                        x >= 95.0
                    )
                )
        ),

        windows_99_zero=(
            "zero_percentage",
            lambda x:
                int(
                    np.sum(
                        x >= 99.0
                    )
                )
        ),

        windows_100_zero=(
            "zero_percentage",
            lambda x:
                int(
                    np.sum(
                        x == 100.0
                    )
                )
        ),
    )
    .reset_index()
)


config_summary = (
    config_summary
    .merge(
        sparsity_summary,
        on=[
            "low_frequency",
            "noise_floor_db",
        ],
        how="left",
    )
)


# ============================================================
# PRINT GLOBAL TABLE
# ============================================================

print(
    "\n============================================================"
)

print(
    "GLOBAL JOINT SWEEP SUMMARY"
)

print(
    "============================================================"
)


columns_to_show = [
    "low_frequency",
    "noise_floor_db",

    "zero_mean",
    "zero_median",

    "complete_loss_count",
    "complete_loss_percent",

    "windows_95_zero",
    "windows_99_zero",
    "windows_100_zero",

    "mean_feature_median",
    "max_feature_median",

    "strongest_center_median",
]


print(
    config_summary[
        columns_to_show
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# MEDIAN ZERO % MATRIX
# ============================================================

print(
    "\n============================================================"
)

print(
    "MEDIAN FINAL ZERO %"
)

print(
    "rows = low_frequency"
)

print(
    "columns = noise_floor"
)

print(
    "============================================================"
)


zero_matrix = (
    result_df
    .groupby(
        [
            "low_frequency",
            "noise_floor_db",
        ]
    )[
        "zero_percentage"
    ]
    .median()
    .unstack()
)


print(
    zero_matrix.to_string()
)


# ============================================================
# COMPLETE LOSS % MATRIX
# ============================================================

print(
    "\n============================================================"
)

print(
    "COMPLETE LOSS %"
)

print(
    "rows = low_frequency"
)

print(
    "columns = noise_floor"
)

print(
    "============================================================"
)


loss_matrix = (
    result_df
    .groupby(
        [
            "low_frequency",
            "noise_floor_db",
        ]
    )[
        "complete_loss"
    ]
    .mean()
    .mul(
        100.0
    )
    .unstack()
)


print(
    loss_matrix.to_string()
)


# ============================================================
# WINDOWS >=95% ZERO
# ============================================================

print(
    "\n============================================================"
)

print(
    "NUMBER OF WINDOWS >=95% ZERO"
)

print(
    "============================================================"
)


very_sparse_matrix = (
    result_df
    .assign(
        very_sparse=(
            result_df[
                "zero_percentage"
            ] >= 95.0
        )
    )
    .groupby(
        [
            "low_frequency",
            "noise_floor_db",
        ]
    )[
        "very_sparse"
    ]
    .sum()
    .unstack()
)


print(
    very_sparse_matrix.to_string()
)


# ============================================================
# COMPLETE LOSS % BY SPECIES
# ============================================================

print(
    "\n============================================================"
)

print(
    "COMPLETE LOSS % BY SPECIES AND CONFIG"
)

print(
    "============================================================"
)


species_loss = (
    result_df
    .groupby(
        [
            "species",
            "low_frequency",
            "noise_floor_db",
        ]
    )[
        "complete_loss"
    ]
    .mean()
    .mul(
        100.0
    )
    .reset_index()
)


for species in sorted(
    result_df[
        "species"
    ].unique()
):

    print(
        f"\n--- {species} ---"
    )

    species_table = (
        species_loss[
            species_loss[
                "species"
            ] == species
        ]
        .pivot(
            index="low_frequency",
            columns="noise_floor_db",
            values="complete_loss",
        )
    )

    print(
        species_table.to_string()
    )


# ============================================================
# MEDIAN ZERO % BY SPECIES
# ============================================================

print(
    "\n============================================================"
)

print(
    "MEDIAN ZERO % BY SPECIES AND CONFIG"
)

print(
    "============================================================"
)


species_zero = (
    result_df
    .groupby(
        [
            "species",
            "low_frequency",
            "noise_floor_db",
        ]
    )[
        "zero_percentage"
    ]
    .median()
    .reset_index()
)


for species in sorted(
    result_df[
        "species"
    ].unique()
):

    print(
        f"\n--- {species} ---"
    )

    species_table = (
        species_zero[
            species_zero[
                "species"
            ] == species
        ]
        .pivot(
            index="low_frequency",
            columns="noise_floor_db",
            values="zero_percentage",
        )
    )

    print(
        species_table.to_string()
    )


# ============================================================
# SAVE FULL RESULTS
# ============================================================

result_df.to_csv(
    output_path,
    index=False
)


summary_path = Path(
    "D:/VGU-27/Edge AI Project/"
    "data/train_joint_preprocessing_summary.csv"
)


config_summary.to_csv(
    summary_path,
    index=False
)


print(
    "\n============================================================"
)

print(
    "SAVED"
)

print(
    "============================================================"
)

print(
    "Full results:"
)

print(
    output_path
)

print(
    "\nConfiguration summary:"
)

print(
    summary_path
)