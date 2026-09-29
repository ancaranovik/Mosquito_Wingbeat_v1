from pathlib import Path
import sys

import numpy as np
import pandas as pd

from scipy.io import wavfile
from scipy import signal


# ============================================================
# STEP 1C.7D
# LOW-FREQUENCY AUDIT
# ============================================================
#
# Goal:
#
# Test whether very-low-frequency content is dominating MFE.
#
# We keep EVERYTHING ELSE fixed:
#
# sample rate     = 16 kHz
# frame length    = 20 ms
# frame stride    = 10 ms
# FFT             = 256
# Mel filters     = 40
# noise floor     = -52 dB
#
# Only low_frequency changes:
#
# 0 / 100 / 150 / 200 / 300 Hz
#
# IMPORTANT:
#
# Train split only.
# Test set remains untouched.
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
    "data/train_low_frequency_audit.csv"
)


# ============================================================
# EDGE IMPULSE SPEECHPY
# ============================================================
#
# Use the official Edge Impulse fork that we already cloned.
#
# This avoids introducing a new hand-written Mel filterbank
# implementation for this experiment.
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
# CONSTANTS
# ============================================================

SOURCE_SAMPLE_RATE = 44100

TARGET_SAMPLE_RATE = 16000

FRAME_LENGTH_SECONDS = 0.020

FRAME_STRIDE_SECONDS = 0.010

FFT_LENGTH = 256

NUM_FILTERS = 40

HIGH_FREQUENCY = 8000

NOISE_FLOOR_DB = -52.0


# ============================================================
# LOW FREQUENCIES TO TEST
# ============================================================

LOW_FREQUENCIES = [
    0,
    100,
    150,
    200,
    300,
]


# ============================================================
# PRE-COMPUTE OFFICIAL EDGE IMPULSE MEL FILTERBANKS
# ============================================================
#
# FFT 256:
#
# 256 / 2 + 1
# =
# 129 power-spectrum coefficients
#
# ============================================================

NUM_FFT_COEFFICIENTS = (
    FFT_LENGTH // 2
    + 1
)


filterbanks = {}

filter_frequencies = {}


for low_frequency in LOW_FREQUENCIES:

    filterbank, center_frequencies = (
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


    filter_frequencies[
        low_frequency
    ] = center_frequencies


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
    "STEP 1C.7D - LOW FREQUENCY AUDIT"
)

print(
    "============================================================"
)

print(
    "Train windows:",
    len(train_df)
)

print(
    "Low-frequency settings:",
    LOW_FREQUENCIES
)


# ============================================================
# RESULTS
# ============================================================

records = []


# ============================================================
# PROCESS EVERY TRAIN WINDOW
# ============================================================

for count, (_, row) in enumerate(
    train_df.iterrows(),
    start=1
):

    # --------------------------------------------------------
    # Audio path
    # --------------------------------------------------------

    audio_path = (
        audio_dir
        / f"{int(row['id'])}.wav"
    )


    # --------------------------------------------------------
    # Load WAV
    # --------------------------------------------------------

    sample_rate, audio = wavfile.read(
        audio_path
    )


    if sample_rate != SOURCE_SAMPLE_RATE:

        raise ValueError(
            f"Unexpected sample rate "
            f"{sample_rate} Hz "
            f"for ID {row['id']}"
        )


    # --------------------------------------------------------
    # PCM24 -> normalized float
    #
    # scipy represents PCM24 inside int32.
    # --------------------------------------------------------

    audio_float = (
        audio.astype(np.float32)
        / 2147483648.0
    )


    # --------------------------------------------------------
    # Extract SAME deterministic 1-second audit window
    # --------------------------------------------------------

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
            f"Invalid window length "
            f"for ID {row['id']}: "
            f"{len(window)}"
        )


    # ========================================================
    # RESAMPLE
    # ========================================================

    resampled = signal.resample_poly(
        window,
        up=160,
        down=441
    )


    # ========================================================
    # PRE-EMPHASIS
    #
    # Official Edge Impulse SpeechPy implementation.
    # ========================================================

    emphasized = (
        speechpy.processing.preemphasis(
            resampled,
            cof=0.98,
            shift=1
        )
    )


    # ========================================================
    # FRAMING
    #
    # Official Edge Impulse framing.
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
    # POWER SPECTRUM
    #
    # This does NOT depend on low_frequency.
    #
    # Calculate once per window.
    # ========================================================

    power_spectrum = (
        speechpy.processing.power_spectrum(
            frames,
            FFT_LENGTH
        )
    )


    # ========================================================
    # TEST EACH LOW-FREQUENCY SETTING
    # ========================================================

    for low_frequency in LOW_FREQUENCIES:

        mel_filterbank = (
            filterbanks[
                low_frequency
            ]
        )

        mel_centers = (
            filter_frequencies[
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
        # RAW STRONGEST BAND
        # ====================================================
        #
        # IMPORTANT:
        #
        # Determine strongest frequency BEFORE:
        #
        # -52 dB floor
        # normalization
        # quantization
        #
        # Therefore an all-zero final MFE cannot falsely
        # become "band 0 is strongest".
        #
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


        first_band_is_strongest = (
            strongest_band == 0
        )


        # ====================================================
        # MFE ENERGY -> dB
        # ====================================================

        mfe_db = (
            10.0
            * np.log10(
                mfe_energy
            )
        )


        # ====================================================
        # % BINS BELOW -52 dB
        # ====================================================

        below_floor_percentage = (
            100.0
            * np.mean(
                mfe_db
                < NOISE_FLOOR_DB
            )
        )


        # ====================================================
        # COMPLETE LOSS
        #
        # True if not a single MFE bin reaches -52 dB.
        # ====================================================

        complete_loss = (
            np.max(
                mfe_db
            )
            < NOISE_FLOOR_DB
        )


        # ====================================================
        # EDGE IMPULSE NORMALIZATION
        # ====================================================

        normalized = (
            mfe_db
            - NOISE_FLOOR_DB
        ) / (
            (-NOISE_FLOOR_DB)
            + 12.0
        )


        normalized = np.clip(
            normalized,
            0.0,
            1.0
        )


        # ====================================================
        # EDGE IMPULSE QUANTIZATION
        # ====================================================

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


        # ====================================================
        # FINAL ZERO %
        # ====================================================

        zero_percentage = (
            100.0
            * np.mean(
                quantized
                == 0
            )
        )


        # ====================================================
        # STORE RESULT
        # ====================================================

        records.append(
            {
                "id":
                    int(row["id"]),

                "species":
                    row["species"],

                "name":
                    row["name"],

                "low_frequency":
                    low_frequency,

                "zero_percentage":
                    float(
                        zero_percentage
                    ),

                "below_52db_percentage":
                    float(
                        below_floor_percentage
                    ),

                "complete_loss":
                    bool(
                        complete_loss
                    ),

                "max_mfe_db":
                    float(
                        np.max(
                            mfe_db
                        )
                    ),

                "strongest_band":
                    strongest_band,

                "strongest_center_hz":
                    strongest_center_hz,

                "first_band_is_strongest":
                    bool(
                        first_band_is_strongest
                    ),
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
# GLOBAL SUMMARY
# ============================================================

print(
    "\n============================================================"
)

print(
    "GLOBAL LOW-FREQUENCY SUMMARY"
)

print(
    "============================================================"
)


global_summary = (
    result_df
    .groupby(
        "low_frequency"
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

        below_52_median=(
            "below_52db_percentage",
            "median"
        ),

        complete_loss_count=(
            "complete_loss",
            "sum"
        ),

        complete_loss_percent=(
            "complete_loss",
            "mean"
        ),

        strongest_center_median=(
            "strongest_center_hz",
            "median"
        ),

        first_band_strongest_percent=(
            "first_band_is_strongest",
            "mean"
        ),
    )
)


global_summary[
    "complete_loss_percent"
] *= 100.0


global_summary[
    "first_band_strongest_percent"
] *= 100.0


print(
    global_summary.to_string()
)


# ============================================================
# FREQUENCY REGION
# ============================================================

def frequency_region(freq):

    if freq < 100:

        return "<100 Hz"

    elif freq < 300:

        return "100-300 Hz"

    elif freq < 800:

        return "300-800 Hz"

    elif freq < 2000:

        return "800-2000 Hz"

    else:

        return ">=2000 Hz"


result_df[
    "strongest_frequency_region"
] = (
    result_df[
        "strongest_center_hz"
    ]
    .apply(
        frequency_region
    )
)


# ============================================================
# STRONGEST FREQUENCY REGION BY LOW-FREQUENCY SETTING
# ============================================================

print(
    "\n============================================================"
)

print(
    "STRONGEST RAW MFE FREQUENCY REGION"
)

print(
    "PERCENT OF WINDOWS"
)

print(
    "============================================================"
)


region_table = pd.crosstab(
    result_df[
        "low_frequency"
    ],
    result_df[
        "strongest_frequency_region"
    ],
    normalize="index"
) * 100.0


print(
    region_table.to_string()
)


# ============================================================
# COMPLETE LOSS BY SPECIES
# ============================================================

print(
    "\n============================================================"
)

print(
    "COMPLETE LOSS % BY SPECIES"
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
    species_loss.to_string()
)


# ============================================================
# ZERO % BY SPECIES
# ============================================================

print(
    "\n============================================================"
)

print(
    "MEDIAN FINAL ZERO % BY SPECIES"
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
        ]
    )[
        "zero_percentage"
    ]
    .median()
    .unstack()
)


print(
    species_zero.to_string()
)


# ============================================================
# STRONGEST CENTER FREQUENCY BY SPECIES
# ============================================================

print(
    "\n============================================================"
)

print(
    "MEDIAN STRONGEST RAW MFE CENTER Hz BY SPECIES"
)

print(
    "============================================================"
)


species_frequency = (
    result_df
    .groupby(
        [
            "species",
            "low_frequency",
        ]
    )[
        "strongest_center_hz"
    ]
    .median()
    .unstack()
)


print(
    species_frequency.to_string()
)


# ============================================================
# SAVE
# ============================================================

result_df.to_csv(
    output_path,
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
    output_path
)