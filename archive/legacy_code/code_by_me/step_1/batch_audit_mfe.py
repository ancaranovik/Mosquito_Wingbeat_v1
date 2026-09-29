from pathlib import Path

import numpy as np
import pandas as pd

from scipy.io import wavfile
from scipy import signal


# ============================================================
# STEP 1C.7B
# BATCH MFE AUDIT + NOISE FLOOR SWEEP
# ============================================================
#
# Purpose:
#
# 1. Process all TRAIN audit windows.
# 2. Calculate the same MFE pipeline already validated
#    against Edge Impulse.
# 3. Measure sparsity at the official -52 dB setting.
# 4. Test alternative noise floors WITHOUT changing the
#    final preprocessing configuration yet.
#
# IMPORTANT:
#
# Only TRAIN split is inspected here.
# Validation and Test remain untouched for parameter decisions.
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
    "data/train_mfe_audit_summary.csv"
)


# ============================================================
# DSP CONFIGURATION
# ============================================================

SOURCE_SAMPLE_RATE = 44100
TARGET_SAMPLE_RATE = 16000

FRAME_LENGTH_SECONDS = 0.020
FRAME_STRIDE_SECONDS = 0.010

FRAME_LENGTH_SAMPLES = int(
    FRAME_LENGTH_SECONDS
    * TARGET_SAMPLE_RATE
)

FRAME_STRIDE_SAMPLES = int(
    FRAME_STRIDE_SECONDS
    * TARGET_SAMPLE_RATE
)

FFT_LENGTH = 256

NUM_FILTERS = 40

LOW_FREQUENCY = 0
HIGH_FREQUENCY = 8000


# ============================================================
# CURRENT EDGE IMPULSE NOISE FLOOR
# ============================================================

NOISE_FLOOR_DB = -52.0


# ============================================================
# NOISE FLOORS TO DIAGNOSE
# ============================================================
#
# We are NOT choosing a new value yet.
#
# We only ask:
#
# "How sparse would the MFE representation be
#  if the floor were lower?"
#
# ============================================================

TEST_NOISE_FLOORS = [
    -52.0,
    -55.0,
    -60.0,
    -65.0,
    -70.0,
]


# ============================================================
# HZ <-> MEL
# ============================================================

def hz_to_mel(frequency):

    return (
        2595.0
        * np.log10(
            1.0
            + frequency / 700.0
        )
    )


def mel_to_hz(mel):

    return (
        700.0
        * (
            10.0 ** (
                mel / 2595.0
            )
            - 1.0
        )
    )


# ============================================================
# PRE-COMPUTE MEL FILTERBANK
# ============================================================

low_mel = hz_to_mel(
    LOW_FREQUENCY
)

high_mel = hz_to_mel(
    HIGH_FREQUENCY
)


# 40 filters require 42 boundary points.
mel_points = np.linspace(
    low_mel,
    high_mel,
    NUM_FILTERS + 2
)


hz_points = mel_to_hz(
    mel_points
)


# Center frequency of each of the 40 Mel filters.
mel_center_frequencies = (
    hz_points[1:-1]
)


# Convert Mel boundary frequencies to FFT-bin indices.
bin_points = np.floor(
    (FFT_LENGTH + 1)
    * hz_points
    / TARGET_SAMPLE_RATE
).astype(int)


# 256-point real FFT:
#
# 256 / 2 + 1
# =
# 129 frequency bins.
mel_filterbank = np.zeros(
    (
        NUM_FILTERS,
        FFT_LENGTH // 2 + 1
    ),
    dtype=np.float64
)


for i in range(NUM_FILTERS):

    left = bin_points[i]
    center = bin_points[i + 1]
    right = bin_points[i + 2]


    # --------------------------------------------------------
    # Rising side
    # --------------------------------------------------------

    if center != left:

        for j in range(
            left,
            center
        ):

            mel_filterbank[i, j] = (
                (j - left)
                /
                (center - left)
            )


    # --------------------------------------------------------
    # Falling side
    # --------------------------------------------------------

    if right != center:

        for j in range(
            center,
            right
        ):

            mel_filterbank[i, j] = (
                (right - j)
                /
                (right - center)
            )


# ============================================================
# MFE FUNCTION
# ============================================================

def calculate_mfe(
    one_second_audio
):

    # ========================================================
    # 1. RESAMPLE
    #
    # 44.1 kHz -> 16 kHz
    #
    # ratio:
    #
    # 160 / 441
    #
    # ========================================================

    resampled = signal.resample_poly(
        one_second_audio,
        up=160,
        down=441
    )


    # ========================================================
    # 2. PRE-EMPHASIS
    #
    # Same implementation used by Edge Impulse:
    #
    # y[n] = x[n] - 0.98*x[n-1]
    #
    # ========================================================

    previous = np.roll(
        resampled,
        1
    )

    emphasized = (
        resampled
        - 0.98 * previous
    )


    # ========================================================
    # 3. FRAMING
    #
    # 20 ms frame
    # 10 ms stride
    #
    # 1 second at 16 kHz:
    #
    # -> 99 frames
    #
    # ========================================================

    num_frames = 1 + (
        len(emphasized)
        - FRAME_LENGTH_SAMPLES
    ) // FRAME_STRIDE_SAMPLES


    frames = np.empty(
        (
            num_frames,
            FRAME_LENGTH_SAMPLES
        ),
        dtype=np.float32
    )


    for i in range(
        num_frames
    ):

        start = (
            i
            * FRAME_STRIDE_SAMPLES
        )

        end = (
            start
            + FRAME_LENGTH_SAMPLES
        )

        frames[i] = (
            emphasized[
                start:end
            ]
        )


    # ========================================================
    # 4. FFT
    #
    # Input:
    #
    # 99 x 320
    #
    # 256-point rFFT:
    #
    # 99 x 129
    #
    # ========================================================

    fft_values = np.fft.rfft(
        frames,
        n=FFT_LENGTH,
        axis=1
    )


    magnitude = np.abs(
        fft_values
    )


    # ========================================================
    # 5. POWER SPECTRUM
    #
    # Edge Impulse / SpeechPy:
    #
    # power = magnitude^2 / FFT_length
    #
    # ========================================================

    power = (
        np.square(
            magnitude
        )
        / FFT_LENGTH
    )


    # ========================================================
    # 6. MEL FILTERBANK
    #
    # (99 x 129)
    #
    # @
    #
    # (129 x 40)
    #
    # =
    #
    # (99 x 40)
    #
    # ========================================================

    mfe_energy = np.dot(
        power,
        mel_filterbank.T
    )


    # Prevent log10(0).
    mfe_energy = np.clip(
        mfe_energy,
        1e-30,
        None
    )


    # ========================================================
    # 7. MEL ENERGY -> dB
    #
    # Power/energy uses:
    #
    # 10 * log10(E)
    #
    # ========================================================

    mfe_db = (
        10.0
        * np.log10(
            mfe_energy
        )
    )


    # ========================================================
    # 8. CURRENT EDGE IMPULSE NORMALIZATION
    #
    # Current baseline:
    #
    # noise_floor = -52 dB
    #
    # normalized =
    #
    # (dB - noise_floor)
    # ------------------
    # (-noise_floor)+12
    #
    # then clip [0, 1]
    #
    # ========================================================

    mfe_normalized = (
        mfe_db
        - NOISE_FLOOR_DB
    ) / (
        (-NOISE_FLOOR_DB)
        + 12.0
    )


    mfe_normalized = np.clip(
        mfe_normalized,
        0.0,
        1.0
    )


    # ========================================================
    # 9. 8-BIT-LIKE QUANTIZATION
    #
    # Same as Edge Impulse:
    #
    # round(x * 256)
    # -> uint8
    # -> /256
    #
    # ========================================================

    mfe_quantized = np.uint8(
        np.around(
            mfe_normalized
            * 256.0
        )
    )


    mfe_quantized = np.clip(
        mfe_quantized,
        0,
        255
    )


    mfe_quantized = np.float32(
        mfe_quantized
        / 256.0
    )


    return (
        mfe_db,
        mfe_quantized
    )


# ============================================================
# LOAD AUDIT-WINDOW MANIFEST
# ============================================================

manifest = pd.read_csv(
    manifest_path
)


# ============================================================
# ONLY TRAIN
# ============================================================
#
# Preprocessing decisions must not use test data.
#
# ============================================================

train_df = manifest[
    manifest["split"]
    == "train"
].copy()


print(
    "============================================================"
)

print(
    "STEP 1C.7B - TRAIN MFE + NOISE FLOOR SWEEP"
)

print(
    "============================================================"
)

print(
    "Train windows:",
    len(train_df)
)


# ============================================================
# PROCESS TRAIN WINDOWS
# ============================================================

records = []


for count, (_, row) in enumerate(
    train_df.iterrows(),
    start=1
):

    # ========================================================
    # WAV PATH
    # ========================================================

    audio_path = (
        audio_dir
        / f"{int(row['id'])}.wav"
    )


    # ========================================================
    # LOAD WAV
    # ========================================================

    sample_rate, audio = wavfile.read(
        audio_path
    )


    # Safety check.
    if sample_rate != SOURCE_SAMPLE_RATE:

        raise ValueError(
            f"Unexpected sample rate "
            f"{sample_rate} Hz "
            f"for ID {row['id']}"
        )


    # ========================================================
    # PCM24 -> NORMALIZED FLOAT
    #
    # scipy loads 24-bit PCM into left-justified int32.
    #
    # Therefore:
    #
    # int32 / 2^31
    #
    # gives normalized waveform.
    #
    # ========================================================

    audio_float = (
        audio.astype(np.float32)
        / 2147483648.0
    )


    # ========================================================
    # EXTRACT EXACT 1-SECOND AUDIT WINDOW
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
    # CALCULATE MFE
    # ========================================================

    mfe_db, mfe = calculate_mfe(
        window
    )


    # ========================================================
    # CURRENT -52 dB FEATURE STATISTICS
    # ========================================================

    total_bins = (
        mfe.size
    )


    zero_bins = np.sum(
        mfe == 0
    )


    zero_percentage = (
        100.0
        * zero_bins
        / total_bins
    )


    # --------------------------------------------------------
    # Mean feature energy in every Mel band.
    #
    # Shape:
    #
    # 99 x 40
    #
    # mean over time
    #
    # -> 40
    # --------------------------------------------------------

    band_mean = np.mean(
        mfe,
        axis=0
    )


    strongest_band = int(
        np.argmax(
            band_mean
        )
    )


    # ========================================================
    # NOISE FLOOR SWEEP
    # ========================================================
    #
    # IMPORTANT:
    #
    # Use RAW MFE dB.
    #
    # Do NOT recompute FFT/Mel.
    #
    # This isolates only the effect of noise floor.
    #
    # ========================================================

    floor_stats = {}


    for floor_db in TEST_NOISE_FLOORS:

        below_percentage = (
            100.0
            * np.mean(
                mfe_db
                < floor_db
            )
        )


        # e.g.
        #
        # -52 -> below_52db_percent
        #
        column_name = (
            f"below_"
            f"{abs(int(floor_db))}"
            f"db_percent"
        )


        floor_stats[
            column_name
        ] = float(
            below_percentage
        )


    # ========================================================
    # SAVE WINDOW STATISTICS
    # ========================================================

    nonzero_values = (
        mfe[
            mfe > 0
        ]
    )


    records.append(
        {
            "id":
                int(row["id"]),

            "species":
                row["species"],

            "name":
                row["name"],

            "zero_percentage":
                float(
                    zero_percentage
                ),

            "max_feature":
                float(
                    np.max(mfe)
                ),

            "mean_feature":
                float(
                    np.mean(mfe)
                ),

            "mean_nonzero_feature":
                float(
                    np.mean(
                        nonzero_values
                    )
                )
                if len(nonzero_values) > 0
                else 0.0,

            "strongest_mel_band":
                strongest_band,

            "strongest_mel_center_hz":
                float(
                    mel_center_frequencies[
                        strongest_band
                    ]
                ),

            "min_mfe_db":
                float(
                    np.min(
                        mfe_db
                    )
                ),

            "max_mfe_db":
                float(
                    np.max(
                        mfe_db
                    )
                ),

            **floor_stats,
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
    "GLOBAL TRAIN SUMMARY"
)

print(
    "============================================================"
)


summary_columns = [
    "zero_percentage",
    "max_feature",
    "mean_feature",
    "mean_nonzero_feature",
    "strongest_mel_center_hz",
    "min_mfe_db",
    "max_mfe_db",
]


print(
    result_df[
        summary_columns
    ]
    .describe()
    .to_string()
)


# ============================================================
# SUMMARY BY SPECIES
# ============================================================

print(
    "\n============================================================"
)

print(
    "SUMMARY BY SPECIES"
)

print(
    "============================================================"
)


species_summary = (
    result_df
    .groupby(
        "species"
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

        zero_min=(
            "zero_percentage",
            "min"
        ),

        zero_max=(
            "zero_percentage",
            "max"
        ),

        max_feature_mean=(
            "max_feature",
            "mean"
        ),

        mean_feature_mean=(
            "mean_feature",
            "mean"
        ),

        strongest_freq_median=(
            "strongest_mel_center_hz",
            "median"
        ),

        max_db_median=(
            "max_mfe_db",
            "median"
        ),
    )
)


print(
    species_summary.to_string()
)


# ============================================================
# EXTREME SPARSITY CHECK
# ============================================================

print(
    "\n============================================================"
)

print(
    "EXTREME SPARSITY CHECK"
)

print(
    "============================================================"
)


print(
    "Windows >= 99% zero:",
    np.sum(
        result_df[
            "zero_percentage"
        ] >= 99.0
    )
)


print(
    "Windows >= 95% zero:",
    np.sum(
        result_df[
            "zero_percentage"
        ] >= 95.0
    )
)


print(
    "Windows < 50% zero:",
    np.sum(
        result_df[
            "zero_percentage"
        ] < 50.0
    )
)


print(
    "Windows exactly 100% zero:",
    np.sum(
        result_df[
            "zero_percentage"
        ] == 100.0
    )
)


# ============================================================
# NOISE FLOOR SWEEP
# ============================================================

floor_columns = [
    "below_52db_percent",
    "below_55db_percent",
    "below_60db_percent",
    "below_65db_percent",
    "below_70db_percent",
]


print(
    "\n============================================================"
)

print(
    "NOISE FLOOR SWEEP"
)

print(
    "============================================================"
)


print(
    result_df[
        floor_columns
    ]
    .describe()
    .loc[
        [
            "mean",
            "25%",
            "50%",
            "75%",
            "min",
            "max",
        ]
    ]
    .to_string()
)


# ============================================================
# NOISE FLOOR SWEEP BY SPECIES
# ============================================================

print(
    "\n============================================================"
)

print(
    "NOISE FLOOR SWEEP BY SPECIES"
)

print(
    "MEDIAN % OF MFE BINS BELOW EACH FLOOR"
)

print(
    "============================================================"
)


floor_species_summary = (
    result_df
    .groupby(
        "species"
    )[
        floor_columns
    ]
    .median()
)


print(
    floor_species_summary.to_string()
)


# ============================================================
# COUNT VERY-SPARSE WINDOWS AT EACH FLOOR
# ============================================================
#
# Example:
#
# If 99% of raw MFE bins are below -60 dB,
# then a -60 dB hard floor would still produce
# an extremely sparse representation.
#
# ============================================================

print(
    "\n============================================================"
)

print(
    "VERY-SPARSE WINDOWS BY NOISE FLOOR"
)

print(
    "============================================================"
)


for floor_db in TEST_NOISE_FLOORS:

    column_name = (
        f"below_"
        f"{abs(int(floor_db))}"
        f"db_percent"
    )


    count_95 = np.sum(
        result_df[
            column_name
        ] >= 95.0
    )


    count_99 = np.sum(
        result_df[
            column_name
        ] >= 99.0
    )


    print(
        f"{int(floor_db)} dB:"
    )

    print(
        "  Windows >=95% below floor:",
        count_95
    )

    print(
        "  Windows >=99% below floor:",
        count_99
    )


# ============================================================
# FULLY BELOW CURRENT -52 dB FLOOR
# ============================================================

print(
    "\n============================================================"
)

print(
    "CURRENT -52 dB COMPLETE-LOSS CHECK"
)

print(
    "============================================================"
)


fully_below_52 = (
    result_df[
        "max_mfe_db"
    ]
    < -52.0
)


print(
    "Windows whose ENTIRE MFE is below -52 dB:",
    fully_below_52.sum()
)


print(
    "Percentage:",
    100.0
    * fully_below_52.mean()
)


# ============================================================
# SAVE RESULTS
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