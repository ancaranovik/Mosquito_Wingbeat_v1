from pathlib import Path

import numpy as np
import pandas as pd

from scipy.io import wavfile
from scipy import signal


# ============================================================
# STEP 1C.7C
# DIAGNOSE WHY -52 dB PRODUCES EXTREME SPARSITY
# ============================================================


# ============================================================
# PATHS
# ============================================================

manifest_path = Path(
    "D:/VGU-27/Edge AI Project/"
    "data/core_1s_audit_windows.csv"
)

mfe_summary_path = Path(
    "D:/VGU-27/Edge AI Project/"
    "data/train_mfe_audit_summary.csv"
)

audio_dir = Path(
    "D:/VGU-27/Edge AI Project/"
    "data/audio"
)

output_path = Path(
    "D:/VGU-27/Edge AI Project/"
    "data/train_signal_floor_diagnostic.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

SOURCE_SAMPLE_RATE = 44100
TARGET_SAMPLE_RATE = 16000

EPSILON = 1e-12


# ============================================================
# LOAD DATA
# ============================================================

manifest = pd.read_csv(
    manifest_path
)

mfe_summary = pd.read_csv(
    mfe_summary_path
)


# Only train
train_manifest = manifest[
    manifest["split"] == "train"
].copy()


# Merge with results from Step 1C.7B
df = train_manifest.merge(
    mfe_summary,
    on=[
        "id",
        "species",
        "name",
    ],
    how="inner"
)


print(
    "============================================================"
)

print(
    "STEP 1C.7C - SIGNAL vs NOISE FLOOR DIAGNOSTIC"
)

print(
    "============================================================"
)

print(
    "Train windows:",
    len(df)
)


# ============================================================
# CALCULATE SIGNAL-LEVEL STATISTICS
# ============================================================

records = []


for count, (_, row) in enumerate(
    df.iterrows(),
    start=1
):

    audio_path = (
        audio_dir
        / f"{int(row['id'])}.wav"
    )


    # --------------------------------------------------------
    # LOAD WAV
    # --------------------------------------------------------

    sample_rate, audio = wavfile.read(
        audio_path
    )


    if sample_rate != SOURCE_SAMPLE_RATE:

        raise ValueError(
            f"Unexpected sample rate "
            f"for ID {row['id']}"
        )


    # PCM24 loaded as left-justified int32.
    audio_float = (
        audio.astype(np.float32)
        / 2147483648.0
    )


    # --------------------------------------------------------
    # EXTRACT SAME 1-SECOND WINDOW
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


    # ========================================================
    # RAW WINDOW LEVEL
    # ========================================================

    raw_rms = np.sqrt(
        np.mean(
            np.square(window)
        )
    )

    raw_rms_dbfs = (
        20.0
        * np.log10(
            raw_rms
            + EPSILON
        )
    )

    raw_peak = np.max(
        np.abs(window)
    )


    # ========================================================
    # RESAMPLE TO 16 kHz
    # ========================================================

    resampled = signal.resample_poly(
        window,
        up=160,
        down=441
    )


    resampled_rms = np.sqrt(
        np.mean(
            np.square(resampled)
        )
    )

    resampled_rms_dbfs = (
        20.0
        * np.log10(
            resampled_rms
            + EPSILON
        )
    )


    # ========================================================
    # PRE-EMPHASIS
    # ========================================================

    previous = np.roll(
        resampled,
        1
    )

    emphasized = (
        resampled
        - 0.98 * previous
    )


    emphasized_rms = np.sqrt(
        np.mean(
            np.square(emphasized)
        )
    )

    emphasized_rms_dbfs = (
        20.0
        * np.log10(
            emphasized_rms
            + EPSILON
        )
    )


    # ========================================================
    # COMPLETE LOSS AT -52 dB?
    # ========================================================
    #
    # max_mfe_db < -52 means:
    #
    # every one of the 3960 MFE bins is below
    # the current noise floor.
    #
    # ========================================================

    complete_loss_52 = (
        row["max_mfe_db"]
        < -52.0
    )


    records.append(
        {
            "id":
                int(row["id"]),

            "species":
                row["species"],

            "name":
                row["name"],

            "raw_rms_dbfs":
                float(raw_rms_dbfs),

            "raw_peak":
                float(raw_peak),

            "resampled_rms_dbfs":
                float(resampled_rms_dbfs),

            "preemphasized_rms_dbfs":
                float(
                    emphasized_rms_dbfs
                ),

            "max_mfe_db":
                float(
                    row["max_mfe_db"]
                ),

            "zero_percentage":
                float(
                    row["zero_percentage"]
                ),

            "strongest_mel_center_hz":
                float(
                    row[
                        "strongest_mel_center_hz"
                    ]
                ),

            "complete_loss_52":
                complete_loss_52,
        }
    )


    if (
        count % 100 == 0
        or count == len(df)
    ):

        print(
            f"Processed "
            f"{count}/{len(df)}"
        )


# ============================================================
# RESULT TABLE
# ============================================================

result_df = pd.DataFrame(
    records
)


# ============================================================
# 1. COMPLETE LOSS SUMMARY
# ============================================================

print(
    "\n============================================================"
)

print(
    "COMPLETE LOSS AT -52 dB"
)

print(
    "============================================================"
)


print(
    result_df[
        "complete_loss_52"
    ]
    .value_counts()
)


print(
    "\nPercentage complete loss:"
)

print(
    100.0
    * result_df[
        "complete_loss_52"
    ]
    .mean()
)


# ============================================================
# 2. SIGNAL LEVEL:
#    LOST vs SURVIVED
# ============================================================

print(
    "\n============================================================"
)

print(
    "SIGNAL LEVEL: LOST vs SURVIVED"
)

print(
    "============================================================"
)


level_summary = (
    result_df
    .groupby(
        "complete_loss_52"
    )
    [
        [
            "raw_rms_dbfs",
            "resampled_rms_dbfs",
            "preemphasized_rms_dbfs",
            "max_mfe_db",
        ]
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
)


print(
    level_summary.to_string()
)


# ============================================================
# 3. COMPLETE LOSS BY SPECIES
# ============================================================

print(
    "\n============================================================"
)

print(
    "COMPLETE LOSS BY SPECIES"
)

print(
    "============================================================"
)


species_loss = (
    result_df
    .groupby(
        "species"
    )
    .agg(
        windows=(
            "id",
            "count"
        ),

        complete_loss_windows=(
            "complete_loss_52",
            "sum"
        ),

        complete_loss_percent=(
            "complete_loss_52",
            "mean"
        ),

        median_raw_rms_dbfs=(
            "raw_rms_dbfs",
            "median"
        ),

        median_max_mfe_db=(
            "max_mfe_db",
            "median"
        ),
    )
)


species_loss[
    "complete_loss_percent"
] *= 100.0


print(
    species_loss.to_string()
)


# ============================================================
# 4. CORRELATION:
#    SIGNAL LEVEL vs MFE STRENGTH
# ============================================================

print(
    "\n============================================================"
)

print(
    "SIGNAL LEVEL CORRELATION"
)

print(
    "============================================================"
)


raw_correlation = (
    result_df[
        "raw_rms_dbfs"
    ]
    .corr(
        result_df[
            "max_mfe_db"
        ]
    )
)


preemphasis_correlation = (
    result_df[
        "preemphasized_rms_dbfs"
    ]
    .corr(
        result_df[
            "max_mfe_db"
        ]
    )
)


print(
    "Correlation raw RMS dBFS vs max MFE dB:",
    raw_correlation
)

print(
    "Correlation pre-emphasized RMS dBFS vs max MFE dB:",
    preemphasis_correlation
)


# ============================================================
# 5. STRONGEST MEL FREQUENCY DISTRIBUTION
# ============================================================
#
# We do not assume which range is mosquito wingbeat here.
#
# We only ask:
#
# "Where does the strongest MFE band tend to occur?"
#
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
        "strongest_mel_center_hz"
    ]
    .apply(
        frequency_region
    )
)


print(
    "\n============================================================"
)

print(
    "STRONGEST MEL FREQUENCY REGIONS"
)

print(
    "============================================================"
)


frequency_counts = (
    result_df[
        "strongest_frequency_region"
    ]
    .value_counts()
)


print(
    frequency_counts.to_string()
)


print(
    "\nPercent:"
)

print(
    (
        100.0
        * frequency_counts
        / len(result_df)
    )
    .to_string()
)


# ============================================================
# 6. FREQUENCY REGION BY SPECIES
# ============================================================

print(
    "\n============================================================"
)

print(
    "STRONGEST FREQUENCY REGION BY SPECIES"
)

print(
    "============================================================"
)


species_frequency = pd.crosstab(
    result_df["species"],
    result_df[
        "strongest_frequency_region"
    ],
    normalize="index"
) * 100.0


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