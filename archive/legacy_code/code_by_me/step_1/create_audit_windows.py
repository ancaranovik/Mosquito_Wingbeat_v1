import pandas as pd
from pathlib import Path


# ============================================================
# STEP 1C.6B
# CREATE DETERMINISTIC 1-SECOND AUDIT WINDOWS
# ============================================================


# ============================================================
# PATHS
# ============================================================

metadata_path = Path(
    "D:/VGU-27/Edge AI Project/"
    "data/core_single_4species_frozen_split.csv"
)

audit_path = Path(
    "D:/VGU-27/Edge AI Project/"
    "data/core_audio_audit.csv"
)

output_path = Path(
    "D:/VGU-27/Edge AI Project/"
    "data/core_1s_audit_windows.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

metadata_df = pd.read_csv(
    metadata_path
)

audio_audit_df = pd.read_csv(
    audit_path
)


# ============================================================
# MERGE AUDIO INFORMATION
# ============================================================
#
# We use the ACTUAL number of WAV frames from the audio audit,
# rather than metadata duration.
#
# This guarantees the 1-second window stays inside the WAV.
# ============================================================

df = metadata_df.merge(
    audio_audit_df[
        [
            "id",
            "frames",
            "duration_wav",
        ]
    ],
    on="id",
    how="left"
)


print("============================================================")
print("STEP 1C.6B - 1 SECOND AUDIT WINDOWS")
print("============================================================")

print("Core clips:", len(df))


# ============================================================
# WINDOW CONFIGURATION
# ============================================================

WINDOW_SECONDS = 1.0

EXPECTED_SAMPLE_RATE = 44100

WINDOW_SAMPLES = int(
    WINDOW_SECONDS
    * EXPECTED_SAMPLE_RATE
)


print(
    "Window length:",
    WINDOW_SECONDS,
    "second"
)

print(
    "Window samples:",
    WINDOW_SAMPLES
)


# ============================================================
# CREATE CENTERED WINDOW
# ============================================================
#
# Example:
#
# total WAV:
#
# |---------------------------------------|
#
# centered audit window:
#
#                  |----1 sec----|
#
#
# Integer sample indices are used so every window is
# EXACTLY 44100 samples.
# ============================================================

df["frames"] = (
    df["frames"]
    .astype(int)
)


df["start_sample"] = (
    (
        df["frames"]
        - WINDOW_SAMPLES
    )
    // 2
)


df["end_sample"] = (
    df["start_sample"]
    + WINDOW_SAMPLES
)


# Also store times for easier inspection
df["start_second"] = (
    df["start_sample"]
    / EXPECTED_SAMPLE_RATE
)

df["end_second"] = (
    df["end_sample"]
    / EXPECTED_SAMPLE_RATE
)


# ============================================================
# VALIDATION
# ============================================================

df["window_samples"] = (
    df["end_sample"]
    - df["start_sample"]
)


invalid_start = (
    df["start_sample"] < 0
)

invalid_end = (
    df["end_sample"]
    > df["frames"]
)

wrong_window_length = (
    df["window_samples"]
    != WINDOW_SAMPLES
)

wrong_sample_rate = (
    df["sample_rate"]
    != EXPECTED_SAMPLE_RATE
)


print("\n============================================================")
print("WINDOW VALIDATION")
print("============================================================")

print(
    "Invalid start positions:",
    invalid_start.sum()
)

print(
    "Invalid end positions:",
    invalid_end.sum()
)

print(
    "Wrong window lengths:",
    wrong_window_length.sum()
)

print(
    "Wrong sample rates:",
    wrong_sample_rate.sum()
)


# ============================================================
# SUMMARY BY CLASS
# ============================================================

print("\n============================================================")
print("WINDOWS BY CLASS")
print("============================================================")

print(
    df["species"]
    .value_counts()
    .to_string()
)


# ============================================================
# SUMMARY BY SPLIT
# ============================================================

print("\n============================================================")
print("WINDOWS BY SPLIT")
print("============================================================")

print(
    df["split"]
    .value_counts()
    .to_string()
)


# ============================================================
# CLASS × SPLIT
# ============================================================

print("\n============================================================")
print("WINDOWS BY CLASS AND SPLIT")
print("============================================================")

class_split = pd.crosstab(
    df["species"],
    df["split"]
)

print(
    class_split.to_string()
)


# ============================================================
# SHOW EXAMPLES
# ============================================================

print("\n============================================================")
print("EXAMPLE WINDOWS")
print("============================================================")

print(
    df[
        [
            "id",
            "species",
            "split",
            "name",
            "duration_wav",
            "start_second",
            "end_second",
            "start_sample",
            "end_sample",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# FINAL CHECK
# ============================================================

all_valid = (
    invalid_start.sum() == 0
    and invalid_end.sum() == 0
    and wrong_window_length.sum() == 0
    and wrong_sample_rate.sum() == 0
)


print("\n============================================================")
print("FINAL CHECK")
print("============================================================")

print(
    "Total audit windows:",
    len(df)
)

print(
    "All windows valid:",
    all_valid
)


# ============================================================
# SAVE MANIFEST
# ============================================================

columns_to_save = [
    "id",
    "species",
    "split",
    "name",
    "sample_rate",
    "frames",
    "duration_wav",
    "start_sample",
    "end_sample",
    "start_second",
    "end_second",
    "window_samples",
]

df[
    columns_to_save
].to_csv(
    output_path,
    index=False
)


print("\nSaved audit-window manifest:")
print(output_path)