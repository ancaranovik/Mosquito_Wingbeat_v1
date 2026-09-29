from pathlib import Path
import wave
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

metadata_path = Path(
    "D:/VGU-27/Edge AI Project/"
    "data/core_single_4species_frozen_split.csv"
)

audio_dir = Path(
    "D:/VGU-27/Edge AI Project/"
    "data/audio"
)


# ============================================================
# LOAD FROZEN METADATA
# ============================================================

df = pd.read_csv(metadata_path)


print("============================================================")
print("STEP 1C.6A - CORE AUDIO AUDIT")
print("============================================================")

print("Metadata rows:", len(df))


# ============================================================
# AUDIT EVERY WAV FILE
# ============================================================

records = []


for _, row in df.iterrows():

    audio_path = (
        audio_dir
        / f"{int(row['id'])}.wav"
    )

    record = {
        "id": int(row["id"]),
        "species": row["species"],
        "split": row["split"],
        "exists": audio_path.exists(),
        "sample_rate": np.nan,
        "channels": np.nan,
        "sample_width_bytes": np.nan,
        "frames": np.nan,
        "duration_wav": np.nan,
        "duration_metadata": row["length"],
        "duration_difference": np.nan,
    }


    # --------------------------------------------------------
    # Missing file
    # --------------------------------------------------------

    if not audio_path.exists():

        records.append(record)
        continue


    # --------------------------------------------------------
    # Read WAV HEADER only
    #
    # This is fast and does not load the whole audio signal.
    # --------------------------------------------------------

    try:

        with wave.open(
            str(audio_path),
            "rb"
        ) as wav:

            channels = (
                wav.getnchannels()
            )

            sample_width = (
                wav.getsampwidth()
            )

            sample_rate = (
                wav.getframerate()
            )

            num_frames = (
                wav.getnframes()
            )


        duration_wav = (
            num_frames
            / sample_rate
        )


        record[
            "sample_rate"
        ] = sample_rate

        record[
            "channels"
        ] = channels

        record[
            "sample_width_bytes"
        ] = sample_width

        record[
            "frames"
        ] = num_frames

        record[
            "duration_wav"
        ] = duration_wav

        record[
            "duration_difference"
        ] = abs(
            duration_wav
            - row["length"]
        )


    except Exception as error:

        print(
            "ERROR reading",
            audio_path,
            error
        )


    records.append(record)


audit_df = pd.DataFrame(
    records
)


# ============================================================
# BASIC RESULTS
# ============================================================

print("\n============================================================")
print("FILE EXISTENCE")
print("============================================================")

print(
    audit_df["exists"]
    .value_counts(dropna=False)
)


print("\n============================================================")
print("SAMPLE RATES")
print("============================================================")

print(
    audit_df["sample_rate"]
    .value_counts(dropna=False)
)


print("\n============================================================")
print("CHANNELS")
print("============================================================")

print(
    audit_df["channels"]
    .value_counts(dropna=False)
)


print("\n============================================================")
print("SAMPLE WIDTH")
print("============================================================")

print(
    audit_df["sample_width_bytes"]
    .value_counts(dropna=False)
)


# ============================================================
# DURATION CONSISTENCY
# ============================================================

print("\n============================================================")
print("DURATION DIFFERENCE")
print("============================================================")

print(
    audit_df[
        "duration_difference"
    ].describe()
)


# Files whose WAV duration differs from metadata
# by more than 10 ms.
duration_problem = audit_df[
    audit_df[
        "duration_difference"
    ] > 0.01
]


print(
    "\nFiles differing by >10 ms:",
    len(duration_problem)
)


if len(duration_problem) > 0:

    print(
        duration_problem[
            [
                "id",
                "species",
                "duration_metadata",
                "duration_wav",
                "duration_difference",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# FINAL AUDIT CHECKS
# ============================================================

missing_files = np.sum(
    audit_df["exists"] == False
)

wrong_sample_rate = np.sum(
    audit_df["sample_rate"] != 44100
)

non_mono = np.sum(
    audit_df["channels"] != 1
)


print("\n============================================================")
print("FINAL CHECK")
print("============================================================")

print(
    "Missing files:",
    missing_files
)

print(
    "Files not 44.1 kHz:",
    wrong_sample_rate
)

print(
    "Non-mono files:",
    non_mono
)

print(
    "Duration mismatch >10 ms:",
    len(duration_problem)
)


# ============================================================
# SAVE AUDIT
# ============================================================

output_path = Path(
    "D:/VGU-27/Edge AI Project/"
    "data/core_audio_audit.csv"
)

audit_df.to_csv(
    output_path,
    index=False
)

print("\nSaved audit:")
print(output_path)