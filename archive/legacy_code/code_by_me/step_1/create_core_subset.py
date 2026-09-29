import pandas as pd


# ============================================================
# LOAD METADATA
# ============================================================

csv_path = (
    "D:/VGU-27/Edge AI Project/"
    "data/neurips_2021_zenodo_0_0_1.csv"
)

df = pd.read_csv(csv_path)


# ============================================================
# DEFINE CORE CLASSES
# ============================================================

core_species = [
    "an arabiensis",
    "culex pipiens complex",
    "an funestus ss",
    "an squamosus",
]


# ============================================================
# STRICT CORE FILTER
# ============================================================
#
# Requirements:
#
# 1. mosquito
# 2. known target species
# 3. explicitly Single
# 4. native sample rate = 44.1 kHz
#
# ============================================================

core_df = df[
    (df["sound_type"] == "mosquito")
    & (df["species"].isin(core_species))
    & (df["plurality"] == "Single")
    & (df["sample_rate"] == 44100)
].copy()


print("============================================================")
print("CORE DATASET")
print("============================================================")

print("Total rows:", len(core_df))

print(
    "Unique IDs:",
    core_df["id"].nunique()
)

print(
    "Unique source recordings:",
    core_df["name"].nunique()
)


# ============================================================
# SUMMARY BY CLASS
# ============================================================

summary = (
    core_df
    .groupby("species")
    .agg(
        clips=("id", "count"),
        source_recordings=("name", "nunique"),
        total_duration_seconds=("length", "sum"),
        min_duration=("length", "min"),
        median_duration=("length", "median"),
        max_duration=("length", "max"),
    )
)

print("\n============================================================")
print("BY CLASS")
print("============================================================")

print(
    summary.to_string()
)


# ============================================================
# CHECK 1-SECOND WINDOW FEASIBILITY
# ============================================================

print("\n============================================================")
print("1-SECOND WINDOW CHECK")
print("============================================================")

duration_check = (
    core_df
    .assign(
        at_least_1_second=core_df["length"] >= 1.0
    )
    .groupby("species")
    ["at_least_1_second"]
    .agg(["sum", "count"])
)

duration_check["percentage"] = (
    100
    * duration_check["sum"]
    / duration_check["count"]
)

print(
    duration_check.to_string()
)


# ============================================================
# SAVE CORE METADATA
# ============================================================

output_path = (
    "D:/VGU-27/Edge AI Project/"
    "data/core_single_4species_metadata.csv"
)

core_df.to_csv(
    output_path,
    index=False
)

print("\nSaved to:")
print(output_path)