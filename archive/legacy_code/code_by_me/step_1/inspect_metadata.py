import pandas as pd

csv_path = (
    "D:/VGU-27/Edge AI Project/"
    "data/neurips_2021_zenodo_0_0_1.csv"
)

df = pd.read_csv(csv_path)

print("============================================================")
print("METADATA OVERVIEW")
print("============================================================")

print("Shape:", df.shape)

print("\nColumns:")
for column in df.columns:
    print("-", column)

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\n============================================================")
print("BASIC DISTRIBUTIONS")
print("============================================================")

print("\nSound types:")
print(df["sound_type"].value_counts(dropna=False))

print("\nSpecies:")
print(df["species"].value_counts(dropna=False))

print("\nSample rates:")
print(df["sample_rate"].value_counts(dropna=False))

print("\nUnique source names:")
print(df["name"].nunique())

print("\nRows per source name - top 20:")
print(df["name"].value_counts().head(20))


print("\n============================================================")
print("CHECK OUR TEST FILE: ID 220025")
print("============================================================")

print(
    df[
        df["id"] == 220025
    ].to_string(index=False)
)

print("\n============================================================")
print("SPECIES SOURCE SUMMARY")
print("============================================================")

mosquito_df = df[df["sound_type"] == "mosquito"].copy()

species_summary = (
    mosquito_df
    .groupby("species")
    .agg(
        clips=("id", "count"),
        source_recordings=("name", "nunique"),
        total_duration_seconds=("length", "sum"),
        sample_rates=("sample_rate", lambda x: sorted(x.unique().tolist()))
    )
    .sort_values(
        by=["source_recordings", "clips"],
        ascending=False
    )
)

print(
    species_summary.to_string()
)


print("\n============================================================")
print("SOURCE COUNTS BY SAMPLE RATE")
print("============================================================")

source_rate_summary = (
    mosquito_df
    .groupby(["species", "sample_rate"])
    .agg(
        clips=("id", "count"),
        source_recordings=("name", "nunique"),
        duration_seconds=("length", "sum")
    )
)

print(
    source_rate_summary.to_string()
)

print("\n============================================================")
print("SOURCE GROUP CONSISTENCY")
print("============================================================")

mosquito_df = df[
    df["sound_type"] == "mosquito"
].copy()


# ------------------------------------------------------------
# Check whether one source name is associated with
# multiple species.
# ------------------------------------------------------------

species_per_source = (
    mosquito_df
    .groupby("name")["species"]
    .nunique(dropna=True)
)

multi_species_sources = (
    species_per_source[
        species_per_source > 1
    ]
)


print(
    "Total mosquito source names:",
    mosquito_df["name"].nunique()
)

print(
    "Sources containing >1 species:",
    len(multi_species_sources)
)


if len(multi_species_sources) > 0:

    print("\nExamples:")

    problem_names = (
        multi_species_sources
        .head(20)
        .index
    )

    print(
        mosquito_df[
            mosquito_df["name"].isin(
                problem_names
            )
        ][
            [
                "name",
                "species",
                "sample_rate",
                "device_type",
                "place"
            ]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )


# ------------------------------------------------------------
# Check whether one source name appears with
# multiple sample rates.
# ------------------------------------------------------------

rates_per_source = (
    mosquito_df
    .groupby("name")["sample_rate"]
    .nunique()
)

multi_rate_sources = (
    rates_per_source[
        rates_per_source > 1
    ]
)

print(
    "\nSources containing >1 sample rate:",
    len(multi_rate_sources)
)

print("\n============================================================")
print("SINGLE-MOSQUITO ASSUMPTION AUDIT")
print("============================================================")

# Only rows labelled as mosquito
mosquito_df = df[
    df["sound_type"] == "mosquito"
].copy()


# ------------------------------------------------------------
# 1. Are IDs unique?
# ------------------------------------------------------------

print(
    "Total mosquito rows:",
    len(mosquito_df)
)

print(
    "Unique mosquito IDs:",
    mosquito_df["id"].nunique()
)

print(
    "Duplicated mosquito IDs:",
    mosquito_df["id"].duplicated().sum()
)


# ------------------------------------------------------------
# 2. How is plurality labelled?
# ------------------------------------------------------------

print("\nPlurality distribution:")

print(
    mosquito_df["plurality"]
    .value_counts(dropna=False)
    .to_string()
)


# ------------------------------------------------------------
# 3. Plurality by species
# ------------------------------------------------------------

print("\nPlurality by species:")

plurality_species = pd.crosstab(
    mosquito_df["species"],
    mosquito_df["plurality"],
    dropna=False
)

print(
    plurality_species.to_string()
)


# ------------------------------------------------------------
# 4. How many labelled mosquito clips
#    have no species?
# ------------------------------------------------------------

missing_species = mosquito_df[
    mosquito_df["species"].isna()
]

print(
    "\nMosquito rows with missing species:",
    len(missing_species)
)


# ------------------------------------------------------------
# 5. Clip duration statistics
# ------------------------------------------------------------

print("\nMosquito clip duration statistics:")

print(
    mosquito_df["length"].describe()
)

import numpy as np
print("\n============================================================")
print("STRICT SINGLE-MOSQUITO DATASET")
print("============================================================")

single_df = df[
    (df["sound_type"] == "mosquito")
    & (df["species"].notna())
    & (df["plurality"] == "Single")
].copy()


print(
    "Strict single-mosquito rows:",
    len(single_df)
)

print(
    "Unique IDs:",
    single_df["id"].nunique()
)

print(
    "Unique source names:",
    single_df["name"].nunique()
)


# ============================================================
# SPECIES SUMMARY AFTER STRICT FILTER
# ============================================================

single_summary = (
    single_df
    .groupby("species")
    .agg(
        clips=("id", "count"),
        source_recordings=("name", "nunique"),
        total_duration_seconds=("length", "sum"),
        median_duration=("length", "median"),
        min_duration=("length", "min"),
        sample_rates=(
            "sample_rate",
            lambda x: sorted(x.unique().tolist())
        )
    )
    .sort_values(
        ["source_recordings", "clips"],
        ascending=False
    )
)


print("\nSpecies summary:")

print(
    single_summary.to_string()
)


# ============================================================
# SAMPLE RATE DISTRIBUTION AFTER STRICT FILTER
# ============================================================

print("\n============================================================")
print("STRICT SINGLE BY SAMPLE RATE")
print("============================================================")

single_rate_summary = (
    single_df
    .groupby(
        ["species", "sample_rate"]
    )
    .agg(
        clips=("id", "count"),
        source_recordings=("name", "nunique"),
        duration_seconds=("length", "sum")
    )
)

print(
    single_rate_summary.to_string()
)


# ============================================================
# DURATION CHECK
# ============================================================

print("\n============================================================")
print("STRICT SINGLE DURATION CHECK")
print("============================================================")

print(
    "Clips >= 1 second:",
    np.sum(
        single_df["length"] >= 1.0
    )
)

print(
    "Clips < 1 second:",
    np.sum(
        single_df["length"] < 1.0
    )
)

print(
    "Percentage >= 1 second:",
    100
    * np.mean(
        single_df["length"] >= 1.0
    )
)

