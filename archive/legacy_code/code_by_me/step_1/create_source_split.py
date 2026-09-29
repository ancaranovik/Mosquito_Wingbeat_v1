import pandas as pd
import numpy as np


# ============================================================
# LOAD CORE METADATA
# ============================================================

input_path = (
    "D:/VGU-27/Edge AI Project/"
    "data/core_single_4species_metadata.csv"
)

df = pd.read_csv(input_path)


print("============================================================")
print("STEP 1C.5 - SOURCE GROUP SPLIT")
print("============================================================")

print("Rows:", len(df))
print("Unique sources:", df["name"].nunique())


# ============================================================
# CHECK WHETHER ONE SOURCE CONTAINS MULTIPLE CORE SPECIES
# ============================================================

species_per_source = (
    df
    .groupby("name")["species"]
    .nunique()
)

multi_species_sources = (
    species_per_source[
        species_per_source > 1
    ]
)


print(
    "Sources containing >1 CORE species:",
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
        df[
            df["name"].isin(problem_names)
        ][
            ["name", "species", "id"]
        ]
        .sort_values(["name", "species"])
        .to_string(index=False)
    )

    raise SystemExit(
        "\nSTOP: Need to handle multi-species source groups before splitting."
    )


print(
    "\nPASS: Each source recording belongs "
    "to only one Core species."
)

# ============================================================
# SPLIT CONFIGURATION
# ============================================================
#
# Split by SOURCE RECORDING, not by individual clips.
#
# 70% train
# 15% validation
# 15% test
#
# The random seed makes the split reproducible.
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# ASSIGN SOURCES TO SPLITS
# ============================================================

source_split_records = []


for species in sorted(
    df["species"].unique()
):

    species_sources = (
        df[
            df["species"] == species
        ]["name"]
        .drop_duplicates()
        .to_numpy()
    )

    # Deterministic shuffle
    rng.shuffle(
        species_sources
    )

    n_sources = len(
        species_sources
    )


    # --------------------------------------------------------
    # Number of sources in each split
    # --------------------------------------------------------

    n_train = int(
        np.floor(
            n_sources
            * TRAIN_RATIO
        )
    )

    n_val = int(
        np.floor(
            n_sources
            * VAL_RATIO
        )
    )

    n_test = (
        n_sources
        - n_train
        - n_val
    )


    train_sources = (
        species_sources[
            :n_train
        ]
    )

    val_sources = (
        species_sources[
            n_train:
            n_train + n_val
        ]
    )

    test_sources = (
        species_sources[
            n_train + n_val:
        ]
    )


    for source in train_sources:

        source_split_records.append(
            {
                "name": source,
                "species": species,
                "split": "train",
            }
        )


    for source in val_sources:

        source_split_records.append(
            {
                "name": source,
                "species": species,
                "split": "validation",
            }
        )


    for source in test_sources:

        source_split_records.append(
            {
                "name": source,
                "species": species,
                "split": "test",
            }
        )


# ============================================================
# CREATE SOURCE SPLIT TABLE
# ============================================================

source_split_df = pd.DataFrame(
    source_split_records
)


# ============================================================
# MERGE SPLIT BACK INTO CLIP METADATA
# ============================================================

split_df = df.merge(
    source_split_df[
        ["name", "split"]
    ],
    on="name",
    how="left"
)


# ============================================================
# VERIFY NO SOURCE LEAKAGE
# ============================================================

splits_per_source = (
    split_df
    .groupby("name")["split"]
    .nunique()
)

leaking_sources = (
    splits_per_source[
        splits_per_source > 1
    ]
)


print("\n============================================================")
print("LEAKAGE CHECK")
print("============================================================")

print(
    "Sources appearing in >1 split:",
    len(leaking_sources)
)


# ============================================================
# SOURCE COUNTS BY CLASS AND SPLIT
# ============================================================

print("\n============================================================")
print("SOURCE COUNTS")
print("============================================================")

source_summary = (
    source_split_df
    .groupby(
        ["species", "split"]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

print(
    source_summary.to_string()
)


# ============================================================
# CLIP COUNTS BY CLASS AND SPLIT
# ============================================================

print("\n============================================================")
print("CLIP COUNTS")
print("============================================================")

clip_summary = (
    split_df
    .groupby(
        ["species", "split"]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

print(
    clip_summary.to_string()
)


# ============================================================
# TOTAL COUNTS
# ============================================================

print("\n============================================================")
print("TOTAL SPLIT COUNTS")
print("============================================================")

print(
    split_df["split"]
    .value_counts()
)


# ============================================================
# SAVE FROZEN SPLIT
# ============================================================

output_path = (
    "D:/VGU-27/Edge AI Project/"
    "data/core_single_4species_frozen_split.csv"
)

split_df.to_csv(
    output_path,
    index=False
)


print("\nSaved frozen split to:")
print(output_path)