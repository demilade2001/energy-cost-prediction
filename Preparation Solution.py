# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 16:23:24 2025

@author: Demilade
"""

from pathlib import Path
import pandas as pd


def ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected a file at {path}")


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


data_dir = Path.home() / "Downloads"
output_dir = Path.home() / "Desktop" / "Data Analytics Assessment"
output_dir.mkdir(parents=True, exist_ok=True)

demand_path = data_dir / "demand.csv"
gen_path = data_dir / "generation_costs.csv"
plants_path = data_dir / "plants.csv"
merged_output_path = output_dir / "cleaned merged data.csv"

for path in (demand_path, gen_path, plants_path):
    ensure_file(path)


demand_df = pd.read_csv(demand_path)
gen_df = pd.read_csv(gen_path)
plants_df = pd.read_csv(plants_path)

require_columns(demand_df, ["DF_region", "Demand ID"], "demand.csv")
require_columns(gen_df, ["Demand ID", "Plant ID", "Cost_USD_per_MWh"], "generation_costs.csv")
require_columns(plants_df, ["Plant ID", "Region"], "plants.csv")

# Clean demand
if "DF_region" in demand_df.columns:
    demand_df["DF_region"] = demand_df["DF_region"].fillna("NAM")

demand_cols = [
    col for col in demand_df.columns
    if col.startswith("DF") and col != "DF_region"
]
if demand_cols:
    demand_df[demand_cols] = demand_df[demand_cols].fillna(
        demand_df[demand_cols].mean(numeric_only=True)
    )
else:
    print("Warning: no DF_* demand columns were found.")

# Clean generation cost
if gen_df["Cost_USD_per_MWh"].dtype.kind not in "biufc":
    gen_df["Cost_USD_per_MWh"] = pd.to_numeric(
        gen_df["Cost_USD_per_MWh"], errors="coerce"
    )
gen_df["Cost_USD_per_MWh"] = gen_df["Cost_USD_per_MWh"].fillna(
    gen_df["Cost_USD_per_MWh"].median()
)

# Clean plant data
plants_df["Region"] = plants_df["Region"].fillna("NAM")

# Merge datasets
merged_df = gen_df.merge(demand_df, on="Demand ID", how="left")
merged_df = merged_df.merge(plants_df, on="Plant ID", how="left")

# Reorder columns with strings first
string_cols = merged_df.select_dtypes(include="object").columns.tolist()
other_cols = [col for col in merged_df.columns if col not in string_cols]
merged_df = merged_df[string_cols + other_cols]

# Move cost to the end
if "Cost_USD_per_MWh" in merged_df.columns:
    cols = [c for c in merged_df.columns if c != "Cost_USD_per_MWh"] + [
        "Cost_USD_per_MWh"
    ]
    merged_df = merged_df[cols]
else:
    print("Warning: Cost_USD_per_MWh column is missing from merged data.")

required_merge_cols = ["Plant ID", "Cost_USD_per_MWh"]
if all(col in merged_df.columns for col in required_merge_cols):
    plant_stats = (
        merged_df.groupby("Plant ID")["Cost_USD_per_MWh"]
        .mean()
        .reset_index(name="mean_cost")
    )
    cutoff = plant_stats["mean_cost"].quantile(0.75)
    under_plants = plant_stats.loc[
        plant_stats["mean_cost"] >= cutoff, "Plant ID"
    ]
    merged_df = merged_df.loc[~merged_df["Plant ID"].isin(under_plants)].copy()
else:
    print("Skipping plant cost filtering because required columns are missing.")

merged_df.info()
merged_df.to_csv(merged_output_path, index=False)
print(f"Saved cleaned merged data to: {merged_output_path}")