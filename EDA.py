# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 14:15:29 2025

@author: Demilade
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error
from scipy.stats import f_oneway

"""## IMPORTING THE MERGED DATASET"""
data_path = Path.home() / "Desktop" / "Data Analytics Assessment" / "cleaned merged data.csv"
if not data_path.exists():
    raise FileNotFoundError(f"Input CSV not found: {data_path}")

merged_df = pd.read_csv(data_path)
# print(merged_df.info())

"""## CORRELATION BETWEEN DEMAND FEATURES AND GENERATION COST"""
plt.figure(figsize=(17, 10))
demand_cost_cols = [
    col for col in merged_df.columns
    if col.startswith("DF") or col == "Cost_USD_per_MWh"
]
correlation_df = merged_df[demand_cost_cols].select_dtypes(include=[np.number])
if correlation_df.empty:
    raise ValueError("No numeric demand feature columns found for correlation analysis.")
corr_matrix = correlation_df.corr()
sns.heatmap(corr_matrix, cmap="Greens", annot=True, fmt=".2f", square=True,
    linewidths=0.2, annot_kws={"size": 12})
plt.title("Correlation Heatmap (Demand Features and Generation Cost)", 
          fontsize=16, fontweight='bold', pad=14)
plt.xticks(rotation=25, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



"""# COST PATTERN ANALYSIS"""
"""## HISTOGRAM OF THE SPREAD OF GENERATION COST"""
plt.figure(figsize=(10,5))
sns.histplot(data=merged_df, x="Cost_USD_per_MWh", bins=55, kde=True)
plt.title("Distribution of Generation Cost (USD/MWh)")
plt.show()

# CREATING A COPY OF MERGED_DF DATASET FOR GENERATION COST ANALYSIS
merged1 = merged_df.copy()

# BUILD AN OVERALL DEMAND INDEX
merged1["Demand_Index"] = merged1.iloc[:, 6:18].mean(axis=1)

# CREATE DEMAND CONTEXT BUCKETS
merged1["Demand_Context"] = pd.qcut(merged1["Demand_Index"], q=3, labels=[
    "Low Demand", "Medium Demand", "High Demand"])

print(merged1[["Demand_Index", "Demand_Context"]].head())

"""## COST-EFFECTIVENESS BY PLANT TYPE ACROSS DEMAND CONTEXTS"""
cost_by_type_context = merged1.groupby(
    ["Plant Type", "Demand_Context"])["Cost_USD_per_MWh"].mean().reset_index()
print("\nAverage Cost by Plant Type & Demand Context:")
print(cost_by_type_context)
plt.figure(figsize=(12,6))
sns.barplot(data=cost_by_type_context, x="Plant Type",
            y="Cost_USD_per_MWh", hue="Demand_Context")
plt.title("Cost-Effectiveness of Plant Types Under Different Demand Contexts")
plt.ylabel("Avg Cost (USD/MWh)")
plt.show()

"""## COST-EFFECTIVENESS BY REGION ACROSS DEMAND CONTEXTS"""
cost_by_region_context = merged1.groupby(
    ["DF_region", "Demand_Context"])["Cost_USD_per_MWh"].mean().reset_index()
print("\nAverage Cost by Region & Demand Context:")
print(cost_by_region_context)
plt.figure(figsize=(14,6))
sns.barplot(data=cost_by_region_context, x="DF_region", 
            y="Cost_USD_per_MWh", hue="Demand_Context")
plt.title("Regional Cost-Effectiveness Under Different Demand Contexts")
plt.ylabel("Avg Cost (USD/MWh)")
plt.xticks(rotation=45)
plt.show()

"""## ANOVA TEST ON HOW DEMAND CONTEXT SIGNIFICANTLY CHANGE COST"""
groups = [merged1[merged1["Demand_Context"] == "Low Demand"]["Cost_USD_per_MWh"],
          merged1[merged1["Demand_Context"] == "Medium Demand"]["Cost_USD_per_MWh"],
          merged1[merged1["Demand_Context"] == "High Demand"]["Cost_USD_per_MWh"]]
anova_result = f_oneway(*groups)

print("\nANOVA Result Comparing Cost Across Demand Contexts:")
print(anova_result)

"""## HEATMAP OF high demand make certain plant types cheaper/more expensive?"""
pivot_analysis = merged1.pivot_table(values="Cost_USD_per_MWh", index="Plant Type", 
                                columns="Demand_Context",aggfunc="mean")
print("\nCost Comparison by Plant Type Across Demand Contexts:")
print(pivot_analysis)
plt.figure(figsize=(10,5))
sns.heatmap(pivot_analysis, annot=True, cmap="GnBu_r")
plt.title("Heatmap: Plant Type Cost vs. Demand Level")
plt.show()


"""## RMSE BASELINE MODEL (Cheapest plant per Demand ID)"""
# Compute cheapest cost per demand ID
cheapest_cost = merged1.groupby("Demand ID")["Cost_USD_per_MWh"].min().reset_index()
cheapest_cost = cheapest_cost.rename(columns={"Cost_USD_per_MWh": "Baseline_Cost"})

# Merge back
baseline_df = merged1.merge(cheapest_cost, on="Demand ID", how="left")

# Baseline prediction = Baseline_Cost
baseline_df["Predicted_Cost"] = baseline_df["Baseline_Cost"]
baseline_df["Actual_Cost"] = baseline_df["Cost_USD_per_MWh"]

"""## RMSE"""
rmse = np.sqrt(mean_squared_error(
    baseline_df["Actual_Cost"], baseline_df["Predicted_Cost"]))
print("\nBaseline RMSE:", rmse)

""""## Error distribution"""
baseline_df["Error"] = baseline_df["Actual_Cost"] - baseline_df["Predicted_Cost"]

plt.figure(figsize=(10,5))
sns.histplot(data=baseline_df, x="Error", bins=35, kde=True)
plt.title("Prediction Error Distribution (Baseline Model)")
plt.show()

