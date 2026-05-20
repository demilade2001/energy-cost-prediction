# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 16:59:38 2025
@author: Demilade
OPTIMIZED VERSION: Faster hyperparameter tuning and reduced model complexity
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, ParameterGrid
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

start_time = time.time()

"""# CLEANED DATA"""
data_path = Path.home() / "Desktop" / "Data Analytics Assessment" / "cleaned merged data.csv"
if not data_path.exists():
    raise FileNotFoundError(f"Input CSV not found: {data_path}")
merged_df = pd.read_csv(data_path)

"""# FEATURE SEPARATION"""
groups = merged_df["Demand ID"]
target = "Cost_USD_per_MWh"
ID_COLS = ["Demand ID", "Plant ID"]

cat_cols = merged_df.select_dtypes(include="object").columns.tolist()
cat_cols = [c for c in cat_cols if c not in ID_COLS]

num_cols = merged_df.select_dtypes(include="number").columns.tolist()
num_cols = [col for col in num_cols if col != target and col not in ID_COLS]

"""# GROUPED TRAIN / TEST SPLIT"""
np.random.seed(42)
test_groups = np.random.choice(groups.unique(), size=20, replace=False)

train_mask = ~groups.isin(test_groups)
test_mask = groups.isin(test_groups)

X_train_raw = merged_df.loc[train_mask]
X_test_raw = merged_df.loc[test_mask]

y_train = X_train_raw[target]
y_test = X_test_raw[target]

"""# SCALING NUMERICAL DATA AND ENCODING CATEGORICAL DATA"""
scaler = MinMaxScaler()
scaler.fit(X_train_raw[num_cols])

X_train_num = pd.DataFrame(scaler.transform(X_train_raw[num_cols]), columns=num_cols,
                           index=X_train_raw.index)

X_test_num = pd.DataFrame(scaler.transform(X_test_raw[num_cols]), columns=num_cols,
                          index=X_test_raw.index)

X_train_cat = pd.get_dummies(X_train_raw[cat_cols], dtype='uint8')
X_test_cat = pd.get_dummies(X_test_raw[cat_cols], dtype='uint8')

X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join="outer", axis=1, fill_value=0)

X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)

"""# DECISION-LEVEL SCORER"""
# Use 3-fold CV (faster than 5) and tune on full training set
gkf = GroupKFold(n_splits=3)

def decision_rmse(y_true, y_pred, current_groups):
    current_groups = pd.Series(current_groups).reset_index(drop=True)
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    if len(current_groups) != len(y_true):
        raise ValueError(
            f"Group labels length {len(current_groups)} does not match y length {len(y_true)}"
        )
    temp = pd.DataFrame({"Demand ID": current_groups,
                         "Actual": y_true,
                         "Predicted": y_pred})
    selected = temp.loc[temp.groupby("Demand ID")["Predicted"].idxmin()]
    optimal = temp.groupby("Demand ID")["Actual"].min()
    errors = optimal - selected.set_index("Demand ID")["Actual"]
    return np.sqrt(np.mean(errors ** 2))


def get_max_random_iters(param_grid, max_iter):
    total_combinations = len(list(ParameterGrid(param_grid)))
    return min(max_iter, total_combinations)


def decision_scorer(estimator, X, y):
    preds = estimator.predict(X)
    current_groups = groups.loc[X.index]
    return -decision_rmse(y, preds, current_groups)

# Initialize results list
results_list = []



"""# MODEL 1: RANDOM FOREST MODEL (OPTIMIZED)"""
# Reduced parameter grid for speed
rf_param_grid = {"n_estimators": [80, 150], "max_depth": [10, 15],
                 "min_samples_split": [5, 10],
                 "min_samples_leaf": [2, 4],
                 "max_features": ["sqrt"]}

print(f"Training Random Forest (10 iterations)...")
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    rf_param_grid,
    n_iter=10,  # Reduced from 20
    cv=gkf.split(X_train_final, y_train, groups.loc[X_train_final.index]),
    scoring=decision_scorer,
    refit=True, n_jobs=-1, verbose=0, random_state=42)

rf_search.fit(X_train_final, y_train)
rf_best = rf_search.best_estimator_

print("Best RF params:", rf_search.best_params_)

rf_test_rmse = decision_rmse(
    y_test,
    rf_search.best_estimator_.predict(X_test_final),
    groups.loc[X_test_final.index]
)

results_list.append({"Model": "Random Forest", 
                     "Best Params": str(rf_search.best_params_), 
                     "CV Decision RMSE": -rf_search.best_score_, 
                     "Test Decision RMSE": rf_test_rmse})

"""# MODEL 2: GRADIENT BOOSTING MODEL (OPTIMIZED)"""
# Reduced parameter grid for speed
gb_param_grid = {"n_estimators": [100, 150], "learning_rate": [0.05, 0.1],
                 "max_depth": [3, 4], "subsample": [0.8],
                 "max_features": ["sqrt"]}

print(f"Training Gradient Boosting (8 iterations)...")
gb_search = RandomizedSearchCV(GradientBoostingRegressor(random_state=42, validation_fraction=0.1),
                         gb_param_grid,
                         n_iter=8,  # Reduced from 20
                         cv=gkf.split(X_train_final, y_train, 
                                       groups.loc[X_train_final.index]),
                         scoring=decision_scorer,
                         refit=True, n_jobs=-1, verbose=0, random_state=42)

gb_search.fit(X_train_final, y_train)
gb_best = gb_search.best_estimator_

gb_test_rmse = decision_rmse(
    y_test,
    gb_search.best_estimator_.predict(X_test_final),
    groups.loc[X_test_final.index]
)

results_list.append({"Model": "Gradient Boosting", 
                     "Best Params": str(gb_search.best_params_), 
                     "CV Decision RMSE": -gb_search.best_score_,
                     "Test Decision RMSE": gb_test_rmse})

"""# MODEL 3: EXTRA TREES REGRESSION (OPTIMIZED)"""
# Reduced parameter grid for speed
et_param_grid = {"n_estimators": [200, 400], "max_depth": [10, 15],
                 "min_samples_leaf": [2, 4]}

print(f"Training Extra Trees (8 iterations)...")
et_search = RandomizedSearchCV(ExtraTreesRegressor(random_state=42, n_jobs=-1),
                         et_param_grid,
                         n_iter=8,  # Reduced from 12
                         cv=gkf.split(X_train_final, y_train, 
                                       groups.loc[X_train_final.index]),
                         scoring=decision_scorer, refit=True,
                         n_jobs=-1, verbose=0, random_state=42)

et_search.fit(X_train_final, y_train)
et_best = et_search.best_estimator_

et_test_rmse = decision_rmse(
    y_test,
    et_search.best_estimator_.predict(X_test_final),
    groups.loc[X_test_final.index]
)

results_list.append({"Model": "Extra Trees", "Best Params": str(et_search.best_params_), 
                     "CV Decision RMSE": -et_search.best_score_, 
                     "Test Decision RMSE": et_test_rmse})

# 9. BASELINE CALCULATION
plant_mean = X_train_raw.groupby(
    "Plant ID")[target].mean().reset_index().rename(columns={target: "PlantMean"})

baseline_df = X_test_raw.merge(plant_mean, on="Plant ID", how="left")
baseline_df["PlantMean"] = baseline_df["PlantMean"].fillna(y_train.mean()) 

baseline_sel = baseline_df.loc[baseline_df.groupby("Demand ID")["PlantMean"].idxmin()]
baseline_opt = X_test_raw.groupby("Demand ID")[target].min()
baseline_rmse = np.sqrt(np.mean((baseline_opt.values - baseline_sel[target].values) ** 2))

results_list.append({"Model": "Plant-Mean Baseline", "Best Params": "N/A", 
                     "CV Decision RMSE": "N/A", "Test Decision RMSE": baseline_rmse})


# TEST-SET DECISION-LEVEL EVALUATION
# Results are already collected in results_list above.

"""# FINAL COMPARISON TABLE"""
results_df = pd.DataFrame(results_list)
print("\n--- Model Comparison Summary ---")
print(results_df)

# Export to CSV for your use
results_df.to_csv("Model_Comparison_Table.csv", index=False)
print("\nFile saved as: Model_Comparison_Table.csv")

elapsed_time = time.time() - start_time
print(f"\n✓ Total runtime: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")