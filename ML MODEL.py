# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 16:59:38 2025

@author: Demilade
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor


"""# CLEANED DATA"""
merged_df = pd.read_csv(
    'C:/Users/Demilade/Desktop/Data Analytics Assessment/cleaned merged data.csv')
# print(merged_df.info())

"""# FEATURE SEPARATION"""
groups = merged_df["Demand ID"]
target = "Cost_USD_per_MWh"
ID_COLS = ["Demand ID", "Plant ID"]

cat_cols = merged_df.select_dtypes(include="object").columns.tolist()
cat_cols = [c for c in cat_cols if c not in ID_COLS]

num_cols = merged_df.select_dtypes(include="number").columns.tolist()
num_cols = [col for col in num_cols if col != target]

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

X_train_cat = pd.get_dummies(X_train_raw[cat_cols])
X_test_cat = pd.get_dummies(X_test_raw[cat_cols])

X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join="outer", axis=1, fill_value=0)

X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)

"""# USE 20% OF DEMAND IDs FOR TUNING"""
unique_demands = groups.loc[X_train_final.index].unique()
train_demands_20 = pd.Series(unique_demands).sample(
    frac=0.2, random_state=42).values

sample_mask = groups.loc[X_train_final.index].isin(train_demands_20)

X_tune = X_train_final.loc[sample_mask]
y_tune = y_train.loc[sample_mask]

groups_tune = groups.loc[X_tune.index]


print(f"Tuning on {len(X_tune)} rows from {len(train_demands_20)} Demand IDs")



"""# DECISION-LEVEL SCORER (LOGO)"""
logo = LeaveOneGroupOut()

def decision_rmse(y_true, y_pred, current_groups):
    temp = pd.DataFrame({"Demand ID": current_groups,
                         "Actual": y_true,
                         "Predicted": y_pred})
    # Identify the actual best plant
    selected = temp.loc[temp.groupby("Demand ID")["Predicted"].idxmin()]
    # Identify the actual best plant
    optimal = temp.groupby("Demand ID")["Actual"].min()
    # Calculate RMSE of the difference
    errors = optimal - selected.set_index("Demand ID")["Actual"]
    return -np.sqrt(np.mean(errors ** 2))

def scorer(estimator, X, y):
    fold_groups = groups.loc[X.index]
    preds = estimator.predict(X)
    return -decision_rmse(y, preds,fold_groups)

# Initialize results list
results_list = []



"""# MODEL 1: RANDOM FOREST MODEL (PRIMARY TUNING MODEL)"""
rf_param_grid = {"n_estimators": [100, 200, 300], "max_depth": [None, 15],
                 "min_samples_split": [2, 5],
                 "min_samples_leaf": [2, 5],
                 "max_features": ["sqrt"]}


rf_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_param_grid,
    cv=logo.split(X_tune, y_tune, groups_tune),
    scoring=decision_rmse,   
    refit=True, n_jobs=-1, verbose=1)


rf_search.fit(X_tune, y_tune)
rf_best = rf_search.best_estimator_


print("Best RF params:", rf_search.best_params_)

rf_test_rmse = decision_rmse(y_test, 
                                       rf_search.best_estimator_.predict(X_test_final), 
                                       groups.loc[X_test_final.index])

results_list.append({"Model": "Random Forest", 
                     "Best Params": str(rf_search.best_params_), 
                     "CV Decision RMSE": -rf_search.best_score_, 
                     "Test Decision RMSE": rf_test_rmse})
print(results_list)

"""# MODEL 2: GRADIENT BOOSTING MODEL"""
gb_param_grid = {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                 "max_depth": [3, 5], "subsample": [0.8],
                 "max_features": ["sqrt"]}


gb_search = GridSearchCV(GradientBoostingRegressor(random_state=42),
                         gb_param_grid,
                         cv=logo.split(X_train_final, y_train, 
                                       groups.loc[X_train_final.index]),
                         scoring=scorer,
                         refit=True, n_jobs=-1, return_train_score=True, 
                         error_score='raise', verbose=1)


gb_search.fit(X_train_final, y_train)
gb_best = gb_search.best_estimator_

gb_test_rmse = decision_rmse(y_test,
                                       gb_search.best_estimator_.predict(X_test_final), 
                                       groups.loc[X_test_final.index])

results_list.append({"Model": "Gradient Boosting", 
                     "Best Params": str(gb_search.best_params_), 
                     "CV Decision RMSE": -gb_search.best_score_,
                     "Test Decision RMSE": gb_test_rmse})

print(results_list)

"""# MODEL 3 — EXTRA TREES REGRESSION"""
et_param_grid = {"n_estimators": [400, 600], "max_depth": [None, 20],
                 "min_samples_leaf": [1, 2]}


et_search = GridSearchCV(ExtraTreesRegressor(random_state=42, n_jobs=-1, warm_start=True),
                         et_param_grid,
                         cv=logo.split(X_train_final, y_train, 
                                       groups.loc[X_train_final.index]),
                         scoring=scorer, refit=True,
                         n_jobs=-1,  error_score='raise', verbose=1)


et_search.fit(X_train_final, y_train)
et_best = et_search.best_estimator_

et_test_rmse = decision_rmse(y_test, et_search.best_estimator_.predict(
    X_test_final), groups.loc[X_test_final.index])

results_list.append({"Model": "Extra Trees", "Best Params": str(et_search.best_params_), 
                     "CV Decision RMSE": -et_search.best_score_, 
                     "Test Decision RMSE": et_test_rmse})

print(results_list)

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
def decision_test_rmse(model):
    preds = model.predict(X_test_final)
    temp = pd.DataFrame({"Demand ID": groups.loc[X_test_final.index],
                         "Actual": y_test.values,"Predicted": preds})
    print(results_df.sort_values("Test Decision RMSE"))



results.append({
    "Model": "Gradient Boosting",
    "Best Params": gb_search.best_params_,
    "LOGO CV RMSE": -gb_search.best_score_,
    "Test Decision RMSE": gb_test_rmse})


"""# FINAL COMPARISON TABLE"""
results_df = pd.DataFrame(results_list)
print("\n--- Model Comparison Summary ---")
print(results_df)

# Export to CSV for your use
results_df.to_csv("Model_Comparison_Table.csv", index=False)
print("\nFile saved as: Model_Comparison_Results_2025.csv")