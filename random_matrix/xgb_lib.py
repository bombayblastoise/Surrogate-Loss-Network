import json
import numpy as np
import pandas as pd
from datetime import datetime

import xgboost as xgb
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# --------------------------------------------------
# Config
# --------------------------------------------------

CSV_PATH = "datasets/application_data.csv"      # <-- change this
TARGET_COL = 0             # target column index
N_TRIALS = 50
RANDOM_STATE = 42

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

y = df.iloc[:, TARGET_COL].values
X = df.drop(df.columns[TARGET_COL], axis=1).values
X = np.nan_to_num(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

# --------------------------------------------------
# Optuna objective
# --------------------------------------------------

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0),
    }

    model = xgb.XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred)

# --------------------------------------------------
# Run Optuna
# --------------------------------------------------

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)

best_params = study.best_params

# --------------------------------------------------
# Train final model
# --------------------------------------------------

final_model = xgb.XGBClassifier(
    **best_params,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

final_model.fit(X_train, y_train)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred),
    "auc": roc_auc_score(y_test, y_proba),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
}

# --------------------------------------------------
# Save results
# --------------------------------------------------

results = {
    "timestamp": datetime.now().isoformat(),
    "best_params": best_params,
    "metrics": metrics,
}

with open("xgb_optuna_results.json", "w") as f:
    json.dump(results, f, indent=2)

# --------------------------------------------------
# Print summary
# --------------------------------------------------

print("\nBest hyperparameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

print("\nTest metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")
