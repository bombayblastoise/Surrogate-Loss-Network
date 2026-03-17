import os
import glob
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, average_precision_score, fbeta_score
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed

# --- CONFIGURATION ---
# CHANGE THIS if your target is the last column (-1) or first column (0)
TARGET_COL_INDEX = 0 

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def calculate_scale_pos_weight(y):
    """Calculate the ratio of negatives to positives."""
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0: return 1.0
    return n_neg / n_pos

def find_best_threshold_f05(y_true, y_prob):
    """Find best threshold maximizing F0.5 (Precision-weighted)."""
    best_thresh = 0.5
    best_score = -1.0
    # Search range
    thresholds = np.arange(0.1, 0.95, 0.01)
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        score = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return best_thresh

def get_metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "threshold": float(threshold)
    }

def run_tuning(X_train, y_train, ratio, n_trials=15):
    """Tune Hyperparameters."""
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Upper bound for weight: Square root of ratio to prevent over-correction
    # e.g., if ratio is 100:1, use max weight 10, not 100.
    max_weight = np.sqrt(ratio)
    if max_weight < 1.0: max_weight = 1.0

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            # CRITICAL FIX: Limit the weight to prevent "Predict All 1s" behavior
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, max_weight),
            'objective': 'binary:logistic',
            'n_jobs': 1,
            'verbosity': 0
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        
        # Optimize Average Precision
        probs = model.predict_proba(X_v)[:, 1]
        return average_precision_score(y_v, probs)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def process_config(file_path, split_type, dim_offset):
    try:
        # 1. Load Data
        df = pd.read_csv(file_path)
        
        if TARGET_COL_INDEX == 0:
            y_raw = df.iloc[:, 0].values
            X_raw = df.iloc[:, 1:].values
        else:
            y_raw = df.iloc[:, -1].values
            X_raw = df.iloc[:, :-1].values

        n_features = X_raw.shape[1]
        imbalance_ratio = calculate_scale_pos_weight(y_raw)

        # 2. Config
        if dim_offset is None:
            config_type = "baseline"
            target_dim = n_features
            X_final = X_raw
        else:
            config_type = "experiment"
            target_dim = int(10 * np.log(n_features)) + dim_offset
            if target_dim < 1: target_dim = 1
            
            # Federated Projection
            indices = np.arange(X_raw.shape[0])
            if split_type == 'random':
                np.random.seed(42)
                np.random.shuffle(indices)
            
            chunks = np.array_split(indices, 5)
            X_parts, y_parts = [], []
            
            for i, idx in enumerate(chunks):
                np.random.seed(i)
                R = np.random.randn(n_features, target_dim)
                X_parts.append(np.dot(X_raw[idx], R))
                y_parts.append(y_raw[idx])
            
            X_final = np.vstack(X_parts)
            y_raw = np.concatenate(y_parts)

        # 3. Split & Impute
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_final, y_raw, test_size=0.2, random_state=42, stratify=y_raw
        )

        if np.isnan(X_train_full).any():
            imputer = SimpleImputer(strategy='median')
            X_train_full = imputer.fit_transform(X_train_full)
            X_test = imputer.transform(X_test)

        # 4. Calibration Split (for Threshold selection)
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )

        # --- Frozen Model ---
        frozen_params = {
            "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
            "objective": "binary:logistic", "n_jobs": 1, 
            # Use strict weight for Frozen too
            "scale_pos_weight": max(1.0, np.sqrt(imbalance_ratio))
        }
        model_frozen = xgb.XGBClassifier(**frozen_params)
        model_frozen.fit(X_fit, y_fit)
        
        probs_val_frozen = model_frozen.predict_proba(X_val)[:, 1]
        thresh_frozen = find_best_threshold_f05(y_val, probs_val_frozen)
        
        probs_test_frozen = model_frozen.predict_proba(X_test)[:, 1]
        metrics_frozen = get_metrics_at_threshold(y_test, probs_test_frozen, thresh_frozen)

        # --- Tuned Model ---
        best_params = run_tuning(X_train_full, y_train_full, imbalance_ratio, n_trials=15)
        best_params.update({'objective': 'binary:logistic', 'n_jobs': 1, 'verbosity': 0})
        
        model_tuned = xgb.XGBClassifier(**best_params)
        model_tuned.fit(X_fit, y_fit)
        
        probs_val_tuned = model_tuned.predict_proba(X_val)[:, 1]
        thresh_tuned = find_best_threshold_f05(y_val, probs_val_tuned)
        
        probs_test_tuned = model_tuned.predict_proba(X_test)[:, 1]
        metrics_tuned = get_metrics_at_threshold(y_test, probs_test_tuned, thresh_tuned)

        return {
            "dataset_name": os.path.basename(file_path).replace('.csv', ''),
            "type": config_type, "split_method": split_type, "target_dim": target_dim,
            "frozen": metrics_frozen, "tuned": metrics_tuned, "best_params": best_params
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

if __name__ == "__main__":
    out_dir = "jl_fl_final_results"
    os.makedirs(out_dir, exist_ok=True)
    datasets = glob.glob("all_datasets/*.csv")
    tasks = []

    for ds in datasets:
        tasks.append((ds, None, None))
        for split in ['random', 'ordered']:
            for d in [-2, -1, 0, 1, 2]:
                tasks.append((ds, split, d))

    print(f"Generated {len(tasks)} tasks. Processing...")
    results_flat = Parallel(n_jobs=-1)(delayed(process_config)(d, s, o) for d, s, o in tasks)

    grouped = {}
    for res in results_flat:
        if not res: continue
        name = res['dataset_name']
        if name not in grouped:
            grouped[name] = {"frozen": {"baseline": {}, "experiments": []}, "tuned": {"baseline": {}, "experiments": []}}
        
        if res['type'] == 'baseline':
            grouped[name]['frozen']['baseline'] = res['frozen']
            grouped[name]['tuned']['baseline'] = res['tuned']
        else:
            meta = {"split_method": res['split_method'], "target_dim": res['target_dim']}
            grouped[name]['frozen']['experiments'].append({**meta, **res['frozen']})
            grouped[name]['tuned']['experiments'].append({**meta, **res['tuned'], "best_params": res['best_params']})

    for name, data in grouped.items():
        with open(f"{out_dir}/{name}.json", 'w') as f:
            json.dump(data, f, indent=4)
            
    print(f"Done. Saved to {out_dir}")