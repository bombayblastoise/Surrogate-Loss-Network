import os
# CRITICAL: Prevent libraries from hijacking all threads. 
# We want 1 thread per task, 32 tasks in parallel.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import json
import logging
import time
import argparse
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from joblib import Parallel, delayed

# Suppress Optuna logging to keep console clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Setup simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("RP_Analysis")

# -----------------------------------------------------------------------------
# 1. Data Processing Helpers
# -----------------------------------------------------------------------------

def load_data(filepath):
    """Load dataset. Assumes first column is target."""
    try:
        df = pd.read_csv(filepath)
        y = df.iloc[:, 0].values  # First column is target
        X = df.iloc[:, 1:].values  # Rest are features
        
        # Handle NaN values by replacing with 0
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        return X, y
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return None, None

def get_data_splits(X, y):
    """Standard stratified split: Train (60%), Val (20%), Test (20%)."""
    # Split 1: 80% Train+Val, 20% Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Split 2: 75% Train (of 80% = 60%), 25% Val (of 80% = 20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    # Scale Data (Crucial for Random Projection)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# -----------------------------------------------------------------------------
# 2. Modeling Helpers (Frozen & Tuned)
# -----------------------------------------------------------------------------

def get_frozen_params():
    return {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_jobs': 1,            # Single thread per model
        'thread_count': 1,      # Enforce single thread
        'tree_method': 'hist',  # Faster on CPU
        'random_state': 42,
        'verbosity': 0
    }

def train_xgb(X_train, y_train, X_val, y_val, X_test, y_test, params):
    """Train XGBoost and return metrics."""
    try:
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            # early_stopping_rounds=10
        )
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        return {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred, average='binary', zero_division=0)),
            'auc': float(roc_auc_score(y_test, y_prob)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0))
        }
    except Exception as e:
        return {'error': str(e)}

def run_tuning(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=20):
    """Run Optuna tuning."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            # Fixed params
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_jobs': 1,
            'thread_count': 1,
            'tree_method': 'hist',
            'random_state': 42,
            'verbosity': 0
        }
        
        model = xgb.XGBClassifier(**params)
        # model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=10)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return f1_score(y_val, preds, zero_division=0)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    
    # Re-train with best params on full training set logic if desired, 
    # but here we stick to the split to be consistent with frozen.
    best_params = study.best_params
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_jobs': 1,
        'thread_count': 1,
        'tree_method': 'hist',
        'random_state': 42,
        'verbosity': 0
    })
    
    metrics = train_xgb(X_train, y_train, X_val, y_val, X_test, y_test, best_params)
    return metrics, best_params

# -----------------------------------------------------------------------------
# 3. Task Worker
# -----------------------------------------------------------------------------

def process_task(task):
    """
    Execute a single configuration (Baseline or RP) for a dataset.
    This function runs in its own process.
    """
    dataset_path, config_type, seed, target_dim = task
    dataset_name = Path(dataset_path).stem
    
    # Reload data inside the process (safe & simple)
    X, y = load_data(dataset_path)
    if X is None: return None
    
    # Preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(X, y)
    
    # Apply Projection if not baseline
    if config_type == 'projection':
        try:
            grp = GaussianRandomProjection(n_components=target_dim, random_state=seed)
            X_train = grp.fit_transform(X_train)
            X_val = grp.transform(X_val)
            X_test = grp.transform(X_test)
        except Exception as e:
            logger.error(f"[{dataset_name}] Projection failed (dim={target_dim}): {e}")
            return None

    # 1. Run Frozen
    frozen_metrics = train_xgb(X_train, y_train, X_val, y_val, X_test, y_test, get_frozen_params())
    
    # 2. Run Tuned
    tuned_metrics, best_params = run_tuning(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=20)
    
    return {
        'dataset': dataset_name,
        'type': config_type,
        'seed': seed,
        'dim': target_dim,
        'frozen_metrics': frozen_metrics,
        'tuned_metrics': tuned_metrics,
        'best_params': best_params
    }

# -----------------------------------------------------------------------------
# 4. Main Controller
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", default="datasets", help="Directory with CSVs")
    parser.add_argument("--n-jobs", type=int, default=32, help="Total CPU cores to use")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of datasets to mix in a batch")
    args = parser.parse_args()
    
    datasets_dir = Path(args.datasets_dir)
    files = sorted(list(datasets_dir.glob("*.csv")))
    
    if not files:
        print("No datasets found.")
        sys.exit(1)
        
    logger.info(f"Found {len(files)} datasets. Using {args.n_jobs} cores.")

    # Process in batches of datasets
    # If batch_size = 4, we generate tasks for 4 datasets (approx 40 tasks).
    # We run them on 32 cores.
    
    for i in range(0, len(files), args.batch_size):
        batch_files = files[i : i + args.batch_size]
        batch_tasks = []
        
        logger.info(f"Preparing batch {i//args.batch_size + 1}: {[f.stem for f in batch_files]}")
        
        # --- Task Generation ---
        for filepath in batch_files:
            try:
                # Inspect dimensions to determine projection targets
                df = pd.read_csv(filepath, nrows=2) # efficient peek
                n_features = df.shape[1] - 1
                
                # JL Target Dimensions
                base_jl = int(10 * np.log(n_features))
                # Generate 3 variations, ensure valid range
                dims = sorted(list(set([
                    max(2, int(base_jl * 0.8)),
                    max(2, base_jl),
                    max(2, int(base_jl * 1.2))
                ])))
                dims = [d for d in dims if d < n_features]
                
                # 1. Baseline Task
                batch_tasks.append((str(filepath), 'baseline', None, n_features))
                
                # 2. Projection Tasks (3 seeds * 3 dims = up to 9)
                if dims:
                    for seed in [42, 123, 999]:
                        for d in dims:
                            batch_tasks.append((str(filepath), 'projection', seed, d))
                            
            except Exception as e:
                logger.error(f"Error prepping {filepath.stem}: {e}")

        logger.info(f"Batch contains {len(batch_tasks)} tasks. Executing parallel run...")
        
        # --- Parallel Execution ---
        # "prefer='processes'" is CRITICAL for CPU-bound tasks like Projection/Training
        results = Parallel(n_jobs=args.n_jobs, prefer='processes', verbose=5)(
            delayed(process_task)(t) for t in batch_tasks
        )
        
        # --- Result Aggregation & Saving ---
        # Group flat results back into datasets
        grouped_results = {}
        for res in results:
            if not res: continue
            ds_name = res['dataset']
            if ds_name not in grouped_results:
                grouped_results[ds_name] = {'baseline': {}, 'projections': []}
            
            if res['type'] == 'baseline':
                grouped_results[ds_name]['baseline'] = {
                    'frozen': res['frozen_metrics'],
                    'tuned': res['tuned_metrics'],
                    'best_params': res['best_params']
                }
            else:
                grouped_results[ds_name]['projections'].append({
                    'dim': res['dim'],
                    'seed': res['seed'],
                    'frozen': res['frozen_metrics'],
                    'tuned': res['tuned_metrics'],
                    'best_params': res['best_params']
                })
        
        # Save JSONs
        for ds_name, data in grouped_results.items():
            out_path = Path("jl_results") / f"{ds_name}_results.json"
            out_path.parent.mkdir(exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {out_path}")

    logger.info("All batches completed.")

if __name__ == "__main__":
    main()
