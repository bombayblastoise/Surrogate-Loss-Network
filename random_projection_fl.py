import os
# Force threads to 1 to allow joblib to manage parallelism
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

# Suppress warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Fed_Analysis")

# -----------------------------------------------------------------------------
# 1. Helpers
# -----------------------------------------------------------------------------

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        # Handle cases where target might be first or last. Assuming last for now based on previous context.
        # If your datasets vary, ensure consistency here.
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return None, None

def get_xgb_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_prob)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0))
    }

def train_xgb_frozen(X_train, y_train, X_val, y_val, X_test, y_test):
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_jobs': 1,
        'tree_method': 'hist',
        'random_state': 42,
        'verbosity': 0
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=10)
    
    return get_xgb_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])

def train_xgb_tuned(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=15):
    """Lighter tuning for the massive parallel batch."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_jobs': 1,
            'tree_method': 'hist',
            'random_state': 42,
            'verbosity': 0
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=10)
        return f1_score(y_val, model.predict(X_val), zero_division=0)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_jobs': 1,
        'tree_method': 'hist',
        'random_state': 42,
        'verbosity': 0
    })
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=10)
    
    return get_xgb_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1]), best_params

# -----------------------------------------------------------------------------
# 2. Federated Simulation Logic
# -----------------------------------------------------------------------------

def apply_federated_projection(X, y, split_type='random', n_parts=5, target_dim=10, seeds=[42, 43, 44, 45, 46]):
    """
    Splits data, projects each part with a DIFFERENT seed, then concatenates.
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if split_type == 'random':
        np.random.seed(42)
        np.random.shuffle(indices)
    
    # Split indices into 5 chunks
    chunks = np.array_split(indices, n_parts)
    
    # Storage for projected parts
    # We must store them in a way we can reconstruct the original order 
    # OR just stack them if order doesn't matter for independent rows.
    # XGBoost assumes i.i.d usually, but let's just stack. 
    # NOTE: If we split randomly, 'y' must be reordered similarly.
    
    X_parts_proj = []
    y_parts = []
    
    for i in range(n_parts):
        idx = chunks[i]
        X_part = X[idx]
        y_part = y[idx]
        
        # Project this client's data using their unique key (seed)
        transformer = GaussianRandomProjection(n_components=target_dim, random_state=seeds[i])
        X_part_proj = transformer.fit_transform(X_part)
        
        X_parts_proj.append(X_part_proj)
        y_parts.append(y_part)
        
    # Concatenate back
    X_final = np.vstack(X_parts_proj)
    y_final = np.hstack(y_parts)
    
    return X_final, y_final

# -----------------------------------------------------------------------------
# 3. Worker Task
# -----------------------------------------------------------------------------

def process_task(task_spec):
    """
    Executes one configuration.
    task_spec: dict with keys (filepath, type, split, dim, etc.)
    """
    filepath = task_spec['filepath']
    task_type = task_spec['type']  # 'baseline' or 'experiment'
    dataset_name = Path(filepath).stem
    
    # Reload data to ensure isolation
    X, y = load_data(filepath)
    if X is None: return None
    
    # Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    results = {
        'dataset': dataset_name,
        'type': task_type,
        'config': task_spec.get('config_id', 'baseline')
    }

    try:
        if task_type == 'experiment':
            split_mode = task_spec['split']
            dim = task_spec['dim']
            
            # Apply Federated Logic
            X_fed, y_fed = apply_federated_projection(
                X, y, split_type=split_mode, n_parts=5, target_dim=dim
            )
            
            # Split Train/Test on the NEW projected data
            X_train, X_temp, y_train, y_temp = train_test_split(X_fed, y_fed, test_size=0.4, stratify=y_fed, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
            
            # Record specifics
            results.update({'split': split_mode, 'dim': dim})
            
        else: # Baseline
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
        
        # Train Frozen
        res_frozen = train_xgb_frozen(X_train, y_train, X_val, y_val, X_test, y_test)
        results['frozen'] = res_frozen
        
        # Train Tuned
        res_tuned, params = train_xgb_tuned(X_train, y_train, X_val, y_val, X_test, y_test)
        results['tuned'] = res_tuned
        results['params'] = params
        
        return results

    except Exception as e:
        logger.error(f"Error in {dataset_name} [{task_type}]: {e}")
        return None

# -----------------------------------------------------------------------------
# 4. Main Orchestrator
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", default="datasets", help="Directory with CSVs")
    parser.add_argument("--output-dir", default="federated_results", help="Output directory")
    parser.add_argument("--n-jobs", type=int, default=32, help="Cores")
    args = parser.parse_args()
    
    datasets_dir = Path(args.datasets_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    
    files = sorted(list(datasets_dir.glob("*.csv")))
    if not files:
        print("No datasets found.")
        return

    logger.info(f"Generating tasks for {len(files)} datasets...")
    
    all_tasks = []
    
    for fp in files:
        try:
            # Peek to get dimensions
            df = pd.read_csv(fp, nrows=5)
            n_features = df.shape[1] - 1
            
            # 1. Baseline Task
            all_tasks.append({
                'filepath': str(fp),
                'type': 'baseline',
                'config_id': 'original'
            })
            
            # 2. Experiment Tasks
            # Generate 10 configs linearly spaced
            # Ensure we don't exceed feature count and have at least 2 dims
            max_dim = max(2, int(n_features * 0.8))
            min_dim = 2
            
            if max_dim <= min_dim:
                 dims = [min_dim]
            else:
                dims = np.linspace(min_dim, max_dim, 10, dtype=int)
                dims = np.unique(dims) # Remove duplicates if range is small
            
            for d in dims:
                # Random Split
                all_tasks.append({
                    'filepath': str(fp),
                    'type': 'experiment',
                    'split': 'random',
                    'dim': int(d),
                    'config_id': f'rand_{d}'
                })
                # Sequential Split
                all_tasks.append({
                    'filepath': str(fp),
                    'type': 'experiment',
                    'split': 'sequential',
                    'dim': int(d),
                    'config_id': f'seq_{d}'
                })
                
        except Exception as e:
            logger.error(f"Skipping task gen for {fp}: {e}")

    logger.info(f"Total tasks generated: {len(all_tasks)}")
    logger.info("Starting Parallel Execution...")
    
    # Run all tasks using the global pool
    results_flat = Parallel(n_jobs=args.n_jobs, prefer='processes', verbose=5)(
        delayed(process_task)(t) for t in all_tasks
    )
    
    # Group results by dataset and save JSONs
    logger.info("Aggregating results...")
    
    dataset_groups = {}
    for r in results_flat:
        if r is None: continue
        name = r['dataset']
        if name not in dataset_groups:
            dataset_groups[name] = {'baseline': {}, 'experiments': []}
            
        if r['type'] == 'baseline':
            dataset_groups[name]['baseline'] = {
                'frozen': r['frozen'],
                'tuned': r['tuned'],
                'best_params': r['params']
            }
        else:
            dataset_groups[name]['experiments'].append({
                'split_method': r['split'],
                'target_dim': r['dim'],
                'frozen': r['frozen'],
                'tuned': r['tuned'],
                'best_params': r['params']
            })
            
    # Write files
    for name, data in dataset_groups.items():
        out_path = out_dir / f"{name}_fed_results.json"
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=4)
            
    logger.info(f"Done. Results saved to {out_dir}")

if __name__ == "__main__":
    main()