import os
# Force XGBoost to use single thread per model so we can parallelize the experiments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from joblib import Parallel, delayed
import multiprocessing
import logging
from pathlib import Path
import json
import time
from datetime import datetime
import argparse
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("JL_Projection_Analysis")

# -----------------------------------------------------------------------------
# Data Loading & Helper Functions
# (Replicated locally to ensure script runs standalone without main.py)
# -----------------------------------------------------------------------------

def load_data(filepath):
    """Load dataset from CSV."""
    df = pd.read_csv(filepath)
    # Assume target is the last column
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def get_train_val_test(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Split into Train, Validation, and Test sets."""
    # First split: Train+Val vs Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Second split: Train vs Val (relative to the temp size)
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_xgb_params(device='cpu'):
    """Standard params for XGBoost (frozen/default)."""
    return {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_jobs': 1,  # CRITICAL: 1 thread per model to allow outer parallelization
        'tree_method': 'hist',
        'device': device,
        'random_state': 42,
        'verbosity': 0
    }

# -----------------------------------------------------------------------------
# Core Analysis Class
# -----------------------------------------------------------------------------

class RandomProjectionAnalysis:
    def __init__(self, datasets_dir='datasets', output_dir='jl_results', n_jobs=32):
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_jobs = n_jobs
        
    def train_evaluate(self, X_train, X_val, X_test, y_train, y_val, y_test, params, config_name):
        """Train XGBoost and return metrics."""
        start_time = time.time()
        
        try:
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=10
            )
            
            # Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'config': config_name,
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'f1': float(f1_score(y_test, y_pred, average='binary')),
                'auc': float(roc_auc_score(y_test, y_prob)),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'training_time': time.time() - start_time,
                'n_features': X_train.shape[1]
            }
            return metrics
        except Exception as e:
            logger.error(f"Error in {config_name}: {e}")
            return None

    def run_projection_task(self, seed, n_components, X_train, X_val, X_test, y_train, y_val, y_test):
        """Worker function: Applies RP and trains model."""
        
        # 1. Create Random Matrix & Project
        # GaussianRandomProjection ensures JL Lemma properties
        transformer = GaussianRandomProjection(
            n_components=n_components,
            random_state=seed
        )
        
        start_project = time.time()
        X_train_proj = transformer.fit_transform(X_train)
        X_val_proj = transformer.transform(X_val)
        X_test_proj = transformer.transform(X_test)
        proj_time = time.time() - start_project
        
        # 2. Train Model
        config_name = f"RP_dim{n_components}_seed{seed}"
        params = get_xgb_params()
        
        metrics = self.train_evaluate(
            X_train_proj, X_val_proj, X_test_proj,
            y_train, y_val, y_test,
            params, config_name
        )
        
        if metrics:
            metrics['projection_time'] = proj_time
            metrics['seed'] = seed
            metrics['target_dim'] = n_components
            
        return metrics

    def process_dataset(self, filepath):
        dataset_name = filepath.stem
        logger.info(f"Processing: {dataset_name}")
        
        # Load & Scale Data
        try:
            X, y = load_data(filepath)
            
            # Check if binary classification
            if len(np.unique(y)) != 2:
                logger.warning(f"Skipping {dataset_name}: Not binary classification.")
                return None

            X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(X, y)
            
            # Standard Scaling (Important for Projection distance preservation)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            
            n_features = X_train.shape[1]
            n_samples = X_train.shape[0]
            
            # ---------------------------------------------------------
            # 1. Establish Baseline (Original Features)
            # ---------------------------------------------------------
            logger.info(f"[{dataset_name}] Running Baseline ({n_features} features)...")
            baseline_metrics = self.train_evaluate(
                X_train, X_val, X_test, y_train, y_val, y_test,
                get_xgb_params(), "Baseline_Original"
            )
            
            if not baseline_metrics:
                return None

            # ---------------------------------------------------------
            # 2. Define JL Experiment Configurations
            # ---------------------------------------------------------
            # JL Target: 10 * ln(n_features)
            # Note: We must ensure dim < n_features and dim > 1
            jl_target = int(10 * np.log(n_features))
            
            # Create 3 variations around the target
            target_dims = sorted(list(set([
                max(2, int(jl_target * 0.8)),
                max(2, jl_target),
                max(2, int(jl_target * 1.2))
            ])))
            
            # Ensure we don't project to higher dimensions than original
            target_dims = [d for d in target_dims if d < n_features]
            
            if not target_dims:
                logger.warning(f"[{dataset_name}] Feature count too low for reduction ({n_features}). Skipping.")
                return None

            seeds = [42, 123, 999] # 3 different random seeds
            
            tasks = []
            for dim in target_dims:
                for seed in seeds:
                    tasks.append((seed, dim))
            
            logger.info(f"[{dataset_name}] Queuing {len(tasks)} RP experiments (Dims: {target_dims})...")

            # ---------------------------------------------------------
            # 3. Parallel Execution
            # ---------------------------------------------------------
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.run_projection_task)(
                    seed, dim, X_train, X_val, X_test, y_train, y_val, y_test
                ) for seed, dim in tasks
            )
            
            # Filter failed runs
            results = [r for r in results if r is not None]
            
            # Combine all results
            final_report = {
                'dataset': dataset_name,
                'original_n_features': n_features,
                'n_samples': n_samples,
                'baseline': baseline_metrics,
                'experiments': results
            }
            
            # Save Results
            self.save_results(final_report, dataset_name)
            self.print_summary(final_report)
            
            return final_report

        except Exception as e:
            logger.error(f"Critical error processing {dataset_name}: {e}")
            return None

    def save_results(self, report, dataset_name):
        out_file = self.output_dir / f"{dataset_name}_jl_report.json"
        with open(out_file, 'w') as f:
            json.dump(report, f, indent=4)
        logger.info(f"Saved report to {out_file}")

    def print_summary(self, report):
        base_f1 = report['baseline']['f1']
        best_proj_f1 = max([e['f1'] for e in report['experiments']])
        
        logger.info("-" * 60)
        logger.info(f"SUMMARY: {report['dataset']}")
        logger.info(f"Original Features: {report['original_n_features']}")
        logger.info(f"Baseline F1:       {base_f1:.4f}")
        logger.info(f"Best Projected F1: {best_proj_f1:.4f} (Retained {best_proj_f1/base_f1:.1%} of performance)")
        logger.info("-" * 60)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", default="datasets", help="Folder containing CSV datasets")
    parser.add_argument("--n-jobs", type=int, default=32, help="Number of parallel cores to use")
    args = parser.parse_args()
    
    analysis = RandomProjectionAnalysis(
        datasets_dir=args.datasets_dir, 
        n_jobs=args.n_jobs
    )
    
    # Get all CSV files
    data_files = sorted(list(analysis.datasets_dir.glob("*.csv")))
    
    if not data_files:
        logger.error(f"No CSV files found in {analysis.datasets_dir}")
        sys.exit(1)
        
    logger.info(f"Found {len(data_files)} datasets. Starting Analysis...")
    
    # Process each dataset sequentially (the parallelization happens INSIDE the dataset processing)
    # This prevents memory overload from loading 32 datasets at once.
    for p in data_files:
        analysis.process_dataset(p)
        
    logger.info("All tasks completed.")