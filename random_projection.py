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
from sklearn.impute import SimpleImputer
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
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("JL_Projection_Analysis")

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# -----------------------------------------------------------------------------
# Data Loading & Helper Functions
# (Replicated locally to ensure script runs standalone without main.py)
# -----------------------------------------------------------------------------

def load_data(filepath):
    """Load dataset from CSV."""
    df = pd.read_csv(filepath)
    # Assume target is the last column
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
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

def impute_missing_values(X_train, X_val, X_test, strategy='median'):
    """Impute missing values using training data statistics (avoid data leakage)."""
    imputer = SimpleImputer(strategy=strategy)
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed, X_val_imputed, X_test_imputed

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

# =============================================================================
# Worker Functions for Parallelization
# =============================================================================

def train_evaluate_worker(X_train, X_val, X_test, y_train, y_val, y_test, params, config_name):
    """Train XGBoost and return metrics. Standalone function for parallelization."""
    start_time = time.time()
    
    try:
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'config': config_name,
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred, average='binary', zero_division=0)),
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

def run_frozen_rp_config_worker(seed, n_components, X_train, X_val, X_test, y_train, y_val, y_test):
    """Worker for frozen RP configuration. Applies RP and trains with default params."""
    
    # Apply Random Projection
    transformer = GaussianRandomProjection(
        n_components=n_components,
        random_state=seed
    )
    
    start_project = time.time()
    X_train_proj = transformer.fit_transform(X_train)
    X_val_proj = transformer.transform(X_val)
    X_test_proj = transformer.transform(X_test)
    proj_time = time.time() - start_project
    
    # Train Model with frozen params
    config_name = f"RP_dim{n_components}_seed{seed}"
    metrics = train_evaluate_worker(
        X_train_proj, X_val_proj, X_test_proj,
        y_train, y_val, y_test,
        get_xgb_params(), config_name
    )
    
    if metrics:
        metrics['projection_time'] = proj_time
        metrics['seed'] = seed
        metrics['target_dim'] = n_components
        
    return metrics

def run_tuned_rp_config_worker(seed, n_components, X_train, X_val, X_test, y_train, y_val, y_test, n_optuna_trials, dataset_name):
    """Worker for tuned RP configuration. Runs Optuna optimization then trains."""
    
    def optuna_objective(trial):
        """Optuna objective function for hyperparameter tuning."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_jobs': 1,
            'tree_method': 'hist',
            'device': 'cpu',
            'random_state': 42,
            'verbosity': 0
        }
        
        # Apply RP
        transformer = GaussianRandomProjection(n_components=n_components, random_state=seed)
        X_train_proj = transformer.fit_transform(X_train)
        X_val_proj = transformer.transform(X_val)
        X_test_proj = transformer.transform(X_test)
        
        # Train and evaluate
        metrics = train_evaluate_worker(
            X_train_proj, X_val_proj, X_test_proj,
            y_train, y_val, y_test,
            params, f"trial_dim{n_components}_seed{seed}"
        )
        
        return metrics['f1'] if metrics else 0.0
    
    logger.info(f"[{dataset_name}] Tuning RP config (Dim={n_components}, Seed={seed})...")
    
    # Run Optuna optimization
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=MedianPruner())
    
    study.optimize(
        optuna_objective,
        n_trials=n_optuna_trials,
        show_progress_bar=False
    )
    
    best_params = study.best_params
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_jobs': 1,
        'tree_method': 'hist',
        'device': 'cpu',
        'random_state': 42,
        'verbosity': 0
    })
    
    logger.info(f"[{dataset_name}] Best params for Dim={n_components}: Best F1={study.best_value:.4f}")
    
    # Apply RP with best params
    transformer = GaussianRandomProjection(n_components=n_components, random_state=seed)
    X_train_proj = transformer.fit_transform(X_train)
    X_val_proj = transformer.transform(X_val)
    X_test_proj = transformer.transform(X_test)
    
    config_name = f"RP_Tuned_dim{n_components}_seed{seed}"
    metrics = train_evaluate_worker(
        X_train_proj, X_val_proj, X_test_proj,
        y_train, y_val, y_test,
        best_params, config_name
    )
    
    if metrics:
        metrics['seed'] = seed
        metrics['target_dim'] = n_components
    
    return metrics

# =============================================================================
# Core Analysis Class
# =============================================================================

class RandomProjectionAnalysis:
    def __init__(self, datasets_dir='datasets', output_dir='jl_results', n_jobs=32, n_optuna_trials=30):
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_jobs = n_jobs
        self.n_optuna_trials = n_optuna_trials
        
        # Create subdirectories for different analysis modes
        self.frozen_dir = self.output_dir / 'frozen'
        self.tuned_dir = self.output_dir / 'tuned'
        self.frozen_dir.mkdir(exist_ok=True)
        self.tuned_dir.mkdir(exist_ok=True)
        
        # Store results for cross-dataset analysis
        self.frozen_results = {}
        self.tuned_results = {}
        
    def plot_average_metrics(self, baseline, experiments, dataset_name, analysis_type):
        """Plot average metrics across all 9 configurations."""
        
        metrics_keys = ['f1', 'accuracy', 'auc', 'precision', 'recall']
        
        # Calculate averages for embedded features
        exp_metrics_list = [{k: exp[k] for k in metrics_keys} for exp in experiments]
        avg_exp = {k: np.mean([e[k] for e in exp_metrics_list]) for k in metrics_keys}
        baseline_metrics = {k: baseline[k] for k in metrics_keys}
        
        # Create comparison
        comparison_data = {
            'Original Features': baseline_metrics,
            'Average RP (9 Configs)': avg_exp
        }
        
        df = pd.DataFrame(comparison_data)
        
        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(kind='bar', ax=ax, width=0.7)
        
        ax.set_title(f'{analysis_type} - {dataset_name}\nAverage Metrics: Original vs RP Embedded', 
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Configuration')
        plt.tight_layout()
        
        plot_path = self.get_output_dir(analysis_type) / f'{dataset_name}_average_metrics.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved average metrics plot to {plot_path}")





    def get_output_dir(self, analysis_type):
        """Get output directory for specific analysis type."""
        return self.frozen_dir if analysis_type == 'Frozen' else self.tuned_dir

    def process_dataset_frozen(self, filepath):
        """Process dataset with frozen (default) XGBoost parameters."""
        dataset_name = filepath.stem
        logger.info(f"\n{'='*70}")
        logger.info(f"[FROZEN] Processing: {dataset_name}")
        logger.info(f"{'='*70}")
        
        try:
            X, y = load_data(filepath)
            
            # Data diagnostics
            logger.info(f"[{dataset_name}] Data loaded: X shape={X.shape}, y shape={y.shape}")
            logger.info(f"[{dataset_name}] Class distribution: {np.bincount(y)}")
            logger.info(f"[{dataset_name}] Class proportions: {np.bincount(y) / len(y)}")
            
            if len(np.unique(y)) != 2:
                logger.warning(f"Skipping {dataset_name}: Not binary classification.")
                return None

            X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(X, y)
            
            # Handle missing values
            n_missing_before = np.sum(np.isnan(X_train))
            if n_missing_before > 0:
                logger.info(f"[{dataset_name}] Detected {n_missing_before} missing values. Applying median imputation...")
                X_train, X_val, X_test = impute_missing_values(X_train, X_val, X_test, strategy='median')
            
            # Standard Scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            
            n_features = X_train.shape[1]
            n_samples = X_train.shape[0]
            
            logger.info(f"[{dataset_name}] Split sizes - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
            logger.info(f"[{dataset_name}] Original features: {n_features}")
            
            # Baseline with original features
            logger.info(f"[{dataset_name}] Running Baseline ({n_features} features)...")
            baseline_metrics = train_evaluate_worker(
                X_train, X_val, X_test, y_train, y_val, y_test,
                get_xgb_params(), "Baseline_Original"
            )
            
            if not baseline_metrics:
                return None

            logger.info(f"[{dataset_name}] Baseline metrics: F1={baseline_metrics['f1']:.4f}, Acc={baseline_metrics['accuracy']:.4f}, AUC={baseline_metrics['auc']:.4f}")

            # Define JL configurations
            jl_target = int(10 * np.log(n_features))
            target_dims = sorted(list(set([
                max(2, int(jl_target * 0.8)),
                max(2, jl_target),
                max(2, int(jl_target * 1.2))
            ])))
            target_dims = [d for d in target_dims if d < n_features]
            
            if not target_dims:
                logger.warning(f"[{dataset_name}] Feature count too low for reduction ({n_features}).")
                return None

            seeds = [42, 123, 999]
            tasks = [(seed, dim) for dim in target_dims for seed in seeds]
            
            logger.info(f"[{dataset_name}] Parallelizing {len(tasks)} frozen RP configs across {self.n_jobs} cores...")

            # Parallel execution
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(run_frozen_rp_config_worker)(
                    seed, dim, X_train, X_val, X_test, y_train, y_val, y_test
                ) for seed, dim in tasks
            )
            
            results = [r for r in results if r is not None]
            
            # Store for cross-dataset analysis
            self.frozen_results[dataset_name] = {
                'baseline': baseline_metrics,
                'experiments': results,
                'original_n_features': n_features,
                'n_samples': n_samples
            }
            
            # Generate only the average metrics plot
            self.plot_average_metrics(baseline_metrics, results, dataset_name, 'Frozen')
            
            # Print summary
            if results:
                best_f1 = max([e['f1'] for e in results])
                logger.info("-" * 70)
                logger.info(f"FROZEN SUMMARY: {dataset_name}")
                logger.info(f"Original Features: {n_features}")
                logger.info(f"Baseline F1:       {baseline_metrics['f1']:.4f}")
                logger.info(f"Best Projected F1: {best_f1:.4f} (Retained {best_f1/baseline_metrics['f1']:.1%} of performance)")
                logger.info("-" * 70)
            
            return True

        except Exception as e:
            logger.error(f"Critical error in frozen analysis for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_dataset_tuned(self, filepath):
        """Process dataset with Optuna-tuned XGBoost parameters."""
        dataset_name = filepath.stem
        logger.info(f"\n{'='*70}")
        logger.info(f"[TUNED] Processing: {dataset_name}")
        logger.info(f"{'='*70}")
        
        try:
            X, y = load_data(filepath)
            
            logger.info(f"[{dataset_name}] Data loaded: X shape={X.shape}, y shape={y.shape}")
            logger.info(f"[{dataset_name}] Class distribution: {np.bincount(y)}")
            
            if len(np.unique(y)) != 2:
                logger.warning(f"Skipping {dataset_name}: Not binary classification.")
                return None

            X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(X, y)
            
            # Handle missing values
            n_missing_before = np.sum(np.isnan(X_train))
            if n_missing_before > 0:
                logger.info(f"[{dataset_name}] Detected {n_missing_before} missing values. Applying median imputation...")
                X_train, X_val, X_test = impute_missing_values(X_train, X_val, X_test, strategy='median')
            
            # Standard Scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            
            n_features = X_train.shape[1]
            
            logger.info(f"[{dataset_name}] Running Optuna tuning for baseline (original features)...")
            
            sampler = TPESampler(seed=42)
            study = optuna.create_study(direction='maximize', sampler=sampler, pruner=MedianPruner())
            
            study.optimize(
                lambda trial: self._optuna_objective_baseline(trial, X_train, X_val, X_test, y_train, y_val, y_test),
                n_trials=self.n_optuna_trials,
                show_progress_bar=False
            )
            
            best_params_baseline = study.best_params
            best_params_baseline.update({
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'n_jobs': 1,
                'tree_method': 'hist',
                'device': 'cpu',
                'random_state': 42,
                'verbosity': 0
            })
            
            logger.info(f"[{dataset_name}] Best baseline params: {best_params_baseline}")
            logger.info(f"[{dataset_name}] Best baseline trial F1: {study.best_value:.4f}")
            
            baseline_metrics = train_evaluate_worker(
                X_train, X_val, X_test, y_train, y_val, y_test,
                best_params_baseline, "Baseline_Tuned"
            )
            
            if not baseline_metrics:
                return None

            logger.info(f"[{dataset_name}] Baseline (Tuned) metrics: F1={baseline_metrics['f1']:.4f}, Acc={baseline_metrics['accuracy']:.4f}")

            # Define JL configurations
            jl_target = int(10 * np.log(n_features))
            target_dims = sorted(list(set([
                max(2, int(jl_target * 0.8)),
                max(2, jl_target),
                max(2, int(jl_target * 1.2))
            ])))
            target_dims = [d for d in target_dims if d < n_features]
            
            if not target_dims:
                logger.warning(f"[{dataset_name}] Feature count too low for reduction ({n_features}).")
                return None

            seeds = [42, 123, 999]
            
            # Create list of (seed, dim) tuples for parallelization
            tasks = [(seed, dim) for dim in target_dims for seed in seeds]
            
            logger.info(f"[{dataset_name}] Parallelizing {len(tasks)} tuned RP configs across {self.n_jobs} cores...")

            # Parallel execution of tuned configs
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(run_tuned_rp_config_worker)(
                    seed, dim, X_train, X_val, X_test, y_train, y_val, y_test, 
                    self.n_optuna_trials, dataset_name
                ) for seed, dim in tasks
            )
            
            results = [r for r in results if r is not None]
            
            # Store for cross-dataset analysis
            self.tuned_results[dataset_name] = {
                'baseline': baseline_metrics,
                'experiments': results,
                'original_n_features': n_features
            }
            
            # Generate only the average metrics plot
            self.plot_average_metrics(baseline_metrics, results, dataset_name, 'Tuned')
            
            # Print summary
            if results:
                best_f1 = max([e['f1'] for e in results])
                logger.info("-" * 70)
                logger.info(f"TUNED SUMMARY: {dataset_name}")
                logger.info(f"Original Features: {n_features}")
                logger.info(f"Baseline F1 (Tuned):       {baseline_metrics['f1']:.4f}")
                logger.info(f"Best Projected F1 (Tuned): {best_f1:.4f} (Retained {best_f1/baseline_metrics['f1']:.1%} of performance)")
                logger.info("-" * 70)
            
            
            return True

        except Exception as e:
            logger.error(f"Critical error in tuned analysis for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _optuna_objective_baseline(self, trial, X_train, X_val, X_test, y_train, y_val, y_test):
        """Optuna objective for baseline (original features) tuning."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_jobs': 1,
            'tree_method': 'hist',
            'device': 'cpu',
            'random_state': 42,
            'verbosity': 0
        }
        
        metrics = train_evaluate_worker(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            params, "trial_baseline"
        )
        
        return metrics['f1'] if metrics else 0.0

    def plot_cross_dataset_comparison(self, analysis_type):
        """Plot cross-dataset comparison of average metrics."""
        
        results_dict = self.frozen_results if analysis_type == 'Frozen' else self.tuned_results
        
        if not results_dict:
            logger.info(f"No {analysis_type} results to plot for cross-dataset comparison.")
            return
        
        datasets = []
        baseline_f1 = []
        avg_projected_f1 = []
        
        for dataset_name, data in sorted(results_dict.items()):
            datasets.append(dataset_name)
            baseline_f1.append(data['baseline']['f1'])
            avg_f1 = np.mean([e['f1'] for e in data['experiments']]) if data['experiments'] else 0
            avg_projected_f1.append(avg_f1)
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(datasets))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_f1, width, label='Original Features', alpha=0.8)
        bars2 = ax.bar(x + width/2, avg_projected_f1, width, label='Avg RP (9 Configs)', alpha=0.8)
        
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title(f'{analysis_type} Model - Cross-Dataset F1 Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plot_path = self.get_output_dir(analysis_type) / f'cross_dataset_f1_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved cross-dataset comparison plot to {plot_path}")

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", default="datasets", help="Folder containing CSV datasets")
    parser.add_argument("--n-jobs", type=int, default=32, help="Number of parallel cores to use")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials for tuning")
    parser.add_argument("--run", choices=["frozen", "tuned", "both"], default="both", 
                        help="Which analysis to run: frozen, tuned, or both")
    args = parser.parse_args()
    
    analysis = RandomProjectionAnalysis(
        datasets_dir=args.datasets_dir, 
        n_jobs=args.n_jobs,
        n_optuna_trials=args.n_trials
    )
    
    # Get all CSV files
    data_files = sorted(list(analysis.datasets_dir.glob("*.csv")))
    
    if not data_files:
        logger.error(f"No CSV files found in {analysis.datasets_dir}")
        sys.exit(1)
        
    logger.info(f"Found {len(data_files)} datasets. Starting Analysis...")
    
    # Process each dataset
    for p in data_files:
        if args.run in ["frozen", "both"]:
            analysis.process_dataset_frozen(p)
        
        if args.run in ["tuned", "both"]:
            analysis.process_dataset_tuned(p)
    
    # Generate cross-dataset comparison plots
    if args.run in ["frozen", "both"]:
        analysis.plot_cross_dataset_comparison('Frozen')
    
    if args.run in ["tuned", "both"]:
        analysis.plot_cross_dataset_comparison('Tuned')
    
    logger.info("\n" + "="*70)
    logger.info("All analyses completed!")
    logger.info(f"Results saved to: {analysis.output_dir}")
    logger.info("="*70)
