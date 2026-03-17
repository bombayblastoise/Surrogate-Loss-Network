import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, fbeta_score
import time
import json
from datetime import datetime
import optuna
from optuna.pruners import MedianPruner
from joblib import Parallel, delayed
import multiprocessing
import logging
from pathlib import Path
import argparse
import sys
import xgboost as xgb

# Import functions from main.py (same directory)
from main import (
    load_data,
    train_val_test_split,
    get_default_params,
    find_best_threshold,
)


class PCAAnalysis:
    """
    Comprehensive PCA analysis for XGBoost model with GPU acceleration and optimized parallelization.
    
    Section 1: Frozen Model Analysis (PARALLELIZED AT DATASET LEVEL)
      - Uses default hyperparameters (frozen/unchanged)
      - Tests with varying PCA components
      - Shows impact of feature reduction alone
    
    Section 2: Optuna-Tuned Analysis (PARALLELIZED INTERNALLY BY OPTUNA)
      - Runs Optuna hyperparameter optimization for each PCA configuration
      - Tests with varying PCA components
      - Shows optimal performance with adapted hyperparameters
    """
    
    def __init__(self, datasets_dir='test', n_optuna_trials=30, n_jobs=-1, output_dir='pca_results', 
                 min_recall_threshold=0.90, f2_weight=0.70, use_gpu=True):
        """
        Initialize PCA analysis.
        
        Args:
            datasets_dir: directory containing CSV datasets
            n_optuna_trials: number of Optuna trials per PCA configuration
            n_jobs: number of parallel jobs (-1 = all CPUs, -2 = all but one)
            output_dir: directory to save results
            min_recall_threshold: minimum recall retention (e.g., 0.90 = 90% of baseline)
            f2_weight: weight for F2 score vs training time (e.g., 0.70 = 70% F2, 30% time)
            use_gpu: whether to attempt GPU acceleration (default: True)
        """
        self.datasets_dir = datasets_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results_frozen = {}  # Section 1: Frozen model
        self.results_tuned = {}   # Section 2: Optuna-tuned
        
        # Scoring parameters
        self.min_recall_threshold = min_recall_threshold
        self.f2_weight = f2_weight
        self.time_weight = 1.0 - f2_weight
        
        # Set number of parallel jobs
        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        elif n_jobs == -2:
            self.n_jobs = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.n_jobs = max(1, min(n_jobs, multiprocessing.cpu_count()))
        
        # Setup logging
        self._setup_logging()
        
        # Check GPU availability and set default params
        self.use_gpu = use_gpu
        self.gpu_available = self._check_gpu_availability()
        self.default_params = self._prepare_default_params()
        
        self.n_optuna_trials = n_optuna_trials
        
        self.logger.info(f"Parallelization: Using {self.n_jobs} CPU cores")
        self.logger.info(f"Minimum recall threshold: {self.min_recall_threshold*100:.0f}% of baseline")
        self.logger.info(f"F2 weight: {self.f2_weight*100:.0f}%, Time weight: {self.time_weight*100:.0f}%")
    
    def _check_gpu_availability(self):
        """Check if GPU is available for XGBoost (XGBoost 3.1+)"""
        if not self.use_gpu:
            self.logger.info("GPU acceleration disabled by user")
            return False
        
        try:
            # XGBoost 3.1+ uses tree_method='gpu_hist' and device='cuda'
            test_model = xgb.XGBClassifier(
                n_estimators=1,
                tree_method='gpu_hist',
                device='cuda',
                random_state=42,
                verbosity=0
            )
            test_X = np.random.rand(100, 10)
            test_y = np.random.randint(0, 2, 100)
            test_model.fit(test_X, test_y, verbose=False)
            self.logger.info("✓ GPU acceleration ENABLED - NVIDIA GPU detected and working")
            return True
        except ValueError as e:
            error_str = str(e)
            if "gpu_hist" in error_str or "Invalid Input" in error_str:
                self.logger.warning("⚠ GPU not available - XGBoost not compiled with GPU support")
                self.logger.info("  Install GPU-enabled XGBoost with: pip install xgboost-gpu-cu12")
                self.logger.info("  Then reinstall or rebuild xgboost with GPU support enabled")
                return False
            else:
                self.logger.warning(f"⚠ GPU error: {e}")
                return False
        except Exception as e:
            error_str = str(e)
            if "device" in error_str.lower() or "cuda" in error_str.lower():
                self.logger.warning(f"⚠ GPU initialization failed: {e}")
                return False
            else:
                self.logger.warning(f"⚠ Unexpected error during GPU check: {e}")
                return False
    
    def _prepare_default_params(self):
        """Prepare default params with GPU or CPU settings based on availability"""
        try:
            custom_params = get_default_params()
            base_params = {
                'n_estimators': custom_params.get('n_estimators', 100),
                'max_depth': custom_params.get('max_depth', 6),
                'learning_rate': custom_params.get('learning_rate', 0.1),
                'reg_lambda': custom_params.get('lambda_', 1.0),
                'gamma': custom_params.get('gamma', 0.0),
                'min_child_weight': custom_params.get('min_child_weight', 1.0),
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_bin': custom_params.get('n_bins', 256),
                'random_state': 42,
                'n_jobs': 1,
                'early_stopping_rounds': 10,
                'verbosity': 0,
            }
        except:
            # Fallback to sensible defaults
            base_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'reg_lambda': 1.0,
                'gamma': 0.0,
                'min_child_weight': 1.0,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_bin': 256,
                'random_state': 42,
                'n_jobs': 1,
                'early_stopping_rounds': 10,
                'verbosity': 0,
            }
        
        # Add GPU-specific or CPU-specific parameters (XGBoost 3.1+)
        if self.gpu_available:
            base_params['tree_method'] = 'gpu_hist'
            base_params['device'] = 'cuda'
        else:
            base_params['tree_method'] = 'hist'
            base_params['device'] = 'cpu'
        
        return base_params
    
    def _setup_logging(self):
        """Setup structured logging to file and console"""
        # Create logs directory
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'pca_analysis_{timestamp}.log'
        
        # Configure logging
        self.logger = logging.getLogger('PCAAnalysis')
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # File handler - detailed logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler - clean output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging to: {log_file}")
        self.logger.info("="*80)
    
    def apply_pca(self, X_train, X_val, X_test, n_components):
        """
        Apply PCA to data with specified number of components.
        
        Args:
            X_train, X_val, X_test: input data
            n_components: number of PCA components
        
        Returns:
            Transformed data and fitted PCA object
        """
        # Fit PCA on training data
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        
        # Transform validation and test data
        X_val_pca = pca.transform(X_val)
        X_test_pca = pca.transform(X_test)
        
        # Calculate explained variance
        explained_var = np.sum(pca.explained_variance_ratio_)
        
        return X_train_pca, X_val_pca, X_test_pca, pca, explained_var
    
    def calculate_all_metrics(self, y_true, y_pred_proba, threshold=0.5):
        """
        Calculate all evaluation metrics including F2 score.
        
        Args:
            y_true: true labels
            y_pred_proba: predicted probabilities
            threshold: classification threshold
        
        Returns:
            Dictionary of metrics
        """
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
        
        return {
            'f1': f1_score(y_true, y_pred_binary, zero_division=0),
            'f2': fbeta_score(y_true, y_pred_binary, beta=2, zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        }
    
    def compute_efficiency_score(
        self,
        f2_score,
        baseline_f2,
        recall,
        baseline_recall,
        training_time,
        baseline_time,
    ):
        """
        Compute composite efficiency score for a PCA configuration.

        Args:
            f2_score: F2 score of current configuration
            baseline_f2: F2 score with all features
            recall: recall of current configuration
            baseline_recall: recall with all features
            training_time: training time of current configuration (seconds)
            baseline_time: training time with all features (seconds)

        Returns:
            score: efficiency score in [0, 1] or negative if recall constraint violated
            acceptable: whether this configuration meets minimum recall threshold
        """
        # Recall retention ratio
        recall_retention = recall / baseline_recall if baseline_recall > 0 else 0

        # Check hard constraint: minimum recall threshold
        if recall_retention < self.min_recall_threshold:
            # Return negative score to indicate unacceptable configuration
            penalty = (self.min_recall_threshold - recall_retention) * 10
            return -penalty, False

        # F2 retention (normalized to [0, 1])
        f2_retention = min(f2_score / baseline_f2, 1.0) if baseline_f2 > 0 else 0

        # Time savings (normalized to [0, 1])
        time_savings = (
            max(0, 1.0 - (training_time / baseline_time))
            if baseline_time > 0
            else 0
        )

        # Composite score: weighted combination
        score = self.f2_weight * f2_retention + self.time_weight * time_savings

        return score, True
    
    def train_and_evaluate_silent(self, X_train, X_val, X_test, y_train, y_val, y_test, params):
        """Train model and return all metrics (silent mode) using GPU-enabled XGBoost"""
        try:
            start_time = time.time()
            
            # Create XGBoost classifier with GPU or CPU
            model = xgb.XGBClassifier(**params)
            
            # Train with early stopping
            model.fit(
                X_train, y_train, 
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            training_time = time.time() - start_time
            
            # Get probability predictions
            y_train_pred = model.predict_proba(X_train)[:, 1]
            y_val_pred = model.predict_proba(X_val)[:, 1]
            y_test_pred = model.predict_proba(X_test)[:, 1]
            
            # Find optimal threshold on validation set
            threshold = find_best_threshold(y_val, y_val_pred)
            
            # Calculate all metrics
            train_metrics = self.calculate_all_metrics(y_train, y_train_pred, threshold)
            val_metrics = self.calculate_all_metrics(y_val, y_val_pred, threshold)
            test_metrics = self.calculate_all_metrics(y_test, y_test_pred, threshold)
            
            metrics = {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics,
                'threshold': threshold,
                'training_time': training_time
            }
            
            return metrics
        
        except Exception as e:
            # Log exception for diagnosis and return None to signal failure
            try:
                self.logger.exception(f"Training failed: {e}")
            except Exception:
                print(f"Training failed: {e}")
            return None
    
    # ==================== SECTION 1: FROZEN MODEL ANALYSIS (OPTIMIZED) ====================
    
    def _train_frozen_batch(self, X_train_pca_dict, X_val_pca_dict, X_test_pca_dict,
                           y_train, y_val, y_test, n_components_batch):
        """
        Train frozen model for a batch of PCA configurations (parallelized worker).
        No logging - just returns results.
        
        Args:
            X_train_pca_dict: dict mapping n_components -> transformed X_train
            X_val_pca_dict: dict mapping n_components -> transformed X_val
            X_test_pca_dict: dict mapping n_components -> transformed X_test
            y_train, y_val, y_test: labels
            n_components_batch: list of n_components to process in this batch
            
        Returns:
            List of result dictionaries
        """
        batch_results = []
        
        for n_comp, explained_var in n_components_batch:
            try:
                X_train_pca = X_train_pca_dict[n_comp]
                X_val_pca = X_val_pca_dict[n_comp]
                X_test_pca = X_test_pca_dict[n_comp]
                
                # Train model
                metrics = self.train_and_evaluate_silent(
                    X_train_pca, X_val_pca, X_test_pca,
                    y_train, y_val, y_test,
                    self.default_params
                )
                
                if metrics is None:
                    continue
                
                result = {
                    'n_components': n_comp,
                    'explained_variance': explained_var,
                    'train_f1': metrics['train']['f1'],
                    'train_f2': metrics['train']['f2'],
                    'train_auc': metrics['train']['auc'],
                    'train_accuracy': metrics['train']['accuracy'],
                    'train_precision': metrics['train']['precision'],
                    'train_recall': metrics['train']['recall'],
                    'val_f1': metrics['val']['f1'],
                    'val_f2': metrics['val']['f2'],
                    'val_auc': metrics['val']['auc'],
                    'val_accuracy': metrics['val']['accuracy'],
                    'val_precision': metrics['val']['precision'],
                    'val_recall': metrics['val']['recall'],
                    'test_f1': metrics['test']['f1'],
                    'test_f2': metrics['test']['f2'],
                    'test_auc': metrics['test']['auc'],
                    'test_accuracy': metrics['test']['accuracy'],
                    'test_precision': metrics['test']['precision'],
                    'test_recall': metrics['test']['recall'],
                    'threshold': metrics['threshold'],
                    'training_time': metrics['training_time']
                }
                
                batch_results.append(result)
                
            except Exception as e:
                # Silent failure - just skip this config
                continue
        
        return batch_results
    
    def test_dataset_frozen(self, dataset_name, filepath):
        """
        [SECTION 1] Test model with frozen hyperparameters (OPTIMIZED PARALLELIZATION).
        """
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info(f"[SECTION 1: FROZEN MODEL] Dataset: {dataset_name}")
        self.logger.info("="*80)
        
        # Load data
        X, y = load_data(filepath)
        self.logger.info(f"Dataset shape: {X.shape}")
        # Ensure binary classification target
        unique_y = np.unique(y)
        if unique_y.size != 2:
            self.logger.warning(
                f"Skipping dataset {dataset_name}: target is not binary (unique values: {unique_y[:10]})"
            )
            return
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        n_features = X.shape[1]
        self.logger.info(f"Original features: {n_features}")
        self.logger.info(f"Using frozen default hyperparameters")
        
        # Test with all original features first
        self.logger.info(f"Testing with ALL ORIGINAL FEATURES...")
        metrics_all = self.train_and_evaluate_silent(
            X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test,
            self.default_params
        )
        
        if metrics_all is None:
            self.logger.warning("Failed to train with all features. Skipping this dataset.")
            return
        
        # Initialize results
        results = {
            'n_components': [n_features],
            'explained_variance': [1.0],
            'train_f1': [metrics_all['train']['f1']],
            'train_f2': [metrics_all['train']['f2']],
            'train_auc': [metrics_all['train']['auc']],
            'train_accuracy': [metrics_all['train']['accuracy']],
            'train_precision': [metrics_all['train']['precision']],
            'train_recall': [metrics_all['train']['recall']],
            'val_f1': [metrics_all['val']['f1']],
            'val_f2': [metrics_all['val']['f2']],
            'val_auc': [metrics_all['val']['auc']],
            'val_accuracy': [metrics_all['val']['accuracy']],
            'val_precision': [metrics_all['val']['precision']],
            'val_recall': [metrics_all['val']['recall']],
            'test_f1': [metrics_all['test']['f1']],
            'test_f2': [metrics_all['test']['f2']],
            'test_auc': [metrics_all['test']['auc']],
            'test_accuracy': [metrics_all['test']['accuracy']],
            'test_precision': [metrics_all['test']['precision']],
            'test_recall': [metrics_all['test']['recall']],
            'threshold': [metrics_all['threshold']],
            'training_time': [metrics_all['training_time']],
            'efficiency_score': [1.0],
            'acceptable': [True],
        }
        
        baseline_f1 = metrics_all['test']['f1']
        baseline_f2 = metrics_all['test']['f2']
        baseline_recall = metrics_all['test']['recall']
        baseline_time = metrics_all['training_time']
        
        self.logger.info(
            f"Baseline: F1={baseline_f1:.4f} | F2={baseline_f2:.4f} | "
            f"AUC={metrics_all['test']['auc']:.4f} | "
            f"Time={baseline_time:.2f}s | Recall={baseline_recall:.4f}"
        )
        
        # Generate list of component counts to test
        component_counts = []
        current_components = n_features
        while True:
            current_components = int(current_components * 0.9)
            if current_components < 2:
                break
            component_counts.append(current_components)
        
        if not component_counts:
            self.logger.info("No PCA configurations to test (dataset too small)")
            self.results_frozen[dataset_name] = results
            return
        
        self.logger.info(f"Testing {len(component_counts)} PCA configurations...")
        
        # OPTIMIZATION: Precompute ALL PCA transformations upfront
        self.logger.info("Precomputing PCA transformations...")
        X_train_pca_dict = {}
        X_val_pca_dict = {}
        X_test_pca_dict = {}
        explained_vars = {}
        
        for n_comp in component_counts:
            X_train_pca, X_val_pca, X_test_pca, pca, explained_var = self.apply_pca(
                X_train_scaled, X_val_scaled, X_test_scaled, n_comp
            )
            X_train_pca_dict[n_comp] = X_train_pca
            X_val_pca_dict[n_comp] = X_val_pca
            X_test_pca_dict[n_comp] = X_test_pca
            explained_vars[n_comp] = explained_var
        
        # Create batches for parallel execution (3-5 configs per batch)
        configs_with_variance = [(n, explained_vars[n]) for n in component_counts]
        batch_size = max(3, len(component_counts) // (self.n_jobs * 2))
        batches = [configs_with_variance[i:i+batch_size] 
                   for i in range(0, len(configs_with_variance), batch_size)]
        
        self.logger.info(f"Running {len(batches)} batches in parallel...")
        
        # PARALLEL EXECUTION with batching
        batch_results = Parallel(
            n_jobs=self.n_jobs, 
            verbose=0, 
            prefer="processes",
            batch_size=1
        )(
            delayed(self._train_frozen_batch)(
                X_train_pca_dict, X_val_pca_dict, X_test_pca_dict,
                y_train, y_val, y_test, batch
            )
            for batch in batches
        )
        
        # Flatten results and process
        f1_cutoff = baseline_f1 - 0.10
        for batch in batch_results:
            for result in batch:
                if result is None:
                    continue
                
                # Compute efficiency score
                eff_score, acceptable = self.compute_efficiency_score(
                    f2_score=result['test_f2'],
                    baseline_f2=baseline_f2,
                    recall=result['test_recall'],
                    baseline_recall=baseline_recall,
                    training_time=result['training_time'],
                    baseline_time=baseline_time,
                )
                result['efficiency_score'] = eff_score
                result['acceptable'] = acceptable
                
                # Log after worker returns
                self.logger.debug(
                    f"[{dataset_name}] Frozen | {result['n_components']:3d} comp | "
                    f"F1={result['test_f1']:.4f} | F2={result['test_f2']:.4f} | "
                    f"Eff={eff_score:.4f} | Acc={acceptable} | Var={result['explained_variance']*100:.1f}%"
                )
                
                # Add to results - append field by field
                results['n_components'].append(result['n_components'])
                results['explained_variance'].append(result['explained_variance'])
                results['training_time'].append(result['training_time'])
                results['efficiency_score'].append(eff_score)
                results['acceptable'].append(acceptable)
                results['threshold'].append(result['threshold'])
                
                for split in ['train', 'val', 'test']:
                    for metric in ['f1', 'f2', 'auc', 'accuracy', 'precision', 'recall']:
                        key = f'{split}_{metric}'
                        results[key].append(result[key])
        
        # Store results
        self.results_frozen[dataset_name] = results
        
        # Summary
        best_pca_f2 = max(results['test_f2'][1:]) if len(results['test_f2']) > 1 else 0
        best_idx = results['test_f2'].index(best_pca_f2) if best_pca_f2 > 0 else 0
        best_score = results['efficiency_score'][best_idx] if best_idx < len(results['efficiency_score']) else 0
        drop = ((baseline_f1 - results['test_f1'][best_idx]) / baseline_f1 * 100) if baseline_f1 > 0 else 0
        
        self.logger.info("")
        self.logger.info("Frozen Model Summary:")
        self.logger.info(f"  Baseline F2: {baseline_f2:.4f}")
        self.logger.info(f"  Best PCA F2: {best_pca_f2:.4f}")
        self.logger.info(f"  Best efficiency score: {best_score:.4f}")
        self.logger.info(f"  Performance drop: {drop:.1f}%")
    
    # ==================== SECTION 2: OPTUNA-TUNED ANALYSIS ====================
    
    def optuna_objective_for_pca(self, trial, X_train, X_val, X_test, y_train, y_val, y_test):
        """Optuna objective function for hyperparameter tuning."""
        # Define hyperparameter search space
        base_params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 5.0),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_bin": 256,
            "random_state": 42,
            "n_jobs": 1,
            "early_stopping_rounds": 10,
            "verbosity": 0,
        }
        
        # Add GPU or CPU specific params (XGBoost 3.1+)
        if self.gpu_available:
            base_params["tree_method"] = "gpu_hist"
            base_params["device"] = "cuda"
        else:
            base_params["tree_method"] = "hist"
            base_params["device"] = "cpu"
        
        try:
            metrics = self.train_and_evaluate_silent(
                X_train, X_val, X_test, y_train, y_val, y_test, base_params
            )
            
            if metrics is None:
                return 0.0
            
            trial.report(metrics['test']['f1'], step=0)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return metrics['test']['f1']
        
        except Exception as e:
            return 0.0
    
    def tune_hyperparameters_for_pca(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                                     n_components, explained_var, dataset_name):
        """Run Optuna hyperparameter optimization."""
        def objective(trial):
            return self.optuna_objective_for_pca(
                trial, X_train, X_val, X_test, y_train, y_val, y_test
            )
        
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=0)
        
        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )
        
        # OPTIMIZATION: Limit Optuna parallelism (scales poorly beyond 4)
        optuna_jobs = min(4, self.n_jobs)
        
        study.optimize(
            objective, 
            n_trials=self.n_optuna_trials, 
            n_jobs=optuna_jobs,
            show_progress_bar=False
        )
        
        best_trial = study.best_trial
        best_params = {
            "n_estimators": best_trial.params["n_estimators"],
            "max_depth": best_trial.params["max_depth"],
            "learning_rate": best_trial.params["learning_rate"],
            "reg_lambda": best_trial.params["reg_lambda"],
            "gamma": best_trial.params["gamma"],
            "min_child_weight": best_trial.params["min_child_weight"],
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_bin": 256,
            "random_state": 42,
            "n_jobs": 1,
            "early_stopping_rounds": 10,
            "verbosity": 0,
        }
        
        # Add GPU or CPU specific params (XGBoost 3.1+)
        if self.gpu_available:
            best_params["tree_method"] = "gpu_hist"
            best_params["device"] = "cuda"
        else:
            best_params["tree_method"] = "hist"
            best_params["device"] = "cpu"
        
        # Train final model with best params to get all metrics
        try:
            n_samples = X_train.shape[0]
            n_features = X_train.shape[1]
        except Exception:
            n_samples = None
            n_features = None

        do_pca = True
        if n_features is not None and n_features == n_components:
            # Already in desired dimensionality
            X_train_pca, X_val_pca, X_test_pca = X_train, X_val, X_test
            explained_var = explained_var
            do_pca = False
        elif n_samples is not None and n_components >= min(n_samples, n_features):
            # Cannot request more components than samples; use original data
            X_train_pca, X_val_pca, X_test_pca = X_train, X_val, X_test
            explained_var = 1.0
            do_pca = False

        if do_pca:
            X_train_pca, X_val_pca, X_test_pca, _, explained_var = self.apply_pca(
                X_train, X_val, X_test, n_components
            )

        final_metrics = self.train_and_evaluate_silent(
            X_train_pca, X_val_pca, X_test_pca,
            y_train, y_val, y_test,
            best_params
        )
        
        return best_params, final_metrics
    
    def test_dataset_tuned(self, dataset_name, filepath):
        """[SECTION 2] Test model with Optuna-tuned hyperparameters."""
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info(f"[SECTION 2: OPTUNA-TUNED] Dataset: {dataset_name}")
        self.logger.info("="*80)
        self.logger.info(f"Running Optuna optimization ({self.n_optuna_trials} trials per config)")
        
        # Load data
        X, y = load_data(filepath)
        self.logger.info(f"Dataset shape: {X.shape}")
        # Ensure binary classification target
        unique_y = np.unique(y)
        if unique_y.size != 2:
            self.logger.warning(
                f"Skipping dataset {dataset_name}: target is not binary (unique values: {unique_y[:10]})"
            )
            return
        
        # Split and scale data
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        n_features = X.shape[1]
        self.logger.info(f"Original features: {n_features}")
        
        # Test with all original features first
        self.logger.info(f"Tuning hyperparameters for ALL ORIGINAL FEATURES...")
        
        best_params_all, metrics_all = self.tune_hyperparameters_for_pca(
            X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train, y_val, y_test, n_features, 1.0, dataset_name
        )
        if metrics_all is None:
            self.logger.warning(f"Optuna tuning failed for baseline on {dataset_name}. Skipping tuned analysis.")
            return
        
        # Initialize results
        results = {
            'n_components': [n_features],
            'explained_variance': [1.0],
            'train_f1': [metrics_all['train']['f1']],
            'train_f2': [metrics_all['train']['f2']],
            'train_auc': [metrics_all['train']['auc']],
            'train_accuracy': [metrics_all['train']['accuracy']],
            'train_precision': [metrics_all['train']['precision']],
            'train_recall': [metrics_all['train']['recall']],
            'val_f1': [metrics_all['val']['f1']],
            'val_f2': [metrics_all['val']['f2']],
            'val_auc': [metrics_all['val']['auc']],
            'val_accuracy': [metrics_all['val']['accuracy']],
            'val_precision': [metrics_all['val']['precision']],
            'val_recall': [metrics_all['val']['recall']],
            'test_f1': [metrics_all['test']['f1']],
            'test_f2': [metrics_all['test']['f2']],
            'test_auc': [metrics_all['test']['auc']],
            'test_accuracy': [metrics_all['test']['accuracy']],
            'test_precision': [metrics_all['test']['precision']],
            'test_recall': [metrics_all['test']['recall']],
            'threshold': [metrics_all['threshold']],
            'training_time': [metrics_all['training_time']],
            'best_params': [best_params_all],
            'efficiency_score': [1.0],
            'acceptable': [True],
        }
        
        baseline_f1 = metrics_all['test']['f1']
        baseline_f2 = metrics_all['test']['f2']
        baseline_recall = metrics_all['test']['recall']
        baseline_time = metrics_all['training_time']
        
        self.logger.info(
            f"Baseline: F1={baseline_f1:.4f} | F2={baseline_f2:.4f} | "
            f"AUC={metrics_all['test']['auc']:.4f} | "
            f"Time={baseline_time:.2f}s | Recall={baseline_recall:.4f}"
        )
        
        # Generate component counts
        f1_cutoff = baseline_f1 - 0.10
        
        component_counts = []
        current_components = n_features
        while True:
            current_components = int(current_components * 0.9)
            if current_components < 2:
                break
            component_counts.append(current_components)
        
        if not component_counts:
            self.logger.info("No PCA configurations to test (dataset too small)")
            self.results_tuned[dataset_name] = results
            return
        
        self.logger.info(f"Testing {len(component_counts)} PCA configurations...")
        
        # Sequential execution (Optuna parallelizes internally)
        for n_comp in component_counts:
            self.logger.info(f"Optimizing for {n_comp} components...")
            
            X_train_pca, X_val_pca, X_test_pca, pca, explained_var = self.apply_pca(
                X_train_scaled, X_val_scaled, X_test_scaled, n_comp
            )
            
            best_params, metrics = self.tune_hyperparameters_for_pca(
                X_train_pca, X_val_pca, X_test_pca,
                y_train, y_val, y_test,
                n_comp, explained_var, dataset_name
            )
            
            # Compute efficiency score
            eff_score, acceptable = self.compute_efficiency_score(
                f2_score=metrics['test']['f2'],
                baseline_f2=baseline_f2,
                recall=metrics['test']['recall'],
                baseline_recall=baseline_recall,
                training_time=metrics['training_time'],
                baseline_time=baseline_time,
            )
            
            # Log after optimization
            self.logger.debug(
                f"[{dataset_name}] Tuned | {n_comp:3d} comp | "
                f"F1={metrics['test']['f1']:.4f} | F2={metrics['test']['f2']:.4f} | "
                f"Eff={eff_score:.4f} | Acc={acceptable} | Var={explained_var*100:.1f}%"
            )
            
            # Add to results
            results['n_components'].append(n_comp)
            results['explained_variance'].append(explained_var)
            results['best_params'].append(best_params)
            results['training_time'].append(metrics['training_time'])
            results['efficiency_score'].append(eff_score)
            results['acceptable'].append(acceptable)
            
            for split in ['train', 'val', 'test']:
                for metric in ['f1', 'f2', 'auc', 'accuracy', 'precision', 'recall']:
                    key = f'{split}_{metric}'
                    results[key].append(metrics[split][metric])
            
            results['threshold'].append(metrics['threshold'])
        
        # Store results
        self.results_tuned[dataset_name] = results
        
        # Summary
        best_pca_f2 = max(results['test_f2'][1:]) if len(results['test_f2']) > 1 else baseline_f2
        best_idx = results['test_f2'].index(best_pca_f2) if best_pca_f2 > 0 else 0
        best_score = results['efficiency_score'][best_idx] if best_idx < len(results['efficiency_score']) else 0
        drop = ((baseline_f1 - results['test_f1'][best_idx]) / baseline_f1 * 100) if baseline_f1 > 0 else 0
        
        self.logger.info("")
        self.logger.info("Tuned Model Summary:")
        self.logger.info(f"  Baseline F2: {baseline_f2:.4f}")
        self.logger.info(f"  Best PCA F2: {best_pca_f2:.4f}")
        self.logger.info(f"  Best efficiency score: {best_score:.4f}")
        self.logger.info(f"  Performance drop: {drop:.1f}%")
    
    # ==================== VISUALIZATION & REPORTING ====================
    
    def _make_serializable(self, obj):
        """Helper: make results JSON-serializable"""
        import numpy as _np
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return self._make_serializable(obj.tolist())
        return obj
    
    def plot_dataset_results(self, dataset_name):
        """Generate comprehensive plots for a single dataset and save JSON report."""
        has_frozen = dataset_name in self.results_frozen
        has_tuned = dataset_name in self.results_tuned
        if not has_frozen and not has_tuned:
            return

        # prepare per-dataset directory
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # 2 rows x 2 columns: left = frozen, right = tuned
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'PCA Analysis: {dataset_name}',
            fontsize=16,
            fontweight='bold',
            y=0.995,
        )

        metrics_to_plot = [
            ('test_f1', 'F1 Score'),
            ('test_f2', 'F2 Score'),
            ('test_auc', 'AUC Score'),
            ('test_accuracy', 'Accuracy'),
            ('test_precision', 'Precision'),
            ('test_recall', 'Recall'),
        ]

        def sorted_pairs(results, metric_key):
            """Return sorted x (n_components) and y (metric) by ascending n_components"""
            n_comp = results['n_components']
            vals = results[metric_key]
            pairs = sorted(zip(n_comp, vals), key=lambda x: x[0])
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            return xs, ys

        # Plot 1: Frozen Model - All Metrics vs Components
        ax = axes[0, 0]
        if has_frozen:
            results = self.results_frozen[dataset_name]
            for metric_key, metric_name in metrics_to_plot:
                n_comp, values = sorted_pairs(results, metric_key)
                ax.plot(n_comp, values, marker='o', label=metric_name, linewidth=2, markersize=6)
        ax.set_xlabel('Number of PCA Components', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('[Frozen Model] All Metrics vs Components', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        # Plot 2: Tuned Model - All Metrics vs Components
        ax = axes[0, 1]
        if has_tuned:
            results = self.results_tuned[dataset_name]
            for metric_key, metric_name in metrics_to_plot:
                n_comp, values = sorted_pairs(results, metric_key)
                ax.plot(n_comp, values, marker='o', label=metric_name, linewidth=2, markersize=6)
        ax.set_xlabel('Number of PCA Components', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('[Tuned Model] All Metrics vs Components', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        # Plot 3: Efficiency Score vs Components (Frozen)
        ax = axes[1, 0]
        if has_frozen:
            results = self.results_frozen[dataset_name]
            n_comp, eff_scores = sorted_pairs(results, 'efficiency_score')
            _, acceptable = sorted_pairs(results, 'acceptable')
            
            # Separate acceptable and unacceptable points
            acceptable_n_comp = [n for n, acc in zip(n_comp, acceptable) if acc]
            acceptable_scores = [s for s, acc in zip(eff_scores, acceptable) if acc]
            unacceptable_n_comp = [n for n, acc in zip(n_comp, acceptable) if not acc]
            unacceptable_scores = [s for s, acc in zip(eff_scores, acceptable) if not acc]
            
            ax.plot(acceptable_n_comp, acceptable_scores, marker='o', linewidth=2, markersize=8, color='green', label='Acceptable')
            if unacceptable_n_comp:
                ax.plot(unacceptable_n_comp, unacceptable_scores, marker='x', linewidth=2, markersize=8, color='red', label='Unacceptable')
        
        ax.set_xlabel('Number of PCA Components', fontweight='bold')
        ax.set_ylabel('Efficiency Score', fontweight='bold')
        ax.set_title('[Frozen Model] Efficiency Score vs Components', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        # Plot 4: Efficiency Score vs Components (Tuned)
        ax = axes[1, 1]
        if has_tuned:
            results = self.results_tuned[dataset_name]
            n_comp, eff_scores = sorted_pairs(results, 'efficiency_score')
            _, acceptable = sorted_pairs(results, 'acceptable')
            
            # Separate acceptable and unacceptable points
            acceptable_n_comp = [n for n, acc in zip(n_comp, acceptable) if acc]
            acceptable_scores = [s for s, acc in zip(eff_scores, acceptable) if acc]
            unacceptable_n_comp = [n for n, acc in zip(n_comp, acceptable) if not acc]
            unacceptable_scores = [s for s, acc in zip(eff_scores, acceptable) if not acc]
            
            ax.plot(acceptable_n_comp, acceptable_scores, marker='o', linewidth=2, markersize=8, color='green', label='Acceptable')
            if unacceptable_n_comp:
                ax.plot(unacceptable_n_comp, unacceptable_scores, marker='x', linewidth=2, markersize=8, color='red', label='Unacceptable')

        ax.set_xlabel('Number of PCA Components', fontweight='bold')
        ax.set_ylabel('Efficiency Score', fontweight='bold')
        ax.set_title('[Tuned Model] Efficiency Score vs Components', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        try:
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            img_name = f"pca_summary_{dataset_name}.png"
            img_path = dataset_dir / img_name
            fig.savefig(img_path, dpi=200)
            plt.close(fig)
            self.logger.info(f"Saved plot to: {img_path}")
        except Exception as e:
            self.logger.error(f"Failed to save plot for {dataset_name}: {e}")

        # Save JSON report
        try:
            report = {
                'dataset': dataset_name,
                'generated_at': datetime.now().isoformat(),
                'frozen': self._make_serializable(self.results_frozen.get(dataset_name, {})),
                'tuned': self._make_serializable(self.results_tuned.get(dataset_name, {})),
            }
            json_name = f"pca_report_{dataset_name}.json"
            json_path = dataset_dir / json_name
            with open(json_path, 'w') as jf:
                json.dump(report, jf, indent=2)
            self.logger.info(f"Saved JSON report to: {json_path}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON report for {dataset_name}: {e}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run PCAAnalysis for XGBoost datasets with GPU support.")
    parser.add_argument("--datasets-dir", "-d", type=str, default="datasets",
                        help="Directory containing CSV datasets (default: datasets)")
    parser.add_argument("--dataset-file", "-f", type=str, default=None,
                        help="Run on a single dataset file (overrides --datasets-dir)")
    parser.add_argument("--run", "-r", choices=["frozen", "tuned", "both"], default="both",
                        help="Which section to run: frozen, tuned, or both (default: both)")
    parser.add_argument("--n-trials", "-t", type=int, default=30,
                        help="Number of Optuna trials per PCA config (default: 30)")
    parser.add_argument("--n-jobs", "-j", type=int, default=-1,
                        help="Number of parallel jobs (-1 = all cores, -2 = all but one) (default: -1)")
    parser.add_argument("--output-dir", "-o", type=str, default="pca_results",
                        help="Directory to write logs and plots (default: pca_results)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging to console (INFO level)")
    parser.add_argument("--parallel-datasets", action="store_true",
                        help="Parallelize at dataset level (recommended for multiple datasets)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration (use CPU only)")
    return parser.parse_args(argv)


# Worker function for dataset-level parallelization
def process_dataset_worker(name, path, args):
    """Worker function to process a single dataset (for dataset-level parallelization)"""
    # Create a new PCAAnalysis instance (avoids shared state issues)
    pa = PCAAnalysis(
        datasets_dir=args.datasets_dir,
        n_optuna_trials=args.n_trials,
        n_jobs=1,  # Each dataset gets 1 core in this mode
        output_dir=args.output_dir,
        use_gpu=not args.no_gpu
    )
    
    # Suppress console output in workers
    for handler in pa.logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.WARNING)
    
    results = {'name': name, 'frozen': None, 'tuned': None}
    
    try:
        if args.run in ("frozen", "both"):
            pa.test_dataset_frozen(name, path)
            results['frozen'] = pa.results_frozen.get(name)
    except Exception as e:
        pa.logger.error(f"Error running frozen analysis for {name}: {e}")
    
    try:
        if args.run in ("tuned", "both"):
            pa.test_dataset_tuned(name, path)
            results['tuned'] = pa.results_tuned.get(name)
    except Exception as e:
        pa.logger.error(f"Error running tuned analysis for {name}: {e}")
    
    return results


if __name__ == "__main__":
    args = parse_args()
    
    # Build dataset list
    dataset_paths = []
    if args.dataset_file:
        dataset_paths = [(Path(args.dataset_file).stem, args.dataset_file)]
    else:
        datasets_dir = Path(args.datasets_dir)
        if not datasets_dir.exists():
            print(f"Error: Datasets directory not found: {datasets_dir}")
            sys.exit(1)
        for p in sorted(datasets_dir.glob("*.csv")):
            dataset_paths.append((p.stem, str(p)))
    
    if not dataset_paths:
        print("Error: No datasets found to process.")
        sys.exit(1)
    
    # OPTIMIZATION: Dataset-level parallelization for multiple datasets
    if args.parallel_datasets and len(dataset_paths) > 1:
        print(f"Running in PARALLEL DATASET mode with {len(dataset_paths)} datasets")
        print("="*80)
        
        # Create main analysis object for collecting results
        pa = PCAAnalysis(
            datasets_dir=args.datasets_dir,
            n_optuna_trials=args.n_trials,
            n_jobs=args.n_jobs,
            output_dir=args.output_dir,
            use_gpu=not args.no_gpu
        )
        
        # Process datasets in parallel
        all_results = Parallel(n_jobs=pa.n_jobs, verbose=10)(
            delayed(process_dataset_worker)(name, path, args)
            for name, path in dataset_paths
        )
        
        # Collect results
        for result in all_results:
            name = result['name']
            if result['frozen']:
                pa.results_frozen[name] = result['frozen']
            if result['tuned']:
                pa.results_tuned[name] = result['tuned']
        
        # DEFERRED: Generate all plots and JSON at the end
        pa.logger.info("")
        pa.logger.info("="*80)
        pa.logger.info("Generating plots and reports...")
        pa.logger.info("="*80)
        
        for name, _ in dataset_paths:
            try:
                pa.plot_dataset_results(name)
            except Exception as e:
                pa.logger.error(f"Error saving plot for {name}: {e}")
    
    else:
        # Sequential mode (original behavior)
        pa = PCAAnalysis(
            datasets_dir=args.datasets_dir,
            n_optuna_trials=args.n_trials,
            n_jobs=args.n_jobs,
            output_dir=args.output_dir,
            use_gpu=not args.no_gpu
        )
        
        if args.verbose:
            pa.logger.setLevel(logging.DEBUG)
        
        for name, path in dataset_paths:
            pa.logger.info("")
            pa.logger.info("#"*80)
            pa.logger.info(f"Processing dataset: {name}")
            pa.logger.info("#"*80)
            
            if args.run in ("frozen", "both"):
                try:
                    pa.test_dataset_frozen(name, path)
                except Exception as e:
                    pa.logger.error(f"Error running frozen analysis for {name}: {e}")
            
            if args.run in ("tuned", "both"):
                try:
                    pa.test_dataset_tuned(name, path)
                except Exception as e:
                    pa.logger.error(f"Error running tuned analysis for {name}: {e}")
        
        # DEFERRED: Generate all plots and JSON at the end
        pa.logger.info("")
        pa.logger.info("="*80)
        pa.logger.info("Generating plots and reports...")
        pa.logger.info("="*80)
        
        for name, _ in dataset_paths:
            try:
                pa.plot_dataset_results(name)
            except Exception as e:
                pa.logger.error(f"Error saving plot for {name}: {e}")
    
    print("\nAll done.")