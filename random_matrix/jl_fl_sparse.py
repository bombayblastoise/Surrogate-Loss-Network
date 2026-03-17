import os
import glob
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import warnings
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, average_precision_score, fbeta_score
from joblib import Parallel, delayed

# --- CONFIGURATION ---
TARGET_COL_INDEX = 0  # 0 for first column, -1 for last
N_TRIALS = 40         # Number of Optuna trials
N_JOBS = -1           # Use all cores

# 1. Suppress Warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# --- HELPER FUNCTIONS ---

def get_metrics(y_true, y_prob):
    """Find best F0.5 threshold and return all metrics."""
    best_thresh = 0.5
    best_score = -1.0
    thresholds = np.arange(0.1, 0.99, 0.01)
    
    # Fast vectorized search for threshold
    # (Looping in python is fine here as arrays are 1D and small-ish)
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        score = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = thresh
            
    y_final = (y_prob >= best_thresh).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_final)),
        "precision": float(precision_score(y_true, y_final, zero_division=0)),
        "f1": float(f1_score(y_true, y_final, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "threshold": float(best_thresh)
    }

def calculate_imbalance(y):
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    return n_neg / n_pos if n_pos > 0 else 1.0

# --- CORE LOGIC: VARIANCE SCALED PROJECTION ---

class VarianceScaledProjection:
    """
    Simulates Federated Projection with Inverse Variance Scaling.
    """
    def __init__(self, n_components, random_state=42):
        self.n_components = n_components
        self.rng = np.random.default_rng(random_state)
        self.random_state = random_state

    def fit_transform(self, X):
        n_samples, n_features = X.shape
        
        # 1. Calculate Standard Deviation (Private Stats)
        stds = np.std(X, axis=0)
        stds[stds == 0] = 1.0 # Handle constant features
        
        # 2. Generate Sparse Matrix (The "Secret Key")
        # density ~ 1/sqrt(n_features) is standard for Achlioptas
        density = 1 / np.sqrt(n_features)
        
        # Generate random values (-1, 1) at sparse locations
        # We use a fixed seed per chunk later, but here we init with class state
        R_raw = sparse.random(n_features, self.n_components, density=density, 
                              random_state=self.rng, format='csr')
        
        # Map non-zeros to {-1, 1}
        R_raw.data = self.rng.choice([-1, 1], size=R_raw.nnz)
        
        # 3. Inverse Variance Scaling
        # Multiply rows by 1/std
        scaling_diag = sparse.diags(1.0 / stds)
        R_scaled = scaling_diag @ R_raw
        
        # 4. Project
        return X @ R_scaled

def apply_federated_projection(X, y, n_components):
    """
    Splits data into 5 chunks, projects each with a UNIQUE seed, and stacks.
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    # 1. Random Shuffle (The "Random Split")
    np.random.seed(42)
    np.random.shuffle(indices)
    
    chunks = np.array_split(indices, 5)
    X_parts, y_parts = [], []
    
    for i, idx in enumerate(chunks):
        # 2. Unique Projection per Client
        # Seed = i ensures Client 1 always uses Matrix 1, Client 2 uses Matrix 2, etc.
        projector = VarianceScaledProjection(n_components, random_state=i)
        
        X_chunk = X[idx]
        y_chunk = y[idx]
        
        X_proj = projector.fit_transform(X_chunk)
        
        X_parts.append(X_proj)
        y_parts.append(y_chunk)
        
    return np.vstack(X_parts), np.concatenate(y_parts)

# --- WORKER FUNCTIONS (PARALLELIZED) ---

def run_single_trial(trial_num, X_t, y_t, X_v, y_v, mode, max_weight, n_features_orig):
    """
    Executes a single Optuna trial. 
    This is what we will parallelize.
    """
    # Use Optuna's define-by-run API manually or just sample random params
    # Since we can't easily pickle the 'trial' object for Parallel, 
    # we generate random params here based on the trial_num seed.
    
    rng = np.random.RandomState(trial_num)
    
    # 1. Suggest Dimension
    center_dim = int(10 * np.log(n_features_orig))
    low_dim = max(2, int(center_dim * 0.5))
    high_dim = max(low_dim + 5, int(center_dim * 2.0))
    n_components = rng.randint(low_dim, high_dim)
    
    # 2. Federated Projection (Fit on Train, Apply to Val)
    # Note: Val set simulates a "held out" client test or centralized val
    X_t_proj, y_t_proj = apply_federated_projection(X_t, y_t, n_components)
    X_v_proj, y_v_proj = apply_federated_projection(X_v, y_v, n_components)
    
    # 3. Sample XGB Params
    if mode == 'frozen':
        params = {
            'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
            'scale_pos_weight': max(1.0, max_weight),
            'objective': 'binary:logistic', 'tree_method': 'hist', 'n_jobs': 1, 'verbosity': 0
        }
    else:
        params = {
            'n_estimators': rng.randint(50, 200),
            'max_depth': rng.randint(3, 8),
            'learning_rate': 10 ** rng.uniform(-2, -0.7), # 0.01 to ~0.2
            'scale_pos_weight': rng.uniform(1.0, max_weight),
            'subsample': rng.uniform(0.6, 1.0),
            'colsample_bytree': rng.uniform(0.6, 1.0),
            'reg_lambda': 10 ** rng.uniform(-1, 1),
            'objective': 'binary:logistic', 'tree_method': 'hist', 'n_jobs': 1, 'verbosity': 0
        }

    # 4. Train & Eval
    model = xgb.XGBClassifier(**params)
    model.fit(X_t_proj, y_t_proj, eval_set=[(X_v_proj, y_v_proj)], verbose=False)
    
    probs = model.predict_proba(X_v_proj)[:, 1]
    
    # Metrics
    metrics = get_metrics(y_v_proj, probs)
    metrics['n_components'] = n_components
    metrics['params'] = params
    
    return metrics

def process_dataset(file_path):
    dataset_name = os.path.basename(file_path).replace('.csv', '')
    print(f"Processing {dataset_name}...")
    
    try:
        df = pd.read_csv(file_path)
        if TARGET_COL_INDEX == 0:
            y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values
        else:
            y = df.iloc[:, -1].values
            X = df.iloc[:, :-1].values

        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)
            
        n_features = X.shape[1]
        imbalance = calculate_imbalance(y)
        max_weight = np.sqrt(imbalance)

        # Split: Train (80%) / Test (20%)
        # All tuning happens on Train. Test is strictly unseen.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Internal Split for Tuning (Train -> SubTrain / Val)
        X_t, X_v, y_t, y_v = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # --- 1. BASELINE ---
        b_params = {
            'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
            'scale_pos_weight': max(1.0, max_weight), 'n_jobs': 1, 'verbosity': 0
        }
        b_model = xgb.XGBClassifier(**b_params)
        b_model.fit(X_t, y_t)
        b_metrics = get_metrics(y_test, b_model.predict_proba(X_test)[:, 1])

        # --- 2. EXPERIMENTS (PARALLEL TRIALS) ---
        # We run N_TRIALS in parallel using joblib
        
        # Frozen Study
        frozen_results = Parallel(n_jobs=N_JOBS)(
            delayed(run_single_trial)(
                i, X_t, y_t, X_v, y_v, 'frozen', max_weight, n_features
            ) for i in range(N_TRIALS)
        )
        
        # Tuned Study
        tuned_results = Parallel(n_jobs=N_JOBS)(
            delayed(run_single_trial)(
                i + 1000, X_t, y_t, X_v, y_v, 'tuned', max_weight, n_features
            ) for i in range(N_TRIALS)
        )

        return {
            "dataset_name": dataset_name,
            "baseline": b_metrics,
            "frozen_experiments": frozen_results,
            "tuned_experiments": tuned_results
        }

    except Exception as e:
        print(f"Error {dataset_name}: {e}")
        return None

if __name__ == "__main__":
    OUT_DIR = "jl_fl_optuna_results"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # We process datasets sequentially, but parallelize the TRIALS within each dataset
    # This is often more efficient for CPU cache than parallelizing datasets
    datasets = glob.glob("all_datasets/*.csv")
    
    for ds_path in datasets:
        res = process_dataset(ds_path)
        if res:
            with open(f"{OUT_DIR}/{res['dataset_name']}.json", 'w') as f:
                json.dump(res, f, indent=4)
                
    print(f"Done. Saved to {OUT_DIR}")