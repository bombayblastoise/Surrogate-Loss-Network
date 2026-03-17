import os
import glob
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, fbeta_score
from joblib import Parallel, delayed

# --- CONFIGURATION ---
TARGET_COL_INDEX = 0  
N_JOBS = -1           # Use all cores

warnings.filterwarnings('ignore')

# --- HELPER FUNCTIONS ---

def get_metrics(y_true, y_prob):
    """Find best F0.5 threshold and return all metrics."""
    best_thresh = 0.5
    best_score = -1.0
    # Coarser search for speed since we run many dims
    thresholds = np.arange(0.1, 0.95, 0.05)
    
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
    def __init__(self, n_components, random_state=42):
        self.n_components = n_components
        self.rng = np.random.default_rng(random_state)
        self.random_state = random_state

    def fit_transform(self, X):
        n_samples, n_features = X.shape
        stds = np.std(X, axis=0)
        stds[stds == 0] = 1.0 
        
        density = 1 / np.sqrt(n_features)
        
        # Unique matrix per random_state
        R_raw = sparse.random(n_features, self.n_components, density=density, 
                              random_state=self.rng, format='csr')
        
        R_raw.data = self.rng.choice([-1, 1], size=R_raw.nnz)
        scaling_diag = sparse.diags(1.0 / stds)
        R_scaled = scaling_diag @ R_raw
        
        return X @ R_scaled

def apply_federated_projection(X, y, n_components):
    """
    Splits data into 5 chunks, projects each with a UNIQUE seed.
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
        # CRITICAL: 'i' changes (0, 1, 2, 3, 4), so the matrix R is different for each chunk.
        projector = VarianceScaledProjection(n_components, random_state=i)
        
        X_chunk = X[idx]
        y_chunk = y[idx]
        
        X_proj = projector.fit_transform(X_chunk)
        
        X_parts.append(X_proj)
        y_parts.append(y_chunk)
        
    return np.vstack(X_parts), np.concatenate(y_parts)

# --- WORKER FUNCTION ---

def run_dimension_trial(n_dim, X_t, y_t, X_v, y_v, max_weight):
    """
    Runs one experiment for a specific output dimension `n_dim`.
    """
    # 1. Federated Projection
    # Project Train and Val separately (simulating local processing)
    X_t_proj, y_t_proj = apply_federated_projection(X_t, y_t, n_dim)
    X_v_proj, y_v_proj = apply_federated_projection(X_v, y_v, n_dim)
    
    # 2. Train XGBoost (Frozen Params to isolate Dimension effect)
    params = {
        'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
        'scale_pos_weight': max(1.0, max_weight),
        'objective': 'binary:logistic', 'tree_method': 'hist', 'n_jobs': 1, 'verbosity': 0
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_t_proj, y_t_proj, eval_set=[(X_v_proj, y_v_proj)], verbose=False)
    
    probs = model.predict_proba(X_v_proj)[:, 1]
    
    # Metrics
    metrics = get_metrics(y_v_proj, probs)
    
    # FIX: Explicitly cast NumPy int64 to Python int
    metrics['n_dim'] = int(n_dim)
    
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

        # Split: Train/Test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Internal Split (Train -> SubTrain / Val)
        X_t, X_v, y_t, y_v = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # --- 1. BASELINE (Full Dimensions) ---
        b_params = {
            'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
            'scale_pos_weight': max(1.0, max_weight), 'n_jobs': 1, 'verbosity': 0
        }
        b_model = xgb.XGBClassifier(**b_params)
        b_model.fit(X_t, y_t)
        b_metrics = get_metrics(y_test, b_model.predict_proba(X_test)[:, 1])

        # --- 2. DIMENSION SWEEP (1 to n/2) ---
        max_dim = max(2, n_features // 2)
        dims_to_test = np.unique(np.linspace(1, max_dim, num=20, dtype=int))
        # Ensure at least a few points
        if len(dims_to_test) < 5:
            dims_to_test = np.arange(1, max_dim + 1)
            
        # Parallel Execution over dimensions
        results = Parallel(n_jobs=N_JOBS)(
            delayed(run_dimension_trial)(
                d, X_t, y_t, X_v, y_v, max_weight
            ) for d in dims_to_test
        )

        return {
            "dataset_name": dataset_name,
            "n_features_orig": int(n_features), # FIX: Cast to int
            "baseline": b_metrics,
            "sweep_results": results
        }

    except Exception as e:
        print(f"Error {dataset_name}: {e}")
        return None

if __name__ == "__main__":
    OUT_DIR = "jl_fl_dim_sensitivity"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    datasets = glob.glob("all_datasets/*.csv")
    
    for ds_path in datasets:
        res = process_dataset(ds_path)
        if res:
            with open(f"{OUT_DIR}/{res['dataset_name']}.json", 'w') as f:
                json.dump(res, f, indent=4)
                
    print(f"Done. Saved to {OUT_DIR}")