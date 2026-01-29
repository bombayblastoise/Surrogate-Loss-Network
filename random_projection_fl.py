import os
import glob
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed

# 1. XGBoost Parameters
PARAMS = {
    "n_estimators": 68,
    "max_depth": 4,
    "learning_rate": 0.20616995910873692,
    "reg_lambda": 0.1771611360412841,
    "gamma": 0.38746422872065867,
    "min_child_weight": 4.873868192482736,
    "objective": "binary:logistic",
    "max_bin": 64,
    "n_jobs": 1,  # Single thread per model to allow joblib parallelization
    "verbosity": 0
}

def get_metrics(y_true, y_pred, y_prob):
    """Calculate Accuracy, Precision, F1, and AUC."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob))
    }

def process_config(file_path, split_type, dim_offset):
    """
    Worker function to process one configuration on one core.
    Loads data, imputes, projects (if needed), trains, and returns metrics.
    """
    # Load Data
    try:
        df = pd.read_csv(file_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Impute Missing Values (Median)
        if np.isnan(X).any():
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)

        n_features = X.shape[1]
        target_dim = 0 # Placeholder

        # Determine if Baseline or Experiment
        if dim_offset is None:
            X_final = X
            config_type = "baseline"
        else:
            config_type = "experiment"
            
            # Calculate target dimension: 10 * ln(features) + offset
            target_dim = int(10 * np.log(n_features)) + dim_offset
            if target_dim < 1: target_dim = 1
            
            # Create Split Indices (5 chunks)
            n_rows = X.shape[0]
            indices = np.arange(n_rows)
            
            if split_type == 'random':
                np.random.seed(42)
                np.random.shuffle(indices)
            
            # Split indices into 5 equal parts (ensuring every row is used once)
            chunks = np.array_split(indices, 5)
            
            # Federated Projection
            X_parts = []
            y_parts = []
            
            for i, idx in enumerate(chunks):
                # Generate Random Gaussian Matrix (Seed varies by chunk index i)
                np.random.seed(i)
                R = np.random.randn(n_features, target_dim)
                
                # Project this chunk
                X_parts.append(np.dot(X[idx], R))
                y_parts.append(y[idx])
                
            # Concatenate (vstack) to restore original row count with reduced features
            X_final = np.vstack(X_parts)
            y = np.concatenate(y_parts) # Reorder y to match X shuffle

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

        # Run XGBoost
        model = xgb.XGBClassifier(**PARAMS)
        model.fit(X_train, y_train)

        # Calculate Metrics
        metrics = get_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])
        
        # Return structured result
        return {
            "dataset_name": os.path.basename(file_path).replace('.csv', ''),
            "type": config_type,
            "split_method": split_type,
            "target_dim": target_dim,
            "metrics": metrics
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# 2. Main Execution
if __name__ == "__main__":
    # Setup directories
    os.makedirs("jl_fl_results", exist_ok=True)
    datasets = glob.glob("datasets/*.csv")
    tasks = []

    # 3. Generate Task List
    for ds in datasets:
        # A. Baseline Task
        tasks.append((ds, None, None))
        
        # B. Federated Configs
        # 5 output dimensions * 2 split types = 10 configs per dataset
        dims = [-2, -1, 0, 1, 2]
        for split in ['random', 'ordered']:
            for d in dims:
                tasks.append((ds, split, d))

    print(f"Generated {len(tasks)} tasks. Processing on all cores...")
    
    # 4. Parallel Execution
    results_flat = Parallel(n_jobs=-1)(
        delayed(process_config)(d, s, o) for d, s, o in tasks
    )

    # 5. Group Results by Dataset
    grouped_data = {}
    
    for res in results_flat:
        if not res: continue
        
        ds_name = res['dataset_name']
        if ds_name not in grouped_data:
            grouped_data[ds_name] = {"baseline": {}, "experiments": []}
        
        if res['type'] == 'baseline':
            grouped_data[ds_name]['baseline'] = res['metrics']
        else:
            entry = {
                "split_method": res['split_method'],
                "target_dim": res['target_dim'],
                **res['metrics'] # Unpack accuracy, f1, etc.
            }
            grouped_data[ds_name]['experiments'].append(entry)

    # 6. Save JSON per Dataset
    for ds_name, content in grouped_data.items():
        output_path = os.path.join("jl_fl_results", f"{ds_name}.json")
        with open(output_path, 'w') as f:
            json.dump(content, f, indent=4)
            
    print(f"Successfully saved JSON results for {len(grouped_data)} datasets in 'jl_fl_results/'.")
