import os
import sys
import numpy as np
import pandas as pd
from xgboost_model import XGBoostModel
from objective import Objective
from metric import Summary
from analysis import PostTrainingAnalysis
import json
from datetime import datetime
import optuna
from optuna.pruners import MedianPruner
from sklearn.metrics import f1_score, precision_recall_fscore_support


def load_data(filepath):
    """Load data from CSV file"""
    if not os.path.isabs(filepath):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filepath)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    y = df.iloc[:, 0].values  
    X = df.iloc[:, 1:].values
    X = np.nan_to_num(X, nan=0.0)
    
    return X, y


def train_val_test_split(X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """Split data into train, validation, and test sets"""
    np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_idx = int(n_samples * train_size)
    val_idx = int(n_samples * (train_size + val_size))
    
    train_indices = indices[:train_idx]
    val_indices = indices[train_idx:val_idx]
    test_indices = indices[val_idx:]
    
    X_train = X[train_indices]
    X_val = X[val_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_default_params():
    """Return default hyperparameters"""
    return {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.07,
        "lambda_": 2.0,
        "gamma": 1.0,
        "min_child_weight": 1.0,
        "objective_type": "classification",
        "n_bins": 64,
        "use_sparsity_split": True,
    }


def display_menu():
    """Display main menu"""
    print(f"\n{'='*80}")
    print("XGBoost Model Training & Hyperparameter Optimization")
    print(f"{'='*80}")
    print("\nSelect an option:")
    print("  1. Train with saved/custom parameters")
    print("  2. Optimize hyperparameters with Optuna")
    print("  3. Exit")
    print(f"\n{'-'*80}")


def get_user_params():
    """Get custom parameters from user input or use defaults"""
    # Try to load from saved file first
    params = load_params("best_params_optuna.json")
    if params is not None:
        print("Using previously optimized parameters from Optuna")
        return params
    
    params = load_params("last_params.json")
    if params is not None:
        print("Using previously saved parameters")
        return params
    
    # Otherwise use defaults
    print("Using default parameters")
    return get_default_params()


def save_params(params, filename="best_params.json"):
    """Save parameters to JSON file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "params": params
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Parameters saved to {filename}")


def load_params(filename="best_params.json"):
    """Load parameters from JSON file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"✓ Loaded parameters from {filename}")
        print(f"  Saved at: {data['timestamp']}")
        return data['params']
    except Exception as e:
        print(f"Error loading parameters: {str(e)}")
        return None


def find_best_threshold(y_true, y_pred):
    """Find threshold that maximizes F1 score"""
    from sklearn.metrics import precision_recall_curve
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    
    idx = np.argmax(f1_scores)
    return thresholds[idx] if idx < len(thresholds) else 0.5


def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, params, verbose=True):
    """Train model and evaluate on all sets"""
    
    if verbose:
        print("\n" + "="*80)
        print("[Training Configuration]")
        print("="*80)
        for key, value in params.items():
            if key != "objective_type":
                print(f"  {key:20s}: {value}")
        print("="*80 + "\n")
    
    objective = Objective()
    model = XGBoostModel(params, objective)
    
    if verbose:
        print("Starting training with early stopping...\n")
    else:
        print("Training...", end=" ", flush=True)
    
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, early_stopping_rounds=10)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Get optimal threshold from validation set
    optimal_threshold = find_best_threshold(y_val, y_val_pred)
    
    if verbose:
        # Print results
        print("\n[Training Set Evaluation]")
        train_summary = Summary(y_train, y_train_pred, params["objective_type"])
        train_summary.print_summary()
        
        print("\n[Validation Set Evaluation]")
        val_summary = Summary(y_val, y_val_pred, params["objective_type"], threshold=optimal_threshold)
        val_summary.print_summary()
        
        print("\n[Test Set Evaluation]")
        test_summary = Summary(y_test, y_test_pred, params["objective_type"], threshold=optimal_threshold)
        test_summary.print_summary()
        
        # Performance comparison
        print(f"\n{'='*70}")
        print(f"[Model Performance Comparison]")
        print(f"  Optimal Threshold (from val set): {optimal_threshold:.4f}")
        y_train_pred_binary = (y_train_pred >= optimal_threshold).astype(int)
        y_val_pred_binary = (y_val_pred >= optimal_threshold).astype(int)
        y_test_pred_binary = (y_test_pred >= optimal_threshold).astype(int)
        train_acc = np.mean(y_train_pred_binary == y_train)
        val_acc = np.mean(y_val_pred_binary == y_val)
        test_acc = np.mean(y_test_pred_binary == y_test)
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Val Accuracy:   {val_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        print(f"{'='*70}\n")
        
        # Post-training analysis (optional - commented out)
        # print("[Post-Training Analysis]\n")
        # feature_importance = model.get_feature_importance()
        # analysis = PostTrainingAnalysis(
        #     X_test, y_test, y_test_pred, feature_importance, 
        #     X_test.shape[1], objective_type=params["objective_type"]
        # )
        # analysis.print_all_analysis()
    
    # Return metrics for Optuna
    y_test_pred_binary = (y_test_pred >= optimal_threshold).astype(int)
    test_f1 = f1_score(y_test, y_test_pred_binary)
    
    return model, test_f1, optimal_threshold


def run_optuna_optimization(X_train, X_val, X_test, y_train, y_val, y_test):
    """Run Optuna hyperparameter optimization"""
    
    print("\n" + "="*80)
    print("[Optuna Hyperparameter Optimization]")
    print("="*80 + "\n")
    
    # Get number of trials from user
    while True:
        try:
            n_trials = int(input("Number of trials (default 50, max 200): ") or "50")
            if 5 <= n_trials <= 200:
                break
            else:
                print("  Please enter a value between 5 and 200")
        except ValueError:
            print("  Please enter a valid integer")
    
    # Get number of jobs for parallel execution
    while True:
        try:
            n_jobs = int(input("Number of parallel jobs (default 1): ") or "1")
            if 1 <= n_jobs <= 32:
                break
            else:
                print("  Please enter a value between 1 and 32")
        except ValueError:
            print("  Please enter a valid integer")
    
    print(f"\n{'='*80}")
    print(f"Starting Optuna optimization with {n_trials} trials and {n_jobs} parallel jobs")
    print(f"{'='*80}\n")
    
    # Define the objective function for Optuna
    def optuna_objective(trial):
        """Objective function to optimize"""
        
        # Define hyperparameter search space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "lambda_": trial.suggest_float("lambda_", 0.1, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 5.0),
            "objective_type": "classification",
            "n_bins": 128,
            "use_sparsity_split": True,
        }
        
        try:
            # Train model silently
            model, test_f1, threshold = train_and_evaluate(
                X_train, X_val, X_test, y_train, y_val, y_test, 
                params, verbose=False
            )
            
            # Report intermediate value for pruning
            trial.report(test_f1, step=0)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            print(f"  Trial {trial.number+1:3d}: F1={test_f1:.4f} | "
                  f"depth={params['max_depth']:2d} | "
                  f"lr={params['learning_rate']:.4f} | "
                  f"lambda={params['lambda_']:.2f}")
            
            return test_f1
        
        except Exception as e:
            print(f"  Trial {trial.number+1:3d}: Failed - {str(e)[:40]}")
            return 0.0
    
    # Create Optuna study with pruning
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )
    
    # Run optimization
    study.optimize(optuna_objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)
    
    # Get best trial
    best_trial = study.best_trial
    
    print(f"\n{'='*80}")
    print(f"[Optimization Complete]")
    print(f"{'='*80}")
    print(f"\nBest Trial: #{best_trial.number}")
    print(f"Best F1 Score: {best_trial.value:.4f}\n")
    
    print(f"Best Hyperparameters:")
    best_params = {
        "n_estimators": best_trial.params["n_estimators"],
        "max_depth": best_trial.params["max_depth"],
        "learning_rate": best_trial.params["learning_rate"],
        "lambda_": best_trial.params["lambda_"],
        "gamma": best_trial.params["gamma"],
        "min_child_weight": best_trial.params["min_child_weight"],
        "objective_type": "classification",
        "n_bins": 64,
        "use_sparsity_split": True,
    }
    
    for key, value in best_params.items():
        if key != "objective_type":
            print(f"  {key:20s}: {value}")
    
    # Ask user if they want to train final model with best params
    while True:
        choice = input(f"\nTrain final model with best parameters? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            print(f"\n{'='*80}")
            print("[Training Final Model with Best Parameters]")
            print(f"{'='*80}")
            model, test_f1, threshold = train_and_evaluate(
                X_train, X_val, X_test, y_train, y_val, y_test, 
                best_params, verbose=True
            )
            
            # Save best parameters
            save_params(best_params, "best_params_optuna.json")
            
            return model, test_f1
        elif choice in ['n', 'no']:
            # Still save the best params found
            save_params(best_params, "best_params_optuna.json")
            return None, best_trial.value
        else:
            print("  Please enter 'y' or 'n'")


def main():
    """Main function"""
    # Load data
    X, y = load_data('datasets/application_data.csv')
    print(f"\nRatio of fraud to total: {y.sum() / len(y):.4f}")
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42
    )
    
    print(f"\n{'='*70}")
    print(f"[Data Split]")
    print(f"  Training set:   {X_train.shape[0]:>8d} samples ({X_train.shape[0]/X.shape[0]*100:>5.1f}%)")
    print(f"  Validation set: {X_val.shape[0]:>8d} samples ({X_val.shape[0]/X.shape[0]*100:>5.1f}%)")
    print(f"  Test set:       {X_test.shape[0]:>8d} samples ({X_test.shape[0]/X.shape[0]*100:>5.1f}%)")
    print(f"  Train target range:   [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"  Val target range:     [{y_val.min():.2f}, {y_val.max():.2f}]")
    print(f"  Test target range:    [{y_test.min():.2f}, {y_test.max():.2f}]")
    print(f"{'='*70}\n")
    
    # Main menu loop
    while True:
        display_menu()
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            # Train with saved/custom parameters
            print("\n" + "="*80)
            print("[Training with Saved Parameters]")
            print("="*80)
            
            params = get_user_params()
            print()
            train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, params, verbose=True)
            save_params(params, "last_params.json")
        
        elif choice == "2":
            # Run Optuna optimization
            run_optuna_optimization(X_train, X_val, X_test, y_train, y_val, y_test)
        
        elif choice == "3":
            print("\nExiting... Goodbye!")
            break
        
        else:
            print("  ⚠️  Invalid choice. Please enter 1-3")
            continue
        
        # Ask if user wants to continue
        while True:
            continue_choice = input("\nContinue? (y/n): ").strip().lower()
            if continue_choice in ['y', 'yes']:
                break
            elif continue_choice in ['n', 'no']:
                print("\nExiting... Goodbye!")
                return
            else:
                print("  Please enter 'y' or 'n'")


if __name__ == "__main__":
    main()
