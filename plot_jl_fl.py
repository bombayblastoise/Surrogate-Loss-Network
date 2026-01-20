import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_fed_results(results_dir='federated_results'):
    """Reads all Federated Result JSON files and aggregates metrics."""
    data = []
    
    # metrics to extract
    metric_names = ['accuracy', 'f1', 'auc', 'precision']
    
    # Find all json files
    # We look for *_fed_results.json based on previous script naming
    json_files = glob.glob(os.path.join(results_dir, "*_fed_results.json"))
    
    # Fallback if specific naming isn't found
    if not json_files:
        json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return pd.DataFrame()

    print(f"Found {len(json_files)} result files. Processing...")

    for filepath in json_files:
        # Get filename without extension and cleanup
        full_name = os.path.basename(filepath).replace('_fed_results.json', '').replace('.json', '')
        
        # TRUNCATION: Use only first 10 chars for aesthetics
        short_name = full_name[:10]
        
        try:
            with open(filepath, 'r') as f:
                res = json.load(f)
            
            # ---------------------------
            # 1. Extract Baseline
            # ---------------------------
            if 'baseline' not in res: 
                print(f"Skipping {full_name}: No baseline found")
                continue
                
            base_frozen = res['baseline'].get('frozen', {})
            base_tuned = res['baseline'].get('tuned', {})
            
            # ---------------------------
            # 2. Extract Experiments
            # ---------------------------
            experiments = res.get('experiments', [])
            if not experiments:
                print(f"Skipping {full_name}: No experiments found")
                continue
            
            # Helper to get lists of values for a specific metric across ALL configs
            def get_exp_values(exp_list, model_type, metric):
                vals = []
                for exp in exp_list:
                    # Check if model_type exists and has the metric
                    if model_type in exp and exp[model_type] and metric in exp[model_type]:
                        vals.append(exp[model_type][metric])
                return vals

            # Process Frozen and Tuned separately
            for model_type in ['frozen', 'tuned']:
                # Get baseline dictionary
                base_metrics = base_frozen if model_type == 'frozen' else base_tuned
                
                for metric in metric_names:
                    # Baseline Value (default to 0)
                    base_val = base_metrics.get(metric, 0)
                    
                    # Experiment Values (List of all 20 configs)
                    exp_vals = get_exp_values(experiments, model_type, metric)
                    
                    if not exp_vals:
                        best_val = 0
                        avg_val = 0
                    else:
                        best_val = max(exp_vals)
                        avg_val = np.mean(exp_vals)
                    
                    # Append Row
                    data.append({
                        'Dataset': short_name, 
                        'Model Type': model_type.capitalize(),
                        'Metric': metric.upper(),
                        'Baseline': base_val,
                        'Best Fed': best_val,
                        'Average Fed': avg_val
                    })
                    
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            
    return pd.DataFrame(data)

def plot_fed_comparison(df, model_type, output_dir):
    """Generates the 4-subplot figure for a specific model type."""
    
    # Filter data
    subset = df[df['Model Type'] == model_type]
    
    if subset.empty:
        print(f"No data found for {model_type} models.")
        return

    # Sort datasets alphabetically
    subset = subset.sort_values(by='Dataset')

    # Metrics to plot
    metrics = ['ACCURACY', 'F1', 'AUC', 'PRECISION']
    
    # Setup Plot (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_type} Model: Original vs Federated Projection', fontsize=20, weight='bold')
    
    axes = axes.flatten()
    
    # Melt dataframe for plotting
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Filter for specific metric
        metric_data = subset[subset['Metric'] == metric].copy()
        
        if metric_data.empty:
            continue

        # Melt into long format
        melted = metric_data.melt(
            id_vars=['Dataset'], 
            value_vars=['Baseline', 'Best Fed', 'Average Fed'],
            var_name='Configuration', 
            value_name='Score'
        )
        
        # Plot Bar Chart
        sns.barplot(
            data=melted, 
            x='Dataset', 
            y='Score', 
            hue='Configuration',
            ax=ax,
            palette=['#4c72b0', '#55a868', '#dd8452'], # Blue, Green, Orange
            edgecolor='black',
            linewidth=0.5
        )
        
        # Styling
        ax.set_title(metric, fontsize=15, weight='bold')
        ax.set_ylim(0, 1.15) 
        ax.set_xlabel('')
        ax.set_ylabel('Score' if i % 2 == 0 else '')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Legend management
        if i == 1:
            ax.legend(loc='upper right', frameon=True, ncol=3, fontsize=9)
        else:
            if ax.get_legend():
                ax.get_legend().remove()
        
        # Rotate labels
        ax.tick_params(axis='x', rotation=45, labelsize=10)
            
        # Add values on top of bars (if not too crowded)
        if len(metric_data) < 20: 
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3, fontsize=8, rotation=90)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    out_path = os.path.join(output_dir, f'fed_comparison_{model_type.lower()}.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot: {out_path}")
    plt.close()

if __name__ == "__main__":
    # Settings
    RESULTS_DIR = 'jl_fl_results'  # Folder containing your JSONs
    PLOTS_DIR = 'federated_plots'      # Folder to save the images
    
    # Create output dir
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # 1. Load Data
    print("Loading data from Federated JSON files...")
    df = load_fed_results(RESULTS_DIR)
    
    if not df.empty:
        # 2. Generate Plots
        print("Generating Frozen model plots...")
        plot_fed_comparison(df, 'Frozen', PLOTS_DIR)
        
        print("Generating Tuned model plots...")
        plot_fed_comparison(df, 'Tuned', PLOTS_DIR)
        
        print("\nAll plots generated successfully.")
    else:
        print("No data extracted. Check your directory path and file content.")
