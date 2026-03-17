import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_plot(results_dir='jl_fl_optuna_results'):
    """
    Generates plots comparing Baseline vs Best Federated Trial vs Average Federated Trial.
    Uses 'frozen_experiments' and 'tuned_experiments' lists from the JSON.
    """
    
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return

    data = []
    metric_names = ['accuracy', 'f1', 'auc', 'precision']

    print(f"Loading {len(json_files)} datasets...")

    for filepath in json_files:
        ds_name = os.path.basename(filepath).replace('.json', '')
        
        try:
            with open(filepath, 'r') as f:
                res = json.load(f)
            
            baseline = res.get('baseline', {})
            
            # We treat 'frozen' and 'tuned' as two model types to plot
            # In the JSON, they are stored as lists of trial results
            sources = {
                'Frozen': res.get('frozen_experiments', []),
                'Tuned': res.get('tuned_experiments', [])
            }
            
            for model_type, experiments in sources.items():
                if not experiments: continue
                
                for metric in metric_names:
                    # 1. Baseline
                    base_val = baseline.get(metric, 0)
                    
                    # 2. Experiments (List of trials)
                    # Extract values for this metric from all trials
                    exp_vals = [e.get(metric, 0) for e in experiments]
                    
                    if not exp_vals:
                        best_val = 0
                        avg_val = 0
                    else:
                        best_val = max(exp_vals)
                        avg_val = sum(exp_vals) / len(exp_vals)
                    
                    data.append({
                        'Dataset': ds_name[:12],
                        'Metric': metric.upper(),
                        'Model Type': model_type,
                        'Baseline': base_val,
                        'Best Config': best_val,
                        'Average Config': avg_val
                    })
                    
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            
    if not data:
        print("No valid data found.")
        return

    df_full = pd.DataFrame(data)

    # --- Generate Plots ---
    for model_type in ['Frozen', 'Tuned']:
        df_model = df_full[df_full['Model Type'] == model_type]
        
        if df_model.empty: continue
            
        df_model = df_model.sort_values(by='Dataset')
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'{model_type} Model: Tuned Dimension Projection vs Baseline', fontsize=20, weight='bold')
        axes = axes.flatten()
        
        metrics = ['ACCURACY', 'F1', 'AUC', 'PRECISION']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            metric_data = df_model[df_model['Metric'] == metric].copy()
            
            melted = metric_data.melt(
                id_vars=['Dataset'], 
                value_vars=['Baseline', 'Best Config', 'Average Config'],
                var_name='Configuration', 
                value_name='Score'
            )
            
            sns.barplot(
                data=melted, x='Dataset', y='Score', hue='Configuration',
                ax=ax, palette=['#4c72b0', '#55a868', '#dd8452'], 
                edgecolor='black', linewidth=0.5
            )
            
            ax.set_title(metric, fontsize=14, weight='bold')
            ax.set_ylim(0, 1.15)
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            
            if i == 1:
                ax.legend(loc='upper right', frameon=True, ncol=3, fontsize=9)
            else:
                if ax.get_legend(): ax.get_legend().remove()
            
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3, fontsize=8, rotation=90)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_file = f'federated_optuna_dim_{model_type.lower()}.png'
        plt.savefig(output_file, dpi=300)
        print(f"Saved plot to {output_file}")

import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_plot2(results_dir='jl_fl_dim_sensitivity'):
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    if not json_files:
        print("No results found.")
        return

    print(f"Generating plots for {len(json_files)} datasets...")

    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                res = json.load(f)
            
            ds_name = res['dataset_name']
            n_feats = res['n_features_orig']
            baseline = res['baseline']
            sweep = res['sweep_results']
            
            # Convert sweep to DataFrame
            df = pd.DataFrame(sweep)
            if df.empty: continue
            
            # Sort by dimension
            df = df.sort_values('n_dim')
            
            # Plot Setup
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Impact of Projection Dimension: {ds_name} (Original Feats: {n_feats})', fontsize=16)
            axes = axes.flatten()
            
            metrics = ['accuracy', 'f1', 'auc', 'precision']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for i, metric in enumerate(metrics):
                ax = axes[i]
                
                # 1. Plot Trend Line
                sns.lineplot(data=df, x='n_dim', y=metric, ax=ax, marker='o', color=colors[i], label='Federated Proj')
                
                # 2. Plot Baseline Reference
                base_val = baseline.get(metric, 0)
                ax.axhline(base_val, color='black', linestyle='--', label=f'Baseline ({base_val:.2f})')
                
                ax.set_title(metric.upper())
                ax.set_xlabel("Output Dimension (n_components)")
                ax.set_ylabel("Score")
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            out_file = f"{results_dir}/plot_{ds_name}.png"
            plt.savefig(out_file)
            plt.close()
            
        except Exception as e:
            print(f"Error plotting {filepath}: {e}")

    print(f"Plots saved to {results_dir}/")

if __name__ == "__main__":
    load_and_plot2()