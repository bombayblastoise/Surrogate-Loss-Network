import numpy as np
import pandas as pd


class PostTrainingAnalysis:
    def __init__(self, X, y, y_pred, feature_importance, n_features, objective_type="regression"):
        self.X = X
        self.y = y
        self.y_pred = y_pred
        self.feature_importance = feature_importance
        self.n_features = n_features
        self.objective_type = objective_type
        self.feature_names = [f"feature_{i}" for i in range(n_features)]
    
    def analyze_feature_importance(self):
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*70)
        
        if self.feature_importance is None:
            print("No feature importance data available")
            return
        
        importance_dict = {}
        if isinstance(self.feature_importance, dict):
            for i in range(self.n_features):
                importance_dict[self.feature_names[i]] = self.feature_importance.get(i, 0)
        else:
            for i in range(self.n_features):
                importance_dict[self.feature_names[i]] = self.feature_importance[i]
        
        importance_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        total_importance = importance_df['Importance'].sum()
        if total_importance > 0:
            importance_df['Relative_Importance_%'] = (importance_df['Importance'] / total_importance * 100).round(2)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        if len(importance_df) > 10:
            print(f"\n... and {len(importance_df) - 10} more features")
        
        return importance_df
    
    def analyze_correlation(self):
        print("\n" + "="*70)
        print("CORRELATION ANALYSIS")
        print("="*70)
        
        # Create dataframe with features and target
        data = np.column_stack([self.X, self.y])
        columns = self.feature_names + ['target']
        df = pd.DataFrame(data, columns=columns)
        
        # Calculate correlations with target
        correlations = df.corr()['target'].drop('target').sort_values(ascending=False)
        
        print("\nFeature Correlations with Target (sorted by absolute value):")
        correlations_abs = correlations.abs().sort_values(ascending=False)
        corr_df = pd.DataFrame({
            'Feature': correlations_abs.index,
            'Correlation': correlations[correlations_abs.index].values,
            'Abs_Correlation': correlations_abs.values
        })
        
        print(corr_df.head(10).to_string(index=False))
        
        return corr_df
    
    def analyze_prediction_residuals(self):
        print("\n" + "="*70)
        print("RESIDUAL ANALYSIS")
        print("="*70)
        
        if self.objective_type == "regression":
            self._analyze_regression_residuals()
        else:
            self._analyze_classification_residuals()
    
    def _analyze_regression_residuals(self):
        residuals = self.y - self.y_pred
        
        print(f"\nResidual Statistics (RMSE-based):")
        print(f"  Mean:     {residuals.mean():.4f}")
        print(f"  Std Dev:  {residuals.std():.4f}")
        print(f"  Min:      {residuals.min():.4f}")
        print(f"  Max:      {residuals.max():.4f}")
        print(f"  Median:   {np.median(residuals):.4f}")
        print(f"  Q1:       {np.percentile(residuals, 25):.4f}")
        print(f"  Q3:       {np.percentile(residuals, 75):.4f}")
        
        # Residuals by prediction magnitude
        pred_quartiles = np.percentile(self.y_pred, [25, 50, 75, 100])
        for i in range(len(pred_quartiles) - 1):
            mask = (self.y_pred >= pred_quartiles[i]) & (self.y_pred <= pred_quartiles[i+1])
            residuals_q = residuals[mask]
            if len(residuals_q) > 0:
                rmse = np.sqrt(np.mean(residuals_q**2))
                mae = np.mean(np.abs(residuals_q))
                print(f"  Predictions [{pred_quartiles[i]:.4f} - {pred_quartiles[i+1]:.4f}]: RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        
        return residuals
    
    def _analyze_classification_residuals(self):
        # Calculate log loss
        y_pred_clipped = np.clip(self.y_pred, 1e-7, 1 - 1e-7)
        log_loss = -np.mean(self.y * np.log(y_pred_clipped) + (1 - self.y) * np.log(1 - y_pred_clipped))
        
        # Calculate accuracy at different thresholds
        print(f"\nClassification Error Analysis:")
        print(f"  Log Loss: {log_loss:.4f}")
        
        # Test different thresholds
        print(f"\nPerformance at Different Thresholds:")
        print(f"  {'Threshold':<15} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1 Score':<15}")
        print("-" * 75)
        
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            y_pred_binary = (self.y_pred >= threshold).astype(int)
            accuracy = np.mean(y_pred_binary == self.y)
            
            tp = np.sum((y_pred_binary == 1) & (self.y == 1))
            fp = np.sum((y_pred_binary == 1) & (self.y == 0))
            fn = np.sum((y_pred_binary == 0) & (self.y == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"  {threshold:<15.1f} {accuracy:<15.4f} {precision:<15.4f} {recall:<15.4f} {f1:<15.4f}")
        
        # Prediction distribution analysis
        print(f"\nPrediction Distribution:")
        print(f"  Mean probability:  {self.y_pred.mean():.4f}")
        print(f"  Std probability:   {self.y_pred.std():.4f}")
        print(f"  Min probability:   {self.y_pred.min():.4f}")
        print(f"  Max probability:   {self.y_pred.max():.4f}")
        print(f"  Median probability: {np.median(self.y_pred):.4f}")
        
        # Class distribution
        n_pos = np.sum(self.y == 1)
        n_neg = np.sum(self.y == 0)
        print(f"\nClass Distribution (actual):")
        print(f"  Positive class: {n_pos} ({n_pos/len(self.y)*100:.2f}%)")
        print(f"  Negative class: {n_neg} ({n_neg/len(self.y)*100:.2f}%)")
        
        # Imbalance ratio
        imbalance_ratio = max(n_pos, n_neg) / min(n_pos, n_neg) if min(n_pos, n_neg) > 0 else float('inf')
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    def analyze_feature_distributions(self):
        print("\n" + "="*70)
        print("FEATURE DISTRIBUTION ANALYSIS")
        print("="*70)
        
        print("\nFeature Statistics:")
        print(f"{'Feature':<15} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        print("-" * 63)
        
        for i in range(min(10, self.n_features)):
            feature_data = self.X[:, i]
            print(f"{self.feature_names[i]:<15} {feature_data.mean():>12.4f} {feature_data.std():>12.4f} {feature_data.min():>12.4f} {feature_data.max():>12.4f}")
        
        if self.n_features > 10:
            print(f"... and {self.n_features - 10} more features")
    
    def print_all_analysis(self):
        self.analyze_feature_importance()
        self.analyze_correlation()
        self.analyze_prediction_residuals()
        self.analyze_feature_distributions()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70 + "\n")
