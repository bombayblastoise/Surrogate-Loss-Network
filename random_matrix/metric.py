import numpy as np

class Summary:
    def __init__(self, y_true, y_pred, objective_type="regression", threshold=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.objective_type = objective_type
        
        if objective_type == "regression":
            self.rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            self.mae = np.mean(np.abs(y_true - y_pred))
            self.r2 = self._calculate_r2()
        else:  # classification
            # Find optimal threshold if not provided
            if threshold is None:
                self.optimal_threshold = self._find_optimal_threshold()
            else:
                self.optimal_threshold = threshold
            
            self.y_pred_binary = (y_pred >= self.optimal_threshold).astype(int)
            self.accuracy = np.mean(self.y_pred_binary == y_true)
            self.precision = self._calculate_precision()
            self.recall = self._calculate_recall()
            self.f1 = self._calculate_f1()
            self.auc = self._calculate_auc()
            self.roc_curve_data = self._calculate_roc_curve()
    
    def _calculate_r2(self):
        ss_res = np.sum((self.y_true - self.y_pred) ** 2) # sum of squares of residuals
        ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2) # total sum of squares
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def _find_optimal_threshold(self):
        """Find threshold that maximizes F1 score"""
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in np.arange(0.1, 1.0, 0.01):
            y_pred_binary = (self.y_pred >= threshold).astype(int)
            tp = np.sum((y_pred_binary == 1) & (self.y_true == 1))
            fp = np.sum((y_pred_binary == 1) & (self.y_true == 0))
            fn = np.sum((y_pred_binary == 0) & (self.y_true == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    def _calculate_precision(self):
        tp = np.sum((self.y_pred_binary == 1) & (self.y_true == 1))
        fp = np.sum((self.y_pred_binary == 1) & (self.y_true == 0))
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    
    def _calculate_recall(self):
        tp = np.sum((self.y_pred_binary == 1) & (self.y_true == 1))
        fn = np.sum((self.y_pred_binary == 0) & (self.y_true == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    def _calculate_f1(self):
        precision = self.precision
        recall = self.recall
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    def _calculate_auc(self):
        sorted_indices = np.argsort(self.y_pred)[::-1]
        sorted_y = self.y_true[sorted_indices]
        n_pos = np.sum(self.y_true == 1)
        n_neg = np.sum(self.y_true == 0)
        if n_pos == 0 or n_neg == 0:
            return 0.5
        tp_cumsum = np.cumsum(sorted_y)
        auc = np.sum(tp_cumsum[self.y_true[sorted_indices] == 0]) / (n_pos * n_neg)
        return auc
    
    def _calculate_roc_curve(self):
        """Calculate ROC curve points (TPR, FPR) at different thresholds"""
        thresholds = np.sort(np.unique(self.y_pred))[::-1]
        tpr_list = []
        fpr_list = []
        
        n_pos = np.sum(self.y_true == 1)
        n_neg = np.sum(self.y_true == 0)
        
        for threshold in thresholds:
            y_pred_binary = (self.y_pred >= threshold).astype(int)
            tp = np.sum((y_pred_binary == 1) & (self.y_true == 1))
            fp = np.sum((y_pred_binary == 1) & (self.y_true == 0))
            
            tpr = tp / n_pos if n_pos > 0 else 0
            fpr = fp / n_neg if n_neg > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        return {'fpr': np.array(fpr_list), 'tpr': np.array(tpr_list), 'thresholds': thresholds}
    
    def print_summary(self):
        print("\n" + "="*70)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*70)
        
        if self.objective_type == "regression":
            print(f"RMSE: {self.rmse:.4f}")
            print(f"MAE:  {self.mae:.4f}")
            print(f"R²:   {self.r2:.4f}")
            print(f"Predictions range: [{self.y_pred.min():.2f}, {self.y_pred.max():.2f}]")
            print(f"Actual range:      [{self.y_true.min():.2f}, {self.y_true.max():.2f}]")
        else:  # classification
            print(f"Optimal Threshold: {self.optimal_threshold:.4f}")
            print(f"Accuracy:  {self.accuracy:.4f}")
            print(f"Precision: {self.precision:.4f}")
            print(f"Recall:    {self.recall:.4f}")
            print(f"F1 Score:  {self.f1:.4f}")
            print(f"AUC-ROC:   {self.auc:.4f}")
            print(f"Prediction probabilities range: [{self.y_pred.min():.4f}, {self.y_pred.max():.4f}]")
            print(f"\nROC Curve Info:")
            print(f"  Sample FPR values: {self.roc_curve_data['fpr'][::max(1, len(self.roc_curve_data['fpr'])//5)]}")
            print(f"  Sample TPR values: {self.roc_curve_data['tpr'][::max(1, len(self.roc_curve_data['tpr'])//5)]}")
        
        print("="*70)
