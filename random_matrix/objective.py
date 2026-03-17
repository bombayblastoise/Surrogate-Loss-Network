import numpy as np


class Objective:
    def __init__(self, scale_pos_weight=None):
        self.scale_pos_weight = scale_pos_weight
    
    def set_scale_pos_weight(self, y_true):
        # Calculate scale_pos_weight based on class distribution
        if self.scale_pos_weight is None:
            n_pos = np.sum(y_true == 1)
            n_neg = np.sum(y_true == 0)
            if n_pos > 0:
                self.scale_pos_weight = n_neg / n_pos
            else:
                self.scale_pos_weight = 1.0
    
    def gradient_hessian(self, y_true, y_pred, func):
        if func == "rmse":
            grad = y_pred - y_true
            hess = np.ones_like(y_true)
        elif func == "logloss":
            pred = 1 / (1 + np.exp(-y_pred))
            grad = pred - y_true
            hess = pred * (1 - pred)
            
            # Apply class weights to gradients and hessians
            if self.scale_pos_weight is not None:
                weights = np.where(y_true == 1, self.scale_pos_weight, 1.0)
                grad = grad * weights
                hess = hess * weights
        
        return grad, hess
