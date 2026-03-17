import numpy as np


class TreeNode:
    def __init__(self, is_leaf=False, value=0.0, feature_index=None, threshold=None, 
                 left=None, right=None, gain=0.0, default_direction=None):
        self.is_leaf = is_leaf
        self.value = value
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.default_direction = default_direction


class Tree:
    def __init__(self, max_depth, lambda_, gamma, min_child_weight, n_bins=256, use_sparsity_split=True):
        self.max_depth = max_depth
        self.lambda_ = lambda_
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.n_bins = n_bins
        self.use_sparsity_split = use_sparsity_split
        self.root = None


    def fit(self, X, grad, hess):
        indices = np.arange(X.shape[0])
        self.root = self._build_tree(X, grad, hess, indices, depth=0)


    def _build_tree(self, X, grad, hess, indices, depth):
        node_samples = len(indices)
        
        if node_samples == 0:
            return TreeNode(is_leaf=True, value=0.0)
        
        if depth >= self.max_depth:
            value = self._leaf_value(grad, hess, indices)
            return TreeNode(is_leaf=True, value=value)
        
        hess_sum = np.sum(hess[indices])
        if node_samples < 2 or hess_sum < self.min_child_weight:
            value = self._leaf_value(grad, hess, indices)
            return TreeNode(is_leaf=True, value=value)
        
        # ✅ FIX: Only try sparsity if data is actually sparse
        split_info = None
        use_sparsity = self._should_use_sparsity_split(X, indices)
        
        if use_sparsity:
            split_info = self._best_split_sparsity_aware(X, grad, hess, indices)
        
        # Fall back to binned split
        if split_info is None:
            split_info = self._best_split_binned(X, grad, hess, indices)
        
        if split_info is None:
            value = self._leaf_value(grad, hess, indices)
            return TreeNode(is_leaf=True, value=value)
        
        feature_index, threshold, gain, default_direction = split_info
        
        if gain < self.gamma:
            value = self._leaf_value(grad, hess, indices)
            return TreeNode(is_leaf=True, value=value)
        
        feature_values_subset = X[indices, feature_index]
        left_indices, right_indices = self._split_indices(feature_values_subset, threshold, indices)
        
        left_node = self._build_tree(X, grad, hess, left_indices, depth + 1)
        right_node = self._build_tree(X, grad, hess, right_indices, depth + 1)
        
        return TreeNode(is_leaf=False, feature_index=feature_index, threshold=threshold,
                       left=left_node, right=right_node, gain=gain, default_direction=default_direction)


    def _should_use_sparsity_split(self, X, indices):
        """Check if data is actually sparse before using sparsity-aware split"""
        if not self.use_sparsity_split or len(indices) < 100:
            return False
        
        # Quick sparsity check on a sample of features
        sample_size = min(5, X.shape[1])
        feature_indices = np.random.choice(X.shape[1], sample_size, replace=False)
        
        sparse_count = 0
        for feat_idx in feature_indices:
            feature_vals = X[indices, feat_idx]
            sparsity = np.sum(np.isnan(feature_vals) | (feature_vals == 0)) / len(indices)
            if sparsity > 0.2:  # If >20% sparse
                sparse_count += 1
        
        # Use sparsity-aware if majority are sparse
        return sparse_count >= sample_size // 2


    def _best_split_sparsity_aware(self, X, grad, hess, indices):
        """Find best split using sparsity-aware method"""
        best_gain = -np.inf
        best_split = None
        n_features = X.shape[1]
        grad_total = np.sum(grad[indices])
        hess_total = np.sum(hess[indices])
        
        for feature_idx in range(n_features):
            feature_values = X[indices, feature_idx]
            
            non_sparse_mask = ~np.isnan(feature_values) & (feature_values != 0)
            non_sparse_count = np.sum(non_sparse_mask)
            
            # Skip if too few non-sparse values
            if non_sparse_count < 2:
                continue
            
            sparse_indices = indices[~non_sparse_mask]
            non_sparse_indices = indices[non_sparse_mask]
            non_sparse_values = feature_values[non_sparse_mask]
            
            grad_sparse = np.sum(grad[sparse_indices]) if len(sparse_indices) > 0 else 0.0
            hess_sparse = np.sum(hess[sparse_indices]) if len(sparse_indices) > 0 else 0.0
            grad_non_sparse = grad_total - grad_sparse
            hess_non_sparse = hess_total - hess_sparse
            
            # ✅ FIX: Use efficient bin edge calculation
            bin_edges = self._get_bin_edges(non_sparse_values)
            
            if len(bin_edges) < 2:
                continue
            
            # ✅ FIX: Use np.digitize for binning
            feature_bins = np.digitize(non_sparse_values, bin_edges) - 1
            feature_bins = np.clip(feature_bins, 0, len(bin_edges) - 2)
            
            # ✅ FIX: Use np.add.at() instead of loop
            bin_grad = np.zeros(len(bin_edges) - 1)
            bin_hess = np.zeros(len(bin_edges) - 1)
            
            np.add.at(bin_grad, feature_bins, grad[non_sparse_indices])
            np.add.at(bin_hess, feature_bins, hess[non_sparse_indices])
            
            # Scan through bins
            grad_left = 0.0
            hess_left = 0.0
            
            for bin_idx in range(len(bin_edges) - 2):
                grad_left += bin_grad[bin_idx]
                hess_left += bin_hess[bin_idx]
                
                grad_right = grad_non_sparse - grad_left
                hess_right = hess_non_sparse - hess_left
                
                if hess_left < self.min_child_weight or hess_right < self.min_child_weight:
                    continue
                
                # Try default direction = left
                gain_left = self._compute_gain(
                    grad_left + grad_sparse, hess_left + hess_sparse,
                    grad_right, hess_right, grad_total, hess_total
                )
                
                if gain_left > best_gain:
                    best_gain = gain_left
                    threshold = bin_edges[bin_idx + 1]
                    best_split = (feature_idx, threshold, gain_left, 'left')
                
                # Try default direction = right
                gain_right = self._compute_gain(
                    grad_left, hess_left,
                    grad_right + grad_sparse, hess_right + hess_sparse,
                    grad_total, hess_total
                )
                
                if gain_right > best_gain:
                    best_gain = gain_right
                    threshold = bin_edges[bin_idx + 1]
                    best_split = (feature_idx, threshold, gain_right, 'right')
        
        return best_split


    def _best_split_binned(self, X, grad, hess, indices):
        """Find best split using binned histogram method"""
        best_gain = -np.inf
        best_split = None
        n_features = X.shape[1]
        grad_total = np.sum(grad[indices])
        hess_total = np.sum(hess[indices])
        
        for feature_idx in range(n_features):
            feature_values = X[indices, feature_idx]
            feature_min = np.nanmin(feature_values)
            feature_max = np.nanmax(feature_values)
            
            if feature_min == feature_max or np.isnan(feature_min) or np.isnan(feature_max):
                continue
            
            # ✅ COMPUTE ONCE AND REUSE
            bin_edges = np.linspace(feature_min, feature_max, self.n_bins)
            feature_bins = np.digitize(feature_values, bin_edges) - 1
            feature_bins = np.clip(feature_bins, 0, self.n_bins - 1)
            
            # ✅ FIX: Use np.add.at() instead of Python loop
            bin_grad = np.zeros(self.n_bins)
            bin_hess = np.zeros(self.n_bins)
            bin_count = np.zeros(self.n_bins)
            
            np.add.at(bin_grad, feature_bins, grad[indices])
            np.add.at(bin_hess, feature_bins, hess[indices])
            np.add.at(bin_count, feature_bins, 1)
            
            grad_left = 0.0
            hess_left = 0.0
            
            for bin_idx in range(self.n_bins - 1):
                if bin_count[bin_idx] == 0:
                    continue
                
                grad_left += bin_grad[bin_idx]
                hess_left += bin_hess[bin_idx]
                grad_right = grad_total - grad_left
                hess_right = hess_total - hess_left
                
                if hess_left < self.min_child_weight or hess_right < self.min_child_weight:
                    continue
                
                gain = self._compute_gain(grad_left, hess_left, grad_right, hess_right,
                                        grad_total, hess_total)
                
                if gain > best_gain:
                    best_gain = gain
                    bin_threshold = bin_edges[bin_idx + 1]
                    best_split = (feature_idx, bin_threshold, gain, None)
        
        return best_split


    def _get_bin_edges(self, values):
        """
        Get bin edges efficiently using quantiles.
        
        ✅ Much faster than np.percentile()
        """
        if len(values) == 0:
            return np.array([])
        
        # For small arrays, use simple approach
        if len(values) <= self.n_bins:
            return np.unique(values)
        
        # For large arrays, use sorted quantile approach
        sorted_vals = np.sort(values)
        # Get indices at quantile positions
        quantile_indices = np.linspace(0, len(sorted_vals) - 1, self.n_bins + 1, dtype=int)
        bin_edges = sorted_vals[quantile_indices]
        
        return np.unique(bin_edges)


    def _compute_gain(self, grad_left, hess_left, grad_right, hess_right, grad_total, hess_total):
        """XGBoost gain formula"""
        return ((grad_left**2 / (hess_left + self.lambda_) +
                grad_right**2 / (hess_right + self.lambda_) -
                grad_total**2 / (hess_total + self.lambda_)) / 2)


    def _split_indices(self, X_column, threshold, indices):
        """Split indices into left and right based on threshold"""
        mask = X_column <= threshold
        left_indices = indices[mask]
        right_indices = indices[~mask]
        return left_indices, right_indices


    def _leaf_value(self, grad, hess, indices):
        """Calculate optimal leaf value"""
        if len(indices) == 0:
            return 0.0
        return -np.sum(grad[indices]) / (np.sum(hess[indices]) + self.lambda_)


    def predict(self, X):
        """Predict using the tree"""
        predictions = np.array([self._predict_row(x, self.root) for x in X])
        return predictions


    def _predict_row(self, x, node):
        """Predict for a single row"""
        if node.is_leaf:
            return node.value
        
        feature_val = x[node.feature_index]
        
        # Handle sparse/missing values
        if np.isnan(feature_val) or feature_val == 0:
            if node.default_direction == 'left':
                return self._predict_row(x, node.left)
            elif node.default_direction == 'right':
                return self._predict_row(x, node.right)
        
        # Normal split
        if feature_val <= node.threshold:
            return self._predict_row(x, node.left)
        else:
            return self._predict_row(x, node.right)


    def get_feature_importance(self):
        """Get feature importance scores"""
        importance = {}
        
        def traverse(node):
            if node is None or node.is_leaf:
                return
            feature_idx = node.feature_index
            gain = node.gain
            importance[feature_idx] = importance.get(feature_idx, 0) + gain
            traverse(node.left)
            traverse(node.right)
        
        traverse(self.root)
        return importance
