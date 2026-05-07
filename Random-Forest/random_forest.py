import numpy as np
from collections import Counter

# Decision Tree #
class DecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.tree = None

    def entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def fit(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if depth >= self.max_depth or n_labels == 1:
            return Counter(y).most_common(1)[0][0]

        # Find the best split
        feat_idx = np.random.randint(0, n_features) # Feature randomness
        threshold = np.mean(X[:, feat_idx])
        
        left_indices = np.where(X[:, feat_idx] <= threshold)[0]
        right_indices = np.where(X[:, feat_idx] > threshold)[0]

        # Recursively build branches
        left = self.fit(X[left_indices], y[left_indices], depth + 1)
        right = self.fit(X[right_indices], y[right_indices], depth + 1)
        return {'feature': feat_idx, 'threshold': threshold, 'left': left, 'right': right}

    def predict_one(self, x, tree):
        if not isinstance(tree, dict): return tree
        if x[tree['feature']] <= tree['threshold']:
            return self.predict_one(x, tree['left'])
        return self.predict_one(x, tree['right'])

# Random Forest #
  class RandomForestFromScratch:
    def __init__(self, n_trees=10, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth)
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree.tree = tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Gather predictions from every tree
        tree_preds = np.array([tree.predict_one(x, tree.tree) for x in X for tree in self.trees])
        tree_preds = tree_preds.reshape(len(X), self.n_trees)
        
        # Take the majority vote (mode) for each row
        return np.array([Counter(row).most_common(1)[0][0] for row in tree_preds])
