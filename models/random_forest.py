"""
Random Forest & Decision Tree Classifier Implementation from Scratch.

This module implements a Classification Decision Tree and Random Forest ensemble
from scratch using NumPy, without relying on black-box scikit-learn estimators.
Designed for binary and multi-class classification tasks.
"""

from collections import Counter
import random
import numpy as np


class DecisionTreeNode:
    """
    Node representation in a Binary Decision Tree.

    Attributes:
        feature (int, optional): Index of feature used for splitting at this node.
        threshold (float, optional): Threshold value used for splitting.
        left (DecisionTreeNode, optional): Left child node (feature_value <= threshold).
        right (DecisionTreeNode, optional): Right child node (feature_value > threshold).
        value (any, optional): Predicted class label if this node is a leaf node.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    """
    Binary Decision Tree Classifier using Gini Impurity for split evaluation.

    Parameters:
        min_samples_split (int): Minimum number of samples required to split an internal node.
        max_depth (int): Maximum depth of the decision tree.
    """

    def __init__(self, min_samples_split=2, max_depth=10):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def _calculate_gini(self, y):
        """
        Calculates Gini Impurity for a target array y.
        Formula: 1 - sum(p_i^2)
        """
        classes = np.unique(y)
        total_samples = len(y)
        if total_samples == 0:
            return 0.0

        gini = 1.0
        for c in classes:
            p = np.sum(y == c) / total_samples
            gini -= p ** 2
        return gini

    def _best_split(self, X, y):
        """Find a Gini split efficiently using sorted values and cumulative counts."""
        best_gain = -1.0
        best_feature = None
        best_threshold = None
        n_samples, n_features = X.shape
        feature_count = max(1, int(np.sqrt(n_features)))
        feature_indices = random.sample(range(n_features), feature_count)
        classes, encoded_y = np.unique(y, return_inverse=True)
        n_classes = len(classes)
        total_counts = np.bincount(encoded_y, minlength=n_classes).astype(float)
        parent_gini = 1.0 - np.sum((total_counts / n_samples) ** 2)

        for feature_idx in feature_indices:
            values = X[:, feature_idx]
            order = np.argsort(values, kind="mergesort")
            sorted_values = values[order]
            sorted_y = encoded_y[order]
            # Candidate split positions are the last row of each unique value.
            boundaries = np.flatnonzero(sorted_values[:-1] != sorted_values[1:])
            if boundaries.size == 0:
                continue
            cumulative = np.zeros((boundaries.size, n_classes), dtype=float)
            for class_index in range(n_classes):
                cumulative[:, class_index] = np.cumsum(sorted_y == class_index)[boundaries]
            left_n = boundaries + 1
            right_n = n_samples - left_n
            right_counts = total_counts - cumulative
            left_gini = 1.0 - np.sum((cumulative / left_n[:, None]) ** 2, axis=1)
            right_gini = 1.0 - np.sum((right_counts / right_n[:, None]) ** 2, axis=1)
            weighted_gini = (left_n / n_samples) * left_gini + (right_n / n_samples) * right_gini
            gains = parent_gini - weighted_gini
            best_position = int(np.argmax(gains))
            gain = float(gains[best_position])
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = float(sorted_values[boundaries[best_position]])

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """
        Recursively builds the decision tree by finding best splits until stopping criteria are met.
        """
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Stopping criteria: minimum samples reached, max depth exceeded, or node is pure
        if n_samples < self.min_samples_split or depth >= self.max_depth or n_classes <= 1:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionTreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )

    def fit(self, X, y):
        """
        Fits the Decision Tree to training data (X, y).
        """
        self.root = self._build_tree(X, y)
        return self

    def _predict_single(self, x, node):
        """
        Traverses tree for a single feature vector to return predicted class label.
        """
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X):
        """
        Predicts target labels for array of samples X.
        """
        return np.array([self._predict_single(x, self.root) for x in X])


class RandomForest:
    """
    Random Forest Ensemble Classifier built from scratch.

    Combines bagging (bootstrap sampling) with random feature selection across an ensemble of Decision Trees.

    Parameters:
        n_trees (int): Number of decision trees in the ensemble.
        min_samples_split (int): Minimum samples required to split a node.
        max_depth (int): Maximum depth of each decision tree.
    """

    def __init__(self, n_trees=10, min_samples_split=2, max_depth=10):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []

    def _bootstrap_sample(self, X, y):
        """
        Generates a bootstrap sample (sampling with replacement) of equal size to original dataset.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        """
        Trains the ensemble of decision trees on bootstrap samples.
        """
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        """
        Aggregates predictions from all trees via majority voting.
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([Counter(pred).most_common(1)[0][0] for pred in predictions.T])
