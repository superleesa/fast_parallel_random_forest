__author__ = "Satoshi Kashima"

import numpy as np
from sklearn.tree import DecisionTreeClassifier

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, random_state=None, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.max_features = max_features
        self.estimators = []

    def fit(self, X, y):
        num_samples, num_features = X.shape
        np.random.seed(self.random_state)
        self.estimators = []

        max_features = max(self.max_features, num_features) if self.max_features is not None\
            else int(np.ceil(np.sqrt(num_features)))

        for _ in range(self.n_estimators):
            # random feature selection
            feature_indices = np.random.choice(num_features, size=max_features)

            # Bagging
            sample_indices = np.random.choice(num_samples, size=num_samples, replace=True)


            # generating the subset
            X_subset = X[np.ix_(sample_indices, feature_indices)]
            y_subset = y[sample_indices]

            # Create a decision tree and fit it to the subset
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_subset, y_subset)

            # Add the decision tree to the ensemble
            self.estimators.append((tree, feature_indices))

    def predict(self, X):
        predictions = np.zeros((len(X), len(self.estimators)))

        for i, (tree, feature_indices) in enumerate(self.estimators):
            predictions[:, i] = tree.predict(X[:, feature_indices])

        # Make final predictions by majority voting
        final_predictions = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(),
            axis=1,
            arr=predictions
        )
        return final_predictions