"""
Machine learning classifiers for record linkage.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib


class BaseLinkageClassifier:
    """Base class for linkage classifiers."""

    def __init__(self):
        self.model = None
        self._metrics = {}
        self.feature_names = []

    def fit(self, X: List[List[float]], y: List[int], feature_names: Optional[List[str]] = None):
        """Fit the classifier."""
        raise NotImplementedError

    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict labels."""
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(X).tolist()

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted")
        probs = self.model.predict_proba(X)
        return probs[:, 1].tolist()  # Probability of match (class 1)

    def get_metrics(self) -> Dict[str, float]:
        """Get model metrics."""
        return self._metrics

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        raise NotImplementedError


class LogisticRegressionClassifier(BaseLinkageClassifier):
    """Logistic regression classifier for record linkage."""

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def fit(self, X: List[List[float]], y: List[int], feature_names: Optional[List[str]] = None):
        """Fit logistic regression model."""
        X_arr = np.array(X)
        y_arr = np.array(y)

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]

        # Split for evaluation
        if len(y_arr) > 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
            )
        else:
            X_train, X_test, y_train, y_test = X_arr, X_arr, y_arr, y_arr

        # Train model
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            **self.kwargs
        )
        self.model.fit(X_train, y_train)

        # Calculate metrics
        y_pred = self.model.predict(X_test)
        self._metrics = {
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'accuracy': float((y_pred == y_test).mean())
        }

        # Cross-validation if enough data
        if len(y_arr) >= 30:
            cv_scores = cross_val_score(self.model, X_arr, y_arr, cv=5, scoring='f1')
            self._metrics['cv_f1_mean'] = float(cv_scores.mean())
            self._metrics['cv_f1_std'] = float(cv_scores.std())

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from coefficients."""
        if self.model is None:
            return {}

        importance = {}
        coefficients = self.model.coef_[0]
        for name, coef in zip(self.feature_names, coefficients):
            importance[name] = float(abs(coef))

        return importance


class RandomForestLinkageClassifier(BaseLinkageClassifier):
    """Random forest classifier for record linkage."""

    def __init__(self, n_estimators: int = 100, **kwargs):
        super().__init__()
        self.n_estimators = n_estimators
        self.kwargs = kwargs

    def fit(self, X: List[List[float]], y: List[int], feature_names: Optional[List[str]] = None):
        """Fit random forest model."""
        X_arr = np.array(X)
        y_arr = np.array(y)

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]

        # Split for evaluation
        if len(y_arr) > 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
            )
        else:
            X_train, X_test, y_train, y_test = X_arr, X_arr, y_arr, y_arr

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight='balanced',
            random_state=42,
            **self.kwargs
        )
        self.model.fit(X_train, y_train)

        # Calculate metrics
        y_pred = self.model.predict(X_test)
        self._metrics = {
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'accuracy': float((y_pred == y_test).mean())
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from random forest."""
        if self.model is None:
            return {}

        importance = {}
        for name, imp in zip(self.feature_names, self.model.feature_importances_):
            importance[name] = float(imp)

        return importance


class GradientBoostingLinkageClassifier(BaseLinkageClassifier):
    """Gradient boosting classifier for record linkage."""

    def __init__(self, n_estimators: int = 100, **kwargs):
        super().__init__()
        self.n_estimators = n_estimators
        self.kwargs = kwargs

    def fit(self, X: List[List[float]], y: List[int], feature_names: Optional[List[str]] = None):
        """Fit gradient boosting model."""
        X_arr = np.array(X)
        y_arr = np.array(y)

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]

        # Split for evaluation
        if len(y_arr) > 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
            )
        else:
            X_train, X_test, y_train, y_test = X_arr, X_arr, y_arr, y_arr

        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            random_state=42,
            **self.kwargs
        )
        self.model.fit(X_train, y_train)

        # Calculate metrics
        y_pred = self.model.predict(X_test)
        self._metrics = {
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'accuracy': float((y_pred == y_test).mean())
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from gradient boosting."""
        if self.model is None:
            return {}

        importance = {}
        for name, imp in zip(self.feature_names, self.model.feature_importances_):
            importance[name] = float(imp)

        return importance


# Factory functions

def train_classifier(
    X: List[List[float]],
    y: List[int],
    model_type: str,
    parameters: Dict[str, Any]
) -> BaseLinkageClassifier:
    """
    Train a classifier of the specified type.

    Args:
        X: Comparison vectors
        y: Labels (1=match, 0=non-match)
        model_type: "logistic_regression", "random_forest", or "gradient_boosting"
        parameters: Model hyperparameters

    Returns:
        Trained classifier
    """
    if model_type == "logistic_regression":
        classifier = LogisticRegressionClassifier(**parameters)
    elif model_type == "random_forest":
        classifier = RandomForestLinkageClassifier(**parameters)
    elif model_type == "gradient_boosting":
        classifier = GradientBoostingLinkageClassifier(**parameters)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    classifier.fit(X, y)
    return classifier


def save_classifier(classifier: BaseLinkageClassifier, path: str):
    """Save a classifier to disk."""
    joblib.dump(classifier, path)


def load_classifier(path: str) -> BaseLinkageClassifier:
    """Load a classifier from disk."""
    return joblib.load(path)
