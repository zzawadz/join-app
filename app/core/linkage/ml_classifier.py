"""
Machine learning classifiers for record linkage.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import joblib
import hmac
import hashlib
from pathlib import Path


class BaseLinkageClassifier:
    """Base class for linkage classifiers."""

    def __init__(self):
        self.model = None
        self._metrics = {}
        self._train_metrics = {}
        self._test_metrics = {}
        self._confusion_matrix = {}
        self.feature_names = []
        self._train_samples = 0
        self._test_samples = 0

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
        """Get model metrics (backward compatibility - returns test metrics)."""
        return self._metrics

    def get_train_metrics(self) -> Dict[str, float]:
        """Get in-sample (training set) metrics."""
        return self._train_metrics

    def get_test_metrics(self) -> Dict[str, float]:
        """Get out-of-sample (test set) metrics."""
        return self._test_metrics

    def get_confusion_matrix(self) -> Dict[str, int]:
        """Get confusion matrix from test set."""
        return self._confusion_matrix

    def get_train_samples(self) -> int:
        """Get number of training samples."""
        return self._train_samples

    def get_test_samples(self) -> int:
        """Get number of test samples."""
        return self._test_samples

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        raise NotImplementedError

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute classification metrics."""
        return {
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'accuracy': float(accuracy_score(y_true, y_pred))
        }

    def _compute_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
        """Compute confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        return {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }


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

        # Split for evaluation - need enough samples and both classes in each split
        has_test_set = False
        if len(y_arr) > 20 and len(np.unique(y_arr)) > 1:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
                )
                has_test_set = True
            except ValueError:
                # Not enough samples for stratified split
                X_train, y_train = X_arr, y_arr
                X_test, y_test = None, None
        else:
            X_train, y_train = X_arr, y_arr
            X_test, y_test = None, None

        self._train_samples = len(y_train)
        self._test_samples = len(y_test) if has_test_set else 0

        # Train model
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            **self.kwargs
        )
        self.model.fit(X_train, y_train)

        # Calculate in-sample (training) metrics
        y_train_pred = self.model.predict(X_train)
        self._train_metrics = self._compute_metrics(y_train, y_train_pred)

        # Calculate out-of-sample (test) metrics if we have a test set
        if has_test_set:
            y_test_pred = self.model.predict(X_test)
            self._test_metrics = self._compute_metrics(y_test, y_test_pred)
            self._confusion_matrix = self._compute_confusion_matrix(y_test, y_test_pred)
            # For backward compatibility, _metrics uses test metrics
            self._metrics = self._test_metrics.copy()
        else:
            self._test_metrics = {}
            self._confusion_matrix = {}
            # Fall back to training metrics for backward compatibility
            self._metrics = self._train_metrics.copy()

        # Cross-validation if enough data
        if len(y_arr) >= 30 and len(np.unique(y_arr)) > 1:
            try:
                cv_scores = cross_val_score(self.model, X_arr, y_arr, cv=5, scoring='f1')
                self._metrics['cv_f1_mean'] = float(cv_scores.mean())
                self._metrics['cv_f1_std'] = float(cv_scores.std())
            except ValueError:
                pass  # Skip CV if not enough samples

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

        # Split for evaluation - need enough samples and both classes
        has_test_set = False
        if len(y_arr) > 20 and len(np.unique(y_arr)) > 1:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
                )
                has_test_set = True
            except ValueError:
                X_train, y_train = X_arr, y_arr
                X_test, y_test = None, None
        else:
            X_train, y_train = X_arr, y_arr
            X_test, y_test = None, None

        self._train_samples = len(y_train)
        self._test_samples = len(y_test) if has_test_set else 0

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight='balanced',
            random_state=42,
            **self.kwargs
        )
        self.model.fit(X_train, y_train)

        # Calculate in-sample (training) metrics
        y_train_pred = self.model.predict(X_train)
        self._train_metrics = self._compute_metrics(y_train, y_train_pred)

        # Calculate out-of-sample (test) metrics if we have a test set
        if has_test_set:
            y_test_pred = self.model.predict(X_test)
            self._test_metrics = self._compute_metrics(y_test, y_test_pred)
            self._confusion_matrix = self._compute_confusion_matrix(y_test, y_test_pred)
            self._metrics = self._test_metrics.copy()
        else:
            self._test_metrics = {}
            self._confusion_matrix = {}
            self._metrics = self._train_metrics.copy()

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

        # Split for evaluation - need enough samples and both classes
        has_test_set = False
        if len(y_arr) > 20 and len(np.unique(y_arr)) > 1:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
                )
                has_test_set = True
            except ValueError:
                X_train, y_train = X_arr, y_arr
                X_test, y_test = None, None
        else:
            X_train, y_train = X_arr, y_arr
            X_test, y_test = None, None

        self._train_samples = len(y_train)
        self._test_samples = len(y_test) if has_test_set else 0

        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            random_state=42,
            **self.kwargs
        )
        self.model.fit(X_train, y_train)

        # Calculate in-sample (training) metrics
        y_train_pred = self.model.predict(X_train)
        self._train_metrics = self._compute_metrics(y_train, y_train_pred)

        # Calculate out-of-sample (test) metrics if we have a test set
        if has_test_set:
            y_test_pred = self.model.predict(X_test)
            self._test_metrics = self._compute_metrics(y_test, y_test_pred)
            self._confusion_matrix = self._compute_confusion_matrix(y_test, y_test_pred)
            self._metrics = self._test_metrics.copy()
        else:
            self._test_metrics = {}
            self._confusion_matrix = {}
            self._metrics = self._train_metrics.copy()

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


def _sign_model(model_path: str, secret_key: str) -> Path:
    """
    Sign a model file with HMAC-SHA256.

    Args:
        model_path: Path to the model file
        secret_key: Secret key for signing

    Returns:
        Path to the signature file
    """
    with open(model_path, 'rb') as f:
        model_data = f.read()

    signature = hmac.new(
        secret_key.encode(),
        model_data,
        hashlib.sha256
    ).hexdigest()

    sig_path = Path(f"{model_path}.sig")
    with open(sig_path, 'w') as f:
        f.write(signature)

    return sig_path


def _verify_model_signature(model_path: str, secret_key: str) -> bool:
    """
    Verify model signature.

    Args:
        model_path: Path to the model file
        secret_key: Secret key for verification

    Returns:
        True if signature is valid

    Raises:
        ValueError: If signature file is missing or signature is invalid
    """
    sig_path = Path(f"{model_path}.sig")

    if not sig_path.exists():
        raise ValueError(
            f"Model signature file missing: {sig_path}. "
            "This may indicate tampering or an unsigned model."
        )

    # Read model data
    with open(model_path, 'rb') as f:
        model_data = f.read()

    # Read expected signature
    with open(sig_path, 'r') as f:
        expected_sig = f.read().strip()

    # Compute actual signature
    actual_sig = hmac.new(
        secret_key.encode(),
        model_data,
        hashlib.sha256
    ).hexdigest()

    # Use constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(actual_sig, expected_sig):
        raise ValueError(
            "Model signature verification failed. "
            "This may indicate tampering with the model file."
        )

    return True


def save_classifier(classifier: BaseLinkageClassifier, path: str):
    """
    Save a classifier to disk with cryptographic signature.

    Args:
        classifier: The classifier to save
        path: Path where the model should be saved

    Note:
        This function automatically signs the model after saving.
        The signature file will be created at {path}.sig
    """
    from app.config import get_settings
    settings = get_settings()

    # Save model
    joblib.dump(classifier, path)

    # Sign model for integrity verification
    _sign_model(path, settings.model_secret_key)


def load_classifier(path: str) -> BaseLinkageClassifier:
    """
    Load a classifier from disk with signature verification.

    Args:
        path: Path to the model file

    Returns:
        The loaded classifier

    Raises:
        ValueError: If signature is missing or invalid

    Security:
        This function verifies the model's cryptographic signature
        before loading to prevent execution of tampered models.
    """
    from app.config import get_settings
    settings = get_settings()

    # Verify signature before loading
    _verify_model_signature(path, settings.model_secret_key)

    # Load model (signature verified, safe to load)
    return joblib.load(path)
