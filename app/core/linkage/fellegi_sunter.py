"""
Fellegi-Sunter probabilistic record linkage model.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score


class FellegiSunterModel:
    """
    Fellegi-Sunter probabilistic record linkage model.

    The model estimates:
    - m probabilities: P(agree | match) - probability fields agree given records match
    - u probabilities: P(agree | non-match) - probability fields agree given records don't match

    Match weight for a comparison vector is:
    w = sum(log(m_i/u_i) if agree else log((1-m_i)/(1-u_i)))

    Classification:
    - w > upper_threshold -> match
    - w < lower_threshold -> non-match
    - otherwise -> review
    """

    def __init__(self):
        self.m_probs = {}  # P(agree | match) for each field
        self.u_probs = {}  # P(agree | non-match) for each field
        self.upper_threshold = 0.0
        self.lower_threshold = 0.0
        self.field_names = []
        self._metrics = {}

    def fit(
        self,
        comparison_vectors: List[List[float]],
        labels: Optional[List[int]] = None,
        field_names: Optional[List[str]] = None,
        agreement_threshold: float = 0.85
    ):
        """
        Fit the model to estimate m and u probabilities.

        Args:
            comparison_vectors: List of comparison vectors (each is list of similarity scores)
            labels: Optional labels (1=match, 0=non-match). If None, use EM algorithm.
            field_names: Names of fields in comparison vector
            agreement_threshold: Threshold for considering a field "agrees"
        """
        if not comparison_vectors:
            raise ValueError("No comparison vectors provided")

        X = np.array(comparison_vectors)
        n_samples, n_fields = X.shape

        # Generate field names if not provided
        if field_names:
            self.field_names = field_names
        else:
            self.field_names = [f"field_{i}" for i in range(n_fields)]

        # Convert similarity scores to agreement indicators
        agrees = X >= agreement_threshold

        if labels is not None:
            # Supervised estimation
            y = np.array(labels)
            matches = y == 1
            non_matches = y == 0

            n_matches = matches.sum()
            n_non_matches = non_matches.sum()

            if n_matches == 0 or n_non_matches == 0:
                raise ValueError("Need both matches and non-matches for training")

            for i, field in enumerate(self.field_names):
                # m probability: P(agree | match)
                self.m_probs[field] = agrees[matches, i].sum() / n_matches
                # u probability: P(agree | non-match)
                self.u_probs[field] = agrees[non_matches, i].sum() / n_non_matches

                # Clip to avoid log(0) issues
                self.m_probs[field] = np.clip(self.m_probs[field], 0.001, 0.999)
                self.u_probs[field] = np.clip(self.u_probs[field], 0.001, 0.999)

            # Calculate metrics
            weights = self._calculate_weights(comparison_vectors, agreement_threshold)
            predictions = (np.array(weights) > 0).astype(int)
            self._metrics = {
                'precision': float(precision_score(y, predictions, zero_division=0)),
                'recall': float(recall_score(y, predictions, zero_division=0)),
                'f1': float(f1_score(y, predictions, zero_division=0))
            }

        else:
            # Unsupervised EM estimation
            self._fit_em(agrees, n_iterations=20)

        # Set thresholds
        self._set_thresholds(comparison_vectors, agreement_threshold, labels)

    def _fit_em(self, agrees: np.ndarray, n_iterations: int = 20):
        """Fit using Expectation-Maximization algorithm."""
        n_samples, n_fields = agrees.shape

        # Initialize with reasonable priors
        # Assume about 5% of pairs are matches
        p_match = 0.05

        # Initialize m and u probabilities
        for i, field in enumerate(self.field_names):
            self.m_probs[field] = 0.9  # High agreement for matches
            self.u_probs[field] = 0.1  # Low agreement for non-matches

        for iteration in range(n_iterations):
            # E-step: Calculate posterior probability of match for each pair
            log_odds = np.zeros(n_samples)
            for i, field in enumerate(self.field_names):
                m = self.m_probs[field]
                u = self.u_probs[field]

                # Log likelihood ratio for this field
                log_ratio_agree = np.log(m / u)
                log_ratio_disagree = np.log((1 - m) / (1 - u))

                log_odds += np.where(agrees[:, i], log_ratio_agree, log_ratio_disagree)

            # Prior odds
            log_prior_odds = np.log(p_match / (1 - p_match))
            log_posterior_odds = log_odds + log_prior_odds

            # Convert to probability
            p_match_given_data = 1 / (1 + np.exp(-log_posterior_odds))

            # M-step: Update parameters
            sum_p = p_match_given_data.sum()
            sum_not_p = (1 - p_match_given_data).sum()

            for i, field in enumerate(self.field_names):
                # Update m probability
                self.m_probs[field] = np.clip(
                    (p_match_given_data * agrees[:, i]).sum() / (sum_p + 1e-10),
                    0.001, 0.999
                )
                # Update u probability
                self.u_probs[field] = np.clip(
                    ((1 - p_match_given_data) * agrees[:, i]).sum() / (sum_not_p + 1e-10),
                    0.001, 0.999
                )

            # Update prior
            p_match = np.clip(sum_p / n_samples, 0.001, 0.5)

    def _calculate_weights(
        self,
        comparison_vectors: List[List[float]],
        agreement_threshold: float = 0.85
    ) -> List[float]:
        """Calculate match weights for comparison vectors."""
        weights = []

        for vector in comparison_vectors:
            weight = 0.0
            for i, (field, score) in enumerate(zip(self.field_names, vector)):
                m = self.m_probs.get(field, 0.9)
                u = self.u_probs.get(field, 0.1)

                if score >= agreement_threshold:
                    # Field agrees
                    weight += np.log(m / u)
                else:
                    # Field disagrees
                    weight += np.log((1 - m) / (1 - u))

            weights.append(weight)

        return weights

    def _set_thresholds(
        self,
        comparison_vectors: List[List[float]],
        agreement_threshold: float,
        labels: Optional[List[int]] = None
    ):
        """Set classification thresholds."""
        weights = self._calculate_weights(comparison_vectors, agreement_threshold)

        if labels is not None:
            # Use labeled data to find optimal thresholds
            y = np.array(labels)
            match_weights = [w for w, l in zip(weights, y) if l == 1]
            non_match_weights = [w for w, l in zip(weights, y) if l == 0]

            if match_weights and non_match_weights:
                self.upper_threshold = np.percentile(match_weights, 10)
                self.lower_threshold = np.percentile(non_match_weights, 90)
        else:
            # Use percentiles for unsupervised
            self.upper_threshold = np.percentile(weights, 95)
            self.lower_threshold = np.percentile(weights, 50)

    def predict(
        self,
        comparison_vectors: List[List[float]],
        agreement_threshold: float = 0.85
    ) -> List[str]:
        """
        Predict classifications for comparison vectors.

        Returns:
            List of "match", "non_match", or "review"
        """
        weights = self._calculate_weights(comparison_vectors, agreement_threshold)
        predictions = []

        for weight in weights:
            if weight > self.upper_threshold:
                predictions.append("match")
            elif weight < self.lower_threshold:
                predictions.append("non_match")
            else:
                predictions.append("review")

        return predictions

    def predict_proba(
        self,
        comparison_vectors: List[List[float]],
        agreement_threshold: float = 0.85
    ) -> List[float]:
        """
        Predict match probabilities for comparison vectors.

        Returns:
            List of probabilities (0-1)
        """
        weights = self._calculate_weights(comparison_vectors, agreement_threshold)

        # Convert weights to probabilities using sigmoid
        probabilities = []
        for weight in weights:
            # Scale weight and apply sigmoid
            prob = 1 / (1 + np.exp(-weight / 5))  # Scaling factor for smoother probabilities
            probabilities.append(float(prob))

        return probabilities

    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters for serialization."""
        return {
            'm_probs': self.m_probs,
            'u_probs': self.u_probs,
            'upper_threshold': float(self.upper_threshold),
            'lower_threshold': float(self.lower_threshold),
            'field_names': self.field_names
        }

    def load_parameters(self, params: Dict[str, Any]):
        """Load model parameters."""
        self.m_probs = params.get('m_probs', {})
        self.u_probs = params.get('u_probs', {})
        self.upper_threshold = params.get('upper_threshold', 0.0)
        self.lower_threshold = params.get('lower_threshold', 0.0)
        self.field_names = params.get('field_names', [])

    def get_metrics(self) -> Dict[str, float]:
        """Get model metrics."""
        return self._metrics

    def get_field_weights(self) -> Dict[str, float]:
        """Get discriminative power of each field."""
        weights = {}
        for field in self.field_names:
            m = self.m_probs.get(field, 0.9)
            u = self.u_probs.get(field, 0.1)
            # Weight is log likelihood ratio for agreement
            weights[field] = float(np.log(m / u))
        return weights
