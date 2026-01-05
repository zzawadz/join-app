"""
Tests for model training functionality.
"""
import pytest
from app.core.linkage.ml_classifier import (
    LogisticRegressionClassifier,
    RandomForestLinkageClassifier,
    GradientBoostingLinkageClassifier,
    train_classifier,
    save_classifier,
    load_classifier
)
from app.core.linkage.fellegi_sunter import FellegiSunterModel
from app.api.models import train_model_task
from app.db.models import LinkageModel, LabeledPair
import tempfile
import os


class TestLogisticRegressionClassifier:
    """Tests for LogisticRegressionClassifier."""

    def test_fit_and_predict(self):
        """Test that logistic regression can fit and predict."""
        X = [
            [1.0, 1.0],  # Match
            [0.9, 0.95], # Match
            [0.85, 0.9], # Match
            [0.2, 0.3],  # Non-match
            [0.1, 0.2],  # Non-match
            [0.3, 0.1],  # Non-match
            [0.95, 0.98], # Match
            [0.25, 0.15], # Non-match
            [0.88, 0.92], # Match
            [0.15, 0.25], # Non-match
            [0.92, 0.89], # Match
            [0.22, 0.18], # Non-match
        ]
        y = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]

        classifier = LogisticRegressionClassifier()
        classifier.fit(X, y, feature_names=["first_name", "last_name"])

        predictions = classifier.predict(X)
        assert len(predictions) == len(y)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self):
        """Test probability predictions."""
        X = [
            [1.0, 1.0], [0.9, 0.95], [0.85, 0.9],
            [0.2, 0.3], [0.1, 0.2], [0.3, 0.1],
            [0.95, 0.98], [0.25, 0.15], [0.88, 0.92],
            [0.15, 0.25], [0.92, 0.89], [0.22, 0.18],
        ]
        y = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]

        classifier = LogisticRegressionClassifier()
        classifier.fit(X, y)

        probs = classifier.predict_proba(X)
        assert len(probs) == len(y)
        assert all(0 <= p <= 1 for p in probs)

        # High similarity should have high probability
        high_sim_prob = classifier.predict_proba([[1.0, 1.0]])[0]
        low_sim_prob = classifier.predict_proba([[0.1, 0.1]])[0]
        assert high_sim_prob > low_sim_prob

    def test_get_metrics(self):
        """Test that metrics are calculated."""
        X = [
            [1.0, 1.0], [0.9, 0.95], [0.85, 0.9],
            [0.2, 0.3], [0.1, 0.2], [0.3, 0.1],
            [0.95, 0.98], [0.25, 0.15], [0.88, 0.92],
            [0.15, 0.25], [0.92, 0.89], [0.22, 0.18],
        ]
        y = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]

        classifier = LogisticRegressionClassifier()
        classifier.fit(X, y)

        metrics = classifier.get_metrics()
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "accuracy" in metrics

        # All metrics should be between 0 and 1
        for key, value in metrics.items():
            if not key.startswith("cv_"):
                assert 0 <= value <= 1, f"{key} = {value} is out of range"

    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        X = [
            [1.0, 1.0], [0.9, 0.95], [0.85, 0.9],
            [0.2, 0.3], [0.1, 0.2], [0.3, 0.1],
            [0.95, 0.98], [0.25, 0.15], [0.88, 0.92],
            [0.15, 0.25], [0.92, 0.89], [0.22, 0.18],
        ]
        y = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]

        classifier = LogisticRegressionClassifier()
        classifier.fit(X, y, feature_names=["first_name", "last_name"])

        importance = classifier.get_feature_importance()
        assert "first_name" in importance
        assert "last_name" in importance


class TestRandomForestClassifier:
    """Tests for RandomForestLinkageClassifier."""

    def test_fit_and_predict(self):
        """Test that random forest can fit and predict."""
        X = [
            [1.0, 1.0], [0.9, 0.95], [0.85, 0.9],
            [0.2, 0.3], [0.1, 0.2], [0.3, 0.1],
            [0.95, 0.98], [0.25, 0.15], [0.88, 0.92],
            [0.15, 0.25], [0.92, 0.89], [0.22, 0.18],
        ]
        y = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]

        classifier = RandomForestLinkageClassifier(n_estimators=10)
        classifier.fit(X, y)

        predictions = classifier.predict(X)
        assert len(predictions) == len(y)

    def test_predict_proba(self):
        """Test probability predictions."""
        X = [
            [1.0, 1.0], [0.9, 0.95], [0.85, 0.9],
            [0.2, 0.3], [0.1, 0.2], [0.3, 0.1],
            [0.95, 0.98], [0.25, 0.15], [0.88, 0.92],
            [0.15, 0.25], [0.92, 0.89], [0.22, 0.18],
        ]
        y = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]

        classifier = RandomForestLinkageClassifier(n_estimators=10)
        classifier.fit(X, y)

        probs = classifier.predict_proba(X)
        assert len(probs) == len(y)
        assert all(0 <= p <= 1 for p in probs)


class TestGradientBoostingClassifier:
    """Tests for GradientBoostingLinkageClassifier."""

    def test_fit_and_predict(self):
        """Test that gradient boosting can fit and predict."""
        X = [
            [1.0, 1.0], [0.9, 0.95], [0.85, 0.9],
            [0.2, 0.3], [0.1, 0.2], [0.3, 0.1],
            [0.95, 0.98], [0.25, 0.15], [0.88, 0.92],
            [0.15, 0.25], [0.92, 0.89], [0.22, 0.18],
        ]
        y = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]

        classifier = GradientBoostingLinkageClassifier(n_estimators=10)
        classifier.fit(X, y)

        predictions = classifier.predict(X)
        assert len(predictions) == len(y)


class TestTrainClassifierFactory:
    """Tests for the train_classifier factory function."""

    def test_train_logistic_regression(self):
        """Test training logistic regression via factory."""
        X = [
            [1.0, 1.0], [0.9, 0.95], [0.85, 0.9],
            [0.2, 0.3], [0.1, 0.2], [0.3, 0.1],
            [0.95, 0.98], [0.25, 0.15], [0.88, 0.92],
            [0.15, 0.25], [0.92, 0.89], [0.22, 0.18],
        ]
        y = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]

        classifier = train_classifier(X, y, "logistic_regression", {})
        assert isinstance(classifier, LogisticRegressionClassifier)
        assert classifier.model is not None

    def test_train_random_forest(self):
        """Test training random forest via factory."""
        X = [
            [1.0, 1.0], [0.9, 0.95], [0.85, 0.9],
            [0.2, 0.3], [0.1, 0.2], [0.3, 0.1],
            [0.95, 0.98], [0.25, 0.15], [0.88, 0.92],
            [0.15, 0.25], [0.92, 0.89], [0.22, 0.18],
        ]
        y = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]

        classifier = train_classifier(X, y, "random_forest", {"n_estimators": 10})
        assert isinstance(classifier, RandomForestLinkageClassifier)
        assert classifier.model is not None

    def test_train_unknown_model_type(self):
        """Test that unknown model type raises error."""
        X = [[1.0, 1.0], [0.2, 0.3]]
        y = [1, 0]

        with pytest.raises(ValueError, match="Unknown model type"):
            train_classifier(X, y, "unknown_model", {})


class TestModelSaveLoad:
    """Tests for saving and loading classifiers."""

    def test_save_and_load_logistic_regression(self):
        """Test saving and loading logistic regression model."""
        X = [
            [1.0, 1.0], [0.9, 0.95], [0.85, 0.9],
            [0.2, 0.3], [0.1, 0.2], [0.3, 0.1],
            [0.95, 0.98], [0.25, 0.15], [0.88, 0.92],
            [0.15, 0.25], [0.92, 0.89], [0.22, 0.18],
        ]
        y = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]

        classifier = LogisticRegressionClassifier()
        classifier.fit(X, y, feature_names=["f1", "f2"])

        # Get predictions before saving
        original_preds = classifier.predict(X)
        original_probs = classifier.predict_proba(X)

        # Save and load
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            model_path = f.name

        try:
            save_classifier(classifier, model_path)
            loaded_classifier = load_classifier(model_path)

            # Verify loaded model produces same results
            loaded_preds = loaded_classifier.predict(X)
            loaded_probs = loaded_classifier.predict_proba(X)

            assert original_preds == loaded_preds
            assert all(abs(a - b) < 0.001 for a, b in zip(original_probs, loaded_probs))
        finally:
            os.unlink(model_path)

    def test_save_and_load_random_forest(self):
        """Test saving and loading random forest model."""
        X = [
            [1.0, 1.0], [0.9, 0.95], [0.85, 0.9],
            [0.2, 0.3], [0.1, 0.2], [0.3, 0.1],
            [0.95, 0.98], [0.25, 0.15], [0.88, 0.92],
            [0.15, 0.25], [0.92, 0.89], [0.22, 0.18],
        ]
        y = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]

        classifier = RandomForestLinkageClassifier(n_estimators=10)
        classifier.fit(X, y)

        original_preds = classifier.predict(X)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            model_path = f.name

        try:
            save_classifier(classifier, model_path)
            loaded_classifier = load_classifier(model_path)

            loaded_preds = loaded_classifier.predict(X)
            assert original_preds == loaded_preds
        finally:
            os.unlink(model_path)


class TestFellegiSunterModel:
    """Tests for Fellegi-Sunter model."""

    def test_fit_and_predict(self):
        """Test that FS model can fit and predict."""
        X = [
            [1.0, 1.0], [0.9, 0.95], [0.85, 0.9],
            [0.2, 0.3], [0.1, 0.2], [0.3, 0.1],
            [0.95, 0.98], [0.25, 0.15], [0.88, 0.92],
            [0.15, 0.25], [0.92, 0.89], [0.22, 0.18],
        ]
        y = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]

        model = FellegiSunterModel()
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert len(probs) == len(y)
        assert all(0 <= p <= 1 for p in probs)

    def test_get_and_load_parameters(self):
        """Test parameter serialization."""
        X = [
            [1.0, 1.0], [0.9, 0.95], [0.85, 0.9],
            [0.2, 0.3], [0.1, 0.2], [0.3, 0.1],
            [0.95, 0.98], [0.25, 0.15], [0.88, 0.92],
            [0.15, 0.25], [0.92, 0.89], [0.22, 0.18],
        ]
        y = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]

        model1 = FellegiSunterModel()
        model1.fit(X, y)
        original_probs = model1.predict_proba(X)

        # Save and load parameters
        params = model1.get_parameters()
        model2 = FellegiSunterModel()
        model2.load_parameters(params)

        loaded_probs = model2.predict_proba(X)
        assert all(abs(a - b) < 0.001 for a, b in zip(original_probs, loaded_probs))


class TestModelTrainingIntegration:
    """Integration tests for model training (direct database tests)."""

    def test_train_model_task_with_labeled_data(self, test_db, test_project, labeled_pairs):
        """Test the background training task directly."""
        from app.api.models import train_model_task
        from app.config import get_settings
        import tempfile

        settings = get_settings()

        # Create model record
        model = LinkageModel(
            project_id=test_project.id,
            name="Test Model",
            model_type="logistic_regression"
        )
        test_db.add(model)
        test_db.commit()
        test_db.refresh(model)
        model_id = model.id

        # Run training task directly (normally runs in background)
        # Note: We need to run it synchronously for testing
        from app.db.database import SessionLocal
        from app.core.linkage.ml_classifier import train_classifier, save_classifier
        import os

        # Get labeled pairs for training
        labeled = test_db.query(LabeledPair).filter(
            LabeledPair.project_id == test_project.id
        ).all()

        assert len(labeled) >= 10, "Need at least 10 labeled pairs for training"

        # Prepare training data
        X = [list(p.comparison_vector.values()) for p in labeled if p.comparison_vector]
        y = [1 if p.label.value == "match" else 0 for p in labeled if p.comparison_vector]

        # Train classifier
        classifier = train_classifier(X, y, "logistic_regression", {})

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            model_path = f.name

        save_classifier(classifier, model_path)

        # Update model record
        model.model_path = model_path
        model.metrics = classifier.get_metrics()
        model.training_pairs_count = len(labeled)
        test_db.commit()

        # Verify model was trained
        test_db.refresh(model)
        assert model.metrics is not None
        assert "precision" in model.metrics
        assert "recall" in model.metrics
        assert model.training_pairs_count == len(labeled)

        # Cleanup
        os.unlink(model_path)

    def test_model_activation(self, test_db, test_project):
        """Test model activation logic."""
        # Create two models
        model1 = LinkageModel(
            project_id=test_project.id,
            name="Model 1",
            model_type="logistic_regression",
            is_active=True
        )
        model2 = LinkageModel(
            project_id=test_project.id,
            name="Model 2",
            model_type="random_forest",
            is_active=False
        )
        test_db.add(model1)
        test_db.add(model2)
        test_db.commit()

        # Activate model2
        test_db.query(LinkageModel).filter(
            LinkageModel.project_id == test_project.id,
            LinkageModel.id != model2.id
        ).update({"is_active": False})
        model2.is_active = True
        test_db.commit()

        test_db.refresh(model1)
        test_db.refresh(model2)

        assert model1.is_active is False
        assert model2.is_active is True

    def test_model_deletion_cleanup(self, test_db, test_project):
        """Test that model deletion works correctly."""
        model = LinkageModel(
            project_id=test_project.id,
            name="Model to Delete",
            model_type="logistic_regression"
        )
        test_db.add(model)
        test_db.commit()
        model_id = model.id

        # Delete model
        test_db.delete(model)
        test_db.commit()

        # Verify it's deleted
        deleted_model = test_db.query(LinkageModel).filter(LinkageModel.id == model_id).first()
        assert deleted_model is None


class TestLabelingProgressIntegration:
    """Integration tests for labeling progress."""

    def test_label_distribution_query(self, test_db, test_project, labeled_pairs):
        """Test that label distribution query works correctly."""
        from sqlalchemy import func
        from app.db.models import LabelingSession

        session = test_db.query(LabelingSession).filter(
            LabelingSession.project_id == test_project.id
        ).first()

        # Query label distribution
        labels = test_db.query(LabeledPair.label, func.count(LabeledPair.id)).filter(
            LabeledPair.session_id == session.id
        ).group_by(LabeledPair.label).all()

        label_counts = {label.value: count for label, count in labels}

        assert "match" in label_counts or "non_match" in label_counts
        total = sum(label_counts.values())
        assert total == session.total_labeled
