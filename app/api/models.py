from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime

from app.db.database import get_db
from app.db.models import User, Project, LinkageModel, LabeledPair, TrainingHistory
from app.api.auth import get_current_active_user
from app.api.deps import get_project_or_404
from app.schemas.linkage import (
    ModelCreate, ModelResponse, ModelDetail,
    TrainingHistoryResponse, ModelStatisticsResponse, HistoryRetentionUpdate
)
from app.config import get_settings

settings = get_settings()
router = APIRouter()


def enforce_history_retention(model: LinkageModel, db: Session):
    """Delete old training history entries beyond retention limit."""
    if model.history_retention is None or model.history_retention == 0:
        return  # Keep all

    # Count existing history entries
    count = db.query(TrainingHistory).filter(
        TrainingHistory.model_id == model.id
    ).count()

    if count > model.history_retention:
        # Get IDs to delete (oldest first)
        to_delete = db.query(TrainingHistory.id).filter(
            TrainingHistory.model_id == model.id
        ).order_by(TrainingHistory.iteration.asc()).limit(count - model.history_retention).all()

        if to_delete:
            db.query(TrainingHistory).filter(
                TrainingHistory.id.in_([t.id for t in to_delete])
            ).delete(synchronize_session=False)


def train_model_task(model_id: int, db_url: str):
    """Background task to train a model."""
    from app.db.database import SessionLocal
    from app.core.linkage.fellegi_sunter import FellegiSunterModel
    from app.core.linkage.ml_classifier import train_classifier, save_classifier
    import os

    db = SessionLocal()
    try:
        model = db.query(LinkageModel).filter(LinkageModel.id == model_id).first()
        if not model:
            return

        project = model.project

        # Get labeled pairs for training
        labeled_pairs = db.query(LabeledPair).filter(
            LabeledPair.project_id == project.id
        ).all()

        if len(labeled_pairs) < 10:
            model.metrics = {"error": "Need at least 10 labeled pairs for training"}
            db.commit()
            return

        # Get feature names from comparison vector keys
        feature_names = None
        if labeled_pairs and labeled_pairs[0].comparison_vector:
            feature_names = list(labeled_pairs[0].comparison_vector.keys())

        # Prepare training data
        X = [list(p.comparison_vector.values()) for p in labeled_pairs if p.comparison_vector]
        y = [1 if p.label.value == "match" else 0 for p in labeled_pairs if p.comparison_vector]

        # Determine next iteration number
        last_history = db.query(TrainingHistory).filter(
            TrainingHistory.model_id == model_id
        ).order_by(TrainingHistory.iteration.desc()).first()
        next_iteration = (last_history.iteration + 1) if last_history else 1

        train_metrics = {}
        test_metrics = {}
        confusion_matrix = {}
        feature_importance = {}
        cv_metrics = {}
        train_samples = 0
        test_samples = 0

        if model.model_type == "fellegi_sunter":
            fs_model = FellegiSunterModel()
            fs_model.fit(X, y)
            model.fs_parameters = fs_model.get_parameters()
            model.metrics = fs_model.get_metrics()
            train_metrics = fs_model.get_metrics()
        else:
            classifier = train_classifier(X, y, model.model_type, model.parameters or {})
            model_path = os.path.join(settings.models_dir, f"model_{model_id}.joblib")
            save_classifier(classifier, model_path)
            model.model_path = model_path

            # Get detailed metrics from classifier
            train_metrics = classifier.get_train_metrics()
            test_metrics = classifier.get_test_metrics()
            confusion_matrix = classifier.get_confusion_matrix()
            train_samples = classifier.get_train_samples()
            test_samples = classifier.get_test_samples()

            try:
                feature_importance = classifier.get_feature_importance()
            except NotImplementedError:
                feature_importance = {}

            # Store combined metrics for backward compatibility
            combined_metrics = classifier.get_metrics().copy()
            combined_metrics['train_metrics'] = train_metrics
            combined_metrics['test_metrics'] = test_metrics
            combined_metrics['confusion_matrix'] = confusion_matrix
            combined_metrics['feature_importance'] = feature_importance
            combined_metrics['train_samples'] = train_samples
            combined_metrics['test_samples'] = test_samples
            model.metrics = combined_metrics

            cv_metrics = {
                'cv_f1_mean': combined_metrics.get('cv_f1_mean'),
                'cv_f1_std': combined_metrics.get('cv_f1_std')
            }

        model.training_pairs_count = len(labeled_pairs)
        model.trained_at = datetime.utcnow()

        # Record training history
        history = TrainingHistory(
            model_id=model_id,
            iteration=next_iteration,
            training_pairs_count=len(labeled_pairs),
            train_precision=train_metrics.get('precision'),
            train_recall=train_metrics.get('recall'),
            train_f1=train_metrics.get('f1'),
            train_accuracy=train_metrics.get('accuracy'),
            train_samples=train_samples,
            test_precision=test_metrics.get('precision'),
            test_recall=test_metrics.get('recall'),
            test_f1=test_metrics.get('f1'),
            test_accuracy=test_metrics.get('accuracy'),
            test_samples=test_samples,
            cv_f1_mean=cv_metrics.get('cv_f1_mean'),
            cv_f1_std=cv_metrics.get('cv_f1_std'),
            feature_importance=feature_importance if feature_importance else None,
            confusion_matrix=confusion_matrix if confusion_matrix else None
        )
        db.add(history)
        db.commit()

        # Enforce history retention limit
        enforce_history_retention(model, db)
        db.commit()

    except Exception as e:
        model.metrics = {"error": str(e)}
        db.commit()
    finally:
        db.close()


@router.get("/projects/{project_id}/models", response_model=List[ModelResponse])
def list_project_models(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all models for a project."""
    project = get_project_or_404(project_id, current_user, db)

    return db.query(LinkageModel).filter(
        LinkageModel.project_id == project_id
    ).order_by(LinkageModel.created_at.desc()).all()


@router.post("/projects/{project_id}/train", response_model=ModelResponse)
def train_model(
    project_id: int,
    model_data: ModelCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Train a new model on labeled data."""
    project = get_project_or_404(project_id, current_user, db)

    # Check for labeled data
    labeled_count = db.query(LabeledPair).filter(
        LabeledPair.project_id == project_id
    ).count()

    if labeled_count < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 10 labeled pairs for training. Currently have {labeled_count}."
        )

    # Create model record
    model = LinkageModel(
        project_id=project_id,
        name=model_data.name,
        model_type=model_data.model_type,
        parameters=model_data.parameters or {}
    )

    db.add(model)
    db.commit()
    db.refresh(model)

    # Start training in background
    background_tasks.add_task(train_model_task, model.id, settings.database_url)

    return model


@router.get("/{model_id}", response_model=ModelDetail)
def get_model(
    model_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get model details."""
    model = db.query(LinkageModel).filter(LinkageModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    get_project_or_404(model.project_id, current_user, db)

    return model


@router.post("/{model_id}/activate")
def activate_model(
    model_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Set model as the active model for the project."""
    model = db.query(LinkageModel).filter(LinkageModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    get_project_or_404(model.project_id, current_user, db)

    # Deactivate other models
    db.query(LinkageModel).filter(
        LinkageModel.project_id == model.project_id,
        LinkageModel.id != model_id
    ).update({"is_active": False})

    model.is_active = True
    db.commit()

    return {"message": "Model activated", "model_id": model_id}


@router.delete("/{model_id}")
def delete_model(
    model_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a model."""
    model = db.query(LinkageModel).filter(LinkageModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    get_project_or_404(model.project_id, current_user, db)

    # Delete model file if exists
    if model.model_path:
        import os
        if os.path.exists(model.model_path):
            os.remove(model.model_path)

    db.delete(model)
    db.commit()

    return {"message": "Model deleted"}


@router.get("/{model_id}/statistics", response_model=ModelStatisticsResponse)
def get_model_statistics(
    model_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive model statistics including train/test metrics."""
    model = db.query(LinkageModel).filter(LinkageModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    get_project_or_404(model.project_id, current_user, db)

    # Count training history entries
    history_count = db.query(TrainingHistory).filter(
        TrainingHistory.model_id == model_id
    ).count()

    # Get latest training history for metrics
    latest_history = db.query(TrainingHistory).filter(
        TrainingHistory.model_id == model_id
    ).order_by(TrainingHistory.iteration.desc()).first()

    # Extract metrics from model.metrics or latest history
    metrics = model.metrics or {}
    train_metrics = metrics.get('train_metrics', {})
    test_metrics = metrics.get('test_metrics', {})
    confusion_matrix = metrics.get('confusion_matrix')
    feature_importance = metrics.get('feature_importance')

    # Use history if available and metrics are empty
    if latest_history and not train_metrics:
        train_metrics = {
            'precision': latest_history.train_precision,
            'recall': latest_history.train_recall,
            'f1': latest_history.train_f1,
            'accuracy': latest_history.train_accuracy
        }
        test_metrics = {
            'precision': latest_history.test_precision,
            'recall': latest_history.test_recall,
            'f1': latest_history.test_f1,
            'accuracy': latest_history.test_accuracy
        }
        confusion_matrix = latest_history.confusion_matrix
        feature_importance = latest_history.feature_importance

    # Check for overfitting
    overfitting_warning = False
    overfitting_details = None
    train_f1 = train_metrics.get('f1') if train_metrics else None
    test_f1 = test_metrics.get('f1') if test_metrics else None

    if train_f1 is not None and test_f1 is not None:
        if train_f1 > test_f1 + 0.15:
            overfitting_warning = True
            overfitting_details = f"Train F1 ({train_f1:.2%}) significantly higher than Test F1 ({test_f1:.2%})"

    return ModelStatisticsResponse(
        model_id=model.id,
        model_name=model.name,
        model_type=model.model_type or "unknown",
        is_active=model.is_active,
        trained_at=model.trained_at,
        training_pairs_count=model.training_pairs_count or 0,
        train_precision=train_metrics.get('precision'),
        train_recall=train_metrics.get('recall'),
        train_f1=train_metrics.get('f1'),
        train_accuracy=train_metrics.get('accuracy'),
        train_samples=metrics.get('train_samples') or (latest_history.train_samples if latest_history else None),
        test_precision=test_metrics.get('precision'),
        test_recall=test_metrics.get('recall'),
        test_f1=test_metrics.get('f1'),
        test_accuracy=test_metrics.get('accuracy'),
        test_samples=metrics.get('test_samples') or (latest_history.test_samples if latest_history else None),
        cv_f1_mean=metrics.get('cv_f1_mean'),
        cv_f1_std=metrics.get('cv_f1_std'),
        feature_importance=feature_importance,
        confusion_matrix=confusion_matrix,
        overfitting_warning=overfitting_warning,
        overfitting_details=overfitting_details,
        history_retention=model.history_retention,
        history_count=history_count
    )


@router.get("/{model_id}/training-history", response_model=List[TrainingHistoryResponse])
def get_training_history(
    model_id: int,
    limit: int = 10,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get training history for a model."""
    model = db.query(LinkageModel).filter(LinkageModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    get_project_or_404(model.project_id, current_user, db)

    history = db.query(TrainingHistory).filter(
        TrainingHistory.model_id == model_id
    ).order_by(TrainingHistory.iteration.desc()).limit(limit).all()

    return history


@router.put("/{model_id}/history-retention")
def update_history_retention(
    model_id: int,
    update: HistoryRetentionUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update history retention setting for a model."""
    model = db.query(LinkageModel).filter(LinkageModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    get_project_or_404(model.project_id, current_user, db)

    model.history_retention = update.retention if update.retention > 0 else None
    db.commit()

    # Enforce new retention limit immediately
    enforce_history_retention(model, db)
    db.commit()

    return {
        "message": "History retention updated",
        "retention": model.history_retention
    }
