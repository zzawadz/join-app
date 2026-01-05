from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.db.database import get_db
from app.db.models import User, Project, LinkageModel, LabeledPair
from app.api.auth import get_current_active_user
from app.api.deps import get_project_or_404
from app.schemas.linkage import ModelCreate, ModelResponse, ModelDetail
from app.config import get_settings

settings = get_settings()
router = APIRouter()


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

        # Prepare training data
        X = [list(p.comparison_vector.values()) for p in labeled_pairs if p.comparison_vector]
        y = [1 if p.label.value == "match" else 0 for p in labeled_pairs if p.comparison_vector]

        if model.model_type == "fellegi_sunter":
            fs_model = FellegiSunterModel()
            fs_model.fit(X, y)
            model.fs_parameters = fs_model.get_parameters()
            model.metrics = fs_model.get_metrics()
        else:
            classifier = train_classifier(X, y, model.model_type, model.parameters or {})
            model_path = os.path.join(settings.models_dir, f"model_{model_id}.joblib")
            save_classifier(classifier, model_path)
            model.model_path = model_path
            model.metrics = classifier.get_metrics()

        model.training_pairs_count = len(labeled_pairs)
        model.trained_at = datetime.utcnow()
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
