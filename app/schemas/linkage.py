from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.db.models import JobStatus


class LinkageJobCreate(BaseModel):
    model_id: Optional[int] = None
    job_type: Optional[str] = "full_linkage"


class LinkageJobResponse(BaseModel):
    id: int
    project_id: int
    model_id: Optional[int] = None
    status: JobStatus
    job_type: Optional[str] = None
    total_pairs: int = 0
    processed_pairs: int = 0
    matched_pairs: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


class LinkageJobDetail(LinkageJobResponse):
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class RecordPairResponse(BaseModel):
    id: int
    job_id: int
    left_record_idx: int
    right_record_idx: int
    comparison_vector: Optional[Dict[str, float]] = None
    match_score: Optional[float] = None
    classification: Optional[str] = None

    class Config:
        from_attributes = True


class LinkageResultsExport(BaseModel):
    format: str = "csv"


class ModelCreate(BaseModel):
    name: str
    model_type: str = "logistic_regression"  # or "random_forest", "fellegi_sunter"
    parameters: Optional[Dict[str, Any]] = None


class ModelResponse(BaseModel):
    id: int
    project_id: int
    name: str
    model_type: Optional[str] = None
    is_active: bool = False
    training_pairs_count: int = 0
    trained_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ModelDetail(ModelResponse):
    parameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class TrainingHistoryResponse(BaseModel):
    """Training history entry."""
    id: int
    model_id: int
    iteration: int
    training_pairs_count: int

    # In-sample metrics
    train_precision: Optional[float] = None
    train_recall: Optional[float] = None
    train_f1: Optional[float] = None
    train_accuracy: Optional[float] = None
    train_samples: Optional[int] = None

    # Out-of-sample metrics
    test_precision: Optional[float] = None
    test_recall: Optional[float] = None
    test_f1: Optional[float] = None
    test_accuracy: Optional[float] = None
    test_samples: Optional[int] = None

    # Cross-validation
    cv_f1_mean: Optional[float] = None
    cv_f1_std: Optional[float] = None

    feature_importance: Optional[Dict[str, float]] = None
    confusion_matrix: Optional[Dict[str, int]] = None
    trained_at: datetime

    class Config:
        from_attributes = True


class ModelStatisticsResponse(BaseModel):
    """Comprehensive model statistics."""
    model_id: int
    model_name: str
    model_type: str
    is_active: bool
    trained_at: Optional[datetime] = None
    training_pairs_count: int

    # Current train metrics
    train_precision: Optional[float] = None
    train_recall: Optional[float] = None
    train_f1: Optional[float] = None
    train_accuracy: Optional[float] = None
    train_samples: Optional[int] = None

    # Current test metrics
    test_precision: Optional[float] = None
    test_recall: Optional[float] = None
    test_f1: Optional[float] = None
    test_accuracy: Optional[float] = None
    test_samples: Optional[int] = None

    # Cross-validation
    cv_f1_mean: Optional[float] = None
    cv_f1_std: Optional[float] = None

    feature_importance: Optional[Dict[str, float]] = None
    confusion_matrix: Optional[Dict[str, int]] = None

    # Overfitting indicator
    overfitting_warning: bool = False
    overfitting_details: Optional[str] = None

    # History info
    history_retention: Optional[int] = None
    history_count: int = 0


class HistoryRetentionUpdate(BaseModel):
    """Request to update history retention setting."""
    retention: int  # 0 = keep all, positive = limit
