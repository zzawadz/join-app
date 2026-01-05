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
