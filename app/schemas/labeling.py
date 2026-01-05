from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.db.models import PairLabel


class LabelingSessionCreate(BaseModel):
    strategy: Optional[str] = "uncertainty"  # or "random"
    target_labels: Optional[int] = 100


class LabelingSessionResponse(BaseModel):
    id: int
    project_id: int
    user_id: int
    status: str
    strategy: Optional[str] = None
    total_labeled: int = 0
    target_labels: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


class LabelingPairResponse(BaseModel):
    session_id: int
    left_record: Dict[str, Any]
    right_record: Dict[str, Any]
    comparison_vector: Dict[str, float]
    column_mappings: Dict[str, str]


class LabelSubmission(BaseModel):
    left_record: Dict[str, Any]
    right_record: Dict[str, Any]
    comparison_vector: Optional[Dict[str, float]] = None
    label: PairLabel
    confidence: Optional[float] = None
    labeling_time_ms: Optional[int] = None


class LabelingProgress(BaseModel):
    session_id: int
    total_labeled: int
    target_labels: Optional[int] = None
    matches: int = 0
    non_matches: int = 0
    uncertain: int = 0
    status: str
