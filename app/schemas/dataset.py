from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class DatasetBase(BaseModel):
    name: str
    role: Optional[str] = "source"


class DatasetCreate(DatasetBase):
    project_id: int


class DatasetResponse(DatasetBase):
    id: int
    project_id: int
    original_filename: Optional[str] = None
    file_size: Optional[int] = None
    row_count: Optional[int] = None
    column_names: Optional[List[str]] = None
    column_types: Optional[Dict[str, str]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class DatasetPreview(BaseModel):
    id: int
    name: str
    column_names: List[str]
    sample_data: List[Dict[str, Any]]
    row_count: int
