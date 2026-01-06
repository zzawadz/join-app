from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.db.models import LinkageType
from app.schemas.dataset import DatasetResponse


class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    linkage_type: LinkageType = LinkageType.LINKAGE


class ProjectCreate(ProjectBase):
    pass


class DemoProjectCreate(BaseModel):
    """Schema for creating a demo project with synthetic data."""
    name: str
    description: Optional[str] = None
    linkage_type: LinkageType = LinkageType.LINKAGE
    demo_domain: str  # "people" or "companies"


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    linkage_type: Optional[LinkageType] = None


class ProjectResponse(ProjectBase):
    id: int
    organization_id: int
    created_by_id: int
    is_demo: bool = False
    demo_domain: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ProjectDetail(ProjectResponse):
    datasets: List[DatasetResponse] = []
    column_mappings: Optional[Dict[str, str]] = None
    comparison_config: Optional[Dict[str, Any]] = None
    blocking_config: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class ColumnMappingRequest(BaseModel):
    mappings: Dict[str, str]  # source_column -> target_column


class ComparisonConfigRequest(BaseModel):
    config: Dict[str, Any]
    # Example: {"name": {"method": "jaro_winkler", "threshold": 0.85}}


class BlockingConfigRequest(BaseModel):
    config: Dict[str, Any]
    # Example: {"method": "standard", "blocking_keys": ["last_name", "zip"]}
