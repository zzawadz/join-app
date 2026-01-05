from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional

from app.db.database import get_db
from app.db.models import User, Project, Dataset, LinkageType
from app.api.auth import get_current_active_user
from app.api.deps import get_user_organization, get_project_or_404
from app.schemas.project import (
    ProjectCreate, ProjectUpdate, ProjectResponse, ProjectDetail,
    ColumnMappingRequest, ComparisonConfigRequest, BlockingConfigRequest
)
from app.services.column_mapper import suggest_column_mappings

router = APIRouter()


@router.get("", response_model=List[ProjectResponse])
def list_projects(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all projects in user's organization."""
    org = get_user_organization(current_user, db)

    return db.query(Project).filter(
        Project.organization_id == org.id
    ).order_by(Project.created_at.desc()).all()


@router.post("", response_model=ProjectResponse)
def create_project(
    project_data: ProjectCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new project."""
    org = get_user_organization(current_user, db)

    project = Project(
        name=project_data.name,
        description=project_data.description,
        organization_id=org.id,
        created_by_id=current_user.id,
        linkage_type=project_data.linkage_type
    )

    db.add(project)
    db.commit()
    db.refresh(project)

    return project


@router.get("/{project_id}", response_model=ProjectDetail)
def get_project(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get project details including datasets and configuration."""
    project = get_project_or_404(project_id, current_user, db)

    return project


@router.put("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: int,
    project_data: ProjectUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update project details."""
    project = get_project_or_404(project_id, current_user, db)

    if project_data.name is not None:
        project.name = project_data.name
    if project_data.description is not None:
        project.description = project_data.description
    if project_data.linkage_type is not None:
        project.linkage_type = project_data.linkage_type

    db.commit()
    db.refresh(project)

    return project


@router.delete("/{project_id}")
def delete_project(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a project and all associated data."""
    project = get_project_or_404(project_id, current_user, db)

    db.delete(project)
    db.commit()

    return {"message": "Project deleted"}


# ============ Column Mapping Endpoints ============

@router.get("/{project_id}/mapping/suggest")
def suggest_mappings(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Auto-suggest column mappings based on column names and content."""
    project = get_project_or_404(project_id, current_user, db)

    # Get datasets
    datasets = db.query(Dataset).filter(Dataset.project_id == project_id).all()

    if project.linkage_type == LinkageType.DEDUPLICATION:
        if not datasets:
            raise HTTPException(status_code=400, detail="No datasets found")
        # For dedup, suggest columns to compare with themselves
        dataset = datasets[0]
        return {
            "suggestions": [
                {"column": col, "confidence": 1.0}
                for col in dataset.column_names
            ]
        }
    else:
        # For linkage, need source and target
        source = next((d for d in datasets if d.role == "source"), None)
        target = next((d for d in datasets if d.role == "target"), None)

        if not source or not target:
            raise HTTPException(
                status_code=400,
                detail="Need both source and target datasets for linkage"
            )

        suggestions = suggest_column_mappings(
            source.column_names, target.column_names,
            source.sample_data, target.sample_data
        )

        return {"suggestions": suggestions}


@router.post("/{project_id}/mapping")
def save_mappings(
    project_id: int,
    mapping_data: ColumnMappingRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Save column mappings for the project."""
    project = get_project_or_404(project_id, current_user, db)

    project.column_mappings = mapping_data.mappings
    db.commit()

    return {"message": "Mappings saved", "mappings": project.column_mappings}


@router.get("/{project_id}/mapping")
def get_mappings(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get current column mappings."""
    project = get_project_or_404(project_id, current_user, db)
    return {"mappings": project.column_mappings or {}}


# ============ Comparison Configuration ============

@router.post("/{project_id}/config/comparisons")
def save_comparison_config(
    project_id: int,
    config_data: ComparisonConfigRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Save comparison configuration for columns."""
    project = get_project_or_404(project_id, current_user, db)

    project.comparison_config = config_data.config
    db.commit()

    return {"message": "Comparison config saved", "config": project.comparison_config}


@router.get("/{project_id}/config/comparisons")
def get_comparison_config(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get comparison configuration."""
    project = get_project_or_404(project_id, current_user, db)
    return {"config": project.comparison_config or {}}


# ============ Blocking Configuration ============

@router.post("/{project_id}/config/blocking")
def save_blocking_config(
    project_id: int,
    config_data: BlockingConfigRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Save blocking configuration."""
    project = get_project_or_404(project_id, current_user, db)

    project.blocking_config = config_data.config
    db.commit()

    return {"message": "Blocking config saved", "config": project.blocking_config}


@router.get("/{project_id}/config/blocking")
def get_blocking_config(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get blocking configuration."""
    project = get_project_or_404(project_id, current_user, db)
    return {"config": project.blocking_config or {}}


# ============ Available Comparators ============

@router.get("/{project_id}/config/comparators")
def list_comparators(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List available comparison methods."""
    return {
        "comparators": [
            {
                "id": "exact",
                "name": "Exact Match",
                "description": "Exact string equality",
                "has_threshold": False
            },
            {
                "id": "jaro_winkler",
                "name": "Jaro-Winkler",
                "description": "Jaro-Winkler string similarity (good for names)",
                "has_threshold": True,
                "default_threshold": 0.85
            },
            {
                "id": "levenshtein",
                "name": "Levenshtein",
                "description": "Edit distance based similarity",
                "has_threshold": True,
                "default_threshold": 0.8
            },
            {
                "id": "soundex",
                "name": "Soundex",
                "description": "Phonetic matching (English names)",
                "has_threshold": False
            },
            {
                "id": "numeric",
                "name": "Numeric",
                "description": "Numeric similarity with tolerance",
                "has_threshold": True,
                "default_threshold": 0.0
            },
            {
                "id": "date",
                "name": "Date",
                "description": "Date comparison with day tolerance",
                "has_threshold": True,
                "default_threshold": 0
            }
        ]
    }
