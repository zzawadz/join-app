from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.db.database import get_db
from app.db.models import User, Project, Dataset, LinkageType, LabelingSession, LabeledPair, PairLabel
from app.api.auth import get_current_active_user
from app.api.deps import get_user_organization, get_project_or_404
from app.schemas.project import (
    ProjectCreate, ProjectUpdate, ProjectResponse, ProjectDetail,
    ColumnMappingRequest, ComparisonConfigRequest, BlockingConfigRequest,
    DemoProjectCreate
)
from app.services.column_mapper import suggest_column_mappings
from app.config import get_settings

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


@router.post("/demo", response_model=ProjectDetail)
def create_demo_project(
    demo_data: DemoProjectCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a demo project with synthetic test data."""
    from app.services.demo_generator import DemoDataGenerator, save_demo_datasets

    org = get_user_organization(current_user, db)
    settings = get_settings()

    # Validate demo domain
    if demo_data.demo_domain not in ["people", "companies"]:
        raise HTTPException(
            status_code=400,
            detail="demo_domain must be 'people' or 'companies'"
        )

    # Create the project
    project = Project(
        name=demo_data.name,
        description=demo_data.description or f"Demo project with {demo_data.demo_domain} data",
        organization_id=org.id,
        created_by_id=current_user.id,
        linkage_type=demo_data.linkage_type,
        is_demo=True,
        demo_domain=demo_data.demo_domain
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    try:
        # Generate demo data
        generator = DemoDataGenerator(seed=42)
        is_dedup = demo_data.linkage_type == LinkageType.DEDUPLICATION

        if is_dedup:
            # Create deduplication dataset
            df = generator.create_dedup_dataset(
                domain=demo_data.demo_domain,
                n_unique=80,
                duplicate_rate=0.3
            )
            source_path, _ = save_demo_datasets(
                settings.storage_path, project.id, df, is_dedup=True
            )

            # Create dataset record
            dataset = Dataset(
                project_id=project.id,
                name=f"Demo {demo_data.demo_domain.title()} Dataset",
                original_filename="demo_dedupe.csv",
                file_path=source_path,
                row_count=len(df),
                column_names=list(df.columns),
                sample_data=df.head(5).to_dict(orient='records'),
                role="dedupe"
            )
            db.add(dataset)

            # For dedup, source and target are the same
            source_df = target_df = df
        else:
            # Create linkage datasets
            source_df, target_df = generator.create_linkage_datasets(
                domain=demo_data.demo_domain,
                source_size=80,
                target_size=100,
                overlap_rate=0.4
            )
            source_path, target_path = save_demo_datasets(
                settings.storage_path, project.id, source_df, target_df
            )

            # Create dataset records
            source_dataset = Dataset(
                project_id=project.id,
                name=f"Demo Source ({demo_data.demo_domain.title()})",
                original_filename="demo_source.csv",
                file_path=source_path,
                row_count=len(source_df),
                column_names=list(source_df.columns),
                sample_data=source_df.head(5).to_dict(orient='records'),
                role="source"
            )
            target_dataset = Dataset(
                project_id=project.id,
                name=f"Demo Target ({demo_data.demo_domain.title()})",
                original_filename="demo_target.csv",
                file_path=target_path,
                row_count=len(target_df),
                column_names=list(target_df.columns),
                sample_data=target_df.head(5).to_dict(orient='records'),
                role="target"
            )
            db.add(source_dataset)
            db.add(target_dataset)

        # Set up column mappings and comparison config
        project.column_mappings = generator.get_column_mappings(demo_data.demo_domain)
        project.comparison_config = generator.get_comparison_config(demo_data.demo_domain)

        # Generate pre-labeled pairs
        labeled_pairs_data = generator.generate_labeled_pairs(
            source_df, target_df, demo_data.demo_domain,
            n_matches=15, n_non_matches=15
        )

        # Create a labeling session for the pre-labeled pairs
        labeling_session = LabelingSession(
            project_id=project.id,
            user_id=current_user.id,
            status="completed",
            strategy="demo",
            total_labeled=len(labeled_pairs_data),
            target_labels=len(labeled_pairs_data),
            started_at=datetime.utcnow()
        )
        db.add(labeling_session)
        db.flush()

        # Add labeled pairs
        for pair_data in labeled_pairs_data:
            labeled_pair = LabeledPair(
                session_id=labeling_session.id,
                project_id=project.id,
                labeled_by_id=current_user.id,
                left_record=pair_data["left_record"],
                right_record=pair_data["right_record"],
                comparison_vector=pair_data["comparison_vector"],
                label=PairLabel.MATCH if pair_data["label"] == "match" else PairLabel.NON_MATCH
            )
            db.add(labeled_pair)

        db.commit()
        db.refresh(project)

        return project

    except Exception as e:
        # Clean up on failure
        db.delete(project)
        db.commit()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create demo project: {str(e)}"
        )


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
