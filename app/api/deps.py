from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User, Project, Dataset, Organization, OrganizationMember
from app.api.auth import get_current_active_user


def get_user_organization(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Organization:
    """Get the user's primary organization (first one they belong to)."""
    membership = db.query(OrganizationMember).filter(
        OrganizationMember.user_id == current_user.id
    ).first()

    if not membership:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User does not belong to any organization"
        )

    return membership.organization


def get_project_or_404(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Project:
    """Get a project by ID, ensuring user has access."""
    # Get user's organizations
    user_org_ids = [
        m.organization_id for m in
        db.query(OrganizationMember).filter(
            OrganizationMember.user_id == current_user.id
        ).all()
    ]

    project = db.query(Project).filter(
        Project.id == project_id,
        Project.organization_id.in_(user_org_ids)
    ).first()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    return project


def get_dataset_or_404(
    dataset_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Dataset:
    """Get a dataset by ID, ensuring user has access through project."""
    # Get user's organizations
    user_org_ids = [
        m.organization_id for m in
        db.query(OrganizationMember).filter(
            OrganizationMember.user_id == current_user.id
        ).all()
    ]

    dataset = db.query(Dataset).join(Project).filter(
        Dataset.id == dataset_id,
        Project.organization_id.in_(user_org_ids)
    ).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )

    return dataset
