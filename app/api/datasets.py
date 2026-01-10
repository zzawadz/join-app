from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import Optional, List
import os
import csv
import aiofiles

from app.db.database import get_db
from app.db.models import User, Dataset, Project
from app.api.auth import get_current_active_user
from app.api.deps import get_user_organization, get_dataset_or_404
from app.schemas.dataset import DatasetCreate, DatasetResponse, DatasetPreview
from app.services.csv_processor import process_csv_upload
from app.services.storage import save_uploaded_file, delete_file
from app.config import get_settings

settings = get_settings()
router = APIRouter()


async def validate_csv_content(file_path: str) -> bool:
    """
    Validate that a file is actually a CSV by inspecting its content.

    Args:
        file_path: Path to the file to validate

    Returns:
        True if valid CSV

    Raises:
        ValueError: If file is not a valid CSV
    """
    try:
        # Read first 4KB to check content
        async with aiofiles.open(file_path, 'rb') as f:
            sample = await f.read(4096)

        if not sample:
            raise ValueError("File is empty")

        # Check for null bytes (binary file indicator)
        if b'\x00' in sample:
            raise ValueError("File appears to be binary, not a valid CSV text file")

        # Try to decode as text
        try:
            sample_text = sample.decode('utf-8')
        except UnicodeDecodeError:
            try:
                sample_text = sample.decode('latin-1')
            except UnicodeDecodeError:
                raise ValueError("File encoding is not supported. Please use UTF-8 or Latin-1")

        # Try to parse as CSV
        try:
            import io
            reader = csv.reader(io.StringIO(sample_text))
            rows = list(reader)
            if len(rows) < 1:
                raise ValueError("CSV file has no data")
            # Check that first row has at least one column
            if not rows[0] or all(cell.strip() == '' for cell in rows[0]):
                raise ValueError("CSV file has no valid columns")
        except csv.Error as e:
            raise ValueError(f"File is not a valid CSV: {str(e)}")

        return True

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to validate CSV file: {str(e)}")


@router.get("", response_model=List[DatasetResponse])
def list_datasets(
    project_id: Optional[int] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all datasets accessible to user, optionally filtered by project."""
    org = get_user_organization(current_user, db)

    query = db.query(Dataset).join(Project).filter(
        Project.organization_id == org.id
    )

    if project_id:
        query = query.filter(Dataset.project_id == project_id)

    return query.order_by(Dataset.created_at.desc()).all()


@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    project_id: int = Form(...),
    role: str = Form("source"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload a CSV file as a new dataset."""
    # Verify project access
    org = get_user_organization(current_user, db)
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.organization_id == org.id
    ).first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Validate file type by extension
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    # Save file with size validation
    try:
        file_path, file_size = await save_uploaded_file(file, project_id)
    except ValueError as e:
        # File size exceeded
        raise HTTPException(status_code=413, detail=str(e))

    # Validate file content is actually CSV
    try:
        await validate_csv_content(file_path)
    except ValueError as e:
        # Not a valid CSV file
        delete_file(file_path)
        raise HTTPException(status_code=400, detail=str(e))

    # Process CSV to extract metadata
    try:
        metadata = process_csv_upload(file_path)
    except Exception as e:
        # Clean up file on error
        delete_file(file_path)
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")

    # Create dataset record
    dataset = Dataset(
        project_id=project_id,
        name=name,
        original_filename=file.filename,
        file_path=file_path,
        file_size=file_size,
        row_count=metadata["row_count"],
        column_names=metadata["column_names"],
        column_types=metadata["column_types"],
        sample_data=metadata["sample_data"],
        role=role
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return dataset


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(
    dataset_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get dataset details."""
    return get_dataset_or_404(dataset_id, current_user, db)


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
def get_dataset_preview(
    dataset_id: int,
    rows: int = 10,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get sample data from dataset."""
    dataset = get_dataset_or_404(dataset_id, current_user, db)

    return {
        "id": dataset.id,
        "name": dataset.name,
        "column_names": dataset.column_names,
        "sample_data": dataset.sample_data[:rows] if dataset.sample_data else [],
        "row_count": dataset.row_count
    }


@router.delete("/{dataset_id}")
def delete_dataset(
    dataset_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a dataset."""
    dataset = get_dataset_or_404(dataset_id, current_user, db)

    # Delete file
    delete_file(dataset.file_path)

    # Delete record
    db.delete(dataset)
    db.commit()

    return {"message": "Dataset deleted"}


@router.get("/{dataset_id}/download")
def download_dataset(
    dataset_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Download the original CSV file."""
    dataset = get_dataset_or_404(dataset_id, current_user, db)

    if not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        dataset.file_path,
        media_type="text/csv",
        filename=dataset.original_filename or f"{dataset.name}.csv"
    )
