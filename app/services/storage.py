import os
import uuid
import aiofiles
from pathlib import Path
from fastapi import UploadFile
from typing import Tuple

from app.config import get_settings

settings = get_settings()


async def save_uploaded_file(
    file: UploadFile,
    project_id: int,
    max_size_mb: int = None
) -> Tuple[str, int]:
    """
    Save an uploaded file to the storage directory with size validation.

    Args:
        file: The uploaded file
        project_id: Project ID for organization
        max_size_mb: Maximum file size in MB (defaults to config setting)

    Returns:
        Tuple of (file_path, file_size)

    Raises:
        ValueError: If file exceeds size limit
    """
    if max_size_mb is None:
        max_size_mb = settings.max_upload_size_mb

    max_size_bytes = max_size_mb * 1024 * 1024

    # Create project directory
    project_dir = Path(settings.upload_dir) / str(project_id)
    project_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    ext = Path(file.filename).suffix if file.filename else '.csv'
    unique_name = f"{uuid.uuid4()}{ext}"
    file_path = project_dir / unique_name

    # Save file with size validation
    file_size = 0
    try:
        async with aiofiles.open(file_path, 'wb') as out_file:
            while content := await file.read(1024 * 1024):  # 1MB chunks
                file_size += len(content)

                # Check size limit during streaming
                if file_size > max_size_bytes:
                    # Clean up partial file
                    await out_file.close()
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    raise ValueError(
                        f"File exceeds maximum size of {max_size_mb}MB "
                        f"(attempted: {file_size / 1024 / 1024:.1f}MB)"
                    )

                await out_file.write(content)

        return str(file_path), file_size

    except Exception as e:
        # Clean up on any error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise


def delete_file(file_path: str) -> bool:
    """Delete a file from storage."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
    except Exception:
        pass
    return False


def get_storage_usage(project_id: int) -> int:
    """Get total storage used by a project in bytes."""
    project_dir = Path(settings.upload_dir) / str(project_id)
    if not project_dir.exists():
        return 0

    total = 0
    for file in project_dir.rglob('*'):
        if file.is_file():
            total += file.stat().st_size
    return total


def save_model_file(model_data: bytes, project_id: int, model_name: str) -> str:
    """Save a serialized model file."""
    model_dir = Path(settings.models_dir) / str(project_id)
    model_dir.mkdir(parents=True, exist_ok=True)

    file_path = model_dir / f"{model_name}.joblib"

    with open(file_path, 'wb') as f:
        f.write(model_data)

    return str(file_path)


def load_model_file(file_path: str) -> bytes:
    """Load a serialized model file."""
    with open(file_path, 'rb') as f:
        return f.read()
