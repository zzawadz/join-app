from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime
import csv
import io
import pandas as pd

from app.db.database import get_db
from app.db.models import User, Project, LinkageJob, LinkageModel, RecordPair, JobStatus
from app.api.auth import get_current_active_user, get_current_user
from app.api.deps import get_project_or_404
from app.schemas.linkage import (
    LinkageJobCreate, LinkageJobResponse, LinkageJobDetail,
    RecordPairResponse, LinkageResultsExport
)
from app.core.security import decode_token

router = APIRouter()


def run_linkage_job_task(job_id: int, db_url: str):
    """Background task to run linkage job."""
    from app.db.database import SessionLocal
    from app.core.linkage.blocking import create_candidate_pairs
    from app.core.linkage.comparators import compare_records
    from app.core.linkage.fellegi_sunter import FellegiSunterModel
    from app.core.linkage.ml_classifier import load_classifier
    import pandas as pd

    db = SessionLocal()
    try:
        job = db.query(LinkageJob).filter(LinkageJob.id == job_id).first()
        if not job:
            return

        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        db.commit()

        project = job.project

        # Load datasets
        datasets = {d.role: d for d in project.datasets}

        if project.linkage_type.value == "deduplication":
            source_df = pd.read_csv(datasets["dedupe"].file_path)
            target_df = source_df
        else:
            source_df = pd.read_csv(datasets["source"].file_path)
            target_df = pd.read_csv(datasets["target"].file_path)

        # Create candidate pairs using blocking
        pairs = create_candidate_pairs(
            source_df, target_df,
            project.blocking_config or {},
            is_dedup=(project.linkage_type.value == "deduplication")
        )

        job.total_pairs = len(pairs)
        db.commit()

        # Compare pairs
        comparison_vectors = []
        for i, (left_idx, right_idx) in enumerate(pairs):
            left_record = source_df.iloc[left_idx].to_dict()
            right_record = target_df.iloc[right_idx].to_dict()

            vector = compare_records(
                left_record, right_record,
                project.column_mappings or {},
                project.comparison_config or {}
            )
            comparison_vectors.append({
                "left_idx": left_idx,
                "right_idx": right_idx,
                "vector": vector
            })

            if i % 1000 == 0:
                job.processed_pairs = i
                db.commit()

        # Classify using model
        model = None
        if job.model_id:
            model_record = db.query(LinkageModel).filter(
                LinkageModel.id == job.model_id
            ).first()
            if model_record:
                if model_record.model_type == "fellegi_sunter":
                    model = FellegiSunterModel()
                    model.load_parameters(model_record.fs_parameters)
                else:
                    model = load_classifier(model_record.model_path)

        # Get user-configured thresholds
        upper_threshold = job.upper_threshold or 0.8
        lower_threshold = job.lower_threshold or 0.3

        # Store results with three-category classification
        matched = 0
        review = 0
        for cv in comparison_vectors:
            if model:
                score = model.predict_proba([list(cv["vector"].values())])[0]
            else:
                # Simple average score without model
                avg_score = sum(cv["vector"].values()) / len(cv["vector"]) if cv["vector"] else 0
                score = avg_score

            # Three-category classification using user-defined thresholds
            if score >= upper_threshold:
                classification = "match"
                matched += 1
            elif score <= lower_threshold:
                classification = "non_match"
            else:
                classification = "review"
                review += 1

            pair = RecordPair(
                job_id=job_id,
                left_record_idx=cv["left_idx"],
                right_record_idx=cv["right_idx"],
                comparison_vector=cv["vector"],
                match_score=score,
                classification=classification
            )
            db.add(pair)

        job.processed_pairs = len(pairs)
        job.matched_pairs = matched
        job.review_pairs = review
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        db.commit()

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error_message = str(e)
        db.commit()
    finally:
        db.close()


@router.post("/projects/{project_id}/run", response_model=LinkageJobResponse)
def start_linkage_job(
    project_id: int,
    job_data: LinkageJobCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Start a new linkage job."""
    project = get_project_or_404(project_id, current_user, db)

    # Validate thresholds
    if job_data.lower_threshold >= job_data.upper_threshold:
        raise HTTPException(
            status_code=400,
            detail="Lower threshold must be less than upper threshold"
        )

    # Create job record
    job = LinkageJob(
        project_id=project_id,
        model_id=job_data.model_id,
        job_type=job_data.job_type or "full_linkage",
        upper_threshold=job_data.upper_threshold,
        lower_threshold=job_data.lower_threshold,
        status=JobStatus.PENDING
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    # Start background task
    from app.config import get_settings
    settings = get_settings()
    background_tasks.add_task(run_linkage_job_task, job.id, settings.database_url)

    return job


@router.get("/projects/{project_id}/jobs", response_model=List[LinkageJobResponse])
def list_project_jobs(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all linkage jobs for a project."""
    project = get_project_or_404(project_id, current_user, db)

    return db.query(LinkageJob).filter(
        LinkageJob.project_id == project_id
    ).order_by(LinkageJob.created_at.desc()).all()


@router.get("/jobs/{job_id}", response_model=LinkageJobDetail)
def get_job(
    job_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get job details and status."""
    job = db.query(LinkageJob).filter(LinkageJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Verify access
    get_project_or_404(job.project_id, current_user, db)

    return job


@router.get("/jobs/{job_id}/results", response_model=List[RecordPairResponse])
def get_job_results(
    job_id: int,
    classification: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get linkage results (matched pairs)."""
    job = db.query(LinkageJob).filter(LinkageJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    get_project_or_404(job.project_id, current_user, db)

    query = db.query(RecordPair).filter(RecordPair.job_id == job_id)

    if classification:
        query = query.filter(RecordPair.classification == classification)

    return query.order_by(RecordPair.match_score.desc()).offset(offset).limit(limit).all()


def get_user_from_token_param(token: str, db: Session) -> User:
    """Authenticate user from token query parameter (for file downloads)."""
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials"
    )
    payload = decode_token(token)
    if payload is None:
        raise credentials_exception

    user_id_str = payload.get("sub")
    if user_id_str is None:
        raise credentials_exception

    try:
        user_id = int(user_id_str)
    except (ValueError, TypeError):
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None or not user.is_active:
        raise credentials_exception

    return user


@router.get("/jobs/{job_id}/results/full")
def get_job_results_with_records(
    job_id: int,
    classification: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get linkage results with full record values."""
    job = db.query(LinkageJob).filter(LinkageJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    get_project_or_404(job.project_id, current_user, db)

    project = job.project

    # Load datasets
    datasets = {d.role: d for d in project.datasets}

    if project.linkage_type.value == "deduplication":
        source_df = pd.read_csv(datasets["dedupe"].file_path)
        target_df = source_df
    else:
        source_df = pd.read_csv(datasets["source"].file_path)
        target_df = pd.read_csv(datasets["target"].file_path)

    query = db.query(RecordPair).filter(RecordPair.job_id == job_id)

    if classification:
        query = query.filter(RecordPair.classification == classification)

    pairs = query.order_by(RecordPair.match_score.desc()).offset(offset).limit(limit).all()

    results = []
    for pair in pairs:
        left_record = source_df.iloc[pair.left_record_idx].to_dict()
        right_record = target_df.iloc[pair.right_record_idx].to_dict()

        # Convert numpy types to Python types
        left_record = {k: (v.item() if hasattr(v, 'item') else v) for k, v in left_record.items()}
        right_record = {k: (v.item() if hasattr(v, 'item') else v) for k, v in right_record.items()}

        results.append({
            "id": pair.id,
            "left_record_idx": pair.left_record_idx,
            "right_record_idx": pair.right_record_idx,
            "left_record": left_record,
            "right_record": right_record,
            "comparison_vector": pair.comparison_vector,
            "match_score": pair.match_score,
            "classification": pair.classification
        })

    return results


@router.get("/jobs/{job_id}/export")
def export_results(
    job_id: int,
    format: str = "csv",
    token: Optional[str] = Query(None, description="JWT token for authentication (alternative to Authorization header)"),
    current_user: Optional[User] = Depends(lambda: None),
    db: Session = Depends(get_db)
):
    """Export linkage results as CSV with full record values.

    Supports authentication via either:
    - Authorization: Bearer <token> header
    - token query parameter (for direct download links)
    """
    # Handle token-based auth for downloads
    from fastapi import Request
    from starlette.requests import Request as StarletteRequest

    # Try to get user from token param first if provided
    if token:
        current_user = get_user_from_token_param(token, db)
    else:
        # Fall back to header-based auth - we need to re-check manually
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide token parameter for downloads."
        )

    job = db.query(LinkageJob).filter(LinkageJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    get_project_or_404(job.project_id, current_user, db)

    project = job.project
    pairs = db.query(RecordPair).filter(
        RecordPair.job_id == job_id,
        RecordPair.classification == "match"
    ).all()

    # Load datasets
    datasets = {d.role: d for d in project.datasets}

    if project.linkage_type.value == "deduplication":
        source_df = pd.read_csv(datasets["dedupe"].file_path)
        target_df = source_df
        source_prefix = "left_"
        target_prefix = "right_"
    else:
        source_df = pd.read_csv(datasets["source"].file_path)
        target_df = pd.read_csv(datasets["target"].file_path)
        source_prefix = "source_"
        target_prefix = "target_"

    # Create CSV with full record values
    output = io.StringIO()
    writer = csv.writer(output)

    # Build header
    source_cols = [f"{source_prefix}{col}" for col in source_df.columns]
    target_cols = [f"{target_prefix}{col}" for col in target_df.columns]
    header = source_cols + target_cols + ["match_score"]
    writer.writerow(header)

    for pair in pairs:
        left_record = source_df.iloc[pair.left_record_idx].tolist()
        right_record = target_df.iloc[pair.right_record_idx].tolist()
        row = left_record + right_record + [pair.match_score]
        writer.writerow(row)

    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=linkage_results_{job_id}.csv"}
    )


@router.delete("/jobs/{job_id}")
def delete_job(
    job_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a linkage job and its results."""
    job = db.query(LinkageJob).filter(LinkageJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    get_project_or_404(job.project_id, current_user, db)

    db.delete(job)
    db.commit()

    return {"message": "Job deleted"}
