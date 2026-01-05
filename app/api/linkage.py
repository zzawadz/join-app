from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import csv
import io

from app.db.database import get_db
from app.db.models import User, Project, LinkageJob, LinkageModel, RecordPair, JobStatus
from app.api.auth import get_current_active_user
from app.api.deps import get_project_or_404
from app.schemas.linkage import (
    LinkageJobCreate, LinkageJobResponse, LinkageJobDetail,
    RecordPairResponse, LinkageResultsExport
)

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

        # Store results
        matched = 0
        for cv in comparison_vectors:
            if model:
                score = model.predict_proba([list(cv["vector"].values())])[0]
                classification = "match" if score > 0.5 else "non_match"
            else:
                # Simple threshold without model
                avg_score = sum(cv["vector"].values()) / len(cv["vector"]) if cv["vector"] else 0
                score = avg_score
                classification = "match" if avg_score > 0.8 else "non_match"

            pair = RecordPair(
                job_id=job_id,
                left_record_idx=cv["left_idx"],
                right_record_idx=cv["right_idx"],
                comparison_vector=cv["vector"],
                match_score=score,
                classification=classification
            )
            db.add(pair)

            if classification == "match":
                matched += 1

        job.processed_pairs = len(pairs)
        job.matched_pairs = matched
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

    # Create job record
    job = LinkageJob(
        project_id=project_id,
        model_id=job_data.model_id,
        job_type=job_data.job_type or "full_linkage",
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


@router.get("/jobs/{job_id}/export")
def export_results(
    job_id: int,
    format: str = "csv",
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Export linkage results as CSV."""
    job = db.query(LinkageJob).filter(LinkageJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    get_project_or_404(job.project_id, current_user, db)

    pairs = db.query(RecordPair).filter(RecordPair.job_id == job_id).all()

    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["left_idx", "right_idx", "match_score", "classification"])

    for pair in pairs:
        writer.writerow([
            pair.left_record_idx,
            pair.right_record_idx,
            pair.match_score,
            pair.classification
        ])

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
