from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime
import random

from app.db.database import get_db
from app.db.models import (
    User, Project, Dataset, LabelingSession, LabeledPair,
    LinkageModel, PairLabel, LinkageJob, RecordPair, JobStatus
)
from app.api.auth import get_current_active_user
from app.api.deps import get_project_or_404
from app.schemas.labeling import (
    LabelingSessionCreate, LabelingSessionResponse,
    LabelingPairResponse, LabelSubmission, LabelingProgress,
    DetailedProgressResponse, PairExplanation, LabelingPairWithExplanation,
    ComparisonConfigUpdate
)
from app.core.linkage.active_learning import (
    select_informative_pair, select_informative_pair_with_explanation,
    count_candidate_pairs, select_from_linkage_results
)
from app.core.linkage.comparators import compare_records

router = APIRouter()


@router.post("/projects/{project_id}/start", response_model=LabelingSessionResponse)
def start_labeling_session(
    project_id: int,
    session_data: LabelingSessionCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Start a new labeling session for active learning."""
    project = get_project_or_404(project_id, current_user, db)

    # Check for existing active session
    existing = db.query(LabelingSession).filter(
        LabelingSession.project_id == project_id,
        LabelingSession.user_id == current_user.id,
        LabelingSession.status == "active"
    ).first()

    if existing:
        return existing

    session = LabelingSession(
        project_id=project_id,
        user_id=current_user.id,
        strategy=session_data.strategy or "uncertainty",
        target_labels=session_data.target_labels or 100,
        started_at=datetime.utcnow(),
        total_labeling_time_ms=0
    )

    db.add(session)
    db.commit()
    db.refresh(session)

    return session


@router.get("/sessions/{session_id}", response_model=LabelingSessionResponse)
def get_session(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get labeling session details."""
    session = db.query(LabelingSession).filter(
        LabelingSession.id == session_id,
        LabelingSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session


@router.get("/sessions/{session_id}/next", response_model=LabelingPairResponse)
def get_next_pair(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get the next pair to label using active learning strategy."""
    session = db.query(LabelingSession).filter(
        LabelingSession.id == session_id,
        LabelingSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != "active":
        raise HTTPException(status_code=400, detail="Session is not active")

    project = session.project

    # Load datasets
    import pandas as pd
    datasets = {d.role: d for d in project.datasets}

    if project.linkage_type.value == "deduplication":
        df = pd.read_csv(datasets["dedupe"].file_path)
        source_df = target_df = df
    else:
        source_df = pd.read_csv(datasets["source"].file_path)
        target_df = pd.read_csv(datasets["target"].file_path)

    # Get already labeled indices to exclude
    labeled = db.query(LabeledPair).filter(
        LabeledPair.project_id == project.id
    ).all()
    labeled_pairs = set((lp.left_record["_idx"], lp.right_record["_idx"])
                       for lp in labeled if "_idx" in lp.left_record)

    # Get active model for uncertainty sampling
    active_model = db.query(LinkageModel).filter(
        LinkageModel.project_id == project.id,
        LinkageModel.is_active == True
    ).first()

    # Select next pair
    pair = select_informative_pair(
        source_df, target_df,
        project.column_mappings or {},
        project.comparison_config or {},
        project.blocking_config or {},
        labeled_pairs,
        active_model,
        strategy=session.strategy,
        is_dedup=(project.linkage_type.value == "deduplication")
    )

    if pair is None:
        raise HTTPException(status_code=204, detail="No more pairs to label")

    left_idx, right_idx, comparison_vector = pair

    left_record = source_df.iloc[left_idx].to_dict()
    left_record["_idx"] = int(left_idx)

    right_record = target_df.iloc[right_idx].to_dict()
    right_record["_idx"] = int(right_idx)

    return {
        "session_id": session_id,
        "left_record": left_record,
        "right_record": right_record,
        "comparison_vector": comparison_vector,
        "column_mappings": project.column_mappings or {}
    }


@router.post("/sessions/{session_id}/label")
def submit_label(
    session_id: int,
    submission: LabelSubmission,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Submit a label for a pair."""
    session = db.query(LabelingSession).filter(
        LabelingSession.id == session_id,
        LabelingSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    labeled_pair = LabeledPair(
        session_id=session_id,
        project_id=session.project_id,
        labeled_by_id=current_user.id,
        left_record=submission.left_record,
        right_record=submission.right_record,
        comparison_vector=submission.comparison_vector,
        label=submission.label,
        confidence=submission.confidence,
        labeling_time_ms=submission.labeling_time_ms
    )

    db.add(labeled_pair)

    session.total_labeled += 1
    # Track cumulative labeling time
    if submission.labeling_time_ms:
        session.total_labeling_time_ms = (session.total_labeling_time_ms or 0) + submission.labeling_time_ms
    if session.target_labels and session.total_labeled >= session.target_labels:
        session.status = "completed"

    db.commit()

    return {
        "message": "Label saved",
        "total_labeled": session.total_labeled,
        "target_labels": session.target_labels
    }


@router.post("/sessions/{session_id}/skip")
def skip_pair(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Skip the current pair without labeling."""
    session = db.query(LabelingSession).filter(
        LabelingSession.id == session_id,
        LabelingSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"message": "Pair skipped"}


@router.get("/sessions/{session_id}/progress", response_model=LabelingProgress)
def get_progress(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get labeling progress for the session."""
    session = db.query(LabelingSession).filter(
        LabelingSession.id == session_id,
        LabelingSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get label distribution
    labels = db.query(LabeledPair.label, func.count(LabeledPair.id)).filter(
        LabeledPair.session_id == session_id
    ).group_by(LabeledPair.label).all()

    label_counts = {label.value: count for label, count in labels}

    return {
        "session_id": session_id,
        "total_labeled": session.total_labeled,
        "target_labels": session.target_labels,
        "matches": label_counts.get("match", 0),
        "non_matches": label_counts.get("non_match", 0),
        "uncertain": label_counts.get("uncertain", 0),
        "status": session.status
    }


@router.post("/sessions/{session_id}/retrain")
def retrain_model(
    session_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Trigger model retraining with current labeled data."""
    session = db.query(LabelingSession).filter(
        LabelingSession.id == session_id,
        LabelingSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get or create active model
    active_model = db.query(LinkageModel).filter(
        LinkageModel.project_id == session.project_id,
        LinkageModel.is_active == True
    ).first()

    if not active_model:
        # Create new model
        active_model = LinkageModel(
            project_id=session.project_id,
            name="Active Learning Model",
            model_type="logistic_regression",
            is_active=True
        )
        db.add(active_model)
        db.commit()
        db.refresh(active_model)

    # Trigger training
    from app.api.models import train_model_task
    from app.config import get_settings
    settings = get_settings()
    background_tasks.add_task(train_model_task, active_model.id, settings.database_url)

    return {"message": "Model retraining started", "model_id": active_model.id}


@router.post("/sessions/{session_id}/complete")
def complete_session(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Mark labeling session as completed."""
    session = db.query(LabelingSession).filter(
        LabelingSession.id == session_id,
        LabelingSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.status = "completed"
    db.commit()

    return {"message": "Session completed", "total_labeled": session.total_labeled}


@router.get("/projects/{project_id}/labeled-pairs")
def get_labeled_pairs(
    project_id: int,
    limit: int = 100,
    offset: int = 0,
    label: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all labeled pairs for a project, optionally filtered by label."""
    project = get_project_or_404(project_id, current_user, db)

    query = db.query(LabeledPair).filter(
        LabeledPair.project_id == project_id
    )

    # Filter by label if provided
    if label:
        try:
            label_enum = PairLabel(label)
            query = query.filter(LabeledPair.label == label_enum)
        except ValueError:
            pass  # Invalid label value, ignore filter

    pairs = query.order_by(LabeledPair.created_at.desc()).offset(offset).limit(limit).all()

    # Get total count with same filter
    count_query = db.query(LabeledPair).filter(
        LabeledPair.project_id == project_id
    )
    if label:
        try:
            label_enum = PairLabel(label)
            count_query = count_query.filter(LabeledPair.label == label_enum)
        except ValueError:
            pass

    total = count_query.count()

    return {
        "pairs": pairs,
        "total": total,
        "limit": limit,
        "offset": offset
    }


@router.get("/sessions/{session_id}/detailed-progress", response_model=DetailedProgressResponse)
def get_detailed_progress(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed labeling progress including timing and candidate stats."""
    session = db.query(LabelingSession).filter(
        LabelingSession.id == session_id,
        LabelingSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    project = session.project

    # Load datasets and count candidates
    import pandas as pd
    datasets = {d.role: d for d in project.datasets}

    try:
        if project.linkage_type.value == "deduplication":
            df = pd.read_csv(datasets["dedupe"].file_path)
            source_df = target_df = df
        else:
            source_df = pd.read_csv(datasets["source"].file_path)
            target_df = pd.read_csv(datasets["target"].file_path)

        # Get labeled pairs for counting
        labeled = db.query(LabeledPair).filter(
            LabeledPair.project_id == project.id
        ).all()
        labeled_pairs = set((lp.left_record["_idx"], lp.right_record["_idx"])
                           for lp in labeled if "_idx" in lp.left_record)

        total_candidates, remaining_candidates = count_candidate_pairs(
            source_df, target_df,
            project.blocking_config or {},
            labeled_pairs,
            is_dedup=(project.linkage_type.value == "deduplication")
        )
    except Exception:
        total_candidates = 0
        remaining_candidates = 0

    # Get label distribution
    labels = db.query(LabeledPair.label, func.count(LabeledPair.id)).filter(
        LabeledPair.session_id == session_id
    ).group_by(LabeledPair.label).all()

    label_counts = {label.value: count for label, count in labels}

    # Calculate average time per pair
    total_time_ms = session.total_labeling_time_ms or 0
    avg_time_per_pair_ms = total_time_ms / session.total_labeled if session.total_labeled > 0 else 0.0

    return {
        "session_id": session_id,
        "total_labeled": session.total_labeled,
        "target_labels": session.target_labels,
        "total_candidates": total_candidates,
        "estimated_remaining": remaining_candidates,
        "total_time_ms": total_time_ms,
        "avg_time_per_pair_ms": avg_time_per_pair_ms,
        "matches": label_counts.get("match", 0),
        "non_matches": label_counts.get("non_match", 0),
        "uncertain": label_counts.get("uncertain", 0),
        "status": session.status,
        "started_at": session.started_at
    }


@router.get("/sessions/{session_id}/next-with-explanation", response_model=LabelingPairWithExplanation)
def get_next_pair_with_explanation(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get the next pair to label with explanation of why it was selected."""
    session = db.query(LabelingSession).filter(
        LabelingSession.id == session_id,
        LabelingSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != "active":
        raise HTTPException(status_code=400, detail="Session is not active")

    project = session.project

    # Load datasets
    import pandas as pd
    datasets = {d.role: d for d in project.datasets}

    if project.linkage_type.value == "deduplication":
        df = pd.read_csv(datasets["dedupe"].file_path)
        source_df = target_df = df
    else:
        source_df = pd.read_csv(datasets["source"].file_path)
        target_df = pd.read_csv(datasets["target"].file_path)

    # Get already labeled indices to exclude
    labeled = db.query(LabeledPair).filter(
        LabeledPair.project_id == project.id
    ).all()
    labeled_pairs = set((lp.left_record["_idx"], lp.right_record["_idx"])
                       for lp in labeled if "_idx" in lp.left_record)

    # Get active model for uncertainty sampling
    active_model = db.query(LinkageModel).filter(
        LinkageModel.project_id == project.id,
        LinkageModel.is_active == True
    ).first()

    # For linkage_priority strategy, first check linkage results
    result = None
    if session.strategy == "linkage_priority":
        result = select_from_linkage_results(
            db, project.id, source_df, target_df,
            project.column_mappings or {},
            project.comparison_config or {},
            labeled_pairs
        )

    # Fall back to regular selection if no linkage results or other strategy
    if result is None:
        result = select_informative_pair_with_explanation(
            source_df, target_df,
            project.column_mappings or {},
            project.comparison_config or {},
            project.blocking_config or {},
            labeled_pairs,
            active_model,
            strategy=session.strategy if session.strategy != "linkage_priority" else "uncertainty",
            is_dedup=(project.linkage_type.value == "deduplication")
        )

    if result is None:
        raise HTTPException(status_code=204, detail="No more pairs to label")

    left_record = source_df.iloc[result.left_idx].to_dict()
    left_record["_idx"] = int(result.left_idx)

    right_record = target_df.iloc[result.right_idx].to_dict()
    right_record["_idx"] = int(result.right_idx)

    explanation = PairExplanation(
        selection_reason=result.selection_reason,
        uncertainty_score=result.uncertainty_score,
        model_probability=result.model_probability,
        candidates_evaluated=result.candidates_evaluated
    )

    return {
        "session_id": session_id,
        "left_record": left_record,
        "right_record": right_record,
        "comparison_vector": result.comparison_vector,
        "column_mappings": project.column_mappings or {},
        "explanation": explanation
    }


@router.put("/projects/{project_id}/comparison-config")
def update_comparison_config(
    project_id: int,
    config_update: ComparisonConfigUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update column mappings and comparison configuration (immediate save)."""
    project = get_project_or_404(project_id, current_user, db)

    project.column_mappings = config_update.column_mappings
    project.comparison_config = config_update.comparison_config
    db.commit()

    return {
        "message": "Configuration updated",
        "column_mappings": project.column_mappings,
        "comparison_config": project.comparison_config
    }


@router.get("/projects/{project_id}/comparison-config")
def get_comparison_config(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get current column mappings and comparison configuration."""
    project = get_project_or_404(project_id, current_user, db)

    # Get available columns from datasets
    datasets = {d.role: d for d in project.datasets}

    source_columns = []
    target_columns = []

    if project.linkage_type.value == "deduplication":
        if "dedupe" in datasets:
            source_columns = target_columns = datasets["dedupe"].column_names or []
    else:
        if "source" in datasets:
            source_columns = datasets["source"].column_names or []
        if "target" in datasets:
            target_columns = datasets["target"].column_names or []

    return {
        "column_mappings": project.column_mappings or {},
        "comparison_config": project.comparison_config or {},
        "source_columns": source_columns,
        "target_columns": target_columns
    }


@router.get("/projects/{project_id}/pending-linkage-matches")
def get_pending_linkage_matches(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get pairs from completed linkage jobs that are classified as matches
    but have not yet been manually labeled for confirmation.
    """
    project = get_project_or_404(project_id, current_user, db)

    # Get the most recent completed linkage job
    latest_job = db.query(LinkageJob).filter(
        LinkageJob.project_id == project_id,
        LinkageJob.status == JobStatus.COMPLETED
    ).order_by(LinkageJob.completed_at.desc()).first()

    if not latest_job:
        return {"pairs": [], "job_id": None, "total": 0}

    # Get already labeled pair indices
    labeled_pairs = db.query(LabeledPair).filter(
        LabeledPair.project_id == project_id
    ).all()
    labeled_indices = set(
        (lp.left_record.get("_idx"), lp.right_record.get("_idx"))
        for lp in labeled_pairs
        if lp.left_record and lp.right_record
    )

    # Get matched pairs from linkage results that haven't been labeled yet
    match_pairs = db.query(RecordPair).filter(
        RecordPair.job_id == latest_job.id,
        RecordPair.classification == "match"
    ).order_by(RecordPair.match_score.desc()).limit(100).all()

    # Filter out already labeled pairs
    pending_pairs = []
    for pair in match_pairs:
        if (pair.left_record_idx, pair.right_record_idx) not in labeled_indices:
            pending_pairs.append({
                "id": pair.id,
                "left_idx": pair.left_record_idx,
                "right_idx": pair.right_record_idx,
                "match_score": pair.match_score,
                "comparison_vector": pair.comparison_vector
            })

    return {
        "pairs": pending_pairs,
        "job_id": latest_job.id,
        "total": len(pending_pairs)
    }


@router.get("/sessions/{session_id}/next-linkage-priority", response_model=LabelingPairWithExplanation)
def get_next_pair_linkage_priority(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get the next pair prioritizing linkage results marked as matches.
    These pairs should be confirmed/rejected by the user first.
    """
    session = db.query(LabelingSession).filter(
        LabelingSession.id == session_id,
        LabelingSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != "active":
        raise HTTPException(status_code=400, detail="Session is not active")

    project = session.project

    # Load datasets
    import pandas as pd
    datasets = {d.role: d for d in project.datasets}

    if project.linkage_type.value == "deduplication":
        df = pd.read_csv(datasets["dedupe"].file_path)
        source_df = target_df = df
    else:
        source_df = pd.read_csv(datasets["source"].file_path)
        target_df = pd.read_csv(datasets["target"].file_path)

    # Get already labeled indices to exclude
    labeled = db.query(LabeledPair).filter(
        LabeledPair.project_id == project.id
    ).all()
    labeled_pairs = set((lp.left_record["_idx"], lp.right_record["_idx"])
                       for lp in labeled if "_idx" in lp.left_record)

    # First try to get from linkage results
    result = select_from_linkage_results(
        db, project.id, source_df, target_df,
        project.column_mappings or {},
        project.comparison_config or {},
        labeled_pairs
    )

    if result is None:
        # Fall back to regular selection
        active_model = db.query(LinkageModel).filter(
            LinkageModel.project_id == project.id,
            LinkageModel.is_active == True
        ).first()

        result = select_informative_pair_with_explanation(
            source_df, target_df,
            project.column_mappings or {},
            project.comparison_config or {},
            project.blocking_config or {},
            labeled_pairs,
            active_model,
            strategy=session.strategy if session.strategy != "linkage_priority" else "uncertainty",
            is_dedup=(project.linkage_type.value == "deduplication")
        )

    if result is None:
        raise HTTPException(status_code=204, detail="No more pairs to label")

    left_record = source_df.iloc[result.left_idx].to_dict()
    left_record["_idx"] = int(result.left_idx)

    right_record = target_df.iloc[result.right_idx].to_dict()
    right_record["_idx"] = int(result.right_idx)

    explanation = PairExplanation(
        selection_reason=result.selection_reason,
        uncertainty_score=result.uncertainty_score,
        model_probability=result.model_probability,
        candidates_evaluated=result.candidates_evaluated
    )

    return {
        "session_id": session_id,
        "left_record": left_record,
        "right_record": right_record,
        "comparison_vector": result.comparison_vector,
        "column_mappings": project.column_mappings or {},
        "explanation": explanation
    }
