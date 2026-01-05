from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean,
    Float, ForeignKey, JSON, Enum
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base
import enum


class MemberRole(str, enum.Enum):
    ADMIN = "admin"
    MEMBER = "member"


class LinkageType(str, enum.Enum):
    DEDUPLICATION = "deduplication"
    LINKAGE = "linkage"


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PairLabel(str, enum.Enum):
    MATCH = "match"
    NON_MATCH = "non_match"
    UNCERTAIN = "uncertain"


# ============ User & Organization Models ============

class Organization(Base):
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    members = relationship("OrganizationMember", back_populates="organization")
    projects = relationship("Project", back_populates="organization")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    memberships = relationship("OrganizationMember", back_populates="user")
    labeled_pairs = relationship("LabeledPair", back_populates="labeled_by_user")
    labeling_sessions = relationship("LabelingSession", back_populates="user")


class OrganizationMember(Base):
    __tablename__ = "organization_members"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    role = Column(Enum(MemberRole), default=MemberRole.MEMBER)
    joined_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="memberships")
    organization = relationship("Organization", back_populates="members")


# ============ Project & Dataset Models ============

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    created_by_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    linkage_type = Column(Enum(LinkageType), default=LinkageType.LINKAGE)

    # Configuration stored as JSON
    column_mappings = Column(JSON, default=dict)
    comparison_config = Column(JSON, default=dict)
    blocking_config = Column(JSON, default=dict)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    organization = relationship("Organization", back_populates="projects")
    created_by = relationship("User")
    datasets = relationship("Dataset", back_populates="project", cascade="all, delete-orphan")
    linkage_models = relationship("LinkageModel", back_populates="project", cascade="all, delete-orphan")
    linkage_jobs = relationship("LinkageJob", back_populates="project", cascade="all, delete-orphan")
    labeled_pairs = relationship("LabeledPair", back_populates="project", cascade="all, delete-orphan")
    labeling_sessions = relationship("LabelingSession", back_populates="project", cascade="all, delete-orphan")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    original_filename = Column(String(255))
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer)
    row_count = Column(Integer)

    column_names = Column(JSON)  # List of column names
    column_types = Column(JSON)  # Dict of column -> type
    sample_data = Column(JSON)   # First N rows for preview

    role = Column(String(50))    # "source", "target", or "dedupe"

    created_at = Column(DateTime, server_default=func.now())

    project = relationship("Project", back_populates="datasets")


# ============ Linkage Model & Job Models ============

class LinkageModel(Base):
    __tablename__ = "linkage_models"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)

    model_type = Column(String(50))  # "fellegi_sunter", "logistic_regression", "random_forest"
    model_path = Column(String(512))

    parameters = Column(JSON, default=dict)
    metrics = Column(JSON, default=dict)     # precision, recall, f1, etc.
    fs_parameters = Column(JSON, default=dict)  # m_probs, u_probs for FS model

    is_active = Column(Boolean, default=False)
    training_pairs_count = Column(Integer, default=0)

    trained_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())

    project = relationship("Project", back_populates="linkage_models")


class LinkageJob(Base):
    __tablename__ = "linkage_jobs"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("linkage_models.id"))

    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    job_type = Column(String(50))  # "full_linkage", "sample", "evaluation"

    # Progress tracking
    total_pairs = Column(Integer, default=0)
    processed_pairs = Column(Integer, default=0)
    matched_pairs = Column(Integer, default=0)

    # Results
    results_path = Column(String(512))
    error_message = Column(Text)

    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())

    project = relationship("Project", back_populates="linkage_jobs")
    model = relationship("LinkageModel")
    record_pairs = relationship("RecordPair", back_populates="job", cascade="all, delete-orphan")


class RecordPair(Base):
    """Stores candidate pairs and their scores from linkage jobs."""
    __tablename__ = "record_pairs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("linkage_jobs.id"), nullable=False)

    left_record_idx = Column(Integer, nullable=False)
    right_record_idx = Column(Integer, nullable=False)

    comparison_vector = Column(JSON)
    match_score = Column(Float)
    classification = Column(String(50))  # "match", "non_match", "review"

    job = relationship("LinkageJob", back_populates="record_pairs")


# ============ Active Learning Models ============

class LabelingSession(Base):
    __tablename__ = "labeling_sessions"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    status = Column(String(50), default="active")  # "active", "paused", "completed"
    strategy = Column(String(50), default="uncertainty")  # "uncertainty", "random"

    total_labeled = Column(Integer, default=0)
    target_labels = Column(Integer)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    project = relationship("Project", back_populates="labeling_sessions")
    user = relationship("User", back_populates="labeling_sessions")
    labeled_pairs = relationship("LabeledPair", back_populates="session", cascade="all, delete-orphan")


class LabeledPair(Base):
    """Training data from active learning - stored independently from source data."""
    __tablename__ = "labeled_pairs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("labeling_sessions.id"))
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    labeled_by_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Store full record data for training independence
    left_record = Column(JSON, nullable=False)
    right_record = Column(JSON, nullable=False)
    comparison_vector = Column(JSON)

    label = Column(Enum(PairLabel), nullable=False)
    confidence = Column(Float)
    labeling_time_ms = Column(Integer)

    created_at = Column(DateTime, server_default=func.now())

    session = relationship("LabelingSession", back_populates="labeled_pairs")
    project = relationship("Project", back_populates="labeled_pairs")
    labeled_by_user = relationship("User", back_populates="labeled_pairs")
