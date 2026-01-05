"""
Pytest configuration and fixtures for testing.
"""
import pytest
import tempfile
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.db.database import Base
from app.main import app
from app.db.database import get_db
from app.core.security import get_password_hash


@pytest.fixture(scope="function")
def test_db():
    """Create a fresh test database for each test."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create all tables
    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
        os.unlink(db_path)


@pytest.fixture(scope="function")
def client(test_db):
    """Create a test client with the test database."""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def test_user(test_db):
    """Create a test user."""
    from app.db.models import User, Organization, OrganizationMember, MemberRole

    user = User(
        email="testuser@example.com",
        hashed_password=get_password_hash("testpass123"),
        full_name="Test User"
    )
    test_db.add(user)
    test_db.flush()

    org = Organization(name="Test Org")
    test_db.add(org)
    test_db.flush()

    membership = OrganizationMember(
        user_id=user.id,
        organization_id=org.id,
        role=MemberRole.ADMIN
    )
    test_db.add(membership)
    test_db.commit()
    test_db.refresh(user)

    return user


@pytest.fixture(scope="function")
def auth_headers(client, test_user):
    """Get authentication headers for the test user."""
    response = client.post(
        "/api/auth/login",
        data={"username": test_user.email, "password": "testpass123"}
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
def test_project(test_db, test_user):
    """Create a test project."""
    from app.db.models import Project, LinkageType

    project = Project(
        name="Test Linkage Project",
        organization_id=test_user.memberships[0].organization_id,
        created_by_id=test_user.id,
        linkage_type=LinkageType.LINKAGE,
        column_mappings={"first_name": "fname", "last_name": "lname"},
        comparison_config={
            "first_name": {"method": "jaro_winkler"},
            "last_name": {"method": "jaro_winkler"}
        }
    )
    test_db.add(project)
    test_db.commit()
    test_db.refresh(project)

    return project


@pytest.fixture(scope="function")
def test_datasets(test_db, test_project):
    """Create test datasets with CSV files."""
    import tempfile
    import csv
    from app.db.models import Dataset

    # Create source CSV
    source_data = [
        {"id": 1, "first_name": "John", "last_name": "Smith", "email": "john@example.com"},
        {"id": 2, "first_name": "Jane", "last_name": "Doe", "email": "jane@example.com"},
        {"id": 3, "first_name": "Bob", "last_name": "Johnson", "email": "bob@example.com"},
        {"id": 4, "first_name": "Alice", "last_name": "Williams", "email": "alice@example.com"},
        {"id": 5, "first_name": "Charlie", "last_name": "Brown", "email": "charlie@example.com"},
    ]

    # Create target CSV
    target_data = [
        {"id": 101, "fname": "John", "lname": "Smith", "contact": "john.smith@mail.com"},
        {"id": 102, "fname": "Janet", "lname": "Doe", "contact": "janet@mail.com"},
        {"id": 103, "fname": "Robert", "lname": "Johnson", "contact": "robert@mail.com"},
        {"id": 104, "fname": "Alicia", "lname": "Williams", "contact": "alicia@mail.com"},
        {"id": 105, "fname": "Charles", "lname": "Brown", "contact": "charles@mail.com"},
    ]

    # Write source CSV
    source_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    writer = csv.DictWriter(source_file, fieldnames=["id", "first_name", "last_name", "email"])
    writer.writeheader()
    writer.writerows(source_data)
    source_file.close()

    # Write target CSV
    target_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    writer = csv.DictWriter(target_file, fieldnames=["id", "fname", "lname", "contact"])
    writer.writeheader()
    writer.writerows(target_data)
    target_file.close()

    # Create dataset records
    source_dataset = Dataset(
        project_id=test_project.id,
        name="Source Dataset",
        file_path=source_file.name,
        role="source",
        row_count=len(source_data),
        column_names=["id", "first_name", "last_name", "email"]
    )
    test_db.add(source_dataset)

    target_dataset = Dataset(
        project_id=test_project.id,
        name="Target Dataset",
        file_path=target_file.name,
        role="target",
        row_count=len(target_data),
        column_names=["id", "fname", "lname", "contact"]
    )
    test_db.add(target_dataset)
    test_db.commit()

    yield {"source": source_dataset, "target": target_dataset}

    # Cleanup files
    os.unlink(source_file.name)
    os.unlink(target_file.name)


@pytest.fixture(scope="function")
def labeled_pairs(test_db, test_project):
    """Create labeled pairs for training."""
    from app.db.models import LabeledPair, LabelingSession, PairLabel

    # Create a labeling session first
    session = LabelingSession(
        project_id=test_project.id,
        user_id=1,
        strategy="random",
        target_labels=20,
        total_labeled=0
    )
    test_db.add(session)
    test_db.flush()

    # Create labeled pairs with comparison vectors
    pairs = [
        # Matches (high similarity)
        {"left": {"_idx": 0, "first_name": "John", "last_name": "Smith"},
         "right": {"_idx": 0, "fname": "John", "lname": "Smith"},
         "vector": {"first_name": 1.0, "last_name": 1.0}, "label": PairLabel.MATCH},
        {"left": {"_idx": 4, "first_name": "Charlie", "last_name": "Brown"},
         "right": {"_idx": 4, "fname": "Charles", "lname": "Brown"},
         "vector": {"first_name": 0.85, "last_name": 1.0}, "label": PairLabel.MATCH},
        {"left": {"_idx": 2, "first_name": "Bob", "last_name": "Johnson"},
         "right": {"_idx": 2, "fname": "Robert", "lname": "Johnson"},
         "vector": {"first_name": 0.5, "last_name": 1.0}, "label": PairLabel.MATCH},
        {"left": {"_idx": 3, "first_name": "Alice", "last_name": "Williams"},
         "right": {"_idx": 3, "fname": "Alicia", "lname": "Williams"},
         "vector": {"first_name": 0.87, "last_name": 1.0}, "label": PairLabel.MATCH},
        {"left": {"_idx": 1, "first_name": "Jane", "last_name": "Doe"},
         "right": {"_idx": 1, "fname": "Janet", "lname": "Doe"},
         "vector": {"first_name": 0.82, "last_name": 1.0}, "label": PairLabel.MATCH},

        # Non-matches (low similarity)
        {"left": {"_idx": 0, "first_name": "John", "last_name": "Smith"},
         "right": {"_idx": 1, "fname": "Janet", "lname": "Doe"},
         "vector": {"first_name": 0.3, "last_name": 0.2}, "label": PairLabel.NON_MATCH},
        {"left": {"_idx": 1, "first_name": "Jane", "last_name": "Doe"},
         "right": {"_idx": 2, "fname": "Robert", "lname": "Johnson"},
         "vector": {"first_name": 0.25, "last_name": 0.35}, "label": PairLabel.NON_MATCH},
        {"left": {"_idx": 2, "first_name": "Bob", "last_name": "Johnson"},
         "right": {"_idx": 3, "fname": "Alicia", "lname": "Williams"},
         "vector": {"first_name": 0.2, "last_name": 0.4}, "label": PairLabel.NON_MATCH},
        {"left": {"_idx": 3, "first_name": "Alice", "last_name": "Williams"},
         "right": {"_idx": 4, "fname": "Charles", "lname": "Brown"},
         "vector": {"first_name": 0.3, "last_name": 0.25}, "label": PairLabel.NON_MATCH},
        {"left": {"_idx": 4, "first_name": "Charlie", "last_name": "Brown"},
         "right": {"_idx": 0, "fname": "John", "lname": "Smith"},
         "vector": {"first_name": 0.35, "last_name": 0.3}, "label": PairLabel.NON_MATCH},
        # Additional pairs to meet 10+ threshold
        {"left": {"_idx": 0, "first_name": "John", "last_name": "Smith"},
         "right": {"_idx": 3, "fname": "Alicia", "lname": "Williams"},
         "vector": {"first_name": 0.2, "last_name": 0.3}, "label": PairLabel.NON_MATCH},
        {"left": {"_idx": 1, "first_name": "Jane", "last_name": "Doe"},
         "right": {"_idx": 4, "fname": "Charles", "lname": "Brown"},
         "vector": {"first_name": 0.25, "last_name": 0.2}, "label": PairLabel.NON_MATCH},
    ]

    labeled_pair_objects = []
    for p in pairs:
        lp = LabeledPair(
            session_id=session.id,
            project_id=test_project.id,
            labeled_by_id=1,
            left_record=p["left"],
            right_record=p["right"],
            comparison_vector=p["vector"],
            label=p["label"]
        )
        test_db.add(lp)
        labeled_pair_objects.append(lp)

    session.total_labeled = len(pairs)
    test_db.commit()

    return labeled_pair_objects
