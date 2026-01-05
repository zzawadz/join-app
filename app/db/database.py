from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import get_settings

settings = get_settings()

# Create engine with SQLite-specific settings
connect_args = {"check_same_thread": False} if "sqlite" in settings.database_url else {}
engine = create_engine(settings.database_url, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency for getting database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    from app.db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)


def create_test_user():
    """Create a test user for development."""
    from app.db.models import User, Organization, OrganizationMember, MemberRole
    from app.core.security import get_password_hash

    db = SessionLocal()
    try:
        # Check if test user exists
        existing = db.query(User).filter(User.email == "test@example.com").first()
        if existing:
            return

        # Create test user
        user = User(
            email="test@example.com",
            hashed_password=get_password_hash("test123"),
            full_name="Test User"
        )
        db.add(user)
        db.flush()

        # Create organization
        org = Organization(name="Test Organization")
        db.add(org)
        db.flush()

        # Add user as admin
        membership = OrganizationMember(
            user_id=user.id,
            organization_id=org.id,
            role=MemberRole.ADMIN
        )
        db.add(membership)

        db.commit()
        print("Test user created: test@example.com / test123")
    except Exception as e:
        db.rollback()
        print(f"Error creating test user: {e}")
    finally:
        db.close()
