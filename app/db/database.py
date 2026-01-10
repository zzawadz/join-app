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
    """
    Create a test user for development ONLY.

    Only runs in debug mode. Uses environment variables for credentials
    or generates a random password if not provided.

    Security: This function will NOT run in production mode.
    """
    import os
    import secrets
    import logging

    settings = get_settings()

    # Only run in debug mode
    if not settings.debug:
        return

    from app.db.models import User, Organization, OrganizationMember, MemberRole
    from app.core.security import get_password_hash

    logger = logging.getLogger(__name__)
    db = SessionLocal()

    try:
        # Get credentials from environment or generate
        test_email = os.getenv("TEST_USER_EMAIL", "test@example.com")
        test_password = os.getenv("TEST_USER_PASSWORD")

        # Check if test user exists
        existing = db.query(User).filter(User.email == test_email).first()
        if existing:
            return

        # Generate random password if not provided
        if not test_password:
            test_password = secrets.token_urlsafe(16)
            logger.info(f"Generated test user password: {test_password}")
            logger.info("Set TEST_USER_PASSWORD environment variable to use a custom password")

        # Create test user
        user = User(
            email=test_email,
            hashed_password=get_password_hash(test_password),
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

        # Log without password if using environment variable
        if os.getenv("TEST_USER_PASSWORD"):
            logger.info(f"Test user created: {test_email} (password from TEST_USER_PASSWORD)")
        else:
            logger.info(f"Test user created: {test_email} / {test_password}")

    except Exception as e:
        db.rollback()
        logger.error(f"Error creating test user: {e}")
    finally:
        db.close()
