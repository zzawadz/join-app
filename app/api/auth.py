from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, Tuple
import re

from app.db.database import get_db
from app.db.models import User, Organization, OrganizationMember, MemberRole
from app.schemas.user import UserCreate, UserResponse, Token, UserLogin
from app.core.security import (
    verify_password, get_password_hash, create_access_token, decode_token
)
from app.core.rate_limit import limiter

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def validate_password_strength(password: str, user_inputs: list = None) -> Tuple[bool, str]:
    """
    Validate password strength using zxcvbn library.

    Args:
        password: Password to validate
        user_inputs: List of user-specific words (email, name) to check against

    Returns:
        Tuple of (is_valid, error_message)
    """
    from zxcvbn import zxcvbn

    # Length check
    if len(password) < 12:
        return False, "Password must be at least 12 characters"
    if len(password) > 128:
        return False, "Password too long (max 128 characters)"

    # Use zxcvbn for strength analysis
    result = zxcvbn(password, user_inputs=user_inputs or [])

    # Require score of 3 or higher (out of 4)
    # 0 = too guessable
    # 1 = very guessable
    # 2 = somewhat guessable
    # 3 = safely unguessable
    # 4 = very unguessable
    if result['score'] < 3:
        # Get feedback message
        feedback_msg = result['feedback'].get('warning') or \
                      (result['feedback'].get('suggestions')[0] if result['feedback'].get('suggestions') else None) or \
                      "Password too weak"
        return False, f"Password too weak: {feedback_msg}"

    return True, ""


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
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
    if user is None:
        raise credentials_exception

    return user


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Ensure user is active."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@router.post("/register", response_model=UserResponse)
@limiter.limit("3/hour")
def register(
    request: Request,
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Register a new user and create their personal organization.

    Rate limit: 3 registrations per hour per IP address.
    """
    # Check if email already exists
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Validate password strength with user context
    user_inputs = [
        user_data.email,
        user_data.email.split('@')[0],  # Username part of email
        user_data.full_name
    ]
    is_valid, error_msg = validate_password_strength(user_data.password, user_inputs)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # Create user
    user = User(
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.full_name
    )
    db.add(user)
    db.flush()

    # Create personal organization
    org = Organization(name=f"{user_data.full_name or user_data.email}'s Workspace")
    db.add(org)
    db.flush()

    # Add user as admin of their organization
    membership = OrganizationMember(
        user_id=user.id,
        organization_id=org.id,
        role=MemberRole.ADMIN
    )
    db.add(membership)

    db.commit()
    db.refresh(user)

    return user


@router.post("/login", response_model=Token)
@limiter.limit("5/minute")
def login(
    request: Request,
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login and receive JWT access token.

    Rate limit: 5 login attempts per minute per IP address.
    """
    user = db.query(User).filter(User.email == form_data.username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )

    access_token = create_access_token(data={"sub": str(user.id)})

    # Set httpOnly cookie (prevents JavaScript access)
    from app.config import get_settings
    settings = get_settings()

    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,
        secure=not settings.debug,  # True in production (HTTPS only)
        samesite="strict" if not settings.debug else "lax",  # Strict CSRF protection in production
        max_age=settings.access_token_expire_minutes * 60  # Convert minutes to seconds
    )

    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/login/json", response_model=Token)
def login_json(
    response: Response,
    user_data: UserLogin,
    db: Session = Depends(get_db)
):
    """Login via JSON body (for HTMX forms)."""
    user = db.query(User).filter(User.email == user_data.email).first()

    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )

    access_token = create_access_token(data={"sub": str(user.id)})

    # Set httpOnly cookie (prevents JavaScript access)
    from app.config import get_settings
    settings = get_settings()

    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,
        secure=not settings.debug,  # True in production (HTTPS only)
        samesite="strict" if not settings.debug else "lax",  # Strict CSRF protection in production
        max_age=settings.access_token_expire_minutes * 60  # Convert minutes to seconds
    )

    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/logout")
def logout(response: Response):
    """Logout by clearing auth cookie."""
    response.delete_cookie(key="access_token")
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_active_user)):
    """Get current user info."""
    return current_user


@router.get("/me/organizations")
def get_my_organizations(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get organizations the current user belongs to."""
    memberships = db.query(OrganizationMember).filter(
        OrganizationMember.user_id == current_user.id
    ).all()

    return [
        {
            "id": m.organization.id,
            "name": m.organization.name,
            "role": m.role.value
        }
        for m in memberships
    ]
