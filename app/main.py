from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.db.database import init_db, create_test_user
from app.api.deps import get_current_user_or_redirect
from app.db.models import User
from app.core.rate_limit import limiter

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate security configuration on startup
    validate_security_config()

    # Startup: Initialize database
    init_db()
    # Create test user for development
    create_test_user()
    # Ensure storage directories exist
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.models_dir).mkdir(parents=True, exist_ok=True)
    yield
    # Shutdown: cleanup if needed


def validate_security_config():
    """Validate security configuration on startup."""
    # Check for insecure default keys
    insecure_patterns = [
        "dev-secret", "your-secret", "change-in-production",
        "secret", "password", "test", "demo"
    ]

    # Validate JWT secret key
    if any(pattern in settings.secret_key.lower() for pattern in insecure_patterns):
        if not settings.debug:
            raise RuntimeError(
                "SECURITY ERROR: Using default/weak SECRET_KEY in production! "
                "Set a strong SECRET_KEY environment variable (32+ chars)."
            )
        else:
            print("⚠️  WARNING: Using default SECRET_KEY in debug mode")

    if len(settings.secret_key) < 32:
        print("⚠️  WARNING: SECRET_KEY should be at least 32 characters")

    # Validate model signing key
    if any(pattern in settings.model_secret_key.lower() for pattern in insecure_patterns):
        if not settings.debug:
            raise RuntimeError(
                "SECURITY ERROR: Using default/weak MODEL_SECRET_KEY in production! "
                "Set a strong MODEL_SECRET_KEY environment variable (32+ chars)."
            )
        else:
            print("⚠️  WARNING: Using default MODEL_SECRET_KEY in debug mode")

    if len(settings.model_secret_key) < 32:
        print("⚠️  WARNING: MODEL_SECRET_KEY should be at least 32 characters")

    if settings.debug:
        print("⚠️  WARNING: Debug mode enabled - disable in production!")


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan
)

# Configure rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
allowed_origins = []
if settings.debug:
    # Development: allow localhost
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:3000"
    ]
else:
    # Production: specific domains only
    if settings.allowed_origins:
        allowed_origins = [origin.strip() for origin in settings.allowed_origins.split(",") if origin.strip()]
    else:
        raise RuntimeError(
            "SECURITY ERROR: ALLOWED_ORIGINS must be configured in production! "
            "Set ALLOWED_ORIGINS environment variable with comma-separated origins."
        )

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=3600,
)


# Security headers middleware
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Prevent MIME sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://unpkg.com; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )

        # Force HTTPS in production
        if not settings.debug:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        # Legacy XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Control referer header
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions policy
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=()"
        )

        return response

app.add_middleware(SecurityHeadersMiddleware)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="app/templates")


# ============ Import and include routers ============
from app.api import auth, datasets, projects, linkage, models as ml_models, labeling

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(linkage.router, prefix="/api/linkage", tags=["linkage"])
app.include_router(ml_models.router, prefix="/api/models", tags=["models"])
app.include_router(labeling.router, prefix="/api/labeling", tags=["labeling"])


# ============ Page Routes ============

@app.get("/")
async def root():
    return RedirectResponse(url="/login")


@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("auth/login.html", {"request": request})


@app.get("/register")
async def register_page(request: Request):
    return templates.TemplateResponse("auth/register.html", {"request": request})


@app.get("/dashboard")
async def dashboard_page(
    request: Request,
    current_user: User = Depends(get_current_user_or_redirect)
):
    return templates.TemplateResponse("dashboard/index.html", {
        "request": request,
        "user": current_user
    })


@app.get("/datasets")
async def datasets_page(
    request: Request,
    current_user: User = Depends(get_current_user_or_redirect)
):
    return templates.TemplateResponse("datasets/list.html", {
        "request": request,
        "user": current_user
    })


@app.get("/datasets/upload")
async def datasets_upload_page(
    request: Request,
    current_user: User = Depends(get_current_user_or_redirect)
):
    return templates.TemplateResponse("datasets/upload.html", {
        "request": request,
        "user": current_user
    })


@app.get("/datasets/{dataset_id}")
async def dataset_detail_page(
    request: Request,
    dataset_id: int,
    current_user: User = Depends(get_current_user_or_redirect)
):
    return templates.TemplateResponse("datasets/preview.html", {
        "request": request,
        "dataset_id": dataset_id,
        "user": current_user
    })


@app.get("/projects")
async def projects_page(
    request: Request,
    current_user: User = Depends(get_current_user_or_redirect)
):
    return templates.TemplateResponse("projects/list.html", {
        "request": request,
        "user": current_user
    })


@app.get("/projects/new")
async def projects_new_page(
    request: Request,
    current_user: User = Depends(get_current_user_or_redirect)
):
    return templates.TemplateResponse("projects/create.html", {
        "request": request,
        "user": current_user
    })


@app.get("/projects/{project_id}")
async def project_detail_page(
    request: Request,
    project_id: int,
    current_user: User = Depends(get_current_user_or_redirect)
):
    return templates.TemplateResponse("projects/detail.html", {
        "request": request,
        "project_id": project_id,
        "user": current_user
    })


@app.get("/projects/{project_id}/mapping")
async def project_mapping_page(
    request: Request,
    project_id: int,
    current_user: User = Depends(get_current_user_or_redirect)
):
    return templates.TemplateResponse("projects/mapping.html", {
        "request": request,
        "project_id": project_id,
        "user": current_user
    })


@app.get("/projects/{project_id}/config")
async def project_config_page(
    request: Request,
    project_id: int,
    current_user: User = Depends(get_current_user_or_redirect)
):
    return templates.TemplateResponse("projects/config.html", {
        "request": request,
        "project_id": project_id,
        "user": current_user
    })


@app.get("/projects/{project_id}/linkage")
async def project_linkage_page(
    request: Request,
    project_id: int,
    current_user: User = Depends(get_current_user_or_redirect)
):
    return templates.TemplateResponse("linkage/run.html", {
        "request": request,
        "project_id": project_id,
        "user": current_user
    })


@app.get("/projects/{project_id}/results")
async def project_results_page(
    request: Request,
    project_id: int,
    current_user: User = Depends(get_current_user_or_redirect)
):
    return templates.TemplateResponse("linkage/results.html", {
        "request": request,
        "project_id": project_id,
        "user": current_user
    })


@app.get("/projects/{project_id}/labeling")
async def project_labeling_page(
    request: Request,
    project_id: int,
    current_user: User = Depends(get_current_user_or_redirect)
):
    return templates.TemplateResponse("labeling/session.html", {
        "request": request,
        "project_id": project_id,
        "user": current_user
    })


@app.get("/projects/{project_id}/models")
async def project_models_page(
    request: Request,
    project_id: int,
    current_user: User = Depends(get_current_user_or_redirect)
):
    return templates.TemplateResponse("models/list.html", {
        "request": request,
        "project_id": project_id,
        "user": current_user
    })
