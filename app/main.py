from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from pathlib import Path

from app.config import get_settings
from app.db.database import init_db, create_test_user
from app.api.deps import get_current_user_or_redirect
from app.db.models import User

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
