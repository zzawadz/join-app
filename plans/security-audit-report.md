# Security Audit Report & Remediation Plan - Record Linkage Web Application

**Audit Date:** January 2026
**Application:** Record Linkage Web Application
**Audit Status:** Comprehensive Security Assessment Completed

---

# Executive Summary

This document contains a comprehensive security audit of the Record Linkage Web Application. The audit identified **20 security vulnerabilities** across critical, high, medium, and low severity levels.

**Immediate Action Required:**
- 3 CRITICAL vulnerabilities requiring urgent remediation
- 6 HIGH priority vulnerabilities
- 11 MEDIUM/LOW priority issues

The most critical findings include:
1. **Arbitrary Code Execution** via unsafe deserialization (RCE vulnerability)
2. **Token Leakage** via query parameters
3. **Denial of Service** via unlimited file uploads

---

# Table of Contents

1. [Critical Vulnerabilities](#critical-vulnerabilities) (3 issues)
2. [High Priority Vulnerabilities](#high-priority-vulnerabilities) (6 issues)
3. [Medium Priority Vulnerabilities](#medium-priority-vulnerabilities) (7 issues)
4. [Low Priority Vulnerabilities](#low-priority-vulnerabilities) (4 issues)
5. [Summary & Remediation Priority](#summary--remediation-priority)
6. [Implementation Guides](#implementation-guides)

---

# Critical Vulnerabilities

## CRITICAL-1: Unsafe Deserialization - Remote Code Execution (RCE)

**Severity:** CRITICAL
**File:** `app/core/linkage/ml_classifier.py:348`
**CWE:** CWE-502 (Deserialization of Untrusted Data)

### Vulnerable Code
```python
def load_classifier(path: str) -> BaseLinkageClassifier:
    """Load a classifier from disk."""
    return joblib.load(path)  # VULNERABLE TO ARBITRARY CODE EXECUTION
```

### Attack Vector
1. Attacker trains a model or uploads malicious file
2. System loads the model using `joblib.load()`
3. Malicious pickle payload executes arbitrary Python code
4. Full server compromise possible

### Impact
- **Complete System Compromise**: Arbitrary code execution as the application user
- **Data Breach**: Access to all database contents and user data
- **Lateral Movement**: Potential to compromise other systems
- **Persistence**: Ability to install backdoors

### Exploitation Difficulty
**Medium** - Requires ability to create/modify model files, but exploitation is well-documented.

### Remediation

**Option 1: Disable Pickle (Recommended)**
```python
def load_classifier(path: str) -> BaseLinkageClassifier:
    """Load a classifier from disk safely."""
    try:
        # Disable pickle protocol to prevent code execution
        return joblib.load(path, allow_pickle=False)
    except Exception as e:
        logger.error(f"Failed to load classifier from {path}: {e}")
        raise ValueError("Failed to load model - file may be corrupted")
```

**Option 2: Cryptographic Signing (More Secure)**
```python
import hmac
import hashlib

def sign_model(model_path: str, secret_key: str):
    """Sign a model file with HMAC."""
    with open(model_path, 'rb') as f:
        model_data = f.read()
    signature = hmac.new(secret_key.encode(), model_data, hashlib.sha256).hexdigest()
    with open(f"{model_path}.sig", 'w') as f:
        f.write(signature)

def load_classifier(path: str) -> BaseLinkageClassifier:
    """Load and verify signed classifier."""
    # Verify signature before loading
    with open(path, 'rb') as f:
        model_data = f.read()
    with open(f"{path}.sig", 'r') as f:
        expected_sig = f.read()

    actual_sig = hmac.new(settings.model_secret_key.encode(), model_data, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(actual_sig, expected_sig):
        raise ValueError("Model signature verification failed - possible tampering")

    return joblib.load(path)
```

**Files to Modify:**
- `app/core/linkage/ml_classifier.py` (line 348)
- `app/api/models.py` (anywhere load_classifier is called)

---

## CRITICAL-2: JWT Token Exposed in Query Parameters

**Severity:** CRITICAL
**File:** `app/api/linkage.py:327-353`
**CWE:** CWE-598 (Use of GET Request Method With Sensitive Query Strings)

### Vulnerable Code
```python
@router.get("/jobs/{job_id}/export")
def export_results(
    job_id: int,
    format: str = "csv",
    token: Optional[str] = Query(None, description="JWT token for authentication..."),
    db: Session = Depends(get_db)
):
    # Line 346: Token accepted from query parameter
    if token:
        current_user = get_user_from_token_param(token, db)
```

### Attack Vector
Tokens in query parameters are logged and exposed in multiple locations:
1. **Browser History**: Permanently stored in user's browser
2. **Server Access Logs**: Plain text in Apache/Nginx logs
3. **Proxy Logs**: Visible to all intermediate proxies
4. **Referer Headers**: Leaked to third-party sites via links
5. **Browser Extensions**: Accessible to malicious extensions
6. **Shared Links**: Users may accidentally share URLs with tokens

### Impact
- **Account Takeover**: Stolen tokens grant full access
- **Session Hijacking**: Long-lived tokens increase risk
- **Log Poisoning**: Sensitive data in logs

### Remediation

**Remove Query Parameter Authentication:**
```python
@router.get("/jobs/{job_id}/export")
def export_results(
    job_id: int,
    format: str = "csv",
    current_user: User = Depends(get_current_active_user),  # Use standard auth
    db: Session = Depends(get_db)
):
    """Export linkage results. Authentication via Authorization header only."""
    # Remove token parameter entirely
    # Use standard header-based authentication
```

**Alternative: POST-based Download**
```python
@router.post("/jobs/{job_id}/export")
def export_results(
    job_id: int,
    export_request: ExportRequest,  # Token in body if needed
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # POST requests don't cache URLs
    # Body is not logged in access logs
```

**Files to Modify:**
- `app/api/linkage.py` (lines 327-353)
- Any frontend code that calls this endpoint

---

## CRITICAL-3: No File Size Validation - Denial of Service

**Severity:** CRITICAL
**File:** `app/api/datasets.py:39-92`, `app/services/storage.py:13-36`
**CWE:** CWE-400 (Uncontrolled Resource Consumption)

### Vulnerable Code

**datasets.py:**
```python
@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    # ... no file size check before reading!
    file_path, file_size = await save_uploaded_file(file, project_id)
```

**storage.py:**
```python
async def save_uploaded_file(file: UploadFile, project_id: int):
    # Streams entire file to disk without size limit
    async with aiofiles.open(full_path, 'wb') as f:
        while True:
            chunk = await file.read(1024 * 1024)  # No cumulative size check!
            if not chunk:
                break
            await f.write(chunk)
```

### Attack Vector
1. Attacker uploads massive file (e.g., 100GB)
2. Server streams entire file to disk
3. Disk space exhausted
4. Application crashes or becomes unavailable

### Impact
- **Denial of Service**: Application becomes unavailable
- **Disk Exhaustion**: System-wide impact
- **Cost Overruns**: Cloud storage costs

### Remediation

**Step 1: Add Size Check in Endpoint**
```python
@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    project_id: int = Form(...),
    name: str = Form(...),
    role: str = Form(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Validate project access
    project = get_project_or_404(project_id, current_user, db)

    # NEW: Check file size before processing
    max_size = settings.max_upload_size_mb * 1024 * 1024  # Convert MB to bytes

    # Read small chunk to verify it's readable
    first_chunk = await file.read(1024)
    await file.seek(0)

    if file.size and file.size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb}MB"
        )

    # ... rest of upload logic
```

**Step 2: Add Streaming Size Limit**
```python
async def save_uploaded_file(
    file: UploadFile,
    project_id: int,
    max_size_bytes: int = None
) -> Tuple[str, int]:
    """Save uploaded file with size limit enforcement."""
    if max_size_bytes is None:
        max_size_bytes = get_settings().max_upload_size_mb * 1024 * 1024

    project_dir = Path(settings.upload_dir) / str(project_id)
    project_dir.mkdir(parents=True, exist_ok=True)

    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    full_path = project_dir / filename

    total_size = 0

    try:
        async with aiofiles.open(full_path, 'wb') as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break

                total_size += len(chunk)

                # NEW: Check size limit during streaming
                if total_size > max_size_bytes:
                    # Clean up partial file
                    await f.close()
                    full_path.unlink(missing_ok=True)
                    raise ValueError(
                        f"File exceeds maximum size of {max_size_bytes / 1024 / 1024:.1f}MB"
                    )

                await f.write(chunk)

        return str(full_path), total_size

    except Exception as e:
        # Clean up on any error
        full_path.unlink(missing_ok=True)
        raise
```

**Files to Modify:**
- `app/api/datasets.py` (lines 39-92)
- `app/services/storage.py` (lines 13-36)

---

# High Priority Vulnerabilities

## HIGH-1: Insecure Cookie Configuration for Production

**Severity:** HIGH
**File:** `app/api/auth.py:148-155, 184-191`
**CWE:** CWE-614 (Sensitive Cookie Without 'Secure' Flag)

### Vulnerable Code
```python
response.set_cookie(
    key="access_token",
    value=f"Bearer {access_token}",
    httponly=True,    # Good
    secure=False,     # ⚠️ VULNERABLE in production!
    samesite="lax",   # Weak - should be "strict"
    max_age=1800
)
```

### Issues
1. **secure=False**: Allows cookie transmission over HTTP (man-in-the-middle)
2. **samesite="lax"**: Weak CSRF protection (allows GET requests from other sites)

### Remediation
```python
response.set_cookie(
    key="access_token",
    value=f"Bearer {access_token}",
    httponly=True,
    secure=not settings.debug,  # True in production, False in dev
    samesite="strict" if not settings.debug else "lax",
    max_age=1800
)
```

**Files to Modify:**
- `app/api/auth.py` (lines 119-126, 155-162)

---

## HIGH-2: Hardcoded Test User Credentials

**Severity:** HIGH
**File:** `app/db/database.py:31-71`
**CWE:** CWE-798 (Use of Hard-coded Credentials)

### Vulnerable Code
```python
def create_test_user():
    """Create a test user for development."""
    # Runs on EVERY startup
    user = User(
        email="test@example.com",
        hashed_password=get_password_hash("test123"),  # Hardcoded!
        full_name="Test User"
    )
    print("Test user created: test@example.com / test123")  # Logs credentials!
```

### Issues
1. Default account accessible to everyone
2. Credentials printed to logs
3. Runs in all environments (dev/prod)
4. No way to disable

### Remediation
```python
def create_test_user():
    """Create a test user for development ONLY."""
    settings = get_settings()

    # Only run in debug mode
    if not settings.debug:
        return

    # Use environment variables for credentials
    test_email = os.getenv("TEST_USER_EMAIL", "test@example.com")
    test_password = os.getenv("TEST_USER_PASSWORD")

    if not test_password:
        # Generate random password if not provided
        test_password = secrets.token_urlsafe(16)
        logger.info(f"Generated test user password: {test_password}")

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == test_email).first()
        if existing:
            return  # Don't recreate

        user = User(
            email=test_email,
            hashed_password=get_password_hash(test_password),
            full_name="Test User"
        )
        # ... rest of logic ...

        # Log without password
        logger.info(f"Test user created: {test_email}")
    finally:
        db.close()
```

**Files to Modify:**
- `app/db/database.py` (lines 31-71)

---

## HIGH-3: Sensitive Data in Error Messages

**Severity:** HIGH
**Files:** `app/api/datasets.py:72`, `app/api/linkage.py:142`, `app/api/projects.py:203-206`
**CWE:** CWE-209 (Generation of Error Message Containing Sensitive Information)

### Vulnerable Code
```python
# datasets.py:72
raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")

# linkage.py:142
job.error_message = str(e)  # Full exception stored in database

# projects.py:206
raise HTTPException(status_code=500, detail=f"Failed to create demo project: {str(e)}")
```

### Issues
Exception messages may contain:
- SQL queries
- File paths
- Stack traces
- Internal configuration details
- Database schema information

### Remediation

**Create Secure Error Handler:**
```python
# app/core/errors.py (new file)
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

def safe_error_response(
    operation: str,
    exception: Exception,
    status_code: int = 500,
    user_message: str = None
) -> HTTPException:
    """
    Log full exception details, return safe message to user.

    Args:
        operation: Description of what was being attempted
        exception: The caught exception
        status_code: HTTP status code
        user_message: Optional custom message for user
    """
    # Log full details for debugging (secure logs only)
    logger.exception(f"Error during {operation}: {exception}")

    # Return generic message to user
    if user_message is None:
        user_message = f"An error occurred during {operation}. Please try again or contact support."

    return HTTPException(status_code=status_code, detail=user_message)
```

**Usage:**
```python
# In datasets.py
try:
    df = pd.read_csv(file_path)
    # ... processing ...
except Exception as e:
    raise safe_error_response(
        operation="CSV processing",
        exception=e,
        status_code=400,
        user_message="Failed to process CSV file. Please check file format."
    )
```

**Files to Modify:**
- Create `app/core/errors.py`
- `app/api/datasets.py` (line 72)
- `app/api/linkage.py` (line 142)
- `app/api/projects.py` (lines 203-206)
- All other endpoints with `detail=str(e)`

---

## HIGH-4: Missing CORS Configuration

**Severity:** HIGH
**File:** `app/main.py`
**CWE:** CWE-346 (Origin Validation Error)

### Issue
No CORS middleware configured. By default, FastAPI may allow all origins.

### Impact
- Cross-origin requests from malicious sites
- CSRF attacks (despite SameSite cookies)
- Data exfiltration

### Remediation
```python
# app/main.py - Add after app creation

from fastapi.middleware.cors import CORSMiddleware

# CORS configuration
allowed_origins = []
if settings.debug:
    # Development: allow localhost
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]
else:
    # Production: specific domains only
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
    if not allowed_origins or allowed_origins == [""]:
        raise RuntimeError("ALLOWED_ORIGINS must be configured in production")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=3600,
)
```

**Files to Modify:**
- `app/main.py`
- `app/config.py` (add allowed_origins setting)
- `.env.example` (add ALLOWED_ORIGINS=https://yourdomain.com)

---

## HIGH-5: Remaining XSS Vulnerabilities in Templates

**Severity:** HIGH
**Files:** Multiple HTML templates
**CWE:** CWE-79 (Cross-site Scripting)

### Vulnerable Templates

**`app/templates/linkage/run.html`** (Lines 205, 213, 355, 357)
**`app/templates/datasets/preview.html`** (Lines 89, 97, 108)
**`app/templates/projects/mapping.html`** (Lines 125, 138)

### Example Vulnerable Code
```javascript
// linkage/run.html:205
tbody.innerHTML = Object.entries(mappings).map(([source, target], idx) => `
    <tr>
        <td>${source}</td>  // Not escaped!
        <td>${target}</td>  // Not escaped!
    </tr>
`).join('');
```

### Remediation
Use `escapeHtml()` function (already defined in base.html):

```javascript
tbody.innerHTML = Object.entries(mappings).map(([source, target], idx) => `
    <tr>
        <td>${escapeHtml(source)}</td>
        <td>${escapeHtml(target)}</td>
    </tr>
`).join('');
```

**Files to Modify:**
- `app/templates/linkage/run.html`
- `app/templates/datasets/preview.html`
- `app/templates/projects/mapping.html`

---

## HIGH-6: Weak File Type Validation

**Severity:** HIGH
**File:** `app/api/datasets.py:59-61`
**CWE:** CWE-434 (Unrestricted Upload of File with Dangerous Type)

### Vulnerable Code
```python
# Validate file type
if not file.filename.endswith('.csv'):
    raise HTTPException(status_code=400, detail="Only CSV files are supported")
```

### Issues
- Filename easily spoofed (attacker can rename malicious file to `.csv`)
- No content validation
- Could upload executable with `.csv` extension

### Remediation
```python
import chardet

async def validate_csv_file(file: UploadFile) -> bool:
    """Validate file is actually a CSV by content inspection."""
    # Read first 1KB to check
    sample = await file.read(1024)
    await file.seek(0)  # Reset for later reading

    # Check encoding
    detected = chardet.detect(sample)
    if detected['confidence'] < 0.7:
        return False

    # Try to parse as CSV
    try:
        import io
        import csv
        sample_str = sample.decode(detected['encoding'])
        csv.reader(io.StringIO(sample_str))
        return True
    except Exception:
        return False

# In upload endpoint
if not file.filename.endswith('.csv'):
    raise HTTPException(status_code=400, detail="Only CSV files are supported")

# NEW: Validate actual content
if not await validate_csv_file(file):
    raise HTTPException(status_code=400, detail="File does not appear to be a valid CSV")
```

**Files to Modify:**
- `app/api/datasets.py` (add validation function and use it)

---

# Medium Priority Vulnerabilities

## MEDIUM-1: Missing Rate Limiting on Authentication

**Severity:** MEDIUM
**File:** `app/api/auth.py:94-194`
**CWE:** CWE-307 (Improper Restriction of Excessive Authentication Attempts)

### Issue
No rate limiting on `/login` and `/register` endpoints.

### Impact
- Brute force attacks on user accounts
- Credential stuffing attacks
- Account enumeration
- Resource exhaustion

### Remediation

**Install slowapi:**
```bash
pip install slowapi
```

**Configure rate limiting:**
```python
# app/main.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# app/api/auth.py
from app.main import limiter

@router.post("/login", response_model=Token)
@limiter.limit("5/minute")  # Max 5 login attempts per minute
def login(
    request: Request,  # Required for rate limiting
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    # ... existing code ...

@router.post("/register", response_model=UserResponse)
@limiter.limit("3/hour")  # Max 3 registrations per hour
def register(
    request: Request,
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    # ... existing code ...
```

**Files to Modify:**
- `requirements.txt` (add slowapi)
- `app/main.py` (configure limiter)
- `app/api/auth.py` (apply limits to endpoints)

---

## MEDIUM-2: Insufficient Password Strength Requirements

**Severity:** MEDIUM
**File:** `app/api/auth.py:19-39`
**CWE:** CWE-521 (Weak Password Requirements)

### Current Implementation
```python
def validate_password_strength(password: str) -> Tuple[bool, str]:
    if len(password) < 8:  # Minimum 8 chars
        return False, "Password must be at least 8 characters"
    # Basic regex checks
    # Hardcoded weak password list (only 5 passwords)
    weak = ['password', '12345678', 'qwerty', 'abc123', 'letmein']
```

### Issues
- Minimum length too short (8 chars)
- Weak password list incomplete
- No entropy/complexity scoring
- Allows common patterns (e.g., "Password1!")

### Remediation

**Option 1: Use zxcvbn library (Recommended)**
```bash
pip install zxcvbn
```

```python
from zxcvbn import zxcvbn

def validate_password_strength(password: str, user_inputs: list = None) -> Tuple[bool, str]:
    """
    Validate password strength using zxcvbn.

    Args:
        password: Password to validate
        user_inputs: List of user-specific words (email, name) to check against
    """
    # Length check
    if len(password) < 12:
        return False, "Password must be at least 12 characters"

    if len(password) > 128:
        return False, "Password too long (max 128 characters)"

    # Use zxcvbn for strength analysis
    result = zxcvbn(password, user_inputs=user_inputs or [])

    # Require score of 3 or higher (out of 4)
    if result['score'] < 3:
        feedback = result['feedback']['warning'] or result['feedback']['suggestions'][0]
        return False, f"Password too weak: {feedback}"

    return True, ""
```

**Update registration to pass user context:**
```python
@router.post("/register", response_model=UserResponse)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    # ... email check ...

    # Validate password with user context
    user_inputs = [
        user_data.email,
        user_data.email.split('@')[0],  # Username part of email
        user_data.full_name
    ]
    is_valid, error_msg = validate_password_strength(user_data.password, user_inputs)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # ... rest of registration ...
```

**Files to Modify:**
- `requirements.txt` (add zxcvbn)
- `app/api/auth.py` (update validation function)

---

## MEDIUM-3: Missing CSRF Protection

**Severity:** MEDIUM
**Files:** All HTML forms
**CWE:** CWE-352 (Cross-Site Request Forgery)

### Issue
While `samesite="lax"` provides some protection, it's not sufficient:
- Lax mode allows GET requests from other sites
- No token validation on forms
- Vulnerable to same-site attacks

### Remediation

**Install fastapi-csrf-protect:**
```bash
pip install fastapi-csrf-protect
```

**Configure CSRF protection:**
```python
# app/config.py
csrf_secret_key: str = Field(
    ...,  # Required
    env="CSRF_SECRET_KEY",
    description="Secret key for CSRF protection"
)

# app/main.py
from fastapi_csrf_protect import CsrfProtect
from fastapi_csrf_protect.exceptions import CsrfProtectError
from pydantic import BaseModel

class CsrfSettings(BaseModel):
    secret_key: str = get_settings().csrf_secret_key
    cookie_samesite: str = 'strict'
    cookie_httponly: bool = True

@CsrfProtect.load_config
def get_csrf_config():
    return CsrfSettings()

@app.exception_handler(CsrfProtectError)
def csrf_protect_exception_handler(request: Request, exc: CsrfProtectError):
    return JSONResponse(status_code=403, content={"detail": "CSRF validation failed"})

# app/api/auth.py - Add token endpoint
@router.get("/csrf-token")
async def get_csrf_token(csrf_protect: CsrfProtect = Depends()):
    """Generate CSRF token for forms."""
    csrf_token, signed_token = csrf_protect.generate_csrf_tokens()
    response = JSONResponse(content={"csrf_token": csrf_token})
    csrf_protect.set_csrf_cookie(signed_token, response)
    return response
```

**Update frontend to include tokens:**
```javascript
// app/templates/base.html
async function getCsrfToken() {
    const response = await fetch('/api/auth/csrf-token');
    const data = await response.json();
    return data.csrf_token;
}

async function authenticatedFetch(url, options = {}) {
    const token = getAuthToken();
    const headers = {'Authorization': 'Bearer ' + token, ...options.headers};

    // Add CSRF token for state-changing requests
    if (['POST', 'PUT', 'DELETE', 'PATCH'].includes(options.method?.toUpperCase())) {
        const csrfToken = await getCsrfToken();
        if (csrfToken) headers['X-CSRF-Token'] = csrfToken;
    }

    return fetch(url, {...options, headers});
}
```

**Files to Modify:**
- `requirements.txt`
- `app/config.py`
- `app/main.py`
- `app/api/auth.py`
- `app/templates/base.html`
- All templates with forms

---

## MEDIUM-4: Missing Security Headers

**Severity:** MEDIUM
**File:** `app/main.py`
**CWE:** CWE-693 (Protection Mechanism Failure)

### Missing Headers
- `X-Frame-Options` (Clickjacking protection)
- `X-Content-Type-Options` (MIME sniffing protection)
- `Content-Security-Policy` (XSS/Injection protection)
- `Strict-Transport-Security` (Force HTTPS)
- `X-XSS-Protection` (Legacy XSS filter)
- `Referrer-Policy` (Control referer header)

### Remediation

**Add security headers middleware:**
```python
# app/main.py
from fastapi.middleware import Middleware
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
```

**Files to Modify:**
- `app/main.py`

---

## MEDIUM-5: Verbose Error Messages

**Severity:** MEDIUM
**Files:** Various API endpoints
**CWE:** CWE-209

### Issue
Multiple endpoints return detailed exception messages to clients (already covered in HIGH-3).

### Remediation
Implement centralized error handling (see HIGH-3 remediation).

---

## MEDIUM-6: No Input Length Validation

**Severity:** MEDIUM
**Files:** `app/schemas/*.py`
**CWE:** CWE-1284 (Improper Validation of Specified Quantity in Input)

### Issue
Pydantic schemas lack maximum length constraints:

```python
class ProjectCreate(BaseModel):
    name: str  # No max length
    description: str  # No max length
```

### Impact
- Database errors on insert
- Memory exhaustion
- Display issues

### Remediation
```python
from pydantic import Field

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=5000)
    linkage_type: LinkageType

class UserCreate(BaseModel):
    email: EmailStr = Field(..., max_length=255)
    password: str = Field(..., min_length=12, max_length=128)
    full_name: str = Field(..., max_length=255)
```

**Files to Modify:**
- `app/schemas/user.py`
- `app/schemas/project.py`
- `app/schemas/dataset.py`
- All other schema files

---

## MEDIUM-7: Path Traversal Risk in Storage

**Severity:** MEDIUM
**File:** `app/services/storage.py:13-36`
**CWE:** CWE-22 (Path Traversal)

### Vulnerable Code
```python
project_dir = Path(settings.upload_dir) / str(project_id)
```

### Issue
While `project_id` comes from database (integer), defense in depth requires validation.

### Remediation
```python
def save_uploaded_file(file: UploadFile, project_id: int):
    """Save uploaded file with path traversal protection."""
    # Validate project_id is positive integer
    if not isinstance(project_id, int) or project_id <= 0:
        raise ValueError("Invalid project ID")

    # Resolve to absolute path and verify it's within upload dir
    base_dir = Path(settings.upload_dir).resolve()
    project_dir = (base_dir / str(project_id)).resolve()

    # Security check: ensure resolved path is within base directory
    if not str(project_dir).startswith(str(base_dir)):
        raise ValueError("Path traversal attempt detected")

    # ... rest of function ...
```

**Files to Modify:**
- `app/services/storage.py`

---

# Low Priority Vulnerabilities

## LOW-1: No Audit Logging

**Severity:** LOW
**Files:** All API endpoints
**CWE:** CWE-778 (Insufficient Logging)

### Issue
No audit trail for security events:
- Login attempts (success/failure)
- Data access
- Configuration changes
- Administrative actions

### Impact
- Cannot detect security incidents
- No forensic trail
- Compliance issues

### Remediation

**Create audit logging system:**
```python
# app/core/audit.py (new file)
import logging
from datetime import datetime
from typing import Optional

audit_logger = logging.getLogger('audit')

def log_security_event(
    event_type: str,
    user_id: Optional[int],
    ip_address: str,
    details: dict,
    status: str = "success"
):
    """
    Log security-relevant events.

    event_type: login, logout, access, create, update, delete, config_change
    status: success, failure, blocked
    """
    audit_logger.info({
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "user_id": user_id,
        "ip_address": ip_address,
        "status": status,
        "details": details
    })

# Usage in auth.py
def login(...):
    try:
        # ... authentication logic ...
        log_security_event(
            event_type="login",
            user_id=user.id,
            ip_address=request.client.host,
            details={"email": user.email},
            status="success"
        )
        return token
    except HTTPException:
        log_security_event(
            event_type="login",
            user_id=None,
            ip_address=request.client.host,
            details={"email": form_data.username},
            status="failure"
        )
        raise
```

**Configure separate audit log file:**
```python
# In logging configuration
audit_handler = logging.FileHandler('logs/audit.log')
audit_handler.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger('audit').addHandler(audit_handler)
logging.getLogger('audit').setLevel(logging.INFO)
```

**Files to Create/Modify:**
- Create `app/core/audit.py`
- Modify `app/api/auth.py`
- Modify `app/api/projects.py`
- Modify `app/api/datasets.py`
- Configure logging in `app/main.py`

---

## LOW-2: Long Session Timeout

**Severity:** LOW
**File:** `app/config.py:18-21`
**CWE:** CWE-613 (Insufficient Session Expiration)

### Issue
```python
access_token_expire_minutes: int = Field(
    default=30,  # 30 minutes
)
```

30 minutes provides a long window for token theft/misuse.

### Remediation
```python
# Reduce to 15 minutes
access_token_expire_minutes: int = Field(default=15, env="ACCESS_TOKEN_EXPIRE_MINUTES")

# Implement refresh tokens for longer sessions
refresh_token_expire_days: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")
```

Implement refresh token flow:
- Short-lived access tokens (15 min)
- Long-lived refresh tokens (7 days)
- Separate endpoint for token refresh

---

## LOW-3: No Database Connection Pooling Configuration

**Severity:** LOW
**File:** `app/db/database.py:1-12`
**CWE:** CWE-770 (Allocation of Resources Without Limits)

### Issue
```python
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
```

No pool size limits or timeout configuration.

### Remediation
```python
from sqlalchemy.pool import QueuePool

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    poolclass=QueuePool,
    pool_size=20,           # Max permanent connections
    max_overflow=40,        # Max temporary connections
    pool_timeout=30,        # Timeout waiting for connection
    pool_recycle=3600,      # Recycle connections after 1 hour
    pool_pre_ping=True,     # Test connections before use
)
```

**Files to Modify:**
- `app/db/database.py`

---

## LOW-4: Debug Mode Warnings

**Severity:** LOW
**File:** `app/main.py:49-53`
**CWE:** CWE-489 (Active Debug Code)

### Issue
Debug mode may expose sensitive information.

### Current Mitigation
Already has warnings, but should enforce production settings.

### Additional Remediation
```python
def validate_security_config():
    """Validate security configuration on startup."""
    # ... existing checks ...

    # Block startup if debug in production with no override
    if settings.debug and os.getenv("FORCE_DEBUG") != "true":
        if os.getenv("ENV") == "production":
            raise RuntimeError(
                "DEBUG mode is enabled in production! "
                "Set DEBUG=false or ENV=development"
            )
```

---

# Summary & Remediation Priority

## Vulnerability Summary

| Severity | Count | Status |
|----------|-------|--------|
| CRITICAL | 3 | ⚠️ URGENT |
| HIGH | 6 | ⚠️ Important |
| MEDIUM | 7 | ⚫ Plan Soon |
| LOW | 4 | ⚫ Consider |
| **TOTAL** | **20** | |

## Recommended Implementation Order

### Phase 1: Critical Fixes (URGENT - Complete Within 1 Week)

1. **CRITICAL-1**: Fix unsafe deserialization in ml_classifier.py
   - Add `allow_pickle=False` to joblib.load()
   - Implement model signing/verification
   - Test with existing models

2. **CRITICAL-2**: Remove token from query parameters
   - Refactor export endpoint
   - Update frontend download logic
   - Test file downloads

3. **CRITICAL-3**: Implement file size validation
   - Add size checks before upload
   - Add streaming size limits
   - Test with large files

### Phase 2: High Priority (Complete Within 2 Weeks)

4. **HIGH-1**: Fix cookie security settings
5. **HIGH-2**: Remove hardcoded credentials
6. **HIGH-3**: Implement safe error handling
7. **HIGH-4**: Configure CORS properly
8. **HIGH-5**: Fix remaining XSS issues
9. **HIGH-6**: Strengthen file validation

### Phase 3: Medium Priority (Complete Within 1 Month)

10. **MEDIUM-1**: Add rate limiting
11. **MEDIUM-2**: Improve password requirements
12. **MEDIUM-3**: Implement CSRF protection
13. **MEDIUM-4**: Add security headers
14. **MEDIUM-5-7**: Input validation, error handling

### Phase 4: Low Priority (Complete Within 2 Months)

15. **LOW-1-4**: Audit logging, session timeout, connection pooling, debug enforcement

---

# Implementation Guides

## Guide 1: Fixing Unsafe Deserialization (CRITICAL-1)

### Files to Modify
1. `app/core/linkage/ml_classifier.py` (line 348)
2. `app/api/models.py` (all load_classifier calls)
3. `app/config.py` (add model_secret_key)

### Step-by-Step

**Step 1: Update Config**
```python
# app/config.py
model_secret_key: str = Field(
    ...,  # Required
    env="MODEL_SECRET_KEY",
    description="Secret key for model signing"
)
```

**Step 2: Add Signing Function**
```python
# app/core/linkage/ml_classifier.py
import hmac
import hashlib
from pathlib import Path

def sign_model(model_path: str, secret_key: str):
    """Sign a model file with HMAC-SHA256."""
    with open(model_path, 'rb') as f:
        model_data = f.read()

    signature = hmac.new(
        secret_key.encode(),
        model_data,
        hashlib.sha256
    ).hexdigest()

    sig_path = Path(f"{model_path}.sig")
    with open(sig_path, 'w') as f:
        f.write(signature)

    return sig_path
```

**Step 3: Update Load Function**
```python
def load_classifier(path: str) -> BaseLinkageClassifier:
    """Load and verify signed classifier."""
    from app.config import get_settings
    settings = get_settings()

    # Verify signature
    sig_path = Path(f"{path}.sig")
    if not sig_path.exists():
        raise ValueError("Model signature file missing - possible tampering")

    with open(path, 'rb') as f:
        model_data = f.read()

    with open(sig_path, 'r') as f:
        expected_sig = f.read().strip()

    actual_sig = hmac.new(
        settings.model_secret_key.encode(),
        model_data,
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(actual_sig, expected_sig):
        raise ValueError("Model signature verification failed - possible tampering")

    # Load with pickle disabled
    return joblib.load(path, allow_pickle=False)
```

**Step 4: Update Save Function**
```python
def save_classifier(classifier: BaseLinkageClassifier, path: str):
    """Save and sign classifier."""
    from app.config import get_settings
    settings = get_settings()

    # Save model
    joblib.dump(classifier, path)

    # Sign model
    sign_model(path, settings.model_secret_key)
```

**Step 5: Update Training Task**
```python
# app/api/models.py - in train_model_task
# Change save call to:
save_classifier(classifier, model_path)  # Will auto-sign
```

**Step 6: Test**
```python
# Test that old unsigned models are rejected
# Test that tampered signatures are detected
# Test that valid signed models load correctly
```

---

## Guide 2: Testing the Fixes

### Manual Testing Checklist

**Critical Vulnerabilities:**
- [ ] Attempt to load unsigned model (should fail)
- [ ] Tamper with model signature (should detect)
- [ ] Upload 500MB file (should reject)
- [ ] Try export with token in URL (should not work)

**High Vulnerabilities:**
- [ ] Check cookie security flags in DevTools
- [ ] Verify test user doesn't exist in production
- [ ] Trigger error, verify no stack trace in response
- [ ] Test XSS payloads in inputs

**Medium Vulnerabilities:**
- [ ] Attempt 10 logins in 1 minute (should block)
- [ ] Test weak password (should reject)
- [ ] Check security headers in response

### Automated Testing

Create `tests/test_security_fixes.py`:

```python
import pytest
from fastapi.testclient import TestClient

class TestCriticalFixes:
    def test_model_signature_required(self):
        """Test that unsigned models cannot be loaded."""
        # Save model without signature
        # Attempt to load
        # Should raise ValueError
        pass

    def test_file_size_limit(self, client, auth_headers):
        """Test file upload size limit."""
        # Create 200MB file
        # Attempt upload
        # Should get 413 error
        pass

    def test_no_token_in_query_param(self, client, auth_headers):
        """Test export doesn't accept token in query."""
        # Call export with token in query
        # Should not authenticate
        pass

class TestHighPriorityFixes:
    def test_cookie_security_production(self):
        """Test cookies have secure flag in production."""
        # Set DEBUG=false
        # Login
        # Check cookie flags
        pass

    def test_no_test_user_in_production(self):
        """Test user is not created in production."""
        # Set DEBUG=false
        # Start app
        # Check test user doesn't exist
        pass
```

---

# Appendix

## Environment Variables Needed

Add to `.env`:
```bash
# Security (REQUIRED)
SECRET_KEY=your-secret-key-32-chars-minimum
MODEL_SECRET_KEY=your-model-key-32-chars-minimum
CSRF_SECRET_KEY=your-csrf-key-32-chars-minimum

# CORS (REQUIRED in production)
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# Debug (MUST be false in production)
DEBUG=false

# Session
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# File Upload
MAX_UPLOAD_SIZE_MB=100
```

## Dependencies to Add

```txt
# requirements.txt additions
slowapi>=0.1.8          # Rate limiting
fastapi-csrf-protect>=0.3.0  # CSRF protection
zxcvbn>=4.4.28         # Password strength
chardet>=5.2.0          # File encoding detection (already present)
```

---

# Record Linkage Web Application - Implementation Plan

## Overview
Build a full-featured web application for record linkage supporting deduplication, Fellegi-Sunter model, ML classification, active learning, CSV upload, fuzzy matching, and team collaboration.

## Tech Stack
- **Backend**: FastAPI (Python 3.11+)
- **Frontend**: Plain HTML + HTMX + Tailwind CSS
- **Database**: SQLite (with SQLAlchemy for easy PostgreSQL migration)
- **ML Libraries**: scikit-learn, jellyfish (string distances)
- **Background Tasks**: FastAPI BackgroundTasks (simple, sufficient for medium scale)

## Project Structure
```
join-app/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI entry point
│   ├── config.py                  # Settings management
│   ├── api/                       # API routes
│   │   ├── __init__.py
│   │   ├── deps.py                # Dependencies (auth, db)
│   │   ├── auth.py                # Auth endpoints
│   │   ├── datasets.py            # Dataset CRUD
│   │   ├── projects.py            # Project management
│   │   ├── linkage.py             # Linkage operations
│   │   ├── models.py              # ML model endpoints
│   │   └── labeling.py            # Active learning
│   ├── core/                      # Business logic
│   │   ├── __init__.py
│   │   ├── security.py            # Auth utilities
│   │   └── linkage/
│   │       ├── __init__.py
│   │       ├── blocking.py        # Blocking strategies
│   │       ├── comparators.py     # String similarity functions
│   │       ├── fellegi_sunter.py  # FS model
│   │       ├── ml_classifier.py   # ML classifiers
│   │       └── active_learning.py # AL strategies
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py            # DB session
│   │   └── models.py              # SQLAlchemy models
│   ├── schemas/                   # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── dataset.py
│   │   ├── project.py
│   │   └── linkage.py
│   ├── services/                  # Business services
│   │   ├── __init__.py
│   │   ├── csv_processor.py
│   │   ├── column_mapper.py
│   │   └── storage.py
│   └── templates/                 # Jinja2 templates
│       ├── base.html
│       ├── auth/
│       ├── dashboard/
│       ├── datasets/
│       ├── projects/
│       ├── linkage/
│       ├── labeling/
│       └── components/            # HTMX partials
├── static/
│   ├── css/
│   └── js/
├── storage/                       # Uploaded files & models
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Database Schema (Key Models)

### Users & Organizations
- **User**: id, email, hashed_password, full_name, organization_id
- **Organization**: id, name, created_at
- **OrganizationMember**: user_id, organization_id, role (admin/member)

### Core Data
- **Project**: id, name, organization_id, linkage_type (dedup/link), column_mappings (JSON), comparison_config (JSON), blocking_config (JSON)
- **Dataset**: id, project_id, name, file_path, row_count, column_names (JSON), role (source/target/dedupe)
- **LinkageModel**: id, project_id, model_type, model_path, parameters (JSON), metrics (JSON), is_active

### Linkage Results
- **LinkageJob**: id, project_id, model_id, status, total_pairs, matched_pairs, results_path
- **LabeledPair**: id, project_id, left_record (JSON), right_record (JSON), comparison_vector (JSON), label, labeled_by

## Implementation Phases

### Phase 1: Project Foundation
1. Initialize FastAPI project with proper structure
2. Set up SQLAlchemy + SQLite database
3. Create base HTML template with Tailwind CSS
4. Add HTMX for interactivity

**Files to create:**
- `app/main.py`
- `app/config.py`
- `app/db/database.py`
- `app/db/models.py`
- `app/templates/base.html`
- `requirements.txt`

### Phase 2: Authentication & Organizations
1. User model with password hashing (bcrypt)
2. JWT token authentication
3. Organization model with member roles
4. Login/Register pages
5. Protected routes

**Files to create:**
- `app/core/security.py`
- `app/api/auth.py`
- `app/api/deps.py`
- `app/schemas/user.py`
- `app/templates/auth/login.html`
- `app/templates/auth/register.html`

### Phase 3: Dataset Management
1. CSV upload with validation
2. Encoding/delimiter detection
3. Column metadata extraction
4. Dataset preview
5. Storage service

**Files to create:**
- `app/api/datasets.py`
- `app/services/csv_processor.py`
- `app/services/storage.py`
- `app/schemas/dataset.py`
- `app/templates/datasets/upload.html`
- `app/templates/datasets/list.html`
- `app/templates/datasets/preview.html`

### Phase 4: Project Management
1. Project CRUD with organization scoping
2. Dataset attachment
3. Project creation wizard

**Files to create:**
- `app/api/projects.py`
- `app/schemas/project.py`
- `app/templates/projects/list.html`
- `app/templates/projects/create.html`
- `app/templates/projects/detail.html`
- `app/templates/dashboard/index.html`

### Phase 5: Column Mapping
1. Auto-suggest mappings based on column names
2. Content-based compatibility analysis
3. Interactive mapping UI

**Files to create:**
- `app/services/column_mapper.py`
- `app/templates/projects/mapping.html`

### Phase 6: Comparison Functions
1. Implement fuzzy matchers:
   - Jaro-Winkler
   - Levenshtein
   - Soundex
   - Exact match
   - Numeric similarity
2. Comparison configuration UI
3. Sample pair testing

**Files to create:**
- `app/core/linkage/comparators.py`
- `app/templates/projects/config.html`

### Phase 7: Blocking Strategies
1. Standard blocking
2. Sorted neighborhood blocking
3. Blocking configuration

**Files to create:**
- `app/core/linkage/blocking.py`

### Phase 8: Fellegi-Sunter Model
1. M and U probability estimation
2. EM algorithm for unsupervised estimation
3. Match weight calculation
4. Classification with thresholds

**Files to create:**
- `app/core/linkage/fellegi_sunter.py`

### Phase 9: Linkage Execution
1. Background job processing
2. Progress tracking with HTMX polling
3. Results storage and display
4. CSV export

**Files to create:**
- `app/api/linkage.py`
- `app/schemas/linkage.py`
- `app/templates/linkage/run.html`
- `app/templates/linkage/results.html`

### Phase 10: ML Classification
1. Classifier base class
2. Logistic regression classifier
3. Random forest classifier
4. Model training UI
5. Model serialization with joblib

**Files to create:**
- `app/core/linkage/ml_classifier.py`
- `app/api/models.py`
- `app/templates/models/train.html`
- `app/templates/models/list.html`

### Phase 11: Active Learning
1. Uncertainty sampling strategy
2. Labeling session management
3. Interactive labeling UI with keyboard shortcuts
4. Automatic model retraining
5. Progress tracking

**Files to create:**
- `app/core/linkage/active_learning.py`
- `app/api/labeling.py`
- `app/templates/labeling/session.html`

### Phase 12: Deduplication Mode
1. Single-dataset handling
2. Self-comparison prevention
3. Duplicate clustering

**Files to update:**
- `app/core/linkage/blocking.py` (add dedup support)
- `app/api/linkage.py` (dedup mode)

## Key Dependencies (requirements.txt)
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
sqlalchemy>=2.0.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-multipart>=0.0.6
jinja2>=3.1.2
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
jellyfish>=1.0.0
python-Levenshtein>=0.23.0
joblib>=1.3.0
chardet>=5.2.0
aiofiles>=23.2.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
```

## API Endpoints Summary

### Auth
- `POST /api/auth/register` - Create account
- `POST /api/auth/login` - Get JWT token
- `GET /api/auth/me` - Current user

### Datasets
- `GET /api/datasets` - List datasets
- `POST /api/datasets/upload` - Upload CSV
- `GET /api/datasets/{id}` - Dataset details
- `GET /api/datasets/{id}/preview` - Sample data

### Projects
- `GET /api/projects` - List projects
- `POST /api/projects` - Create project
- `GET /api/projects/{id}` - Project details
- `POST /api/projects/{id}/mapping/suggest` - Auto-suggest mappings
- `POST /api/projects/{id}/mapping` - Save mappings

### Linkage
- `POST /api/projects/{id}/linkage/run` - Start job
- `GET /api/linkage/jobs/{id}` - Job status
- `GET /api/linkage/jobs/{id}/results` - Get results
- `GET /api/linkage/jobs/{id}/export` - Export CSV

### Models
- `POST /api/projects/{id}/models/train` - Train model
- `GET /api/projects/{id}/models` - List models
- `POST /api/models/{id}/activate` - Set active

### Labeling (Active Learning)
- `POST /api/projects/{id}/labeling/start` - Start session
- `GET /api/labeling/sessions/{id}/next` - Get next pair
- `POST /api/labeling/sessions/{id}/label` - Submit label
- `POST /api/labeling/sessions/{id}/retrain` - Retrain model

### Pages (HTML)
- `GET /` - Redirect to dashboard
- `GET /login`, `/register` - Auth pages
- `GET /dashboard` - Main dashboard
- `GET /datasets`, `/datasets/upload`, `/datasets/{id}` - Dataset pages
- `GET /projects/new`, `/projects/{id}` - Project pages
- `GET /projects/{id}/mapping` - Column mapping
- `GET /projects/{id}/config` - Comparison config
- `GET /projects/{id}/linkage` - Run/view linkage
- `GET /projects/{id}/labeling` - Active learning UI
- `GET /projects/{id}/models` - Model management

## Notes
- Use HTMX `hx-post`/`hx-get` for dynamic updates without full page reloads
- Polling with `hx-trigger="every 2s"` for job progress
- Keyboard shortcuts (y/n/s) for fast labeling
- Store labeled pairs independently from source data for training persistence

---

# Feature Expansion Plan

## Overview
Three new features to enhance the application:
1. **Demo Project Creation** - Quick onboarding with synthetic test data
2. **Active Learning Enhancements** - Progress tracking, timing stats, configuration modal
3. **Model Statistics Tab** - Training history, performance trends, in-sample/out-of-sample metrics

---

## Phase 1: Database Schema Changes

### New Table: TrainingHistory
```python
class TrainingHistory(Base):
    __tablename__ = "training_history"

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("linkage_models.id"))
    iteration = Column(Integer)  # Sequential training number
    training_pairs_count = Column(Integer)

    # In-sample metrics (training set)
    train_precision = Column(Float)
    train_recall = Column(Float)
    train_f1 = Column(Float)
    train_accuracy = Column(Float)

    # Out-of-sample metrics (test set)
    test_precision = Column(Float)
    test_recall = Column(Float)
    test_f1 = Column(Float)
    test_accuracy = Column(Float)

    # Cross-validation
    cv_f1_mean = Column(Float)
    cv_f1_std = Column(Float)

    feature_importance = Column(JSON)
    confusion_matrix = Column(JSON)  # {"tp": x, "tn": y, "fp": z, "fn": w}
    trained_at = Column(DateTime, server_default=func.now())
```

### Modifications to Existing Tables

**Project** - add demo fields:
- `is_demo = Column(Boolean, default=False)`
- `demo_domain = Column(String(50))` - "people" or "companies"

**LinkageModel** - add history tracking:
- `history_retention = Column(Integer, default=10)`
- `training_history = relationship("TrainingHistory", ...)`

**LabelingSession** - add timing:
- `started_at = Column(DateTime)`
- `total_labeling_time_ms = Column(Integer, default=0)`

**Files to modify:** `app/db/models.py`

---

## Phase 2: Demo Project Feature

### New File: `app/services/demo_generator.py`

```python
class DemoDataGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_people_dataset(self, n_records: int = 200) -> pd.DataFrame
    def generate_companies_dataset(self, n_records: int = 150) -> pd.DataFrame
    def create_duplicates_with_variations(self, df, duplicate_rate=0.3) -> pd.DataFrame
    def generate_labeled_pairs(self, source_df, target_df, n_pairs=30) -> List[LabeledPair]
```

### Demo Data Fields

**People dataset:**
| Field | Variations |
|-------|------------|
| first_name | Nicknames (Robert/Bob, William/Bill) |
| last_name | Typos, hyphenation |
| email | Missing dots, different domains |
| phone | Format variations |
| address | Abbreviations (Street/St) |
| city, state, zip | Typos, full vs abbreviated |

**Companies dataset:**
| Field | Variations |
|-------|------------|
| company_name | Inc/Inc./Incorporated variations |
| industry | Synonyms |
| address, city, state, zip | Same as people |
| phone, website | Format variations |

### API Endpoint
```python
# app/api/projects.py
@router.post("/demo", response_model=ProjectResponse)
def create_demo_project(demo_data: DemoProjectCreate, ...):
    # 1. Create project with is_demo=True
    # 2. Generate synthetic CSV files (stored in storage/demo/)
    # 3. Create Dataset records
    # 4. Set up column mappings and comparison config
    # 5. Create 20-30 pre-labeled pairs
```

### Schema Addition
```python
# app/schemas/project.py
class DemoProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    linkage_type: LinkageType = LinkageType.LINKAGE
    demo_domain: str  # "people" or "companies"
```

### Template Update: `app/templates/projects/create.html`
- Add checkbox: "Create as demo project"
- Add radio buttons for demo domain (visible when demo checked)
- Update JavaScript to toggle demo options and submit to `/api/projects/demo`

**Files to modify:**
- `app/db/models.py` (Project fields)
- `app/schemas/project.py` (DemoProjectCreate)
- `app/api/projects.py` (demo endpoint)
- `app/templates/projects/create.html` (UI)

**Files to create:**
- `app/services/demo_generator.py`

---

## Phase 3: Active Learning Enhancements

### New API Endpoints
```python
# app/api/labeling.py

@router.get("/sessions/{session_id}/detailed-progress")
def get_detailed_progress(session_id: int, ...):
    """Returns: total_candidates, labeled_count, estimated_remaining,
       total_time_ms, avg_time_per_pair, label_distribution"""

@router.get("/sessions/{session_id}/pair-explanation")
def get_pair_explanation(session_id: int, ...):
    """Returns: selection_reason, uncertainty_score, model_probability"""

@router.put("/projects/{project_id}/comparison-config")
def update_comparison_config(project_id: int, config: dict, ...):
    """Update mappings and comparison methods (immediate save)"""
```

### Schema Additions
```python
# app/schemas/labeling.py
class DetailedProgressResponse(BaseModel):
    total_candidates: int
    labeled_count: int
    estimated_remaining: int
    total_time_ms: int
    avg_time_per_pair_ms: float
    matches: int
    non_matches: int
    uncertain: int

class PairExplanation(BaseModel):
    selection_reason: str  # "uncertainty_sampling" or "random"
    uncertainty_score: Optional[float]
    model_probability: Optional[float]
```

### Active Learning Module Update
```python
# app/core/linkage/active_learning.py

def select_informative_pair(...) -> Tuple[pair, explanation]:
    # Return both the pair AND explanation for why it was selected

def count_candidate_pairs(source_df, target_df, blocking_config, labeled_pairs):
    # Estimate total candidate pairs for progress display
```

### Template Update: `app/templates/labeling/session.html`

**Enhanced Progress Section:**
```html
<!-- Progress stats -->
<div>Labeled: X / Y candidates (Z% complete)</div>
<div>Time: Xm Ys total | ~Xs per pair</div>
```

**"Why this pair?" Collapsible:**
```html
<details>
  <summary>Why this pair?</summary>
  <p>Selected via uncertainty sampling (score: 0.48)</p>
</details>
```

**Configuration Modal:**
```html
<div id="config-modal" class="hidden">
  <!-- Column Mappings Table -->
  <table>
    <tr><th>Source</th><th>Target</th><th>Method</th><th></th></tr>
    <!-- Dynamic rows with dropdowns -->
  </table>
  <button onclick="saveConfig()">Save</button>
</div>
```

**Files to modify:**
- `app/api/labeling.py` (new endpoints, timing tracking)
- `app/schemas/labeling.py` (new response schemas)
- `app/core/linkage/active_learning.py` (explanation, candidate counting)
- `app/templates/labeling/session.html` (UI enhancements)

---

## Phase 4: Model Statistics Tab

### ML Classifier Updates
```python
# app/core/linkage/ml_classifier.py

def fit(self, X, y, feature_names=None):
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(...)

    # Fit model
    self.model.fit(X_train, y_train)

    # Compute SEPARATE metrics
    self._train_metrics = compute_metrics(y_train, self.model.predict(X_train))
    self._test_metrics = compute_metrics(y_test, self.model.predict(X_test))

    # Confusion matrix on test set
    self._confusion_matrix = compute_confusion_matrix(y_test, y_pred)

def get_train_metrics(self) -> Dict[str, float]
def get_test_metrics(self) -> Dict[str, float]
def get_confusion_matrix(self) -> Dict[str, int]
```

### New API Endpoints
```python
# app/api/models.py

@router.get("/{model_id}/statistics")
def get_model_statistics(model_id: int, ...):
    """Returns: train_metrics, test_metrics, feature_importance,
       confusion_matrix, overfitting_warning"""

@router.get("/{model_id}/training-history")
def get_training_history(model_id: int, limit: int = 10, ...):
    """Returns: list of TrainingHistory entries for trending"""

@router.put("/{model_id}/history-retention")
def update_history_retention(model_id: int, retention: int, ...):
    """Update how many iterations to keep (configurable)"""
```

### Training Task Update
```python
# app/api/models.py - train_model_task()

def train_model_task(model_id: int, db_url: str):
    # ... train model ...

    # Record training history
    history = TrainingHistory(
        model_id=model_id,
        iteration=next_iteration,
        training_pairs_count=len(pairs),
        train_precision=classifier.get_train_metrics()['precision'],
        # ... other metrics ...
        feature_importance=classifier.get_feature_importance(),
        confusion_matrix=classifier.get_confusion_matrix()
    )
    db.add(history)

    # Enforce retention limit
    enforce_history_retention(model, db)
```

### Template Update: `app/templates/labeling/session.html`

**Tab Navigation:**
```html
<div class="tabs">
  <button onclick="showTab('labeling')" class="active">Labeling</button>
  <button onclick="showTab('model-stats')">Model Stats</button>
</div>
```

**Model Stats Tab Content:**
```html
<div id="model-stats-tab" class="hidden">
  <!-- Empty state if no model -->

  <!-- Metrics Cards: In-Sample vs Out-of-Sample -->
  <div class="grid grid-cols-2 gap-4">
    <div class="card">
      <h3>Training Set (In-Sample)</h3>
      <p>Precision: X% | Recall: Y% | F1: Z%</p>
    </div>
    <div class="card">
      <h3>Test Set (Out-of-Sample)</h3>
      <p>Precision: X% | Recall: Y% | F1: Z%</p>
      <!-- Empty if no test data -->
    </div>
  </div>

  <!-- Overfitting Warning -->
  <div class="warning" v-if="train_f1 > test_f1 + 0.15">
    Warning: Model may be overfitting
  </div>

  <!-- Training History Chart (Chart.js) -->
  <canvas id="history-chart"></canvas>

  <!-- Feature Importance Bar Chart -->
  <canvas id="importance-chart"></canvas>

  <!-- Confusion Matrix Grid -->
  <div class="confusion-matrix">
    <div>TP: X</div><div>FP: Y</div>
    <div>FN: Z</div><div>TN: W</div>
  </div>

  <!-- History Retention Setting -->
  <select onchange="updateRetention(this.value)">
    <option value="5">Keep last 5</option>
    <option value="10" selected>Keep last 10</option>
    <option value="20">Keep last 20</option>
    <option value="0">Keep all</option>
  </select>
</div>
```

**Files to modify:**
- `app/db/models.py` (TrainingHistory model, LinkageModel fields)
- `app/core/linkage/ml_classifier.py` (separate train/test metrics)
- `app/api/models.py` (statistics endpoints, history recording)
- `app/schemas/linkage.py` (new response schemas)
- `app/templates/labeling/session.html` (tabs, charts)

---

## Implementation Order

### Step 1: Database Schema (Foundation)
1. Add `TrainingHistory` model to `app/db/models.py`
2. Add demo fields to `Project` model
3. Add history fields to `LinkageModel` model
4. Add timing fields to `LabelingSession` model
5. Delete existing database and let it recreate

### Step 2: Demo Project Feature
1. Create `app/services/demo_generator.py`
2. Add `DemoProjectCreate` schema to `app/schemas/project.py`
3. Add demo endpoint to `app/api/projects.py`
4. Update `app/templates/projects/create.html`

### Step 3: Active Learning Enhancements
1. Add schemas to `app/schemas/labeling.py`
2. Update `app/core/linkage/active_learning.py` (explanation, counting)
3. Add endpoints to `app/api/labeling.py`
4. Update `app/templates/labeling/session.html` (progress, modal)

### Step 4: Model Statistics Tab
1. Update `app/core/linkage/ml_classifier.py` (train/test split metrics)
2. Add schemas to `app/schemas/linkage.py`
3. Add endpoints to `app/api/models.py`
4. Update training task to record history
5. Update `app/templates/labeling/session.html` (tabs, charts)

---

## Critical Files Summary

| File | Changes |
|------|---------|
| `app/db/models.py` | TrainingHistory model, Project/LinkageModel/LabelingSession fields |
| `app/services/demo_generator.py` | NEW - synthetic data generation |
| `app/api/projects.py` | Demo project endpoint |
| `app/api/labeling.py` | Detailed progress, pair explanation, config update endpoints |
| `app/api/models.py` | Statistics, history endpoints, training task update |
| `app/core/linkage/ml_classifier.py` | Separate train/test metrics, confusion matrix |
| `app/core/linkage/active_learning.py` | Selection explanation, candidate counting |
| `app/schemas/project.py` | DemoProjectCreate |
| `app/schemas/labeling.py` | DetailedProgressResponse, PairExplanation |
| `app/schemas/linkage.py` | TrainingHistoryResponse, ModelStatistics |
| `app/templates/projects/create.html` | Demo toggle, domain selector |
| `app/templates/labeling/session.html` | Progress stats, config modal, model stats tab |

---

## Dependencies

- **Chart.js** (CDN) - for training history and feature importance charts
- **Faker** (add to requirements.txt) - for realistic demo data generation

---

# Current Task: Labeling UX Improvements + Threshold Selection

## User Requirements

1. **Model score in Labeled Pairs view** - Show model prediction score for each labeled pair
2. **User-selectable thresholds in linkage** - Allow users to set match/non-match thresholds before running linkage
3. **Remove session concept from labeling** - Labeling should start immediately without needing to "start a session"
4. **Live strategy switching** - Strategy dropdown should be always visible and switchable
5. **Show labeled/total progress** - Display "X labeled out of Y total candidate pairs"

---

## Implementation Plan

### Part 1: Remove Labeling Session Concept

**Goal**: Labeling starts immediately when user visits the page. No "Start Session" step.

**Changes to `app/templates/labeling/session.html`:**
- Remove the "Start Labeling Session" section entirely
- Show the labeling interface and tab navigation immediately on page load
- Move strategy dropdown to the header/controls area (always visible)
- Load first pair automatically on page load
- Remove target_labels concept (labeling continues until user stops)

**Changes to `app/api/labeling.py`:**
- Modify or create endpoint that auto-creates/reuses session on first label submission
- Keep session internally for tracking but make it transparent to user
- New endpoint: `GET /api/labeling/projects/{project_id}/next-pair` that:
  - Auto-creates session if none exists
  - Returns next pair based on current strategy parameter
- Modify `submit_label` to accept strategy parameter and create session if needed

**Key behavior:**
- Strategy is a query parameter, not stored in session
- User can change strategy anytime via dropdown
- Session created lazily on first label submission

---

### Part 2: Model Score in Labeled Pairs History

**Goal**: Show current model's prediction for each labeled pair in the history view.

**Approach**: Compute on-the-fly using the active model (shows current predictions, not historical)

**Changes to `app/api/labeling.py` - `get_labeled_pairs` endpoint:**
```python
@router.get("/projects/{project_id}/labeled-pairs")
def get_labeled_pairs(
    project_id: int,
    limit: int = 100,
    offset: int = 0,
    label: Optional[str] = None,
    include_model_scores: bool = True,  # NEW - default True
    ...
):
    # ... existing query logic ...

    # If model scores requested and active model exists:
    if include_model_scores:
        active_model = db.query(LinkageModel).filter(
            LinkageModel.project_id == project_id,
            LinkageModel.is_active == True
        ).first()

        if active_model:
            classifier = load_model(active_model.model_path)
            for pair in pairs:
                if pair.comparison_vector:
                    X = [list(pair.comparison_vector.values())]
                    pair.model_score = classifier.predict_proba(X)[0]
```

**Changes to `app/templates/labeling/session.html` - History rendering:**
```javascript
// Add model score badge between comparison scores and label
${pair.model_score !== undefined ? `
  <span class="inline-flex items-center rounded px-2 py-0.5 text-xs font-medium
    ${pair.model_score >= 0.7 ? 'bg-green-100 text-green-700' :
      pair.model_score >= 0.4 ? 'bg-yellow-100 text-yellow-700' :
      'bg-red-100 text-red-700'}">
    Model: ${Math.round(pair.model_score * 100)}%
  </span>
` : ''}
```

---

### Part 3: User-Selectable Thresholds for Linkage

**Goal**: Allow users to set upper_threshold (match) and lower_threshold (review) before running linkage.

**Changes to `app/templates/linkage/run.html`:**
```html
<!-- Add threshold configuration section -->
<div class="threshold-config">
  <h3>Classification Thresholds</h3>
  <div>
    <label>Match threshold (above this = match)</label>
    <input type="range" min="0" max="1" step="0.05" value="0.8" id="upper-threshold">
    <span id="upper-value">0.80</span>
  </div>
  <div>
    <label>Non-match threshold (below this = non-match)</label>
    <input type="range" min="0" max="1" step="0.05" value="0.3" id="lower-threshold">
    <span id="lower-value">0.30</span>
  </div>
  <p class="help-text">Scores between thresholds will be marked for "review"</p>
</div>
```

**Changes to `app/api/linkage.py`:**
```python
class LinkageJobCreate(BaseModel):
    model_id: Optional[int] = None
    upper_threshold: float = 0.8  # NEW
    lower_threshold: float = 0.3  # NEW

# In run_linkage_job:
# Use user-provided thresholds instead of hardcoded 0.5
if score >= job.upper_threshold:
    classification = "match"
elif score <= job.lower_threshold:
    classification = "non_match"
else:
    classification = "review"
```

**Changes to `app/db/models.py` - LinkageJob:**
```python
upper_threshold = Column(Float, default=0.8)
lower_threshold = Column(Float, default=0.3)
```

---

### Part 4: Show Labeled vs Total Candidate Pairs

**Goal**: Display "23 labeled / 1,247 candidates" in the labeling UI.

**Already partially implemented** via `count_candidate_pairs()` in active_learning.py.

**Changes to `app/templates/labeling/session.html`:**
- Move the progress display to always be visible (not just after session starts)
- Show: "23 labeled / 1,247 total candidates (1.8%)"
- Update count after each label submission

**Changes to API:**
- Ensure `get_detailed_progress` or a new lightweight endpoint returns:
  - `total_labeled` (all time for this project, not just session)
  - `total_candidates` (from blocking)

---

## Files to Modify

| File | Changes |
|------|---------|
| `app/templates/labeling/session.html` | Remove start section, auto-load, strategy dropdown in header, model scores in history |
| `app/api/labeling.py` | Session-less pair fetching, model score computation, project-level progress |
| `app/templates/linkage/run.html` | Threshold sliders |
| `app/api/linkage.py` | Accept threshold parameters, use in classification |
| `app/db/models.py` | Add threshold fields to LinkageJob |
| `app/schemas/linkage.py` | Update LinkageJobCreate schema |

---

## Implementation Order

1. **Threshold selection in linkage** (isolated change)
   - Add fields to LinkageJob model
   - Update API to accept thresholds
   - Add UI sliders to run.html

2. **Remove session concept from labeling**
   - Refactor API for session-less operation
   - Update template to load immediately
   - Move strategy to always-visible dropdown

3. **Model scores in history**
   - Add model score computation to get_labeled_pairs
   - Update history rendering

4. **Progress display improvements**
   - Show project-level labeled count
   - Display total candidates from blocking

---

# Security Fixes - Critical Vulnerabilities

## Overview
Fix critical security vulnerabilities discovered during security assessment:
1. **Unprotected HTML page routes** (CRITICAL)
2. **XSS vulnerabilities via innerHTML** (CRITICAL)
3. **Insecure token storage in localStorage** (CRITICAL)
4. **Missing CSRF protection** (HIGH)
5. **Hardcoded secret keys** (HIGH)
6. **Debug mode enabled by default** (HIGH)

---

## Critical Issues Summary

### Issue #1: Unprotected HTML Routes ⚠️ CRITICAL
**Problem:** All page routes in `app/main.py` (lines 53-155) lack server-side authentication.
- No `Depends(get_current_active_user)` on HTML routes
- Anyone can load pages without authentication
- Security relies only on client-side JavaScript

**Impact:** Complete bypass of authentication by disabling JavaScript or using curl

---

### Issue #2: XSS via innerHTML ⚠️ CRITICAL
**Problem:** Multiple templates use `innerHTML` with unsanitized user data.
- `app/templates/dashboard/index.html` (lines 159-185)
- `app/templates/projects/list.html` (lines 61-65)
- `app/templates/labeling/session.html` (multiple locations)
- `app/templates/linkage/run.html` (multiple locations)

**Attack Vector:** User creates project with name `<script>alert(document.cookie)</script>` → XSS executes → steals tokens

---

### Issue #3: localStorage Token Storage ⚠️ CRITICAL
**Problem:** JWT tokens in localStorage (`app/templates/base.html`, lines 40-58) are vulnerable to XSS.
- Combined with Issue #2 = token theft possible
- localStorage accessible to all JavaScript
- Should use httpOnly cookies instead

---

### Issue #4: Missing CSRF Protection ⚠️ HIGH
**Problem:** No CSRF tokens on forms or API endpoint validation.
- All state-changing operations at risk
- Need FastAPI CSRF middleware

---

### Issue #5: Hardcoded Secret Keys ⚠️ HIGH
**Problem:** Development keys in production code.
- `app/config.py` line 11: `"dev-secret-key-change-in-production"`
- `.env` line 6: `"your-secret-key-change-in-production"`
- Anyone can forge JWT tokens with known key

---

### Issue #6: Debug Mode Enabled ⚠️ HIGH
**Problem:** Debug mode exposes stack traces and internals.
- `app/config.py` line 8: `debug: bool = True`
- `.env` line 3: `DEBUG=true`

---

## Implementation Plan

### Phase 1: Protect HTML Routes (HIGHEST PRIORITY)

#### Step 1.1: Create Redirect-Based Auth Dependency
**File:** `app/api/deps.py`

Add new function (after existing dependencies):
```python
from fastapi.responses import RedirectResponse
from typing import Union

async def get_current_user_or_redirect(
    request: Request,
    db: Session = Depends(get_db)
) -> Union[User, RedirectResponse]:
    """
    Get current user for HTML pages, redirect to login if not authenticated.
    Checks httpOnly cookie first, then Authorization header as fallback.
    """
    # Try cookie first (httpOnly, more secure)
    token = request.cookies.get("access_token")

    if not token:
        # Fallback: check Authorization header (for API compatibility)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]

    if not token:
        return RedirectResponse(url="/login", status_code=303)

    try:
        # Remove "Bearer " prefix if cookie format
        if token.startswith("Bearer "):
            token = token[7:]

        payload = decode_token(token)
        if payload is None:
            return RedirectResponse(url="/login", status_code=303)

        user_id = payload.get("sub")
        if user_id is None:
            return RedirectResponse(url="/login", status_code=303)

        user = db.query(User).filter(User.id == int(user_id)).first()
        if user is None or not user.is_active:
            return RedirectResponse(url="/login", status_code=303)

        return user
    except Exception:
        return RedirectResponse(url="/login", status_code=303)
```

#### Step 1.2: Add Authentication to All HTML Routes
**File:** `app/main.py` (lines 53-118)

Add `current_user` parameter to EVERY protected route:

```python
from app.api.deps import get_current_user_or_redirect
from app.db.models import User

@app.get("/dashboard")
async def dashboard_page(
    request: Request,
    current_user: User = Depends(get_current_user_or_redirect)
):
    return templates.TemplateResponse("dashboard/index.html", {
        "request": request,
        "user": current_user  # Can use in template if needed
    })
```

**Apply to all routes:**
- `/dashboard` (line 53)
- `/projects` (line 58)
- `/projects/new` (line 63)
- `/projects/{project_id}` (line 68)
- `/projects/{project_id}/configure` (line 73)
- `/datasets` (line 78)
- `/datasets/upload` (line 83)
- `/datasets/{dataset_id}` (line 88)
- `/datasets/{dataset_id}/preview` (line 93)
- `/linkage/jobs` (line 98)
- `/linkage/jobs/{job_id}` (line 103)
- `/labeling/projects/{project_id}` (line 108)
- `/models` (line 119)
- `/models/{model_id}` (line 124)

**Keep UNPROTECTED:**
- `/` (redirect)
- `/login` (line 129)
- `/register` (line 138)

---

### Phase 2: Fix XSS Vulnerabilities

#### Step 2.1: Add Safe HTML Utilities
**File:** `app/templates/base.html`

Add after auth functions (around line 60):

```javascript
/**
 * Safely escape HTML to prevent XSS attacks.
 * Use this when you must use innerHTML.
 */
function escapeHtml(unsafe) {
    if (unsafe === null || unsafe === undefined) {
        return '';
    }
    return String(unsafe)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

/**
 * Create DOM elements safely (preferred over innerHTML).
 */
function createSafeElement(tag, text, className = '') {
    const el = document.createElement(tag);
    el.textContent = text;  // XSS-safe
    if (className) el.className = className;
    return el;
}
```

#### Step 2.2: Fix Dashboard XSS
**File:** `app/templates/dashboard/index.html` (lines 159-185)

Replace innerHTML with safe DOM manipulation:

```javascript
// BEFORE (UNSAFE):
projectsList.innerHTML = projects.map(project => `
    <a href="/projects/${project.id}">
        <p>${project.name}</p>
    </a>
`).join('');

// AFTER (SAFE):
projectsList.innerHTML = '';  // Clear
projects.forEach(project => {
    const link = document.createElement('a');
    link.href = `/projects/${project.id}`;
    link.className = 'block hover:bg-gray-50';

    const container = document.createElement('div');
    container.className = 'px-4 py-4 sm:px-6';

    const nameP = document.createElement('p');
    nameP.className = 'truncate text-sm font-medium text-indigo-600';
    nameP.textContent = project.name;  // XSS-safe!

    const descP = document.createElement('p');
    descP.className = 'flex items-center text-sm text-gray-500';
    descP.textContent = project.description || 'No description';  // XSS-safe!

    container.appendChild(nameP);
    container.appendChild(descP);
    link.appendChild(container);
    projectsList.appendChild(link);
});
```

#### Step 2.3: Fix Projects List XSS
**File:** `app/templates/projects/list.html` (around lines 61-65)

Replace innerHTML with textContent:
```javascript
// Find all innerHTML usages with user data
// Replace with: element.textContent = userInput
```

#### Step 2.4: Fix Labeling/Linkage XSS
**Files:**
- `app/templates/labeling/session.html`
- `app/templates/linkage/run.html`

**Pattern:** Replace `element.innerHTML = userInput` with `element.textContent = userInput` for ALL user-generated content (names, descriptions, etc.).

---

### Phase 3: Move to HttpOnly Cookies

#### Step 3.1: Update Login to Set Cookies
**File:** `app/api/auth.py`

Update login endpoint (lines 94-117):

```python
from fastapi import Response

@router.post("/login")
def login(
    response: Response,  # Add this parameter
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    # ... existing validation code ...

    access_token = create_access_token(data={"sub": str(user.id)})

    # Set httpOnly cookie (prevents JavaScript access)
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,    # Blocks JavaScript access!
        secure=False,     # Set True in production (HTTPS only)
        samesite="lax",   # CSRF protection
        max_age=1800      # 30 minutes in seconds
    )

    # Still return token for backward compatibility
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": user.id, "email": user.email, "full_name": user.full_name}
    }
```

**Also update `login_json` endpoint (lines 120-139)** with same cookie logic.

#### Step 3.2: Add Logout Endpoint
**File:** `app/api/auth.py`

Add after login endpoints:

```python
@router.post("/logout")
def logout(response: Response):
    """Logout by clearing auth cookie."""
    response.delete_cookie(key="access_token")
    return {"message": "Logged out successfully"}
```

#### Step 3.3: Update Frontend Logout
**File:** `app/templates/base.html` (around line 70)

Update logout function:

```javascript
function logout() {
    // Clear localStorage (backward compat)
    localStorage.removeItem(AUTH_TOKEN_KEY);

    // Call logout endpoint to clear cookie
    fetch('/api/auth/logout', {
        method: 'POST',
        headers: {'Authorization': 'Bearer ' + (getAuthToken() || '')}
    }).finally(() => {
        window.location.href = '/login';
    });
}
```

**Note:** Keep localStorage functions for backward compatibility with API calls that still need the token.

---

### Phase 4: Add CSRF Protection

#### Step 4.1: Install Package
**File:** `requirements.txt`

Add:
```
fastapi-csrf-protect>=0.3.0
```

#### Step 4.2: Configure CSRF
**File:** `app/config.py`

Add field:
```python
csrf_secret_key: str = Field(
    default="csrf-secret-change-in-production",
    env="CSRF_SECRET_KEY"
)
```

#### Step 4.3: Add CSRF Middleware
**File:** `app/main.py`

Add after imports:

```python
from fastapi_csrf_protect import CsrfProtect
from fastapi_csrf_protect.exceptions import CsrfProtectError
from pydantic import BaseModel

class CsrfSettings(BaseModel):
    secret_key: str = get_settings().csrf_secret_key
    cookie_samesite: str = 'lax'

@CsrfProtect.load_config
def get_csrf_config():
    return CsrfSettings()

@app.exception_handler(CsrfProtectError)
def csrf_protect_exception_handler(request: Request, exc: CsrfProtectError):
    return JSONResponse(status_code=403, content={"detail": "CSRF validation failed"})
```

#### Step 4.4: Add CSRF Token Endpoint
**File:** `app/api/auth.py`

Add:
```python
from fastapi_csrf_protect import CsrfProtect

@router.get("/csrf-token")
async def get_csrf_token(csrf_protect: CsrfProtect = Depends()):
    """Generate CSRF token for forms."""
    csrf_token, signed_token = csrf_protect.generate_csrf_tokens()
    response = JSONResponse(content={"csrf_token": csrf_token})
    csrf_protect.set_csrf_cookie(signed_token, response)
    return response
```

#### Step 4.5: Update Frontend Fetch
**File:** `app/templates/base.html`

Add CSRF helper:

```javascript
async function getCsrfToken() {
    try {
        const response = await fetch('/api/auth/csrf-token');
        const data = await response.json();
        return data.csrf_token;
    } catch (error) {
        console.error('Failed to get CSRF token:', error);
        return null;
    }
}

// Create authenticated fetch wrapper
async function authenticatedFetch(url, options = {}) {
    const token = getAuthToken();
    const headers = {'Authorization': 'Bearer ' + token, ...options.headers};

    // Add CSRF token for state-changing requests
    if (['POST', 'PUT', 'DELETE', 'PATCH'].includes(options.method?.toUpperCase())) {
        const csrfToken = await getCsrfToken();
        if (csrfToken) headers['X-CSRF-Token'] = csrfToken;
    }

    return fetch(url, {...options, headers});
}
```

**Usage:** Replace `fetch()` calls with `authenticatedFetch()` for POST/PUT/DELETE requests.

---

### Phase 5: Fix Configuration Issues

#### Step 5.1: Require Secret Key
**File:** `app/config.py` (line 11)

Change:
```python
# BEFORE:
secret_key: str = "dev-secret-key-change-in-production"

# AFTER:
secret_key: str = Field(
    ...,  # Required, no default!
    env="SECRET_KEY",
    description="Secret key for JWT - MUST be set"
)
```

#### Step 5.2: Disable Debug by Default
**File:** `app/config.py` (line 8)

Change:
```python
# BEFORE:
debug: bool = True

# AFTER:
debug: bool = Field(default=False, env="DEBUG")
```

#### Step 5.3: Reduce Token Expiration
**File:** `app/config.py` (line 13)

Change:
```python
# BEFORE:
access_token_expire_minutes: int = 1440  # 24 hours

# AFTER:
access_token_expire_minutes: int = Field(
    default=30,  # 30 minutes
    env="ACCESS_TOKEN_EXPIRE_MINUTES"
)
```

#### Step 5.4: Add Startup Validation
**File:** `app/main.py`

Add event handler:

```python
@app.on_event("startup")
async def validate_security_config():
    """Validate security configuration on startup."""
    settings = get_settings()

    # Check for insecure default keys
    insecure_patterns = [
        "dev-secret", "your-secret", "change-in-production",
        "secret", "password", "test", "demo"
    ]

    if any(pattern in settings.secret_key.lower() for pattern in insecure_patterns):
        if not settings.debug:
            raise RuntimeError(
                "SECURITY ERROR: Using default/weak SECRET_KEY in production! "
                "Set a strong SECRET_KEY environment variable (32+ chars)."
            )
        else:
            print("⚠️  WARNING: Using default SECRET_KEY in debug mode")

    if settings.debug:
        print("⚠️  WARNING: Debug mode enabled - disable in production!")

    if len(settings.secret_key) < 32:
        print("⚠️  WARNING: SECRET_KEY should be at least 32 characters")
```

#### Step 5.5: Create .env.example
**File:** `.env.example` (create new)

```bash
# Database
DATABASE_URL=sqlite:///./app.db

# Security - REQUIRED (generate strong random keys!)
SECRET_KEY=your-secret-key-MUST-CHANGE-IN-PRODUCTION-min-32-chars
CSRF_SECRET_KEY=your-csrf-secret-MUST-CHANGE-IN-PRODUCTION

# Debug mode (MUST be false in production)
DEBUG=false

# Token expiration (minutes)
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Storage
STORAGE_PATH=./storage
```

#### Step 5.6: Update .gitignore
**File:** `.gitignore`

Ensure:
```
.env
.env.local
.env.production
```

---

### Phase 6: Password Complexity (Bonus)

#### Step 6.1: Add Validation Function
**File:** `app/api/auth.py` (before register endpoint, around line 50)

```python
import re

def validate_password_strength(password: str) -> tuple[bool, str]:
    """Validate password meets complexity requirements."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if len(password) > 128:
        return False, "Password too long (max 128 characters)"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain lowercase letter"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain uppercase letter"
    if not re.search(r"\d", password):
        return False, "Password must contain digit"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain special character"

    # Check common weak passwords
    weak = ['password', '12345678', 'qwerty', 'abc123', 'letmein']
    if password.lower() in weak:
        return False, "Password too common, choose stronger password"

    return True, ""
```

#### Step 6.2: Apply in Register
**File:** `app/api/auth.py` (register endpoint, around line 56)

Add validation:
```python
@router.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    # ... existing email check ...

    # Validate password strength
    is_valid, error_msg = validate_password_strength(user.password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # ... rest of registration ...
```

---

## Testing & Verification

### Manual Testing

1. **Test Protected Routes:**
   ```bash
   # Should redirect to login
   curl -I http://localhost:8000/dashboard

   # Should work with cookie
   curl -I -b "access_token=Bearer_YOUR_TOKEN" http://localhost:8000/dashboard
   ```

2. **Test XSS Prevention:**
   - Create project with name: `<script>alert('XSS')</script>`
   - View dashboard - should show literal text, not execute script

3. **Test Cookie Auth:**
   - Login via browser
   - Check DevTools > Application > Cookies
   - Verify `access_token` has `HttpOnly` flag
   - Try `document.cookie` in console (shouldn't show auth token)

4. **Test CSRF:**
   ```bash
   # Without CSRF token (should fail)
   curl -X POST http://localhost:8000/api/projects \
     -H "Authorization: Bearer TOKEN" -d '{"name":"test"}'
   ```

5. **Test Password Strength:**
   - Try registering with: `password` (should fail)
   - Try registering with: `MyStr0ng!Pass` (should succeed)

6. **Test Config Validation:**
   ```bash
   export SECRET_KEY="weak"
   export DEBUG=false
   python -m uvicorn app.main:app  # Should fail to start
   ```

### Automated Tests

Create: `tests/test_security.py`

```python
"""Security tests."""
import pytest

class TestRouteProtection:
    def test_dashboard_requires_auth(self, client):
        response = client.get("/dashboard", follow_redirects=False)
        assert response.status_code == 303
        assert "/login" in response.headers["location"]

    def test_authenticated_access_works(self, client, auth_headers):
        client.cookies.set("access_token", f"Bearer {auth_headers['Authorization'].split()[1]}")
        response = client.get("/dashboard")
        assert response.status_code == 200

class TestXSSPrevention:
    def test_project_name_xss_escaped(self, client, auth_headers, test_user):
        malicious_name = "<script>alert('XSS')</script>"
        response = client.post("/api/projects", json={
            "name": malicious_name,
            "linkage_type": "linkage",
            "organization_id": test_user.memberships[0].organization_id
        }, headers=auth_headers)
        assert response.status_code == 201

        # Name stored as-is (will be escaped in frontend)
        projects = client.get("/api/projects", headers=auth_headers).json()
        assert any(p["name"] == malicious_name for p in projects)

class TestPasswordSecurity:
    def test_weak_password_rejected(self, client):
        response = client.post("/api/auth/register", json={
            "email": "test@example.com",
            "password": "password",
            "full_name": "Test"
        })
        assert response.status_code == 400

    def test_strong_password_accepted(self, client):
        response = client.post("/api/auth/register", json={
            "email": "new@example.com",
            "password": "MyStr0ng!Password123",
            "full_name": "Test"
        })
        assert response.status_code == 201

class TestCookieAuth:
    def test_login_sets_cookie(self, client, test_user):
        response = client.post("/api/auth/login", data={
            "username": test_user.email,
            "password": "testpass123"
        })
        assert "access_token" in response.cookies

    def test_logout_clears_cookie(self, client, auth_headers):
        response = client.post("/api/auth/logout", headers=auth_headers)
        assert response.status_code == 200
```

Run:
```bash
python -m pytest tests/test_security.py -v
python -m pytest tests/ -v  # All tests
```

---

## Files Modified

| File | Changes |
|------|---------|
| `app/api/deps.py` | Add `get_current_user_or_redirect` |
| `app/main.py` | Add auth to HTML routes, CSRF config, startup validation |
| `app/api/auth.py` | Cookie auth, logout endpoint, CSRF token, password validation |
| `app/config.py` | Require secret key, disable debug, reduce token expiration, add CSRF secret |
| `app/templates/base.html` | Add `escapeHtml()`, `createSafeElement()`, update logout, add CSRF fetch |
| `app/templates/dashboard/index.html` | Replace innerHTML with safe DOM |
| `app/templates/projects/list.html` | Replace innerHTML with textContent |
| `app/templates/labeling/session.html` | Fix XSS in record display |
| `app/templates/linkage/run.html` | Fix XSS in results display |
| `requirements.txt` | Add `fastapi-csrf-protect` |
| `.env.example` | Create with secure defaults |
| `.gitignore` | Ensure .env excluded |
| `tests/test_security.py` | Create comprehensive security tests |

---

## Deployment Checklist

- [ ] Set strong `SECRET_KEY` env var (32+ random chars)
- [ ] Set strong `CSRF_SECRET_KEY` env var
- [ ] Set `DEBUG=false` in production
- [ ] Set `secure=True` for cookies (HTTPS)
- [ ] Update CORS settings if needed
- [ ] Run security tests
- [ ] Test auth flows after deployment
- [ ] Monitor logs for CSRF/auth errors

---

## Future Enhancements (Not in this PR)

1. Refresh tokens for longer sessions
2. Rate limiting (slowapi)
3. Account lockout after failed attempts
4. 2FA/MFA support
5. Content Security Policy (CSP) headers
6. Security headers middleware (HSTS, X-Frame-Options)
7. Password reset with secure tokens
8. Audit logging for security events
9. IP-based rate limiting
