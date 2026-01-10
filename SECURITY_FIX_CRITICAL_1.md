# CRITICAL-1 Security Fix: Unsafe Deserialization

## Vulnerability Summary

**Severity:** CRITICAL
**CWE:** CWE-502 (Deserialization of Untrusted Data)
**Status:** ✅ FIXED

### Original Vulnerability

The application was vulnerable to Remote Code Execution (RCE) through unsafe deserialization in the ML classifier loading function:

```python
# VULNERABLE CODE (app/core/linkage/ml_classifier.py:348)
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

---

## Fix Implemented

### Solution: Cryptographic Model Signing

Implemented HMAC-SHA256 signing and verification for all ML models to prevent tampering and ensure integrity.

### Changes Made

#### 1. Configuration Update ([app/config.py](app/config.py))

Added required `MODEL_SECRET_KEY` setting:

```python
# Model security
model_secret_key: str = Field(
    ...,  # Required, no default!
    env="MODEL_SECRET_KEY",
    description="Secret key for model signing - MUST be set"
)
```

#### 2. Model Signing Implementation ([app/core/linkage/ml_classifier.py](app/core/linkage/ml_classifier.py))

**Added Functions:**

- `_sign_model(model_path, secret_key)` - Creates HMAC-SHA256 signature
- `_verify_model_signature(model_path, secret_key)` - Verifies signature before loading

**Updated Functions:**

- `save_classifier()` - Automatically signs models after saving
- `load_classifier()` - Verifies signature before loading (prevents RCE)

**Security Features:**

- Uses HMAC-SHA256 for cryptographic signing
- Constant-time comparison (`hmac.compare_digest`) to prevent timing attacks
- Signature file stored as `{model_path}.sig`
- Raises `ValueError` if signature is missing or invalid

#### 3. Environment Configuration

**Updated files:**
- [.env](.env) - Added `MODEL_SECRET_KEY` for development
- [.env.example](.env.example) - Added `MODEL_SECRET_KEY` placeholder for production

#### 4. Startup Validation ([app/main.py](app/main.py))

Added validation for `MODEL_SECRET_KEY`:
- Checks for weak/default keys in production
- Enforces minimum 32-character length
- Fails fast on startup if insecure

---

## Testing

### Unit Tests

Created [test_model_signing.py](test_model_signing.py) to verify:

1. ✅ Valid signatures are accepted
2. ✅ Missing signatures are detected
3. ✅ Tampered models are rejected
4. ✅ Wrong secret keys are detected

**Test Results:**
```
Testing model signing and verification...

1. Testing valid signature...
   ✓ Valid signature verified successfully

2. Testing missing signature detection...
   ✓ Correctly detected missing signature

3. Testing tampered model detection...
   ✓ Correctly detected tampering

4. Testing wrong secret key detection...
   ✓ Correctly detected wrong key

✅ All tests passed! Model signing is working correctly.
```

### Backward Compatibility

**⚠️ BREAKING CHANGE:** Existing unsigned models will fail to load.

**Migration Required:**
1. Re-train all existing models with the new signing system
2. Or manually sign existing models using the `_sign_model()` function

---

## How It Works

### Model Saving Flow

```
1. Train classifier
2. Save model to disk with joblib.dump()
3. Read model file as bytes
4. Compute HMAC-SHA256(model_bytes, MODEL_SECRET_KEY)
5. Save signature to {model_path}.sig
```

### Model Loading Flow

```
1. Check if {model_path}.sig exists (fail if missing)
2. Read model file as bytes
3. Read expected signature from .sig file
4. Compute HMAC-SHA256(model_bytes, MODEL_SECRET_KEY)
5. Compare signatures using constant-time comparison
6. If valid: load model with joblib.load()
7. If invalid: raise ValueError (prevent loading)
```

### Security Properties

- **Integrity**: Detects any modification to model files
- **Authenticity**: Verifies model was created by authorized system
- **No RCE**: Tampered models cannot be loaded, preventing code execution
- **Defense in Depth**: Even if attacker gains file write access, they cannot create valid signatures without the secret key

---

## Production Deployment

### Required Steps

1. **Generate Strong Secret Key:**
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Set Environment Variable:**
   ```bash
   export MODEL_SECRET_KEY="<your-generated-key>"
   ```

3. **Verify Configuration:**
   - Application will fail to start if `MODEL_SECRET_KEY` is not set
   - Warning will appear if using weak/default key

4. **Re-train Models:**
   - All existing models must be re-trained or manually signed
   - Unsigned models will fail to load with clear error message

### Security Best Practices

- Store `MODEL_SECRET_KEY` securely (environment variable, secrets manager)
- Use different keys for development and production
- Rotate keys periodically and re-sign all models
- Monitor logs for signature verification failures (may indicate attack attempts)

---

## Files Modified

| File | Changes |
|------|---------|
| [app/config.py](app/config.py) | Added `model_secret_key` field |
| [app/core/linkage/ml_classifier.py](app/core/linkage/ml_classifier.py) | Added signing/verification, updated save/load |
| [app/main.py](app/main.py) | Added MODEL_SECRET_KEY validation |
| [.env](.env) | Added MODEL_SECRET_KEY for development |
| [.env.example](.env.example) | Added MODEL_SECRET_KEY placeholder |
| [test_model_signing.py](test_model_signing.py) | Created security test suite |

---

## Next Steps

See [plans/security-audit-report.md](plans/security-audit-report.md) for remaining vulnerabilities to fix:

- **CRITICAL-2**: Remove token from query parameters (export endpoint)
- **CRITICAL-3**: Add file size validation (DoS prevention)
- **HIGH-1 to HIGH-6**: Cookie security, hardcoded credentials, XSS, CORS, file validation
- **MEDIUM-1 to MEDIUM-7**: Rate limiting, CSRF, security headers, etc.

---

## References

- [CWE-502: Deserialization of Untrusted Data](https://cwe.mitre.org/data/definitions/502.html)
- [OWASP: Deserialization Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Deserialization_Cheat_Sheet.html)
- [Python HMAC Documentation](https://docs.python.org/3/library/hmac.html)
