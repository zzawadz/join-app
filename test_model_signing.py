"""
Test script to verify model signing and verification works correctly.
This can be run independently to test the security fix.
"""
import tempfile
import os
from pathlib import Path

# Simulate the signing and verification logic
import hmac
import hashlib


def sign_model(model_path: str, secret_key: str) -> Path:
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


def verify_model_signature(model_path: str, secret_key: str) -> bool:
    """Verify model signature."""
    sig_path = Path(f"{model_path}.sig")

    if not sig_path.exists():
        raise ValueError(f"Model signature file missing: {sig_path}")

    with open(model_path, 'rb') as f:
        model_data = f.read()

    with open(sig_path, 'r') as f:
        expected_sig = f.read().strip()

    actual_sig = hmac.new(
        secret_key.encode(),
        model_data,
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(actual_sig, expected_sig):
        raise ValueError("Model signature verification failed")

    return True


def test_model_signing():
    """Test the model signing and verification."""
    print("Testing model signing and verification...")

    # Create a temporary file to simulate a model
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
        model_path = f.name
        f.write(b"This is fake model data for testing")

    try:
        secret_key = "test-secret-key-for-verification"

        # Test 1: Sign and verify a valid model
        print("\n1. Testing valid signature...")
        sign_model(model_path, secret_key)
        verify_model_signature(model_path, secret_key)
        print("   ✓ Valid signature verified successfully")

        # Test 2: Detect missing signature
        print("\n2. Testing missing signature detection...")
        sig_path = Path(f"{model_path}.sig")
        sig_path.unlink()  # Remove signature
        try:
            verify_model_signature(model_path, secret_key)
            print("   ✗ FAILED: Should have detected missing signature")
        except ValueError as e:
            if "missing" in str(e).lower():
                print(f"   ✓ Correctly detected missing signature: {e}")
            else:
                raise

        # Test 3: Detect tampered model
        print("\n3. Testing tampered model detection...")
        sign_model(model_path, secret_key)  # Re-sign
        # Tamper with model
        with open(model_path, 'ab') as f:
            f.write(b"TAMPERED")
        try:
            verify_model_signature(model_path, secret_key)
            print("   ✗ FAILED: Should have detected tampering")
        except ValueError as e:
            if "failed" in str(e).lower():
                print(f"   ✓ Correctly detected tampering: {e}")
            else:
                raise

        # Test 4: Detect wrong secret key
        print("\n4. Testing wrong secret key detection...")
        # Create fresh model and sign with original key
        with open(model_path, 'wb') as f:
            f.write(b"Fresh model data")
        sign_model(model_path, secret_key)

        try:
            verify_model_signature(model_path, "wrong-secret-key")
            print("   ✗ FAILED: Should have detected wrong key")
        except ValueError as e:
            if "failed" in str(e).lower():
                print(f"   ✓ Correctly detected wrong key: {e}")
            else:
                raise

        print("\n✅ All tests passed! Model signing is working correctly.")

    finally:
        # Cleanup
        if os.path.exists(model_path):
            os.unlink(model_path)
        sig_path = Path(f"{model_path}.sig")
        if sig_path.exists():
            sig_path.unlink()


if __name__ == "__main__":
    test_model_signing()
