"""
Pytest fixtures for ML API tests
Creates test JWTs directly (no login endpoint — auth is external)
"""
import os
import sys
import pytest
from datetime import datetime, timedelta
from jose import jwt
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set test environment variables before importing app
os.environ['JWT_SECRET_KEY'] = 'test-secret-key-for-testing-only'
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'


def create_test_token(user_id: str, roles: list = None, expired: bool = False) -> str:
    """Create a test JWT token directly (simulating .NET backend token)."""
    now = datetime.utcnow()
    payload = {
        "sub": user_id,
        "role": roles or [],
        "iat": now,
        "exp": now + timedelta(hours=-1 if expired else 1),
    }
    return jwt.encode(payload, os.environ['JWT_SECRET_KEY'], algorithm="HS256")


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI app."""
    from ml_api.main import app
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def auth_headers():
    """Get authentication headers with a valid user token."""
    token = create_test_token("test-user-123")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def admin_auth_headers():
    """Get authentication headers with a valid admin token."""
    token = create_test_token("admin-user-1", roles=["Admin"])
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_image_bytes():
    """Create a minimal valid JPEG image for testing."""
    import base64
    jpeg_b64 = (
        "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"
        "Ew8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJ"
        "CQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
        "MjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/"
        "xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAB//2Q=="
    )
    return base64.b64decode(jpeg_b64)


@pytest.fixture
def invalid_file_bytes():
    """Create invalid file content for testing."""
    return b"This is not a valid image file"
