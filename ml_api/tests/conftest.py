"""
Pytest fixtures for ML API tests
"""
import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set test environment variables before importing app
os.environ['JWT_SECRET_KEY'] = 'test-secret-key-for-testing-only'
os.environ['ADMIN_USERNAME'] = 'testadmin'
os.environ['ADMIN_EMAIL'] = 'testadmin@test.local'
os.environ['ADMIN_PASSWORD'] = 'testpassword123'
os.environ['DATABASE_PATH'] = ':memory:'


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI app."""
    from ml_api.main import app
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def auth_headers(client):
    """Get authentication headers with valid token."""
    response = client.post(
        "/auth/login",
        json={
            "username": os.environ['ADMIN_USERNAME'],
            "password": os.environ['ADMIN_PASSWORD']
        }
    )
    if response.status_code == 200:
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    return {}


@pytest.fixture
def sample_image_bytes():
    """Create a minimal valid JPEG image for testing."""
    # Minimal valid JPEG (1x1 red pixel)
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
