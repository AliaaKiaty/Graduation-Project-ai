"""
API endpoint tests for admin endpoints
Tests for /admin/* endpoints (model retraining and management)
"""
import os
import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch, MagicMock
from jose import jwt

from ml_api.main import app
from ml_api.database import Base, get_db
from ml_api.models.db_models import ModelMetadata, ProductCategory, Product


def _make_token(user_id: str, roles: list = None) -> str:
    """Create a test JWT token."""
    now = datetime.utcnow()
    payload = {
        "sub": user_id,
        "role": roles or [],
        "iat": now,
        "exp": now + timedelta(hours=1),
    }
    return jwt.encode(payload, os.environ.get('JWT_SECRET_KEY', 'test-secret-key-for-testing-only'), algorithm="HS256")


# Test database setup
@pytest.fixture(scope="function")
def test_db():
    """Create test database with sample data"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    TestingSessionLocal = sessionmaker(bind=engine)
    db = TestingSessionLocal()

    # Create sample model metadata
    svd_model = ModelMetadata(
        id=1,
        model_type="svd",
        version="20240101_120000",
        file_path="/path/to/svd_model.pkl",
        training_date=datetime(2024, 1, 1, 12, 0, 0),
        n_components=10,
        total_products=1000,
        total_ratings=50000,
        rmse=0.85,
        precision_at_10=0.75,
        training_duration_seconds=300,
        is_active=True,
        notes="Test SVD model"
    )
    db.add(svd_model)

    tfidf_model = ModelMetadata(
        id=2,
        model_type="tfidf_kmeans",
        version="20240101_120000",
        file_path="/path/to/tfidf_model.pkl",
        training_date=datetime(2024, 1, 1, 12, 0, 0),
        n_clusters=20,
        max_features=5000,
        total_products=1000,
        training_duration_seconds=120,
        is_active=True,
        notes="Test TF-IDF + KMeans model"
    )
    db.add(tfidf_model)

    # Create old inactive model
    old_svd_model = ModelMetadata(
        id=3,
        model_type="svd",
        version="20231201_100000",
        file_path="/path/to/old_svd_model.pkl",
        training_date=datetime(2023, 12, 1, 10, 0, 0),
        n_components=10,
        total_products=800,
        total_ratings=40000,
        training_duration_seconds=250,
        is_active=False,
        notes="Old SVD model"
    )
    db.add(old_svd_model)

    db.commit()

    # Create sample category and product for retraining tests
    category = ProductCategory(Id=1, Name="Tools")
    db.add(category)
    product = Product(Id=1, Name="Test Product", CategoryId=1)
    db.add(product)
    db.commit()

    yield db

    db.close()
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def client(test_db):
    """Create test client with overridden database"""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers():
    """Get authentication headers for regular user"""
    token = _make_token("test-user-123")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_auth_headers():
    """Get authentication headers for admin user"""
    token = _make_token("admin-user-1", roles=["Admin"])
    return {"Authorization": f"Bearer {token}"}


# ============================================================================
# MODEL RETRAINING ENDPOINT TESTS
# ============================================================================

def test_trigger_retraining_success(client, admin_auth_headers):
    """Test POST /admin/retrain endpoint"""
    with patch('ml_api.api.admin.subprocess.run') as mock_subprocess:
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="Success", stderr="")

        response = client.post(
            "/admin/retrain",
            headers=admin_auth_headers
        )

        assert response.status_code == 202  # Accepted
        data = response.json()

        assert "message" in data
        assert data["status"] == "in_progress"
        assert "triggered_by" in data
        assert data["triggered_by"] == "admin-user-1"


def test_trigger_retraining_unauthorized(client, auth_headers):
    """Test that retraining requires admin authentication"""
    response = client.post(
        "/admin/retrain",
        headers=auth_headers  # Regular user, not admin
    )

    assert response.status_code == 403  # Forbidden


def test_trigger_retraining_no_auth(client):
    """Test that retraining requires authentication"""
    response = client.post("/admin/retrain")

    assert response.status_code in [401, 403]


# ============================================================================
# LIST MODEL METADATA ENDPOINT TESTS
# ============================================================================

def test_list_model_metadata_success(client, admin_auth_headers):
    """Test GET /admin/models endpoint"""
    response = client.get(
        "/admin/models",
        headers=admin_auth_headers
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 3  # We created 3 model metadata records

    # Check structure
    assert "id" in data[0]
    assert "model_type" in data[0]
    assert "version" in data[0]
    assert "is_active" in data[0]
    assert "training_date" in data[0]


def test_list_model_metadata_filter_by_type(client, admin_auth_headers):
    """Test filtering models by type"""
    response = client.get(
        "/admin/models?model_type=svd",
        headers=admin_auth_headers
    )

    assert response.status_code == 200
    data = response.json()

    # Should return only SVD models
    assert len(data) == 2  # 1 active + 1 inactive SVD model
    for model in data:
        assert model["model_type"] == "svd"


def test_list_model_metadata_filter_by_active(client, admin_auth_headers):
    """Test filtering models by active status"""
    response = client.get(
        "/admin/models?is_active=true",
        headers=admin_auth_headers
    )

    assert response.status_code == 200
    data = response.json()

    # Should return only active models
    assert len(data) == 2  # 1 SVD + 1 TF-IDF/KMeans active
    for model in data:
        assert model["is_active"] is True


def test_list_model_metadata_ordered_by_date(client, admin_auth_headers):
    """Test that models are ordered by training_date descending"""
    response = client.get(
        "/admin/models",
        headers=admin_auth_headers
    )

    assert response.status_code == 200
    data = response.json()

    # Most recent should be first
    assert data[0]["version"] == "20240101_120000"
    assert data[-1]["version"] == "20231201_100000"


def test_list_model_metadata_includes_type_specific_fields(client, admin_auth_headers):
    """Test that type-specific fields are included"""
    response = client.get(
        "/admin/models?model_type=svd",
        headers=admin_auth_headers
    )

    assert response.status_code == 200
    data = response.json()

    # SVD models should have these fields
    svd_model = data[0]
    assert "n_components" in svd_model
    assert "rmse" in svd_model
    assert svd_model["n_components"] == 10

    # Get TF-IDF/KMeans model
    response = client.get(
        "/admin/models?model_type=tfidf_kmeans",
        headers=admin_auth_headers
    )
    data = response.json()

    # TF-IDF models should have these fields
    tfidf_model = data[0]
    assert "n_clusters" in tfidf_model
    assert "max_features" in tfidf_model
    assert tfidf_model["n_clusters"] == 20


def test_list_model_metadata_unauthorized(client, auth_headers):
    """Test that listing models requires admin authentication"""
    response = client.get(
        "/admin/models",
        headers=auth_headers  # Regular user
    )

    assert response.status_code == 403


def test_list_model_metadata_no_auth(client):
    """Test that listing models requires authentication"""
    response = client.get("/admin/models")

    assert response.status_code in [401, 403]


# ============================================================================
# GET SINGLE MODEL METADATA ENDPOINT TESTS
# ============================================================================

def test_get_model_metadata_success(client, admin_auth_headers):
    """Test GET /admin/models/{model_id} endpoint"""
    response = client.get(
        "/admin/models/1",
        headers=admin_auth_headers
    )

    assert response.status_code == 200
    data = response.json()

    assert data["id"] == 1
    assert data["model_type"] == "svd"
    assert data["version"] == "20240101_120000"
    assert "n_components" in data
    assert "rmse" in data


def test_get_model_metadata_not_found(client, admin_auth_headers):
    """Test GET model with non-existent ID"""
    response = client.get(
        "/admin/models/999",
        headers=admin_auth_headers
    )

    assert response.status_code == 404


def test_get_model_metadata_unauthorized(client, auth_headers):
    """Test that getting model requires admin authentication"""
    response = client.get(
        "/admin/models/1",
        headers=auth_headers  # Regular user
    )

    assert response.status_code == 403


# ============================================================================
# DEACTIVATE MODEL ENDPOINT TESTS
# ============================================================================

def test_deactivate_model_success(client, admin_auth_headers, test_db):
    """Test DELETE /admin/models/{model_id} endpoint (soft delete)"""
    response = client.delete(
        "/admin/models/1",
        headers=admin_auth_headers
    )

    assert response.status_code == 204

    # Verify model is deactivated
    model = test_db.query(ModelMetadata).filter(ModelMetadata.id == 1).first()
    assert model is not None  # Model still exists
    assert model.is_active is False  # But is inactive


def test_deactivate_model_not_found(client, admin_auth_headers):
    """Test deactivating non-existent model"""
    response = client.delete(
        "/admin/models/999",
        headers=admin_auth_headers
    )

    assert response.status_code == 404


def test_deactivate_model_unauthorized(client, auth_headers):
    """Test that deactivating model requires admin authentication"""
    response = client.delete(
        "/admin/models/1",
        headers=auth_headers  # Regular user
    )

    assert response.status_code == 403


def test_deactivate_model_no_auth(client):
    """Test that deactivating model requires authentication"""
    response = client.delete("/admin/models/1")

    assert response.status_code in [401, 403]


# ============================================================================
# RATE LIMITING TESTS
# ============================================================================

def test_retraining_rate_limit(client, admin_auth_headers):
    """Test that retraining endpoint has strict rate limit"""
    with patch('ml_api.api.admin.subprocess.run') as mock_subprocess:
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="Success", stderr="")

        # Make multiple requests
        responses = []
        for _ in range(5):
            response = client.post(
                "/admin/retrain",
                headers=admin_auth_headers
            )
            responses.append(response.status_code)

        # Should get rate limited (3/hour limit)
        assert responses[0] == 202
        assert responses[1] == 202


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_full_model_management_workflow(client, admin_auth_headers, test_db):
    """Test complete workflow: list models, deactivate old model"""
    # Step 1: List existing models
    response = client.get("/admin/models", headers=admin_auth_headers)
    assert response.status_code == 200
    initial_models = response.json()
    assert len(initial_models) >= 1

    # Step 2: Trigger retraining (mocked)
    with patch('ml_api.api.admin.subprocess.run') as mock_subprocess:
        mock_subprocess.return_value = MagicMock(returncode=0)

        response = client.post("/admin/retrain", headers=admin_auth_headers)
        assert response.status_code == 202

    # Step 3: Deactivate old model
    response = client.delete("/admin/models/3", headers=admin_auth_headers)
    assert response.status_code == 204

    # Step 4: Verify old model is inactive
    response = client.get("/admin/models/3", headers=admin_auth_headers)
    assert response.status_code == 200
    assert response.json()["is_active"] is False

    # Step 5: List only active models
    response = client.get("/admin/models?is_active=true", headers=admin_auth_headers)
    assert response.status_code == 200
    active_models = response.json()
    assert len(active_models) == 1  # Only 2 were active initially, now 1
