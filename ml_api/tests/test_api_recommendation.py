"""
API endpoint tests for recommendation system
Tests for /recommend/* endpoints with database-backed implementation
"""
import os
import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import patch, MagicMock
from jose import jwt
import numpy as np

from ml_api.main import app
from ml_api.database import Base, get_db
from ml_api.models.db_models import ProductCategory, Product, UserInteraction, ProductEmbedding


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
    """Create test database and populate with sample data"""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)

    TestingSessionLocal = sessionmaker(bind=engine)
    db = TestingSessionLocal()

    # Create categories
    tools = ProductCategory(Id=1, Name="Tools")
    paint = ProductCategory(Id=2, Name="Paint")
    db.add_all([tools, paint])
    db.commit()

    # Create products
    products = [
        Product(Id=1, Name="Hammer", Description="Professional hammer", CategoryId=1, Price=29.99),
        Product(Id=2, Name="Screwdriver", Description="Phillips screwdriver", CategoryId=1, Price=15.99),
        Product(Id=3, Name="Paint Brush", Description="2-inch paint brush", CategoryId=2, Price=9.99),
    ]
    db.add_all(products)
    db.commit()

    # Create user interactions with ratings
    interactions = [
        UserInteraction(Id=1, UserId="user1", ProductID=1, Rating=5, InteractionDate=datetime.now()),
        UserInteraction(Id=2, UserId="user2", ProductID=1, Rating=4, InteractionDate=datetime.now()),
        UserInteraction(Id=3, UserId="user3", ProductID=1, Rating=4, InteractionDate=datetime.now()),
        UserInteraction(Id=4, UserId="user1", ProductID=2, Rating=4, InteractionDate=datetime.now()),
        UserInteraction(Id=5, UserId="user2", ProductID=2, Rating=3, InteractionDate=datetime.now()),
    ]
    db.add_all(interactions)
    db.commit()

    # Create product embeddings
    embeddings = [
        ProductEmbedding(product_id=1, cluster_id=0),
        ProductEmbedding(product_id=2, cluster_id=0),
        ProductEmbedding(product_id=3, cluster_id=1),
    ]
    db.add_all(embeddings)
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
    """Get authentication headers for test user"""
    token = _make_token("test-user-123")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_auth_headers():
    """Get authentication headers for admin user"""
    token = _make_token("admin-user-1", roles=["Admin"])
    return {"Authorization": f"Bearer {token}"}


# ============================================================================
# POPULAR RECOMMENDATIONS ENDPOINT TESTS
# ============================================================================

def test_get_popular_items_success(client, auth_headers):
    """Test POST /recommend/popular endpoint"""
    response = client.post(
        "/recommend/popular",
        json={"top_n": 3},
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()

    assert "recommendations" in data
    assert "method" in data
    assert "total_results" in data
    assert data["method"] == "popularity"

    # Should return at least 2 products (we have 2 products with ratings)
    assert len(data["recommendations"]) >= 2

    # First item should be Product 1 (3 ratings)
    assert data["recommendations"][0]["product_id"] == 1
    assert data["recommendations"][0]["rating_count"] == 3


def test_get_popular_items_with_category_filter(client, auth_headers):
    """Test popular items with category filter"""
    response = client.post(
        "/recommend/popular",
        json={"top_n": 5, "category_id": 1},  # Tools category
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()

    # All recommendations should be from Tools category
    for rec in data["recommendations"]:
        assert rec["category_name"] == "Tools"


def test_get_popular_items_unauthorized(client):
    """Test that endpoint requires authentication"""
    response = client.post(
        "/recommend/popular",
        json={"top_n": 5}
    )

    assert response.status_code in [401, 403]


# ============================================================================
# COLLABORATIVE FILTERING ENDPOINT TESTS
# ============================================================================

def test_get_collaborative_recommendations_success(client, auth_headers):
    """Test POST /recommend/collaborative endpoint"""
    with patch('ml_api.api.recommendation.get_rec_engine') as mock_get_engine:
        mock_engine = MagicMock()
        mock_engine.get_collaborative_recommendations.return_value = {
            "input_product_id": 1,
            "input_product_name": "Hammer",
            "recommendations": [
                {
                    "product_id": 2,
                    "product_name": "Screwdriver",
                    "correlation_score": 0.95,
                    "category_name": "Tools",
                    "image_url": None,
                    "price": 15.99,
                    "rank": 1
                }
            ]
        }
        mock_get_engine.return_value = mock_engine

        response = client.post(
            "/recommend/collaborative",
            json={
                "product_id": 1,
                "top_n": 5,
                "min_correlation": 0.5
            },
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        assert "input_product_id" in data
        assert "input_product_name" in data
        assert "recommendations" in data
        assert data["method"] == "svd_collaborative_filtering"
        assert data["input_product_id"] == 1


def test_get_collaborative_recommendations_product_not_found(client, auth_headers):
    """Test collaborative endpoint with non-existent product"""
    with patch('ml_api.api.recommendation.get_rec_engine') as mock_get_engine:
        mock_engine = MagicMock()
        mock_engine.get_collaborative_recommendations.side_effect = KeyError("Product not found")
        mock_get_engine.return_value = mock_engine

        response = client.post(
            "/recommend/collaborative",
            json={"product_id": 999, "top_n": 5},
            headers=auth_headers
        )

        assert response.status_code == 404


def test_get_collaborative_recommendations_unauthorized(client):
    """Test that endpoint requires authentication"""
    response = client.post(
        "/recommend/collaborative",
        json={"product_id": 1, "top_n": 5}
    )

    assert response.status_code in [401, 403]


# ============================================================================
# CONTENT-BASED FILTERING ENDPOINT TESTS
# ============================================================================

def test_get_content_based_recommendations_success(client, auth_headers):
    """Test POST /recommend/content-based endpoint"""
    with patch('ml_api.api.recommendation.get_rec_engine') as mock_get_engine:
        mock_engine = MagicMock()
        mock_engine.get_content_based_recommendations.return_value = {
            "search_query": "hammer tool",
            "predicted_cluster": 0,
            "cluster_keywords": ["tool", "hammer", "metal"],
            "recommendations": [
                {
                    "product_id": 1,
                    "product_name": "Hammer",
                    "product_description": "Professional hammer",
                    "category_name": "Tools",
                    "image_url": None,
                    "rank": 1
                }
            ],
            "total_results": 2
        }
        mock_get_engine.return_value = mock_engine

        response = client.post(
            "/recommend/content-based",
            json={"search_query": "hammer tool", "top_n": 5},
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        assert "search_query" in data
        assert "predicted_cluster" in data
        assert "cluster_keywords" in data
        assert "recommendations" in data
        assert data["method"] == "tfidf_kmeans"


def test_get_content_based_recommendations_unauthorized(client):
    """Test that endpoint requires authentication"""
    response = client.post(
        "/recommend/content-based",
        json={"search_query": "test", "top_n": 5}
    )

    assert response.status_code in [401, 403]


# ============================================================================
# PRODUCT LIST ENDPOINT TESTS (read-only)
# ============================================================================

def test_list_products_success(client, auth_headers):
    """Test GET /recommend/products endpoint"""
    response = client.get(
        "/recommend/products",
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 3  # We created 3 products

    # Check product structure
    assert "id" in data[0]
    assert "name" in data[0]


def test_list_products_with_pagination(client, auth_headers):
    """Test product list with pagination"""
    response = client.get(
        "/recommend/products?skip=0&limit=2",
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data) <= 2


def test_list_products_with_search(client, auth_headers):
    """Test product list with search filter"""
    response = client.get(
        "/recommend/products?search=Hammer",
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data) == 1
    assert "Hammer" in data[0]["name"]


def test_list_products_with_category_filter(client, auth_headers):
    """Test product list with category filter"""
    response = client.get(
        "/recommend/products?category_id=1",  # Tools
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()

    for product in data:
        assert product["category_name"] == "Tools"


def test_get_single_product_success(client, auth_headers):
    """Test GET /recommend/products/{product_id} endpoint"""
    response = client.get(
        "/recommend/products/1",
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()

    assert data["id"] == 1
    assert data["name"] == "Hammer"


def test_get_single_product_not_found(client, auth_headers):
    """Test GET product with non-existent ID"""
    response = client.get(
        "/recommend/products/999",
        headers=auth_headers
    )

    assert response.status_code == 404


# ============================================================================
# CATEGORY ENDPOINT TESTS
# ============================================================================

def test_list_categories_success(client, auth_headers):
    """Test GET /recommend/categories endpoint"""
    response = client.get(
        "/recommend/categories",
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 2  # Tools and Paint

    # Check category structure
    assert "id" in data[0]
    assert "name" in data[0]


def test_list_categories_unauthorized(client):
    """Test that endpoint requires authentication"""
    response = client.get("/recommend/categories")

    assert response.status_code in [401, 403]
