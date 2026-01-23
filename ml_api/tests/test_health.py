"""
Tests for health and general API endpoints
"""
import pytest


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "models" in data
        assert "gpu" in data

    def test_health_model_status(self, client):
        """Test health endpoint includes model status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        models = data.get("models", {})

        # Check model categories exist
        assert "recommendation" in models or "error" in models
        assert "image" in models or "error" in models
        assert "chatbot" in models or "error" in models

    def test_health_gpu_info(self, client):
        """Test health endpoint includes GPU info."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        gpu = data.get("gpu", {})

        assert "available" in gpu
        assert isinstance(gpu["available"], bool)


class TestRootEndpoint:
    """Tests for GET / endpoint."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data


class TestDocsEndpoint:
    """Tests for documentation endpoints."""

    def test_swagger_docs(self, client):
        """Test Swagger docs are accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_docs(self, client):
        """Test ReDoc docs are accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_schema(self, client):
        """Test OpenAPI schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
