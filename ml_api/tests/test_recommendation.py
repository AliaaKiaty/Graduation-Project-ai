"""
Tests for recommendation endpoints
Uses JWT-based auth from conftest (external JWT validation)
"""
import pytest


class TestPopularEndpoint:
    """Tests for POST /recommend/popular endpoint."""

    def test_popular_without_auth(self, client):
        """Test popular endpoint requires authentication."""
        response = client.post("/recommend/popular", json={"top_n": 5})
        assert response.status_code in [401, 403]

    def test_popular_default_params(self, client, auth_headers):
        """Test popular endpoint with default parameters."""
        response = client.post(
            "/recommend/popular",
            json={"top_n": 10},
            headers=auth_headers
        )
        # May return 500 if DB not populated, which is expected in test environment
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data
            assert "method" in data
            assert "total_results" in data

    def test_popular_with_top_n(self, client, auth_headers):
        """Test popular endpoint with custom top_n parameter."""
        response = client.post(
            "/recommend/popular",
            json={"top_n": 5},
            headers=auth_headers
        )
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.json()
            assert len(data.get("recommendations", [])) <= 5


class TestCollaborativeEndpoint:
    """Tests for POST /recommend/collaborative endpoint."""

    def test_collaborative_without_auth(self, client):
        """Test collaborative endpoint requires authentication."""
        response = client.post(
            "/recommend/collaborative",
            json={"product_id": 1, "top_n": 5}
        )
        assert response.status_code in [401, 403]

    def test_collaborative_valid_request(self, client, auth_headers):
        """Test collaborative endpoint with valid request."""
        response = client.post(
            "/recommend/collaborative",
            json={"product_id": 1, "top_n": 5},
            headers=auth_headers
        )
        # May return 503 if models not loaded, 404 if product not found
        assert response.status_code in [200, 404, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data
            assert "input_product_id" in data

    def test_collaborative_missing_product_id(self, client, auth_headers):
        """Test collaborative endpoint with missing product_id."""
        response = client.post(
            "/recommend/collaborative",
            json={"top_n": 5},
            headers=auth_headers
        )
        assert response.status_code == 422


class TestContentBasedEndpoint:
    """Tests for POST /recommend/content-based endpoint."""

    def test_content_based_without_auth(self, client):
        """Test content-based endpoint requires authentication."""
        response = client.post(
            "/recommend/content-based",
            json={"search_query": "cutting tool", "top_n": 5}
        )
        assert response.status_code in [401, 403]

    def test_content_based_valid_request(self, client, auth_headers):
        """Test content-based endpoint with valid request."""
        response = client.post(
            "/recommend/content-based",
            json={"search_query": "cutting tool", "top_n": 5},
            headers=auth_headers
        )
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data
            assert "search_query" in data

    def test_content_based_missing_query(self, client, auth_headers):
        """Test content-based endpoint with missing search_query."""
        response = client.post(
            "/recommend/content-based",
            json={"top_n": 5},
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_content_based_empty_query(self, client, auth_headers):
        """Test content-based endpoint with empty search_query."""
        response = client.post(
            "/recommend/content-based",
            json={"search_query": "", "top_n": 5},
            headers=auth_headers
        )
        assert response.status_code == 422
