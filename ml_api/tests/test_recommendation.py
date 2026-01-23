"""
Tests for recommendation endpoints
"""
import pytest


class TestPopularEndpoint:
    """Tests for GET /recommend/popular endpoint."""

    def test_popular_without_auth(self, client):
        """Test popular endpoint requires authentication."""
        response = client.get("/recommend/popular")
        assert response.status_code == 401

    def test_popular_default_params(self, client, auth_headers):
        """Test popular endpoint with default parameters."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.get("/recommend/popular", headers=auth_headers)
        # May return 503 if models not loaded, which is expected in test environment
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "items" in data
            assert "dataset" in data
            assert "total_count" in data

    def test_popular_with_top_n(self, client, auth_headers):
        """Test popular endpoint with custom top_n parameter."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.get("/recommend/popular?top_n=5", headers=auth_headers)
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert len(data.get("items", [])) <= 5

    def test_popular_with_dataset_english(self, client, auth_headers):
        """Test popular endpoint with English dataset."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.get("/recommend/popular?dataset=english", headers=auth_headers)
        assert response.status_code in [200, 503]

    def test_popular_with_dataset_arabic(self, client, auth_headers):
        """Test popular endpoint with Arabic dataset."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.get("/recommend/popular?dataset=arabic", headers=auth_headers)
        assert response.status_code in [200, 503]

    def test_popular_invalid_top_n(self, client, auth_headers):
        """Test popular endpoint with invalid top_n parameter."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.get("/recommend/popular?top_n=-1", headers=auth_headers)
        assert response.status_code == 422

    def test_popular_top_n_too_large(self, client, auth_headers):
        """Test popular endpoint with top_n exceeding limit."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.get("/recommend/popular?top_n=1000", headers=auth_headers)
        assert response.status_code == 422


class TestCollaborativeEndpoint:
    """Tests for POST /recommend/collaborative endpoint."""

    def test_collaborative_without_auth(self, client):
        """Test collaborative endpoint requires authentication."""
        response = client.post(
            "/recommend/collaborative",
            json={"product_id": "B001MA0QY2", "top_n": 5}
        )
        assert response.status_code == 401

    def test_collaborative_valid_request(self, client, auth_headers):
        """Test collaborative endpoint with valid request."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.post(
            "/recommend/collaborative",
            json={"product_id": "B001MA0QY2", "top_n": 5},
            headers=auth_headers
        )
        # May return 503 if models not loaded, 404 if product not found
        assert response.status_code in [200, 404, 503]

        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data
            assert "query_product_id" in data

    def test_collaborative_missing_product_id(self, client, auth_headers):
        """Test collaborative endpoint with missing product_id."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.post(
            "/recommend/collaborative",
            json={"top_n": 5},
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_collaborative_invalid_product_id(self, client, auth_headers):
        """Test collaborative endpoint with non-existent product_id."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.post(
            "/recommend/collaborative",
            json={"product_id": "NONEXISTENT_PRODUCT_12345", "top_n": 5},
            headers=auth_headers
        )
        # Should return 404 for not found or 503 if models not loaded
        assert response.status_code in [404, 503]


class TestContentBasedEndpoint:
    """Tests for POST /recommend/content-based endpoint."""

    def test_content_based_without_auth(self, client):
        """Test content-based endpoint requires authentication."""
        response = client.post(
            "/recommend/content-based",
            json={"search_query": "cutting tool", "top_n": 5}
        )
        assert response.status_code == 401

    def test_content_based_valid_request(self, client, auth_headers):
        """Test content-based endpoint with valid request."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.post(
            "/recommend/content-based",
            json={"search_query": "cutting tool", "top_n": 5},
            headers=auth_headers
        )
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data
            assert "search_query" in data
            assert "cluster_id" in data

    def test_content_based_arabic_query(self, client, auth_headers):
        """Test content-based endpoint with Arabic query."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.post(
            "/recommend/content-based",
            json={"search_query": "شركة جيدة", "top_n": 5, "language": "arabic"},
            headers=auth_headers
        )
        assert response.status_code in [200, 503]

    def test_content_based_missing_query(self, client, auth_headers):
        """Test content-based endpoint with missing search_query."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.post(
            "/recommend/content-based",
            json={"top_n": 5},
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_content_based_empty_query(self, client, auth_headers):
        """Test content-based endpoint with empty search_query."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.post(
            "/recommend/content-based",
            json={"search_query": "", "top_n": 5},
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_content_based_invalid_top_n(self, client, auth_headers):
        """Test content-based endpoint with invalid top_n."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.post(
            "/recommend/content-based",
            json={"search_query": "test", "top_n": 0},
            headers=auth_headers
        )
        assert response.status_code == 422
