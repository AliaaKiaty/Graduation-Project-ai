"""
Tests for chat endpoints
"""
import pytest


class TestChatMessageEndpoint:
    """Tests for POST /chat/message endpoint."""

    def test_message_without_auth(self, client):
        """Test chat message endpoint requires authentication."""
        response = client.post(
            "/chat/message",
            json={"message": "ما هي عاصمة مصر؟"}
        )
        assert response.status_code == 401

    def test_message_valid_request(self, client, auth_headers):
        """Test chat message endpoint with valid request."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.post(
            "/chat/message",
            json={
                "message": "ما هي عاصمة مصر؟",
                "max_tokens": 50,
                "temperature": 0.4
            },
            headers=auth_headers
        )
        # 503 is expected if model not loaded (which is normal in test env)
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "input_message" in data
            assert "response" in data
            assert "model" in data
            assert "tokens_generated" in data
            assert "generation_time_ms" in data

    def test_message_default_params(self, client, auth_headers):
        """Test chat message endpoint with default parameters."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.post(
            "/chat/message",
            json={"message": "مرحبا"},
            headers=auth_headers
        )
        assert response.status_code in [200, 503]

    def test_message_missing_message(self, client, auth_headers):
        """Test chat message endpoint with missing message."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.post(
            "/chat/message",
            json={"max_tokens": 100},
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_message_empty_message(self, client, auth_headers):
        """Test chat message endpoint with empty message."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.post(
            "/chat/message",
            json={"message": ""},
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_message_max_tokens_limit(self, client, auth_headers):
        """Test chat message endpoint with max_tokens at limit."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.post(
            "/chat/message",
            json={"message": "test", "max_tokens": 1024},
            headers=auth_headers
        )
        assert response.status_code in [200, 503]

    def test_message_max_tokens_exceeds_limit(self, client, auth_headers):
        """Test chat message endpoint with max_tokens exceeding limit."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.post(
            "/chat/message",
            json={"message": "test", "max_tokens": 2000},
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_message_invalid_temperature(self, client, auth_headers):
        """Test chat message endpoint with invalid temperature."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.post(
            "/chat/message",
            json={"message": "test", "temperature": 3.0},
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_message_temperature_range(self, client, auth_headers):
        """Test chat message endpoint with various temperatures."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        # Test low temperature
        response = client.post(
            "/chat/message",
            json={"message": "test", "temperature": 0.1},
            headers=auth_headers
        )
        assert response.status_code in [200, 503]

        # Test high temperature
        response = client.post(
            "/chat/message",
            json={"message": "test", "temperature": 1.5},
            headers=auth_headers
        )
        assert response.status_code in [200, 503]

    def test_message_long_input(self, client, auth_headers):
        """Test chat message endpoint with long input."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        long_message = "ما هي " * 100  # Repeat to create long message
        response = client.post(
            "/chat/message",
            json={"message": long_message},
            headers=auth_headers
        )
        # Should succeed or return 503 if model not loaded
        assert response.status_code in [200, 503]


class TestChatStatusEndpoint:
    """Tests for GET /chat/status endpoint."""

    def test_status_without_auth(self, client):
        """Test chat status endpoint requires authentication."""
        response = client.get("/chat/status")
        assert response.status_code == 401

    def test_status_valid_request(self, client, auth_headers):
        """Test chat status endpoint with valid request."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.get("/chat/status", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "llama_base" in data
        assert "lora_adapter" in data
        assert "is_ready" in data


class TestChatLoadEndpoint:
    """Tests for POST /chat/load endpoint."""

    def test_load_without_auth(self, client):
        """Test chat load endpoint requires authentication."""
        response = client.post("/chat/load")
        assert response.status_code == 401

    def test_load_valid_request(self, client, auth_headers):
        """Test chat load endpoint with valid request."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.post("/chat/load", headers=auth_headers)
        # May succeed or fail depending on GPU/model availability
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "message" in data
            assert "status" in data


class TestChatUnloadEndpoint:
    """Tests for POST /chat/unload endpoint."""

    def test_unload_without_auth(self, client):
        """Test chat unload endpoint requires authentication."""
        response = client.post("/chat/unload")
        assert response.status_code == 401

    def test_unload_valid_request(self, client, auth_headers):
        """Test chat unload endpoint with valid request."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.post("/chat/unload", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "status" in data
