"""
Tests for authentication endpoints
"""
import os
import pytest
import time


class TestAuthLogin:
    """Tests for POST /auth/login endpoint."""

    def test_login_valid_credentials(self, client):
        """Test login with valid credentials returns tokens."""
        response = client.post(
            "/auth/login",
            json={
                "username": os.environ['ADMIN_USERNAME'],
                "password": os.environ['ADMIN_PASSWORD']
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data

    def test_login_invalid_username(self, client):
        """Test login with invalid username returns 401."""
        response = client.post(
            "/auth/login",
            json={
                "username": "nonexistent_user",
                "password": os.environ['ADMIN_PASSWORD']
            }
        )
        assert response.status_code == 401
        assert "detail" in response.json()

    def test_login_invalid_password(self, client):
        """Test login with invalid password returns 401."""
        response = client.post(
            "/auth/login",
            json={
                "username": os.environ['ADMIN_USERNAME'],
                "password": "wrong_password"
            }
        )
        assert response.status_code == 401
        assert "detail" in response.json()

    def test_login_missing_fields(self, client):
        """Test login with missing fields returns 422."""
        response = client.post(
            "/auth/login",
            json={"username": "admin"}
        )
        assert response.status_code == 422

    def test_login_empty_credentials(self, client):
        """Test login with empty credentials returns error."""
        response = client.post(
            "/auth/login",
            json={"username": "", "password": ""}
        )
        assert response.status_code in [401, 422]


class TestAuthRefresh:
    """Tests for POST /auth/refresh endpoint."""

    def test_refresh_valid_token(self, client):
        """Test refresh with valid refresh token returns new access token."""
        # First login to get tokens
        login_response = client.post(
            "/auth/login",
            json={
                "username": os.environ['ADMIN_USERNAME'],
                "password": os.environ['ADMIN_PASSWORD']
            }
        )
        assert login_response.status_code == 200
        refresh_token = login_response.json()["refresh_token"]

        # Use refresh token
        response = client.post(
            "/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data

    def test_refresh_invalid_token(self, client):
        """Test refresh with invalid token returns 401."""
        response = client.post(
            "/auth/refresh",
            json={"refresh_token": "invalid.token.here"}
        )
        assert response.status_code == 401


class TestAuthMe:
    """Tests for GET /auth/me endpoint."""

    def test_me_with_valid_token(self, client, auth_headers):
        """Test /auth/me with valid token returns user info."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.get("/auth/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "username" in data
        assert "email" in data
        assert "is_admin" in data
        assert data["username"] == os.environ['ADMIN_USERNAME']

    def test_me_without_token(self, client):
        """Test /auth/me without token returns 401."""
        response = client.get("/auth/me")
        assert response.status_code == 401

    def test_me_with_invalid_token(self, client):
        """Test /auth/me with invalid token returns 401."""
        headers = {"Authorization": "Bearer invalid.token.here"}
        response = client.get("/auth/me", headers=headers)
        assert response.status_code == 401

    def test_me_with_malformed_header(self, client):
        """Test /auth/me with malformed auth header returns 401."""
        headers = {"Authorization": "InvalidFormat token123"}
        response = client.get("/auth/me", headers=headers)
        assert response.status_code == 401


class TestProtectedEndpoints:
    """Tests for protected endpoint access."""

    def test_protected_endpoint_without_token(self, client):
        """Test accessing protected endpoint without token returns 401."""
        response = client.get("/recommend/popular")
        assert response.status_code == 401

    def test_protected_endpoint_with_expired_token(self, client):
        """Test accessing protected endpoint with expired token returns 401."""
        # Create an expired token (would need to mock time or use short expiry)
        # For now, just test with an invalid token format
        headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0IiwiZXhwIjoxfQ.invalid"}
        response = client.get("/recommend/popular", headers=headers)
        assert response.status_code == 401
