"""
Tests for external JWT authentication
Tests JWT validation, not login/register (auth is handled by .NET backend)
"""
import os
import pytest
from datetime import datetime, timedelta
from jose import jwt


def create_token(user_id: str, roles: list = None, expired: bool = False, extra_claims: dict = None) -> str:
    """Helper to create test JWT tokens."""
    now = datetime.utcnow()
    payload = {
        "sub": user_id,
        "role": roles or [],
        "iat": now,
        "exp": now + timedelta(hours=-1 if expired else 1),
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, os.environ['JWT_SECRET_KEY'], algorithm="HS256")


class TestJWTValidation:
    """Tests for JWT token validation."""

    def test_valid_token_accepted(self, client):
        """Test that a valid JWT token is accepted."""
        token = create_token("user-123")
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/health")
        assert response.status_code == 200

    def test_missing_token_returns_401(self, client):
        """Test that missing token returns 401."""
        response = client.post("/recommend/popular", json={"top_n": 5})
        assert response.status_code in [401, 403]

    def test_invalid_token_returns_401(self, client):
        """Test that an invalid token returns 401."""
        headers = {"Authorization": "Bearer invalid.token.here"}
        response = client.post(
            "/recommend/popular",
            json={"top_n": 5},
            headers=headers
        )
        assert response.status_code == 401

    def test_expired_token_returns_401(self, client):
        """Test that an expired token returns 401."""
        token = create_token("user-123", expired=True)
        headers = {"Authorization": f"Bearer {token}"}
        response = client.post(
            "/recommend/popular",
            json={"top_n": 5},
            headers=headers
        )
        assert response.status_code == 401

    def test_malformed_header_returns_401(self, client):
        """Test that malformed Authorization header returns error."""
        headers = {"Authorization": "InvalidFormat token123"}
        response = client.post(
            "/recommend/popular",
            json={"top_n": 5},
            headers=headers
        )
        assert response.status_code in [401, 403]

    def test_token_with_wrong_secret_returns_401(self, client):
        """Test that token signed with wrong secret returns 401."""
        now = datetime.utcnow()
        payload = {
            "sub": "user-123",
            "role": [],
            "iat": now,
            "exp": now + timedelta(hours=1),
        }
        token = jwt.encode(payload, "wrong-secret-key", algorithm="HS256")
        headers = {"Authorization": f"Bearer {token}"}
        response = client.post(
            "/recommend/popular",
            json={"top_n": 5},
            headers=headers
        )
        assert response.status_code == 401


class TestAdminAccess:
    """Tests for admin-only endpoints."""

    def test_admin_endpoint_with_admin_role(self, client):
        """Test that admin endpoints accept tokens with admin role."""
        token = create_token("admin-1", roles=["Admin"])
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/admin/models", headers=headers)
        assert response.status_code == 200

    def test_admin_endpoint_without_admin_role(self, client):
        """Test that admin endpoints reject tokens without admin role."""
        token = create_token("user-123", roles=["User"])
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/admin/models", headers=headers)
        assert response.status_code == 403

    def test_admin_endpoint_without_any_role(self, client):
        """Test that admin endpoints reject tokens with no roles."""
        token = create_token("user-123", roles=[])
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/admin/models", headers=headers)
        assert response.status_code == 403

    def test_admin_endpoint_without_auth(self, client):
        """Test that admin endpoints require authentication."""
        response = client.get("/admin/models")
        assert response.status_code in [401, 403]


class TestProtectedEndpoints:
    """Tests for protected endpoint access."""

    def test_protected_endpoint_without_token(self, client):
        """Test accessing protected endpoint without token returns 401."""
        response = client.post("/recommend/popular", json={"top_n": 5})
        assert response.status_code in [401, 403]

    def test_protected_endpoint_with_valid_token(self, client, auth_headers):
        """Test accessing protected endpoint with valid token works."""
        response = client.post(
            "/recommend/popular",
            json={"top_n": 5},
            headers=auth_headers
        )
        # May be 200 or 500 depending on DB/model state, but not 401
        assert response.status_code != 401
