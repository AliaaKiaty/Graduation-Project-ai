"""
Authentication dependencies for FastAPI
Validates external JWTs from .NET backend — no login/register endpoints
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Annotated

from .. import config
from .models import TokenUser
from .security import decode_and_validate_token, extract_user_id, extract_roles

security = HTTPBearer()


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> TokenUser:
    """
    Dependency to get current authenticated user from external JWT token.
    No database lookup — user info comes entirely from the JWT claims.

    Args:
        credentials: HTTP bearer token credentials

    Returns:
        TokenUser: Authenticated user extracted from JWT

    Raises:
        HTTPException: If token is invalid or missing required claims
    """
    token = credentials.credentials

    # Decode and validate token
    payload = decode_and_validate_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract user ID
    user_id = extract_user_id(payload)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload: missing user ID",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract roles
    roles = extract_roles(payload)

    return TokenUser(
        user_id=user_id,
        roles=roles,
        claims=payload,
    )


def require_admin(
    current_user: Annotated[TokenUser, Depends(get_current_user)]
) -> TokenUser:
    """
    Dependency to require admin privileges.
    Checks the JWT role claim for the configured admin role.

    Args:
        current_user: Current authenticated user from token

    Returns:
        TokenUser: Admin user

    Raises:
        HTTPException: If user does not have admin role
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user
