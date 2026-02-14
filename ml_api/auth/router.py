"""
Authentication dependencies for FastAPI
Validates external JWTs from .NET backend — no login/register endpoints
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Annotated, Optional

from .. import config
from .models import TokenUser
from .security import decode_and_validate_token, extract_user_id, extract_roles

# When BYPASS_AUTH is on, make the Bearer token optional (auto_error=False)
security = HTTPBearer(auto_error=not config.BYPASS_AUTH)

# Dev user returned when BYPASS_AUTH is enabled and no token is provided
_DEV_USER = TokenUser(
    user_id="dev-user",
    roles=[config.JWT_ADMIN_ROLE],
    claims={"sub": "dev-user", "role": [config.JWT_ADMIN_ROLE]},
)


def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
) -> TokenUser:
    """
    Dependency to get current authenticated user from external JWT token.
    No database lookup — user info comes entirely from the JWT claims.

    When BYPASS_AUTH=true, returns a dev user with admin privileges if no
    valid token is provided. This lets developers test endpoints before
    the .NET backend integration is ready.
    """
    # If no credentials provided
    if credentials is None:
        if config.BYPASS_AUTH:
            return _DEV_USER
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # Decode and validate token
    payload = decode_and_validate_token(token)
    if payload is None:
        if config.BYPASS_AUTH:
            return _DEV_USER
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
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user
