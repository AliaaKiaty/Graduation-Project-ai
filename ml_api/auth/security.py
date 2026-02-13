"""
External JWT validation for tokens issued by .NET backend
No token creation — only validation and claim extraction
"""
from typing import Optional, Dict, Any, List
from jose import JWTError, jwt
from .. import config


def decode_and_validate_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT token issued by the .NET backend.

    Validates signature, expiration, and optionally issuer/audience.
    Extracts user_id from the configurable claim.

    Args:
        token: JWT token string to decode

    Returns:
        Optional[Dict]: Token payload if valid, None if invalid
    """
    try:
        # Build options for validation
        options = {}
        kwargs = {
            "algorithms": config.JWT_ALGORITHMS,
        }

        # Optionally validate issuer
        if config.JWT_ISSUER:
            kwargs["issuer"] = config.JWT_ISSUER

        # Optionally validate audience
        if config.JWT_AUDIENCE:
            kwargs["audience"] = config.JWT_AUDIENCE

        payload = jwt.decode(
            token,
            config.JWT_SECRET_KEY,
            options=options,
            **kwargs,
        )
        return payload
    except JWTError:
        return None


def extract_user_id(payload: Dict[str, Any]) -> Optional[str]:
    """
    Extract user ID from token payload using the configured claim name.

    Args:
        payload: Decoded JWT payload

    Returns:
        Optional[str]: User ID if found, None otherwise
    """
    return payload.get(config.JWT_USER_ID_CLAIM)


def extract_roles(payload: Dict[str, Any]) -> List[str]:
    """
    Extract user roles from token payload using the configured claim name.

    Handles both single role (string) and multiple roles (list) formats.

    Args:
        payload: Decoded JWT payload

    Returns:
        List[str]: List of role strings
    """
    roles = payload.get(config.JWT_ROLE_CLAIM, [])
    if isinstance(roles, str):
        return [roles]
    if isinstance(roles, list):
        return roles
    return []
