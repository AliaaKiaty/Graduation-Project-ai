"""
Token user model for external JWT authentication
No database model — user info comes from JWT claims
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class TokenUser:
    """
    Lightweight user representation extracted from JWT claims.
    No database backing — all info comes from the token.
    """
    user_id: str
    roles: List[str] = field(default_factory=list)
    claims: dict = field(default_factory=dict)

    @property
    def is_admin(self) -> bool:
        """Check if user has admin role"""
        from .. import config
        return config.JWT_ADMIN_ROLE in self.roles
