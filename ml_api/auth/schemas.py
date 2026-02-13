"""
Pydantic schemas for authentication
Minimal schemas — no login/register since auth is handled by .NET backend
"""
from pydantic import BaseModel, Field
from typing import List


class UserResponse(BaseModel):
    """Response schema for current user info (from JWT claims)"""
    user_id: str = Field(..., description="User ID from JWT")
    roles: List[str] = Field(default_factory=list, description="User roles from JWT")
    is_admin: bool = Field(..., description="Whether user has admin role")
