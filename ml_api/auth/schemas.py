"""
Pydantic schemas for authentication
"""
from pydantic import BaseModel, Field
from datetime import datetime


class LoginRequest(BaseModel):
    """Request schema for login endpoint"""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=6, description="Password")

    class Config:
        json_schema_extra = {
            "example": {
                "username": "admin",
                "password": "secure_password"
            }
        }


class TokenResponse(BaseModel):
    """Response schema for login and refresh endpoints"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800
            }
        }


class RefreshTokenRequest(BaseModel):
    """Request schema for token refresh"""
    refresh_token: str = Field(..., description="Refresh token")


class UserResponse(BaseModel):
    """Response schema for user information"""
    id: int = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    is_admin: bool = Field(..., description="Admin status")
    is_active: bool = Field(..., description="Account active status")
    created_at: datetime = Field(..., description="Account creation timestamp")

    class Config:
        from_attributes = True  # Allows ORM mode for SQLAlchemy models
        json_schema_extra = {
            "example": {
                "id": 1,
                "username": "admin",
                "email": "admin@api.local",
                "is_admin": True,
                "is_active": True,
                "created_at": "2026-01-16T10:00:00Z"
            }
        }


class TokenPayload(BaseModel):
    """Payload schema for JWT token"""
    sub: str = Field(..., description="Subject (username)")
    exp: datetime = Field(..., description="Expiration time")
    type: str = Field(..., description="Token type (access or refresh)")
