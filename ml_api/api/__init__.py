"""
API routers module
"""
from .recommendation import router as recommendation_router
from .image import router as image_router
from .chat import router as chat_router

__all__ = ["recommendation_router", "image_router", "chat_router"]
