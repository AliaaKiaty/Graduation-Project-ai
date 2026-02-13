"""
Pydantic schemas module
"""
from .recommendation import (
    PopularItem,
    PopularRequest,
    PopularResponse,
    CollaborativeItem,
    CollaborativeRequest,
    CollaborativeResponse,
    ContentBasedItem,
    ContentBasedRequest,
    ContentBasedResponse,
    ProductResponse,
    CategoryResponse,
)
from .image import (
    SimilarImageItem,
    SimilarImagesResponse,
    PredictionItem,
    ClassificationResponse,
)
from .chat import (
    ChatRequest,
    ChatResponse,
    ChatModelStatus,
)

__all__ = [
    "PopularItem",
    "PopularRequest",
    "PopularResponse",
    "CollaborativeItem",
    "CollaborativeRequest",
    "CollaborativeResponse",
    "ContentBasedItem",
    "ContentBasedRequest",
    "ContentBasedResponse",
    "ProductResponse",
    "CategoryResponse",
    "SimilarImageItem",
    "SimilarImagesResponse",
    "PredictionItem",
    "ClassificationResponse",
    "ChatRequest",
    "ChatResponse",
    "ChatModelStatus",
]
