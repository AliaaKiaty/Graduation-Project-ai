"""
Pydantic schemas module
"""
from .recommendation import (
    DatasetType,
    PopularItem,
    PopularResponse,
    CollaborativeItem,
    CollaborativeRequest,
    CollaborativeResponse,
    ContentBasedItem,
    ContentBasedRequest,
    ContentBasedResponse,
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
    "DatasetType",
    "PopularItem",
    "PopularResponse",
    "CollaborativeItem",
    "CollaborativeRequest",
    "CollaborativeResponse",
    "ContentBasedItem",
    "ContentBasedRequest",
    "ContentBasedResponse",
    "SimilarImageItem",
    "SimilarImagesResponse",
    "PredictionItem",
    "ClassificationResponse",
    "ChatRequest",
    "ChatResponse",
    "ChatModelStatus",
]
