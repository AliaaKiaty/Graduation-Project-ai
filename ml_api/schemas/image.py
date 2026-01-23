"""
Pydantic schemas for image API
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


class SimilarImageItem(BaseModel):
    """Schema for a similar image result"""
    rank: int = Field(..., description="Similarity rank (1 = most similar)")
    filename: str = Field(..., description="Filename or identifier of similar image")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (higher = more similar)")
    distance: float = Field(..., ge=0.0, description="Euclidean distance in feature space")


class SimilarImagesResponse(BaseModel):
    """Response schema for similar images endpoint"""
    similar_images: List[SimilarImageItem] = Field(..., description="List of similar images")
    query_info: dict = Field(..., description="Information about the query image")
    total_results: int = Field(..., description="Number of results returned")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "similar_images": [
                    {
                        "rank": 1,
                        "filename": "butterfly_001.jpg",
                        "similarity_score": 0.95,
                        "distance": 0.05
                    },
                    {
                        "rank": 2,
                        "filename": "butterfly_023.jpg",
                        "similarity_score": 0.89,
                        "distance": 0.12
                    }
                ],
                "query_info": {
                    "format": "image/jpeg",
                    "size_bytes": 245678
                },
                "total_results": 2
            }
        }
    )


class PredictionItem(BaseModel):
    """Schema for a classification prediction"""
    rank: int = Field(..., description="Prediction rank (1 = highest confidence)")
    class_name: str = Field(..., description="Predicted class name")
    class_id: Optional[int] = Field(None, description="Class ID if available")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class ClassificationResponse(BaseModel):
    """Response schema for image classification endpoint"""
    predictions: List[PredictionItem] = Field(..., description="Top-K predictions")
    top_prediction: str = Field(..., description="Most confident prediction")
    top_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of top prediction")
    query_info: dict = Field(..., description="Information about the query image")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [
                    {
                        "rank": 1,
                        "class_name": "Monarch",
                        "class_id": 0,
                        "confidence": 0.92
                    },
                    {
                        "rank": 2,
                        "class_name": "Viceroy",
                        "class_id": 1,
                        "confidence": 0.05
                    }
                ],
                "top_prediction": "Monarch",
                "top_confidence": 0.92,
                "query_info": {
                    "format": "image/jpeg",
                    "size_bytes": 245678
                }
            }
        }
    )
