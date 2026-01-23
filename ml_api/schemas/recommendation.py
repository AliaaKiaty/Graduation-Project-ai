"""
Pydantic schemas for recommendation API
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from enum import Enum


class DatasetType(str, Enum):
    """Dataset type enum"""
    english = "english"
    arabic = "arabic"


class PopularItem(BaseModel):
    """Schema for a popular item"""
    product_id: str = Field(..., description="Product ID")
    rating_count: int = Field(..., description="Number of ratings")
    rank: int = Field(..., description="Popularity rank")


class PopularResponse(BaseModel):
    """Response schema for popular items endpoint"""
    recommendations: List[PopularItem] = Field(..., description="List of popular items")
    dataset: str = Field(..., description="Dataset used")
    method: str = Field(default="popularity", description="Recommendation method")
    total_results: int = Field(..., description="Total number of results")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "recommendations": [
                    {"product_id": "B001MA0QY2", "rating_count": 7533, "rank": 1},
                    {"product_id": "B0009V1YR8", "rating_count": 2869, "rank": 2}
                ],
                "dataset": "english",
                "method": "popularity",
                "total_results": 2
            }
        }
    )


class CollaborativeRequest(BaseModel):
    """Request schema for collaborative filtering"""
    product_id: str = Field(..., min_length=1, description="Product ID to get recommendations for")
    dataset: DatasetType = Field(default=DatasetType.english, description="Dataset to use")
    top_n: int = Field(default=10, ge=1, le=100, description="Number of recommendations")
    min_correlation: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum correlation threshold")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "product_id": "6117036094",
                "dataset": "english",
                "top_n": 10,
                "min_correlation": 0.5
            }
        }
    )


class CollaborativeItem(BaseModel):
    """Schema for a collaborative filtering recommendation"""
    product_id: str = Field(..., description="Product ID")
    correlation_score: float = Field(..., description="Correlation score with input product")
    rank: int = Field(..., description="Recommendation rank")


class CollaborativeResponse(BaseModel):
    """Response schema for collaborative filtering"""
    input_product: str = Field(..., description="Input product ID")
    recommendations: List[CollaborativeItem] = Field(..., description="Recommended products")
    dataset: str = Field(..., description="Dataset used")
    method: str = Field(default="svd_collaborative_filtering", description="Recommendation method")
    total_results: int = Field(..., description="Total number of results")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "input_product": "6117036094",
                "recommendations": [
                    {"product_id": "0733001998", "correlation_score": 0.92, "rank": 1},
                    {"product_id": "1304511073", "correlation_score": 0.89, "rank": 2}
                ],
                "dataset": "english",
                "method": "svd_collaborative_filtering",
                "total_results": 2
            }
        }
    )


class ContentBasedRequest(BaseModel):
    """Request schema for content-based recommendations"""
    search_query: str = Field(..., min_length=1, max_length=500, description="Search query text")
    dataset: DatasetType = Field(default=DatasetType.english, description="Dataset to use")
    top_n: int = Field(default=10, ge=1, le=100, description="Number of recommendations")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "search_query": "cutting tool",
                "dataset": "english",
                "top_n": 10
            }
        }
    )


class ContentBasedItem(BaseModel):
    """Schema for a content-based recommendation"""
    product_id: str = Field(..., description="Product ID")
    product_description: Optional[str] = Field(None, description="Product description (truncated)")
    rank: int = Field(..., description="Recommendation rank")


class ContentBasedResponse(BaseModel):
    """Response schema for content-based recommendations"""
    search_query: str = Field(..., description="Input search query")
    predicted_cluster: int = Field(..., description="Predicted cluster ID")
    cluster_keywords: List[str] = Field(..., description="Top keywords for the cluster")
    recommendations: List[ContentBasedItem] = Field(..., description="Recommended products")
    dataset: str = Field(..., description="Dataset used")
    method: str = Field(default="tfidf_kmeans", description="Recommendation method")
    total_results: int = Field(..., description="Total items in cluster")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "search_query": "cutting tool",
                "predicted_cluster": 6,
                "cluster_keywords": ["tool", "cutting", "metal", "paint", "easy", "handle", "blade"],
                "recommendations": [
                    {
                        "product_id": "100123",
                        "product_description": "Professional cutting tool with ergonomic handle...",
                        "rank": 1
                    }
                ],
                "dataset": "english",
                "method": "tfidf_kmeans",
                "total_results": 15
            }
        }
    )
