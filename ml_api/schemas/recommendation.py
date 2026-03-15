"""
Pydantic schemas for recommendation API
Updated for real .NET backend database schema
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from decimal import Decimal


# ============================================================================
# POPULAR RECOMMENDATIONS
# ============================================================================

class PopularRequest(BaseModel):
    """Request schema for popular items"""
    category_id: Optional[int] = Field(None, description="Filter by category ID")
    top_n: int = Field(default=10, ge=1, le=100, description="Number of items to return")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "category_id": None,
                "top_n": 10
            }
        }
    )


class PopularItem(BaseModel):
    """Schema for a popular item"""
    product_id: int = Field(..., description="Product ID")
    product_name: str = Field(..., description="Product name")
    image_url: Optional[str] = Field(None, description="Product image URL")
    price: Optional[float] = Field(None, description="Product price")
    rating_count: int = Field(..., description="Number of ratings")
    average_rating: Optional[float] = Field(None, description="Average rating score")
    category_name: Optional[str] = Field(None, description="Category name")
    rank: int = Field(..., description="Popularity rank")


class PopularResponse(BaseModel):
    """Response schema for popular items endpoint"""
    recommendations: List[PopularItem] = Field(..., description="List of popular items")
    method: str = Field(default="popularity", description="Recommendation method")
    total_results: int = Field(..., description="Total number of results")
    category_filter: Optional[int] = Field(None, description="Category ID filter applied")


# ============================================================================
# COLLABORATIVE FILTERING
# ============================================================================

class CollaborativeRequest(BaseModel):
    """Request schema for collaborative filtering"""
    product_id: int = Field(..., description="Product ID to get recommendations for")
    top_n: int = Field(default=10, ge=1, le=100, description="Number of recommendations")
    min_correlation: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum correlation threshold")
    category_id: Optional[int] = Field(None, description="Filter by category ID")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "product_id": 1,
                "top_n": 10,
                "min_correlation": 0.5,
                "category_id": None
            }
        }
    )


class CollaborativeItem(BaseModel):
    """Schema for a collaborative filtering recommendation"""
    product_id: int = Field(..., description="Product ID")
    product_name: str = Field(..., description="Product name")
    image_url: Optional[str] = Field(None, description="Product image URL")
    price: Optional[float] = Field(None, description="Product price")
    correlation_score: float = Field(..., description="Correlation score with input product")
    category_name: Optional[str] = Field(None, description="Category name")
    rank: int = Field(..., description="Recommendation rank")


class CollaborativeResponse(BaseModel):
    """Response schema for collaborative filtering"""
    input_product_id: int = Field(..., description="Input product ID")
    input_product_name: str = Field(..., description="Input product name")
    recommendations: List[CollaborativeItem] = Field(..., description="Recommended products")
    method: str = Field(default="svd_collaborative_filtering", description="Recommendation method")
    total_results: int = Field(..., description="Total number of results")


# ============================================================================
# CONTENT-BASED FILTERING
# ============================================================================

class ContentBasedRequest(BaseModel):
    """Request schema for content-based recommendations"""
    search_query: str = Field(..., min_length=1, max_length=500, description="Search query text")
    top_n: int = Field(default=10, ge=1, le=100, description="Number of recommendations")
    category_id: Optional[int] = Field(None, description="Filter by category ID")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "search_query": "cutting tool",
                "top_n": 10,
                "category_id": None
            }
        }
    )


class ContentBasedItem(BaseModel):
    """Schema for a content-based recommendation"""
    product_id: int = Field(..., description="Product ID")
    product_name: str = Field(..., description="Product name")
    product_description: Optional[str] = Field(None, description="Product description (truncated)")
    image_url: Optional[str] = Field(None, description="Product image URL")
    category_name: Optional[str] = Field(None, description="Category name")
    rank: int = Field(..., description="Recommendation rank")


class ContentBasedResponse(BaseModel):
    """Response schema for content-based recommendations"""
    search_query: str = Field(..., description="Input search query")
    predicted_cluster: int = Field(..., description="Predicted cluster ID")
    cluster_keywords: List[str] = Field(..., description="Top keywords for the cluster")
    recommendations: List[ContentBasedItem] = Field(..., description="Recommended products")
    method: str = Field(default="tfidf_kmeans", description="Recommendation method")
    total_results: int = Field(..., description="Total items in cluster")


# ============================================================================
# PRODUCT RESPONSE (read-only)
# ============================================================================

class ProductResponse(BaseModel):
    """Schema for product response (read-only from .NET backend)"""
    id: int
    name: str                          # NameAr preferred, falls back to NameEn
    name_en: Optional[str] = None
    name_ar: Optional[str] = None
    description: Optional[str] = None  # DescriptionEn preferred, falls back to Description
    category_id: Optional[int] = None
    category_name: Optional[str] = None
    price: Optional[Decimal] = None
    image_url: Optional[str] = None
    quantity: Optional[int] = None
    seller_id: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# CATEGORY SCHEMAS
# ============================================================================

class CategoryResponse(BaseModel):
    """Schema for category response (read-only from .NET backend)"""
    id: int
    name: str                  # NameAr preferred, falls back to NameEn
    name_en: Optional[str] = None
    name_ar: Optional[str] = None
    image: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)
