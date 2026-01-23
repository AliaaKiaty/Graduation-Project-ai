"""
Recommendation API router
Provides endpoints for popularity, collaborative filtering, and content-based recommendations
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from typing import Annotated

from ..auth import get_current_user, require_admin, User
from ..models.recommendation import RecommendationEngine
from ..schemas.recommendation import (
    PopularResponse,
    CollaborativeRequest,
    CollaborativeResponse,
    ContentBasedRequest,
    ContentBasedResponse,
    DatasetType
)
from ..limiter import limiter
from .. import config

router = APIRouter()

# Lazy-loaded recommendation engine
_rec_engine = None


def get_rec_engine() -> RecommendationEngine:
    """Get or create the recommendation engine instance."""
    global _rec_engine
    if _rec_engine is None:
        _rec_engine = RecommendationEngine()
    return _rec_engine


@router.get("/popular", response_model=PopularResponse)
@limiter.limit(config.RATE_LIMIT_GENERAL)
async def get_popular_items(
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
    dataset: DatasetType = Query(default=DatasetType.english, description="Dataset to use"),
    top_n: int = Query(default=10, ge=1, le=100, description="Number of items to return"),
) -> PopularResponse:
    """
    Get most popular items based on rating counts.

    Requires authentication.

    Args:
        dataset: Dataset to use ("english" or "arabic")
        top_n: Number of top items to return (1-100)

    Returns:
        List of popular items with rating counts
    """
    try:
        rec_engine = get_rec_engine()
        recommendations = rec_engine.get_popular_items(
            dataset=dataset.value,
            top_n=top_n
        )

        return PopularResponse(
            recommendations=recommendations,
            dataset=dataset.value,
            method="popularity",
            total_results=len(recommendations)
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Recommendation model not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {str(e)}"
        )


@router.post("/collaborative", response_model=CollaborativeResponse)
@limiter.limit(config.RATE_LIMIT_GENERAL)
async def get_collaborative_recommendations(
    request: Request,
    req_data: CollaborativeRequest,
    current_user: Annotated[User, Depends(get_current_user)],
) -> CollaborativeResponse:
    """
    Get collaborative filtering recommendations using SVD and correlation matrix.

    Requires authentication.

    Finds similar products based on user rating patterns using matrix factorization.

    Args:
        req_data: Request with product_id, dataset, top_n, and min_correlation

    Returns:
        List of similar products with correlation scores
    """
    try:
        rec_engine = get_rec_engine()
        recommendations = rec_engine.get_collaborative_recommendations(
            product_id=req_data.product_id,
            dataset=req_data.dataset.value,
            top_n=req_data.top_n,
            min_correlation=req_data.min_correlation
        )

        return CollaborativeResponse(
            input_product=req_data.product_id,
            recommendations=recommendations,
            dataset=req_data.dataset.value,
            method="svd_collaborative_filtering",
            total_results=len(recommendations)
        )

    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Recommendation model not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {str(e)}"
        )


@router.post("/content-based", response_model=ContentBasedResponse)
@limiter.limit(config.RATE_LIMIT_GENERAL)
async def get_content_based_recommendations(
    request: Request,
    req_data: ContentBasedRequest,
    current_user: Annotated[User, Depends(get_current_user)],
) -> ContentBasedResponse:
    """
    Get content-based recommendations using TF-IDF and KMeans clustering.

    Requires authentication.

    Finds products similar to the search query by clustering product descriptions.

    Args:
        req_data: Request with search_query, dataset, and top_n

    Returns:
        Predicted cluster, keywords, and list of similar products
    """
    try:
        rec_engine = get_rec_engine()
        result = rec_engine.get_content_based_recommendations(
            search_query=req_data.search_query,
            dataset=req_data.dataset.value,
            top_n=req_data.top_n
        )

        return ContentBasedResponse(
            search_query=result["search_query"],
            predicted_cluster=result["predicted_cluster"],
            cluster_keywords=result["cluster_keywords"],
            recommendations=result["recommendations"],
            dataset=req_data.dataset.value,
            method="tfidf_kmeans",
            total_results=result["total_results"]
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Recommendation model not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {str(e)}"
        )
