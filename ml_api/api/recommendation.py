"""
Recommendation API router
Provides endpoints for popularity, collaborative filtering, content-based recommendations
Products and categories are read-only (managed by .NET backend)
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from sqlalchemy.orm import Session
from typing import Annotated, List, Optional

from ..auth import get_current_user, TokenUser
from ..database import get_db
from ..models.recommendation import RecommendationEngine
from ..models.db_models import Product, ProductCategory
from ..schemas.recommendation import (
    PopularRequest,
    PopularResponse,
    PopularItem,
    CollaborativeRequest,
    CollaborativeResponse,
    CollaborativeItem,
    ContentBasedRequest,
    ContentBasedResponse,
    ContentBasedItem,
    ProductResponse,
    CategoryResponse
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


# ============================================================================
# RECOMMENDATION ENDPOINTS
# ============================================================================

@router.post("/popular", response_model=PopularResponse)
@limiter.limit(config.RATE_LIMIT_GENERAL)
async def get_popular_items(
    request: Request,
    req_data: PopularRequest,
    current_user: Annotated[TokenUser, Depends(get_current_user)],
    db: Session = Depends(get_db)
) -> PopularResponse:
    """
    Get most popular items based on rating counts from database.
    Requires authentication.
    """
    try:
        rec_engine = get_rec_engine()
        recommendations = rec_engine.get_popular_items(
            top_n=req_data.top_n,
            category_id=req_data.category_id,
            db=db
        )

        return PopularResponse(
            recommendations=[PopularItem(**rec) for rec in recommendations],
            method="popularity",
            total_results=len(recommendations),
            category_filter=req_data.category_id
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
    current_user: Annotated[TokenUser, Depends(get_current_user)],
    db: Session = Depends(get_db)
) -> CollaborativeResponse:
    """
    Get collaborative filtering recommendations using SVD and correlation matrix.
    Requires authentication.
    """
    try:
        rec_engine = get_rec_engine()
        result = rec_engine.get_collaborative_recommendations(
            product_id=req_data.product_id,
            top_n=req_data.top_n,
            min_correlation=req_data.min_correlation,
            category_id=req_data.category_id,
            db=db
        )

        return CollaborativeResponse(
            input_product_id=result["input_product_id"],
            input_product_name=result["input_product_name"],
            recommendations=[CollaborativeItem(**rec) for rec in result["recommendations"]],
            method="svd_collaborative_filtering",
            total_results=len(result["recommendations"])
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
    current_user: Annotated[TokenUser, Depends(get_current_user)],
    db: Session = Depends(get_db)
) -> ContentBasedResponse:
    """
    Get content-based recommendations using TF-IDF and KMeans clustering.
    Requires authentication.
    """
    try:
        rec_engine = get_rec_engine()
        result = rec_engine.get_content_based_recommendations(
            search_query=req_data.search_query,
            top_n=req_data.top_n,
            category_id=req_data.category_id,
            db=db
        )

        return ContentBasedResponse(
            search_query=result["search_query"],
            predicted_cluster=result["predicted_cluster"],
            cluster_keywords=result["cluster_keywords"],
            recommendations=[ContentBasedItem(**rec) for rec in result["recommendations"]],
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


# ============================================================================
# READ-ONLY PRODUCT ENDPOINTS
# ============================================================================

@router.get("/products", response_model=List[ProductResponse])
@limiter.limit(config.RATE_LIMIT_GENERAL)
async def list_products(
    request: Request,
    current_user: Annotated[TokenUser, Depends(get_current_user)],
    db: Session = Depends(get_db),
    skip: int = Query(default=0, ge=0, description="Number of products to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of products to return"),
    search: Optional[str] = Query(default=None, description="Search by product name or description"),
    category_id: Optional[int] = Query(default=None, description="Filter by category ID"),
) -> List[ProductResponse]:
    """
    List products with pagination, search, and filtering.
    Read-only — products are managed by the .NET backend.
    Requires authentication.
    """
    try:
        query = db.query(Product, ProductCategory.NameAr.label('category_name')).outerjoin(
            ProductCategory, Product.CategoryId == ProductCategory.Id
        )

        if search:
            search_filter = f"%{search}%"
            query = query.filter(
                (Product.NameEn.ilike(search_filter)) |
                (Product.NameAr.ilike(search_filter)) |
                (Product.DescriptionEn.ilike(search_filter)) |
                (Product.Description.ilike(search_filter))
            )

        if category_id is not None:
            query = query.filter(Product.CategoryId == category_id)

        products = query.offset(skip).limit(limit).all()

        result = []
        for row in products:
            product = row.Product
            category_name = row.category_name
            result.append(ProductResponse(
                id=product.Id,
                name=product.NameAr or product.NameEn or "",
                name_en=product.NameEn,
                name_ar=product.NameAr,
                description=product.DescriptionEn or product.Description,
                category_id=product.CategoryId,
                category_name=category_name,
                price=product.Price,
                image_url=product.ImageUrl,
                quantity=product.Quantity,
                seller_id=product.SellerID,
            ))

        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching products: {str(e)}"
        )


@router.get("/products/{product_id}", response_model=ProductResponse)
@limiter.limit(config.RATE_LIMIT_GENERAL)
async def get_product(
    request: Request,
    product_id: int,
    current_user: Annotated[TokenUser, Depends(get_current_user)],
    db: Session = Depends(get_db)
) -> ProductResponse:
    """
    Get a single product by ID.
    Read-only — products are managed by the .NET backend.
    Requires authentication.
    """
    try:
        result = (
            db.query(Product, ProductCategory.NameAr.label('category_name'))
            .outerjoin(ProductCategory, Product.CategoryId == ProductCategory.Id)
            .filter(Product.Id == product_id)
            .first()
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product with ID {product_id} not found"
            )

        product = result.Product
        category_name = result.category_name

        return ProductResponse(
            id=product.Id,
            name=product.NameAr or product.NameEn or "",
            name_en=product.NameEn,
            name_ar=product.NameAr,
            description=product.DescriptionEn or product.Description,
            category_id=product.CategoryId,
            category_name=category_name,
            price=product.Price,
            image_url=product.ImageUrl,
            quantity=product.Quantity,
            seller_id=product.SellerID,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching product: {str(e)}"
        )


# ============================================================================
# READ-ONLY CATEGORY ENDPOINTS
# ============================================================================

@router.get("/categories", response_model=List[CategoryResponse])
@limiter.limit(config.RATE_LIMIT_GENERAL)
async def list_categories(
    request: Request,
    current_user: Annotated[TokenUser, Depends(get_current_user)],
    db: Session = Depends(get_db),
) -> List[CategoryResponse]:
    """
    List all product categories.
    Read-only — categories are managed by the .NET backend.
    Requires authentication.
    """
    try:
        categories = db.query(ProductCategory).all()

        return [
            CategoryResponse(
                id=cat.Id,
                name=cat.NameAr or cat.NameEn or "",
                name_en=cat.NameEn,
                name_ar=cat.NameAr,
                image=cat.Image,
            )
            for cat in categories
        ]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching categories: {str(e)}"
        )
