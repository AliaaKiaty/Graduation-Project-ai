"""
Image API router
Provides endpoints for visual similarity search and image classification
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request, UploadFile, File, Query
from typing import Annotated

from ..auth import get_current_user, User
from ..models.image import ImageEngine
from ..schemas.image import SimilarImagesResponse, ClassificationResponse
from ..limiter import limiter
from .. import config

router = APIRouter()

# Lazy-loaded image engine
_image_engine = None


def get_image_engine() -> ImageEngine:
    """Get or create the image engine instance."""
    global _image_engine
    if _image_engine is None:
        _image_engine = ImageEngine()
    return _image_engine


@router.post("/similar", response_model=SimilarImagesResponse)
@limiter.limit(config.RATE_LIMIT_GENERAL)
async def find_similar_images(
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
    file: UploadFile = File(..., description="Image file to search for"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of similar images to return"),
) -> SimilarImagesResponse:
    """
    Find visually similar images using ResNet50 features and KNN search.

    Requires authentication.

    Accepts JPEG, PNG, WebP images up to 10MB.

    Args:
        file: Image file upload
        top_k: Number of similar images to return (1-20)

    Returns:
        List of similar images with similarity scores and distances
    """
    image_engine = get_image_engine()

    # Validate content type
    is_valid, error_msg = image_engine.validate_content_type(
        content_type=file.content_type or "application/octet-stream"
    )
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )

    try:
        # Read file contents
        file_bytes = await file.read()
        file_size = len(file_bytes)

        # Validate file size
        is_valid, error_msg = image_engine.validate_file_size(
            file_size=file_size,
            max_size_mb=config.MAX_UPLOAD_SIZE_MB
        )
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=error_msg
            )

        # Find similar images
        similar_images = image_engine.find_similar_images(
            image_bytes=file_bytes,
            top_k=top_k
        )

        return SimilarImagesResponse(
            similar_images=similar_images,
            query_info={
                "filename": file.filename or "unknown",
                "format": file.content_type or "unknown",
                "size_bytes": file_size
            },
            total_results=len(similar_images)
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Image processing service not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )


@router.post("/classify", response_model=ClassificationResponse)
@limiter.limit(config.RATE_LIMIT_GENERAL)
async def classify_image(
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
    file: UploadFile = File(..., description="Image file to classify"),
    top_k: int = Query(default=5, ge=1, le=10, description="Number of top predictions to return"),
) -> ClassificationResponse:
    """
    Classify butterfly image using trained ResNet50 classifier.

    Requires authentication.

    Accepts JPEG, PNG, WebP images up to 10MB.

    Args:
        file: Image file upload
        top_k: Number of top predictions to return (1-10)

    Returns:
        Top-K predictions with class names and confidence scores
    """
    image_engine = get_image_engine()

    # Validate content type
    is_valid, error_msg = image_engine.validate_content_type(
        content_type=file.content_type or "application/octet-stream"
    )
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )

    try:
        # Read file contents
        file_bytes = await file.read()
        file_size = len(file_bytes)

        # Validate file size
        is_valid, error_msg = image_engine.validate_file_size(
            file_size=file_size,
            max_size_mb=config.MAX_UPLOAD_SIZE_MB
        )
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=error_msg
            )

        # Classify image
        predictions = image_engine.classify_image(
            image_bytes=file_bytes,
            top_k=top_k
        )

        # Get top prediction
        top_prediction = predictions[0] if predictions else None
        if not top_prediction:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate predictions"
            )

        return ClassificationResponse(
            predictions=predictions,
            top_prediction=top_prediction["class_name"],
            top_confidence=top_prediction["confidence"],
            query_info={
                "filename": file.filename or "unknown",
                "format": file.content_type or "unknown",
                "size_bytes": file_size
            }
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Image classification service not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error classifying image: {str(e)}"
        )
