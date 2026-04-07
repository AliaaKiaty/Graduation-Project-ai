"""
Admin API router
Provides admin-only endpoints for model retraining and management
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Annotated, List
import subprocess
import sys
from datetime import datetime

from ..auth import require_admin, TokenUser
from ..database import get_db
from ..models.db_models import ModelMetadata
from ..limiter import limiter
from .. import config

router = APIRouter()


# ============================================================================
# DATABASE MIGRATION ENDPOINT
# ============================================================================

@router.post("/migrate-v2", status_code=status.HTTP_200_OK)
@limiter.limit("10/minute")
async def run_migration_v2(
    request: Request,
    current_user: Annotated[TokenUser, Depends(require_admin)],
):
    """
    Run schema v2 migration.

    Alters existing .NET tables to match db schema v2:
    - ProductCategories: add NameEn, NameAr, audit columns; copies Name → NameEn
    - Products: add NameEn, NameAr, DescriptionEn, DescriptionAr, audit columns, RowVersion;
                copies Name → NameEn and Description → DescriptionEn
    - UserInteraction: add audit columns
    - Creates RawMaterialCategories and RawMaterials tables
    - Creates ML-owned tables (product_embeddings, model_metadata)

    Safe to run multiple times — each step is idempotent.
    """
    from ..scripts.migrate_v2 import migrate
    result = migrate(verbose=False)
    return {
        "status": "error" if result["error"] else "ok",
        "error": result["error"],
        "steps": result["steps"],
    }


@router.post("/migrate", status_code=status.HTTP_200_OK)
@limiter.limit("10/minute")
async def run_migration(
    request: Request,
    current_user: Annotated[TokenUser, Depends(require_admin)],
):
    """
    Explicitly create all database tables (idempotent — safe to run repeatedly).
    Use this if the automatic startup migration failed.
    """
    from ..database import Base, engine
    from ..models.db_models import (  # noqa: F401 — imports register with Base.metadata
        ProductEmbedding, ModelMetadata, Product, ProductCategory, UserInteraction
    )
    from sqlalchemy import inspect as sa_inspect

    results = {}
    error = None

    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)

        inspector = sa_inspect(engine)
        for table_name in ["productcategories", "products", "userinteractions",
                           "product_embeddings", "model_metadata"]:
            results[table_name] = inspector.has_table(table_name)

    except Exception as e:
        error = str(e)

    return {
        "status": "error" if error else "ok",
        "error": error,
        "tables_exist": results,
    }


# ============================================================================
# MODEL MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/retrain", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("3/hour")  # Strict rate limit for expensive operation
async def trigger_retraining(
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Annotated[TokenUser, Depends(require_admin)],
    db: Session = Depends(get_db)
):
    """
    Trigger ML model retraining from database.

    Requires admin authentication.

    IMPORTANT: This is a long-running operation that retrains:
    - SVD collaborative filtering model
    - TF-IDF + KMeans content-based model

    The retraining runs in the background. The API will return immediately
    with status 202 (Accepted). Check logs or /admin/models endpoint to
    verify completion.

    Rate limit: 3 requests per hour per user.

    Returns:
        Status message indicating retraining has started
    """
    def run_retraining():
        """Background task to run retraining script"""
        try:
            # Run the retraining script as a subprocess
            result = subprocess.run(
                [sys.executable, "-m", "ml_api.scripts.retrain_models"],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode != 0:
                print(f"Retraining failed with exit code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
            else:
                print("Retraining completed successfully")
                print(f"STDOUT: {result.stdout}")

        except subprocess.TimeoutExpired:
            print("Retraining timeout (exceeded 1 hour)")
        except Exception as e:
            print(f"Retraining error: {e}")

    # Add background task
    background_tasks.add_task(run_retraining)

    return {
        "message": "Model retraining started",
        "status": "in_progress",
        "triggered_by": current_user.user_id,
        "triggered_at": datetime.now().isoformat(),
        "note": "This is a long-running operation. Check /admin/models to verify completion."
    }


@router.get("/models", response_model=List[dict])
@limiter.limit("100/minute")
async def list_model_metadata(
    request: Request,
    current_user: Annotated[TokenUser, Depends(require_admin)],
    db: Session = Depends(get_db),
    model_type: str = None,
    is_active: bool = None
):
    """
    List all trained model metadata.

    Requires admin authentication.

    Provides information about all trained models including:
    - Model type (svd, tfidf_kmeans)
    - Version (timestamp)
    - Training date
    - Evaluation metrics (RMSE, etc.)
    - Training duration
    - Active status

    Args:
        model_type: Optional filter by model type ("svd" or "tfidf_kmeans")
        is_active: Optional filter by active status

    Returns:
        List of model metadata records
    """
    try:
        # Build query
        query = db.query(ModelMetadata)

        # Apply filters
        if model_type:
            query = query.filter(ModelMetadata.model_type == model_type)

        if is_active is not None:
            query = query.filter(ModelMetadata.is_active == is_active)

        # Order by training date (most recent first)
        models = query.order_by(ModelMetadata.training_date.desc()).all()

        # Format response
        result = []
        for model in models:
            model_dict = {
                "id": model.id,
                "model_type": model.model_type,
                "version": model.version,
                "file_path": model.file_path,
                "training_date": model.training_date.isoformat(),
                "is_active": model.is_active,
                "total_products": model.total_products,
                "total_ratings": model.total_ratings,
                "training_duration_seconds": model.training_duration_seconds,
                "notes": model.notes,
                "created_at": model.created_at.isoformat()
            }

            # Add type-specific metrics
            if model.model_type == "svd":
                model_dict["n_components"] = model.n_components
                model_dict["rmse"] = float(model.rmse) if model.rmse else None
                model_dict["precision_at_10"] = float(model.precision_at_10) if model.precision_at_10 else None
                model_dict["recall_at_10"] = float(model.recall_at_10) if model.recall_at_10 else None
                model_dict["ndcg_at_10"] = float(model.ndcg_at_10) if model.ndcg_at_10 else None
                model_dict["coverage"] = float(model.coverage) if model.coverage else None

            elif model.model_type == "tfidf_kmeans":
                model_dict["n_clusters"] = model.n_clusters
                model_dict["max_features"] = model.max_features

            result.append(model_dict)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching model metadata: {str(e)}"
        )


@router.get("/models/{model_id}", response_model=dict)
@limiter.limit("100/minute")
async def get_model_metadata(
    request: Request,
    model_id: int,
    current_user: Annotated[TokenUser, Depends(require_admin)],
    db: Session = Depends(get_db)
):
    """
    Get detailed metadata for a specific model.

    Requires admin authentication.

    Args:
        model_id: Model metadata ID

    Returns:
        Detailed model metadata
    """
    try:
        model = db.query(ModelMetadata).filter(ModelMetadata.id == model_id).first()

        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model with ID {model_id} not found"
            )

        # Format response
        model_dict = {
            "id": model.id,
            "model_type": model.model_type,
            "version": model.version,
            "file_path": model.file_path,
            "training_date": model.training_date.isoformat(),
            "is_active": model.is_active,
            "total_products": model.total_products,
            "total_ratings": model.total_ratings,
            "training_duration_seconds": model.training_duration_seconds,
            "notes": model.notes,
            "created_at": model.created_at.isoformat()
        }

        # Add type-specific metrics
        if model.model_type == "svd":
            model_dict["n_components"] = model.n_components
            model_dict["rmse"] = float(model.rmse) if model.rmse else None
            model_dict["precision_at_10"] = float(model.precision_at_10) if model.precision_at_10 else None
            model_dict["recall_at_10"] = float(model.recall_at_10) if model.recall_at_10 else None
            model_dict["ndcg_at_10"] = float(model.ndcg_at_10) if model.ndcg_at_10 else None
            model_dict["coverage"] = float(model.coverage) if model.coverage else None

        elif model.model_type == "tfidf_kmeans":
            model_dict["n_clusters"] = model.n_clusters
            model_dict["max_features"] = model.max_features

        return model_dict

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching model metadata: {str(e)}"
        )


# ============================================================================
# SEED DATA ENDPOINT
# ============================================================================

@router.post("/seed", status_code=status.HTTP_200_OK)
@limiter.limit("5/hour")
async def seed_database(
    request: Request,
    current_user: Annotated[TokenUser, Depends(require_admin)],
    clear: bool = False,
):
    """
    Insert test seed data into the database.

    Inserts 8 categories, 40 products, ~300 user interactions, and
    product_embeddings (cluster IDs predicted by the live TF-IDF + KMeans model).

    Args:
        clear: If true, delete existing seed records before inserting.

    Returns:
        Summary of inserted/skipped records.
    """
    try:
        from ..scripts.seed_data import seed
        result = seed(clear_existing=clear)
        return {"status": "ok", "summary": result}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Seed failed: {str(e)}"
        )


@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("100/minute")
async def deactivate_model(
    request: Request,
    model_id: int,
    current_user: Annotated[TokenUser, Depends(require_admin)],
    db: Session = Depends(get_db)
):
    """
    Deactivate a model (soft delete by setting is_active=False).

    Requires admin authentication.

    This does NOT delete the model files, only marks the model as inactive
    in the database.

    Args:
        model_id: Model metadata ID

    Returns:
        No content (204)
    """
    try:
        model = db.query(ModelMetadata).filter(ModelMetadata.id == model_id).first()

        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model with ID {model_id} not found"
            )

        # Soft delete (set is_active=False)
        model.is_active = False
        db.commit()

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deactivating model: {str(e)}"
        )
