"""
Database configuration and session management
Connects to the real .NET backend database (read-only for .NET tables)
Only creates ML-specific tables (product_embeddings, model_metadata)
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import StaticPool
from . import config


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models"""
    pass

# Create SQLAlchemy engine (pool params only for non-SQLite)
_engine_kwargs = {}
if config.DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
    _engine_kwargs["poolclass"] = StaticPool
else:
    _engine_kwargs.update(
        pool_size=config.DB_POOL_SIZE,
        max_overflow=config.DB_MAX_OVERFLOW,
        pool_timeout=config.DB_POOL_TIMEOUT,
        pool_recycle=config.DB_POOL_RECYCLE,
    )
engine = create_engine(config.DATABASE_URL, **_engine_kwargs)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """
    Dependency function to get database session.
    Use in FastAPI endpoints with Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database — create all tables if they don't exist.
    .NET-managed tables (Products, ProductCategories, UserInteraction) are created
    here when the .NET backend hasn't run yet (standalone mode).
    Uses checkfirst=True so existing tables are never altered.
    Wrapped in try/except to handle race conditions with multiple workers.
    """
    # Import all models to register them with SQLAlchemy
    from .models.db_models import ProductEmbedding, ModelMetadata, Product, ProductCategory, UserInteraction  # noqa: F401

    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
    except Exception as e:
        # Handle race condition when multiple workers try to create tables simultaneously
        print(f"Note: DB init encountered an error (likely concurrent init): {e}")
