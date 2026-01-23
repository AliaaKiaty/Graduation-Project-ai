"""
Database configuration and session management
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from pathlib import Path
from .config import DATABASE_URL, DATABASE_PATH


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models"""
    pass

# Create data directory if it doesn't exist
data_dir = Path(DATABASE_PATH).parent
data_dir.mkdir(parents=True, exist_ok=True)

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

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
    Initialize database - create all tables.
    Call this once when starting the application.
    """
    from .auth.models import User  # Import all models to register them
    Base.metadata.create_all(bind=engine)
