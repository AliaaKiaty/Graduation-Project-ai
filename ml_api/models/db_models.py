"""
Database models for ML API
Read-only mirrors of the real .NET backend schema + ML-owned tables
"""
from sqlalchemy import Column, Integer, String, Text, Numeric, Boolean, DateTime, BigInteger, Float, ForeignKey, CheckConstraint, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSON
from ..database import Base


# =============================================================================
# READ-ONLY MIRRORS OF .NET BACKEND TABLES
# These tables are managed by the .NET backend (EF Core).
# The ML API only reads from them — never creates/alters/deletes rows.
# Column names use PascalCase to match ASP.NET EF Core conventions.
# =============================================================================

class ProductCategory(Base):
    """Read-only mirror of .NET ProductCategories table"""
    __tablename__ = "ProductCategories"

    Id = Column("Id", Integer, primary_key=True)
    Name = Column("Name", String(255), nullable=False)
    Image = Column("Image", String(1000), nullable=True)

    # Relationships (read-only)
    products = relationship("Product", back_populates="category", viewonly=True)

    def __repr__(self):
        return f"<ProductCategory(Id={self.Id}, Name='{self.Name}')>"


class Product(Base):
    """Read-only mirror of .NET Products table"""
    __tablename__ = "Products"

    Id = Column("Id", Integer, primary_key=True)
    Name = Column("Name", String(500), nullable=False)
    ImageUrl = Column("ImageUrl", String(1000), nullable=True)
    Quantity = Column("Quantity", Integer, nullable=True)
    Price = Column("Price", Numeric(18, 2), nullable=True)
    Description = Column("Description", Text, nullable=True)
    SellerID = Column("SellerID", String(450), nullable=True)
    CategoryId = Column("CategoryId", Integer, ForeignKey("ProductCategories.Id"), nullable=True)

    # Relationships (read-only for .NET tables)
    category = relationship("ProductCategory", back_populates="products", viewonly=True)
    interactions = relationship("UserInteraction", back_populates="product", viewonly=True)

    # ML-owned relationship
    embedding = relationship("ProductEmbedding", back_populates="product", uselist=False)

    def __repr__(self):
        return f"<Product(Id={self.Id}, Name='{self.Name[:50]}')>"


class UserInteraction(Base):
    """Read-only mirror of .NET UserInteraction table (ratings/favourites)"""
    __tablename__ = "UserInteraction"

    Id = Column("Id", Integer, primary_key=True)
    UserId = Column("UserId", String(450), nullable=True)
    ProductID = Column("ProductID", Integer, ForeignKey("Products.Id"), nullable=True)
    RawMaterialID = Column("RawMaterialID", Integer, nullable=True)
    IsFavourite = Column("IsFavourite", Boolean, nullable=True)
    Rating = Column("Rating", Integer, nullable=True)
    Review = Column("Review", Text, nullable=True)
    InteractionDate = Column("InteractionDate", DateTime, nullable=True)
    TargetUserId = Column("TargetUserId", String(450), nullable=True)

    # Relationships (read-only)
    product = relationship("Product", back_populates="interactions", viewonly=True)

    def __repr__(self):
        return f"<UserInteraction(Id={self.Id}, UserId='{self.UserId}', ProductID={self.ProductID}, Rating={self.Rating})>"


# =============================================================================
# ML-OWNED TABLES
# These tables are created and managed by the ML API.
# They use snake_case naming (ML API convention).
# =============================================================================

class ProductEmbedding(Base):
    """ML-owned: Product embedding and cluster assignment for content-based filtering"""
    __tablename__ = "product_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("Products.Id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    cluster_id = Column(Integer, nullable=False, index=True)
    embedding_vector = Column(JSON)
    last_updated = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    product = relationship("Product", back_populates="embedding")

    def __repr__(self):
        return f"<ProductEmbedding(id={self.id}, product_id={self.product_id}, cluster_id={self.cluster_id})>"


class ModelMetadata(Base):
    """ML-owned: ML model metadata and versioning"""
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String(50), nullable=False, index=True)  # 'svd', 'tfidf', 'kmeans'
    version = Column(String(50), nullable=False)
    file_path = Column(String(500), nullable=False)
    training_date = Column(DateTime, nullable=False)

    # Model-specific parameters
    n_components = Column(Integer)
    n_clusters = Column(Integer)
    max_features = Column(Integer)

    # Evaluation metrics
    rmse = Column(Numeric(10, 4))
    precision_at_10 = Column(Numeric(10, 4))
    recall_at_10 = Column(Numeric(10, 4))
    ndcg_at_10 = Column(Numeric(10, 4))
    coverage = Column(Numeric(10, 4))

    # Training stats
    total_products = Column(Integer)
    total_ratings = Column(BigInteger)
    training_duration_seconds = Column(Integer)

    is_active = Column(Boolean, default=False, nullable=False, index=True)
    notes = Column(Text)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Only one active model per type
    __table_args__ = (
        Index('idx_unique_active_model', 'model_type', 'is_active', unique=True, postgresql_where=(is_active == True)),
    )

    def __repr__(self):
        return f"<ModelMetadata(id={self.id}, model_type='{self.model_type}', version='{self.version}', is_active={self.is_active})>"
