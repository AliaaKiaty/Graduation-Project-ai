"""
Database models for ML API
Read-only mirrors of the real .NET backend schema + ML-owned tables

Real backend tables use lowercase names and snake_case columns.
Python attribute names stay PascalCase for readability; Column("db_col_name")
maps them to the actual database column names.
"""
from sqlalchemy import (
    Column, Integer, SmallInteger, String, Text, Numeric, Boolean,
    DateTime, BigInteger, Float, ForeignKey, LargeBinary, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSON
from ..database import Base


# =============================================================================
# READ-ONLY MIRRORS OF .NET BACKEND TABLES  (lowercase tables, lowercase cols)
# These tables are managed by the .NET backend.
# The ML API only reads from them — never creates/alters/deletes rows.
# =============================================================================

class ProductCategory(Base):
    """Read-only mirror of .NET productcategories table"""
    __tablename__ = "productcategories"

    Id = Column("id", Integer, primary_key=True)
    NameEn = Column("nameen", String(255), nullable=False)
    NameAr = Column("namear", String(255), nullable=True)
    Image = Column("image", String(1000), nullable=True)
    CreatedAt = Column("createdat", DateTime, nullable=True)
    CreatedBy = Column("createdby", String(450), nullable=True)
    IsDeleted = Column("isdeleted", Boolean, nullable=False, default=False)
    UpdatedAt = Column("updatedat", DateTime, nullable=True)

    # Relationships (read-only)
    products = relationship("Product", back_populates="category", viewonly=True)

    def __repr__(self):
        return f"<ProductCategory(Id={self.Id}, NameEn='{self.NameEn}')>"


class Product(Base):
    """Read-only mirror of .NET products table"""
    __tablename__ = "products"

    Id = Column("id", Integer, primary_key=True)
    NameEn = Column("nameen", String(500), nullable=False)
    NameAr = Column("namear", String(500), nullable=True)
    ImageUrl = Column("imageurl", String(1000), nullable=True)
    Quantity = Column("quantity", Integer, nullable=True)
    Price = Column("price", Numeric(18, 2), nullable=False)
    DescriptionEn = Column("descriptionen", Text, nullable=True)
    DescriptionAr = Column("descriptionar", Text, nullable=True)
    SellerID = Column("sellerid", String(450), nullable=False)
    CategoryId = Column("categoryid", Integer, ForeignKey("productcategories.id"), nullable=False)
    CreatedAt = Column("createdat", DateTime, nullable=True)
    CreatedBy = Column("createdby", String(450), nullable=True)
    IsDeleted = Column("isdeleted", Boolean, nullable=False, default=False)
    UpdatedAt = Column("updatedat", DateTime, nullable=True)

    # Relationships (read-only for .NET tables)
    category = relationship("ProductCategory", back_populates="products", viewonly=True)
    interactions = relationship("UserInteraction", back_populates="product", viewonly=True)

    # ML-owned relationship
    embedding = relationship("ProductEmbedding", back_populates="product", uselist=False)

    def __repr__(self):
        name = self.NameEn or self.NameAr or ""
        return f"<Product(Id={self.Id}, Name='{name[:50]}')>"


class RawMaterialCategory(Base):
    """Read-only mirror of .NET rawmaterialcategories table"""
    __tablename__ = "rawmaterialcategories"

    Id = Column("id", Integer, primary_key=True)
    NameEn = Column("nameen", String(255), nullable=False)
    NameAr = Column("namear", String(255), nullable=True)
    Image = Column("image", String(1000), nullable=True)
    CreatedAt = Column("createdat", DateTime, nullable=True)
    CreatedBy = Column("createdby", String(450), nullable=True)
    IsDeleted = Column("isdeleted", Boolean, nullable=False, default=False)
    UpdatedAt = Column("updatedat", DateTime, nullable=True)

    # Relationships (read-only)
    raw_materials = relationship("RawMaterial", back_populates="category", viewonly=True)

    def __repr__(self):
        return f"<RawMaterialCategory(Id={self.Id}, NameEn='{self.NameEn}')>"


class RawMaterial(Base):
    """Read-only mirror of .NET rawmaterials table"""
    __tablename__ = "rawmaterials"

    Id = Column("id", Integer, primary_key=True)
    NameEn = Column("nameen", String(500), nullable=False)
    NameAr = Column("namear", String(500), nullable=True)
    ImageUrl = Column("imageurl", String(1000), nullable=True)
    Quantity = Column("quantity", Integer, nullable=True)
    Price = Column("price", Numeric(18, 2), nullable=False)
    DescriptionEn = Column("descriptionen", Text, nullable=True)
    DescriptionAr = Column("descriptionar", Text, nullable=True)
    SupplierID = Column("supplierid", String(450), nullable=False)
    CategoryId = Column("categoryid", Integer, ForeignKey("rawmaterialcategories.id"), nullable=False)
    IsDeleted = Column("isdeleted", Boolean, nullable=False, default=False)
    UpdatedAt = Column("updatedat", DateTime, nullable=True)

    # Relationships (read-only)
    category = relationship("RawMaterialCategory", back_populates="raw_materials", viewonly=True)

    def __repr__(self):
        name = self.NameEn or self.NameAr or ""
        return f"<RawMaterial(Id={self.Id}, Name='{name[:50]}')>"


class UserInteraction(Base):
    """Read-only mirror of .NET userinteractions table (ratings/reviews).
    NOTE: 'favourites' are in a separate 'favourites' table, not here.
    """
    __tablename__ = "userinteractions"   # lowercase, plural

    Id = Column("id", Integer, primary_key=True)
    UserId = Column("userid", String(450), nullable=False)
    ProductID = Column("productid", Integer, ForeignKey("products.id"), nullable=True)
    RawMaterialID = Column("rawmaterialid", Integer, ForeignKey("rawmaterials.id"), nullable=True)
    TargetUserId = Column("targetuserid", String(450), nullable=True)
    Rating = Column("rating", SmallInteger, nullable=True)
    Review = Column("review", Text, nullable=True)
    InteractionDate = Column("interactiondate", DateTime, nullable=False)
    CreatedAt = Column("createdat", DateTime, nullable=False)
    CreatedBy = Column("createdby", String(450), nullable=True)
    IsDeleted = Column("isdeleted", Boolean, nullable=False, default=False)
    UpdatedAt = Column("updatedat", DateTime, nullable=False)

    # Relationships (read-only)
    product = relationship("Product", back_populates="interactions", viewonly=True)

    def __repr__(self):
        return f"<UserInteraction(Id={self.Id}, UserId='{self.UserId}', ProductID={self.ProductID}, Rating={self.Rating})>"


# =============================================================================
# ML-OWNED TABLES
# Created and managed by the ML API (snake_case naming convention).
# =============================================================================

class ProductEmbedding(Base):
    """ML-owned: Product embedding and cluster assignment for content-based filtering"""
    __tablename__ = "product_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(
        Integer,
        ForeignKey("products.id", ondelete="CASCADE"),   # real lowercase products table
        nullable=False,
        unique=True,
        index=True
    )
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
    model_type = Column(String(50), nullable=False, index=True)  # 'svd', 'tfidf_kmeans'
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

    __table_args__ = (
        Index('idx_model_type_active', 'model_type', 'is_active'),
    )

    def __repr__(self):
        return f"<ModelMetadata(id={self.id}, model_type='{self.model_type}', version='{self.version}', is_active={self.is_active})>"
