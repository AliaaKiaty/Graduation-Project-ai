"""
Unit tests for database models
Tests for ProductCategory, Product, UserInteraction, ProductEmbedding, and ModelMetadata models
"""
import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from ml_api.database import Base
from ml_api.models.db_models import ProductCategory, Product, UserInteraction, ProductEmbedding, ModelMetadata


# Test database setup (use in-memory SQLite for tests)
@pytest.fixture(scope="function")
def test_db():
    """Create a test database session"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    TestingSessionLocal = sessionmaker(bind=engine)
    db = TestingSessionLocal()

    yield db

    db.close()
    Base.metadata.drop_all(engine)


# ============================================================================
# PRODUCT CATEGORY MODEL TESTS
# ============================================================================

def test_create_product_category(test_db):
    """Test creating a product category"""
    category = ProductCategory(
        Id=1,
        Name="Tools",
        Image="tools.png"
    )

    test_db.add(category)
    test_db.commit()
    test_db.refresh(category)

    assert category.Id == 1
    assert category.Name == "Tools"
    assert category.Image == "tools.png"


def test_create_category_without_image(test_db):
    """Test creating a category without image"""
    category = ProductCategory(
        Id=1,
        Name="Tools"
    )

    test_db.add(category)
    test_db.commit()
    test_db.refresh(category)

    assert category.Id == 1
    assert category.Name == "Tools"
    assert category.Image is None


# ============================================================================
# PRODUCT MODEL TESTS
# ============================================================================

def test_create_product(test_db):
    """Test creating a product"""
    product = Product(
        Id=1,
        Name="Professional Hammer",
        Description="Heavy-duty professional hammer",
        Price=29.99,
        Quantity=100
    )

    test_db.add(product)
    test_db.commit()
    test_db.refresh(product)

    assert product.Id == 1
    assert product.Name == "Professional Hammer"
    assert product.Description == "Heavy-duty professional hammer"
    assert product.Quantity == 100


def test_create_product_with_category(test_db):
    """Test creating a product with category foreign key"""
    category = ProductCategory(Id=1, Name="Tools")
    test_db.add(category)
    test_db.commit()

    product = Product(
        Id=1,
        Name="Professional Hammer",
        CategoryId=1,
        Price=29.99
    )

    test_db.add(product)
    test_db.commit()
    test_db.refresh(product)

    assert product.CategoryId == 1
    assert product.category.Name == "Tools"
    assert len(category.products) == 1


def test_create_product_with_seller(test_db):
    """Test creating a product with seller ID"""
    product = Product(
        Id=1,
        Name="Professional Hammer",
        SellerID="seller-abc-123",
        ImageUrl="https://example.com/hammer.jpg"
    )

    test_db.add(product)
    test_db.commit()
    test_db.refresh(product)

    assert product.SellerID == "seller-abc-123"
    assert product.ImageUrl == "https://example.com/hammer.jpg"


# ============================================================================
# USER INTERACTION MODEL TESTS
# ============================================================================

def test_create_user_interaction_with_rating(test_db):
    """Test creating a user interaction with rating"""
    product = Product(Id=1, Name="Test Product")
    test_db.add(product)
    test_db.commit()

    interaction = UserInteraction(
        Id=1,
        UserId="user-123",
        ProductID=1,
        Rating=5,
        InteractionDate=datetime.now()
    )

    test_db.add(interaction)
    test_db.commit()
    test_db.refresh(interaction)

    assert interaction.Id == 1
    assert interaction.UserId == "user-123"
    assert interaction.ProductID == 1
    assert interaction.Rating == 5


def test_create_user_interaction_with_favourite(test_db):
    """Test creating a user interaction with favourite"""
    product = Product(Id=1, Name="Test Product")
    test_db.add(product)
    test_db.commit()

    interaction = UserInteraction(
        Id=1,
        UserId="user-123",
        ProductID=1,
        IsFavourite=True
    )

    test_db.add(interaction)
    test_db.commit()
    test_db.refresh(interaction)

    assert interaction.IsFavourite is True
    assert interaction.Rating is None


def test_create_user_interaction_with_review(test_db):
    """Test creating a user interaction with review"""
    product = Product(Id=1, Name="Test Product")
    test_db.add(product)
    test_db.commit()

    interaction = UserInteraction(
        Id=1,
        UserId="user-123",
        ProductID=1,
        Rating=4,
        Review="Great product!"
    )

    test_db.add(interaction)
    test_db.commit()
    test_db.refresh(interaction)

    assert interaction.Review == "Great product!"
    assert interaction.product.Name == "Test Product"


# ============================================================================
# PRODUCT EMBEDDING MODEL TESTS
# ============================================================================

def test_create_product_embedding(test_db):
    """Test creating a product embedding with cluster_id"""
    product = Product(Id=1, Name="Test Product")
    test_db.add(product)
    test_db.commit()

    embedding = ProductEmbedding(
        product_id=1,
        cluster_id=5
    )

    test_db.add(embedding)
    test_db.commit()
    test_db.refresh(embedding)

    assert embedding.id is not None
    assert embedding.product_id == 1
    assert embedding.cluster_id == 5
    assert embedding.product.Name == "Test Product"


def test_product_embedding_unique_product(test_db):
    """Test that each product can only have one embedding"""
    product = Product(Id=1, Name="Test Product")
    test_db.add(product)
    test_db.commit()

    embedding1 = ProductEmbedding(product_id=1, cluster_id=1)
    test_db.add(embedding1)
    test_db.commit()

    embedding2 = ProductEmbedding(product_id=1, cluster_id=2)
    test_db.add(embedding2)

    with pytest.raises(IntegrityError):
        test_db.commit()


# ============================================================================
# MODEL METADATA TESTS
# ============================================================================

def test_create_svd_model_metadata(test_db):
    """Test creating model metadata for SVD model"""
    metadata = ModelMetadata(
        model_type="svd",
        version="20240101_120000",
        file_path="/path/to/svd_model.pkl",
        training_date=datetime.now(),
        n_components=10,
        total_products=1000,
        total_ratings=50000,
        rmse=0.85,
        training_duration_seconds=300,
        is_active=True,
        notes="Test SVD model"
    )

    test_db.add(metadata)
    test_db.commit()
    test_db.refresh(metadata)

    assert metadata.id is not None
    assert metadata.model_type == "svd"
    assert metadata.n_components == 10
    assert float(metadata.rmse) == 0.85
    assert metadata.is_active is True


def test_create_tfidf_kmeans_model_metadata(test_db):
    """Test creating model metadata for TF-IDF + KMeans model"""
    metadata = ModelMetadata(
        model_type="tfidf_kmeans",
        version="20240101_120000",
        file_path="/path/to/tfidf_model.pkl",
        training_date=datetime.now(),
        n_clusters=20,
        max_features=5000,
        total_products=1000,
        training_duration_seconds=120,
        is_active=True,
        notes="Test TF-IDF + KMeans model"
    )

    test_db.add(metadata)
    test_db.commit()
    test_db.refresh(metadata)

    assert metadata.id is not None
    assert metadata.model_type == "tfidf_kmeans"
    assert metadata.n_clusters == 20
    assert metadata.max_features == 5000
    assert metadata.is_active is True


def test_model_metadata_versioning(test_db):
    """Test that multiple versions of same model type can exist"""
    metadata_v1 = ModelMetadata(
        model_type="svd",
        version="20240101_120000",
        file_path="/path/to/v1.pkl",
        training_date=datetime.now(),
        is_active=False
    )
    test_db.add(metadata_v1)
    test_db.commit()

    metadata_v2 = ModelMetadata(
        model_type="svd",
        version="20240102_120000",
        file_path="/path/to/v2.pkl",
        training_date=datetime.now(),
        is_active=True
    )
    test_db.add(metadata_v2)
    test_db.commit()

    svd_models = test_db.query(ModelMetadata).filter(ModelMetadata.model_type == "svd").all()
    assert len(svd_models) == 2

    active_model = test_db.query(ModelMetadata).filter(
        ModelMetadata.model_type == "svd",
        ModelMetadata.is_active == True
    ).first()
    assert active_model.version == "20240102_120000"


# ============================================================================
# INTEGRATION TESTS (Multi-model relationships)
# ============================================================================

def test_full_recommendation_data_flow(test_db):
    """Test complete data flow: ProductCategory -> Product -> UserInteraction -> Embedding"""
    category = ProductCategory(Id=1, Name="Tools")
    test_db.add(category)
    test_db.commit()

    product = Product(
        Id=1,
        Name="Hammer",
        CategoryId=1,
        Price=29.99
    )
    test_db.add(product)
    test_db.commit()

    for i in range(5):
        interaction = UserInteraction(
            Id=i + 1,
            UserId=f"user{i}",
            ProductID=1,
            Rating=4 + i % 2,
            InteractionDate=datetime.now()
        )
        test_db.add(interaction)
    test_db.commit()

    embedding = ProductEmbedding(
        product_id=1,
        cluster_id=3
    )
    test_db.add(embedding)
    test_db.commit()

    # Verify relationships
    assert product.category.Name == "Tools"
    assert len(product.interactions) == 5
    assert product.embedding.cluster_id == 3

    # Verify reverse relationships
    assert len(category.products) == 1
    assert category.products[0].Name == "Hammer"
