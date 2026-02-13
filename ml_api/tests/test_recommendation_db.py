"""
Integration tests for database-backed recommendation engine
Tests for get_popular_items, get_collaborative_recommendations, get_content_based_recommendations
"""
import pytest
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import MagicMock, patch

from ml_api.database import Base
from ml_api.models.db_models import ProductCategory, Product, UserInteraction, ProductEmbedding
from ml_api.models.recommendation import RecommendationEngine


# Test database setup
@pytest.fixture(scope="function")
def test_db():
    """Create a test database session with sample data"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    TestingSessionLocal = sessionmaker(bind=engine)
    db = TestingSessionLocal()

    setup_test_data(db)

    yield db

    db.close()
    Base.metadata.drop_all(engine)


def setup_test_data(db):
    """Create sample data for testing"""
    # Create categories
    tools = ProductCategory(Id=1, Name="Tools")
    paint = ProductCategory(Id=2, Name="Paint")
    db.add_all([tools, paint])
    db.commit()

    # Create products
    products = [
        Product(Id=1, Name="Hammer", CategoryId=1, Price=29.99),
        Product(Id=2, Name="Screwdriver", CategoryId=1, Price=15.99),
        Product(Id=3, Name="Paint Brush", CategoryId=2, Price=9.99),
        Product(Id=4, Name="Drill", CategoryId=1, Price=89.99),
        Product(Id=5, Name="Spray Paint", CategoryId=2, Price=12.99),
    ]
    db.add_all(products)
    db.commit()

    # Create interactions with ratings (Product 1 is most popular with 5 ratings)
    interactions = [
        UserInteraction(Id=1, UserId="user1", ProductID=1, Rating=5, InteractionDate=datetime.now()),
        UserInteraction(Id=2, UserId="user2", ProductID=1, Rating=4, InteractionDate=datetime.now()),
        UserInteraction(Id=3, UserId="user3", ProductID=1, Rating=4, InteractionDate=datetime.now()),
        UserInteraction(Id=4, UserId="user4", ProductID=1, Rating=5, InteractionDate=datetime.now()),
        UserInteraction(Id=5, UserId="user5", ProductID=1, Rating=5, InteractionDate=datetime.now()),
        # Product 2 has 3 ratings
        UserInteraction(Id=6, UserId="user1", ProductID=2, Rating=4, InteractionDate=datetime.now()),
        UserInteraction(Id=7, UserId="user2", ProductID=2, Rating=3, InteractionDate=datetime.now()),
        UserInteraction(Id=8, UserId="user3", ProductID=2, Rating=5, InteractionDate=datetime.now()),
        # Product 3 has 2 ratings
        UserInteraction(Id=9, UserId="user1", ProductID=3, Rating=3, InteractionDate=datetime.now()),
        UserInteraction(Id=10, UserId="user2", ProductID=3, Rating=4, InteractionDate=datetime.now()),
        # Product 4 has 1 rating
        UserInteraction(Id=11, UserId="user1", ProductID=4, Rating=5, InteractionDate=datetime.now()),
    ]
    db.add_all(interactions)
    db.commit()

    # Create product embeddings for content-based filtering
    embeddings = [
        ProductEmbedding(product_id=1, cluster_id=0),  # Tools cluster
        ProductEmbedding(product_id=2, cluster_id=0),  # Tools cluster
        ProductEmbedding(product_id=3, cluster_id=1),  # Paint cluster
        ProductEmbedding(product_id=4, cluster_id=0),  # Tools cluster
        ProductEmbedding(product_id=5, cluster_id=1),  # Paint cluster
    ]
    db.add_all(embeddings)
    db.commit()


# ============================================================================
# POPULARITY-BASED RECOMMENDATIONS TESTS
# ============================================================================

def test_get_popular_items_basic(test_db):
    """Test getting popular items from database"""
    rec_engine = RecommendationEngine()

    recommendations = rec_engine.get_popular_items(top_n=3, db=test_db)

    assert len(recommendations) == 3
    # Most popular product should be Product 1 (5 ratings)
    assert recommendations[0]["product_id"] == 1
    assert recommendations[0]["rating_count"] == 5
    assert recommendations[0]["rank"] == 1
    # Second should be Product 2 (3 ratings)
    assert recommendations[1]["product_id"] == 2
    assert recommendations[1]["rating_count"] == 3


def test_get_popular_items_with_category_filter(test_db):
    """Test getting popular items filtered by category"""
    rec_engine = RecommendationEngine()

    # Get popular tools (category_id=1)
    recommendations = rec_engine.get_popular_items(top_n=5, category_id=1, db=test_db)

    # Should only return tools category products
    assert len(recommendations) <= 3  # Only 3 tools products with ratings
    for rec in recommendations:
        assert rec["category_name"] == "Tools"


def test_get_popular_items_top_n_limit(test_db):
    """Test that top_n limit is respected"""
    rec_engine = RecommendationEngine()

    recommendations = rec_engine.get_popular_items(top_n=2, db=test_db)

    assert len(recommendations) == 2


def test_get_popular_items_includes_average_rating(test_db):
    """Test that average rating is calculated correctly"""
    rec_engine = RecommendationEngine()

    recommendations = rec_engine.get_popular_items(top_n=1, db=test_db)

    # Product 1 has ratings: [5, 4, 4, 5, 5] -> avg = 4.6
    assert recommendations[0]["product_id"] == 1
    assert recommendations[0]["average_rating"] == pytest.approx(4.6, rel=0.01)


# ============================================================================
# COLLABORATIVE FILTERING TESTS (with mocked models)
# ============================================================================

@pytest.fixture
def mock_svd_models():
    """Create mock SVD models for testing"""
    # Create correlation matrix: Product 1 is similar to Product 2 and Product 4 (all tools)
    correlation_matrix = np.array([
        [1.0, 0.9, 0.2, 0.8, 0.1],  # Product 1: highly correlated with P2, P4
        [0.9, 1.0, 0.3, 0.7, 0.2],  # Product 2
        [0.2, 0.3, 1.0, 0.1, 0.9],  # Product 3: highly correlated with P5 (both paint)
        [0.8, 0.7, 0.1, 1.0, 0.2],  # Product 4
        [0.1, 0.2, 0.9, 0.2, 1.0],  # Product 5
    ])

    # product_names is now a list of integer Product.Id values
    product_names = [1, 2, 3, 4, 5]

    return correlation_matrix, product_names


def test_get_collaborative_recommendations_basic(test_db, mock_svd_models):
    """Test collaborative filtering with mocked models"""
    correlation_matrix, product_names = mock_svd_models

    rec_engine = RecommendationEngine()

    with patch.object(rec_engine.manager, 'get_model') as mock_get_model:
        def get_model_side_effect(model_name):
            if model_name == "correlation_matrix":
                return correlation_matrix
            elif model_name == "product_names":
                return product_names
            return None

        mock_get_model.side_effect = get_model_side_effect

        # Get recommendations for Product 1 (product_id=1)
        result = rec_engine.get_collaborative_recommendations(
            product_id=1,
            top_n=3,
            min_correlation=0.0,
            db=test_db
        )

        assert result["input_product_id"] == 1
        assert result["input_product_name"] == "Hammer"
        assert len(result["recommendations"]) <= 3

        # First recommendation should be Product 2 (correlation=0.9)
        if len(result["recommendations"]) > 0:
            assert result["recommendations"][0]["product_id"] == 2
            assert result["recommendations"][0]["correlation_score"] == pytest.approx(0.9, rel=0.01)


def test_get_collaborative_recommendations_with_min_correlation(test_db, mock_svd_models):
    """Test collaborative filtering with minimum correlation threshold"""
    correlation_matrix, product_names = mock_svd_models

    rec_engine = RecommendationEngine()

    with patch.object(rec_engine.manager, 'get_model') as mock_get_model:
        def get_model_side_effect(model_name):
            if model_name == "correlation_matrix":
                return correlation_matrix
            elif model_name == "product_names":
                return product_names
            return None

        mock_get_model.side_effect = get_model_side_effect

        result = rec_engine.get_collaborative_recommendations(
            product_id=1,
            top_n=10,
            min_correlation=0.7,
            db=test_db
        )

        # Only Product 2 (0.9) and Product 4 (0.8) should pass threshold
        assert len(result["recommendations"]) <= 2
        for rec in result["recommendations"]:
            assert rec["correlation_score"] >= 0.7


def test_get_collaborative_recommendations_with_category_filter(test_db, mock_svd_models):
    """Test collaborative filtering with category filter"""
    correlation_matrix, product_names = mock_svd_models

    rec_engine = RecommendationEngine()

    with patch.object(rec_engine.manager, 'get_model') as mock_get_model:
        def get_model_side_effect(model_name):
            if model_name == "correlation_matrix":
                return correlation_matrix
            elif model_name == "product_names":
                return product_names
            return None

        mock_get_model.side_effect = get_model_side_effect

        # Get recommendations for Product 1, filtered to Tools category (id=1)
        result = rec_engine.get_collaborative_recommendations(
            product_id=1,
            top_n=10,
            category_id=1,
            db=test_db
        )

        # All recommendations should be in Tools category
        for rec in result["recommendations"]:
            assert rec["category_name"] == "Tools"


def test_get_collaborative_recommendations_product_not_found(test_db, mock_svd_models):
    """Test collaborative filtering with non-existent product"""
    correlation_matrix, product_names = mock_svd_models

    rec_engine = RecommendationEngine()

    with patch.object(rec_engine.manager, 'get_model') as mock_get_model:
        def get_model_side_effect(model_name):
            if model_name == "correlation_matrix":
                return correlation_matrix
            elif model_name == "product_names":
                return product_names
            return None

        mock_get_model.side_effect = get_model_side_effect

        # Try with non-existent product_id
        with pytest.raises(KeyError, match="not found"):
            rec_engine.get_collaborative_recommendations(
                product_id=999,
                top_n=5,
                db=test_db
            )


# ============================================================================
# CONTENT-BASED FILTERING TESTS (with mocked models)
# ============================================================================

@pytest.fixture
def mock_tfidf_kmeans_models():
    """Create mock TF-IDF and KMeans models for testing"""
    mock_tfidf = MagicMock()
    mock_tfidf.transform.return_value = np.array([[0.5, 0.3, 0.2]])
    mock_tfidf.get_feature_names_out.return_value = np.array(["tool", "cutting", "metal", "paint", "brush"])

    mock_kmeans = MagicMock()
    mock_kmeans.predict.return_value = np.array([0])  # Predict cluster 0 (tools)
    mock_kmeans.cluster_centers_ = np.array([
        [0.8, 0.6, 0.7, 0.1, 0.1],  # Cluster 0: tool-related
        [0.1, 0.1, 0.1, 0.9, 0.8],  # Cluster 1: paint-related
    ])

    return mock_tfidf, mock_kmeans


def test_get_content_based_recommendations_basic(test_db, mock_tfidf_kmeans_models):
    """Test content-based filtering with mocked models"""
    mock_tfidf, mock_kmeans = mock_tfidf_kmeans_models

    rec_engine = RecommendationEngine()

    with patch.object(rec_engine.manager, 'get_model') as mock_get_model:
        def get_model_side_effect(model_name):
            if model_name == "tfidf_vectorizer":
                return mock_tfidf
            elif model_name == "kmeans_model":
                return mock_kmeans
            return None

        mock_get_model.side_effect = get_model_side_effect

        result = rec_engine.get_content_based_recommendations(
            search_query="hammer tool",
            top_n=5,
            db=test_db
        )

        assert result["search_query"] == "hammer tool"
        assert result["predicted_cluster"] == 0
        assert len(result["cluster_keywords"]) > 0

        # Should return products in cluster 0 (Product 1, 2, 4)
        assert result["total_results"] == 3


def test_get_content_based_recommendations_with_category_filter(test_db, mock_tfidf_kmeans_models):
    """Test content-based filtering with category filter"""
    mock_tfidf, mock_kmeans = mock_tfidf_kmeans_models

    # Change prediction to cluster 1 (paint)
    mock_kmeans.predict.return_value = np.array([1])

    rec_engine = RecommendationEngine()

    with patch.object(rec_engine.manager, 'get_model') as mock_get_model:
        def get_model_side_effect(model_name):
            if model_name == "tfidf_vectorizer":
                return mock_tfidf
            elif model_name == "kmeans_model":
                return mock_kmeans
            return None

        mock_get_model.side_effect = get_model_side_effect

        result = rec_engine.get_content_based_recommendations(
            search_query="paint brush",
            top_n=5,
            category_id=2,
            db=test_db
        )

        assert result["predicted_cluster"] == 1

        for rec in result["recommendations"]:
            assert rec["category_name"] == "Paint"


def test_get_content_based_recommendations_cluster_keywords(test_db, mock_tfidf_kmeans_models):
    """Test that cluster keywords are extracted correctly"""
    mock_tfidf, mock_kmeans = mock_tfidf_kmeans_models

    rec_engine = RecommendationEngine()

    with patch.object(rec_engine.manager, 'get_model') as mock_get_model:
        def get_model_side_effect(model_name):
            if model_name == "tfidf_vectorizer":
                return mock_tfidf
            elif model_name == "kmeans_model":
                return mock_kmeans
            return None

        mock_get_model.side_effect = get_model_side_effect

        result = rec_engine.get_content_based_recommendations(
            search_query="test query",
            top_n=5,
            db=test_db
        )

        assert len(result["cluster_keywords"]) > 0
        for keyword in result["cluster_keywords"]:
            assert keyword in ["tool", "cutting", "metal", "paint", "brush"]


def test_get_content_based_recommendations_top_n_limit(test_db, mock_tfidf_kmeans_models):
    """Test that top_n limit is respected"""
    mock_tfidf, mock_kmeans = mock_tfidf_kmeans_models

    rec_engine = RecommendationEngine()

    with patch.object(rec_engine.manager, 'get_model') as mock_get_model:
        def get_model_side_effect(model_name):
            if model_name == "tfidf_vectorizer":
                return mock_tfidf
            elif model_name == "kmeans_model":
                return mock_kmeans
            return None

        mock_get_model.side_effect = get_model_side_effect

        result = rec_engine.get_content_based_recommendations(
            search_query="tool",
            top_n=2,
            db=test_db
        )

        assert len(result["recommendations"]) <= 2


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_collaborative_recommendations_models_not_loaded(test_db):
    """Test error when SVD models are not loaded"""
    rec_engine = RecommendationEngine()

    with patch.object(rec_engine.manager, 'get_model', return_value=None):
        with pytest.raises(ValueError, match="models not loaded"):
            rec_engine.get_collaborative_recommendations(
                product_id=1,
                top_n=5,
                db=test_db
            )


def test_content_based_recommendations_models_not_loaded(test_db):
    """Test error when TF-IDF/KMeans models are not loaded"""
    rec_engine = RecommendationEngine()

    with patch.object(rec_engine.manager, 'get_model', return_value=None):
        with pytest.raises(ValueError, match="models not loaded"):
            rec_engine.get_content_based_recommendations(
                search_query="test",
                top_n=5,
                db=test_db
            )
