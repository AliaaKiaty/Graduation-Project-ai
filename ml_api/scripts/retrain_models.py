"""
Model Retraining Script
Retrains all ML models (SVD, TF-IDF, KMeans) from PostgreSQL database
Queries the real .NET backend tables (Products, UserInteraction)
Saves models with versioning and updates metadata table
"""
import sys
import os
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sqlalchemy import func

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_api.database import SessionLocal
from ml_api.models.db_models import Product, UserInteraction, ProductEmbedding, ModelMetadata, ProductCategory
from ml_api import config


def retrain_svd_model(db, models_dir: Path, version: str):
    """
    Retrain SVD collaborative filtering model from database ratings.

    Queries UserInteraction table for ratings, uses Product.Id as key.

    Args:
        db: Database session
        models_dir: Directory to save models
        version: Version string (timestamp)

    Returns:
        Tuple of (n_products, n_ratings, rmse)
    """
    print("\n" + "=" * 60)
    print("STEP 1: Retraining SVD Collaborative Filtering Model")
    print("=" * 60)

    start_time = time.time()

    # Query all ratings with product IDs (from UserInteraction table)
    print("\n1. Querying ratings from database...")
    ratings_query = (
        db.query(
            UserInteraction.UserId,
            Product.Id.label('product_id'),
            UserInteraction.Rating
        )
        .join(Product, UserInteraction.ProductID == Product.Id)
        .filter(UserInteraction.Rating.isnot(None))
    )

    ratings_data = ratings_query.all()
    print(f"   Fetched {len(ratings_data):,} ratings")

    if len(ratings_data) < 100:
        raise ValueError("Insufficient ratings data (need at least 100 ratings)")

    # Convert to DataFrame
    print("\n2. Building utility matrix...")
    df = pd.DataFrame(ratings_data, columns=['user_id', 'product_id', 'rating'])

    # Create utility matrix (users x products) — keyed by Product.Id (integer)
    utility_matrix = df.pivot_table(
        index='user_id',
        columns='product_id',
        values='rating'
    ).fillna(0)

    print(f"   Matrix shape: {utility_matrix.shape[0]:,} users x {utility_matrix.shape[1]:,} products")

    # Get product IDs in order (integers, not UIDs)
    product_ids = list(utility_matrix.columns)
    n_products = len(product_ids)

    # Apply TruncatedSVD
    print("\n3. Applying TruncatedSVD...")
    n_components = min(10, min(utility_matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    decomposed_matrix = svd.fit_transform(utility_matrix.T)

    print(f"   Reduced to {n_components} components")
    print(f"   Explained variance: {svd.explained_variance_ratio_.sum():.2%}")

    # Compute correlation matrix
    print("\n4. Computing correlation matrix...")
    correlation_matrix = np.corrcoef(decomposed_matrix)
    print(f"   Correlation matrix shape: {correlation_matrix.shape}")

    # Compute RMSE on training data
    print("\n5. Computing evaluation metrics...")
    reconstructed = svd.inverse_transform(decomposed_matrix).T
    original = utility_matrix.values
    mask = original != 0
    rmse = np.sqrt(mean_squared_error(original[mask], reconstructed[mask]))
    print(f"   RMSE (reconstruction): {rmse:.4f}")

    # Save models
    print("\n6. Saving models...")
    recommendation_dir = models_dir / "recommendation"
    recommendation_dir.mkdir(parents=True, exist_ok=True)

    # Save with versioning
    svd_path = recommendation_dir / f"svd_model_{version}.pkl"
    correlation_path = recommendation_dir / f"correlation_matrix_{version}.npy"
    product_names_path = recommendation_dir / f"product_names_{version}.pkl"

    joblib.dump(svd, svd_path)
    np.save(correlation_path, correlation_matrix)
    joblib.dump(product_ids, product_names_path)  # Save Product.Id list (integers)

    # Also save as current (for API to load)
    joblib.dump(svd, recommendation_dir / "svd_model.pkl")
    np.save(recommendation_dir / "correlation_matrix.npy", correlation_matrix)
    joblib.dump(product_ids, recommendation_dir / "product_names.pkl")

    print(f"   Saved SVD model: {svd_path.name}")
    print(f"   Saved correlation matrix: {correlation_path.name}")
    print(f"   Saved product IDs: {product_names_path.name}")

    # Save metadata to database
    print("\n7. Saving metadata to database...")

    # Deactivate previous SVD models
    db.query(ModelMetadata).filter(
        ModelMetadata.model_type == 'svd',
        ModelMetadata.is_active == True
    ).update({'is_active': False})

    metadata = ModelMetadata(
        model_type='svd',
        version=version,
        file_path=str(correlation_path),
        training_date=datetime.now(),
        n_components=n_components,
        total_products=n_products,
        total_ratings=len(ratings_data),
        rmse=float(rmse),
        training_duration_seconds=int(time.time() - start_time),
        is_active=True,
        notes=f"Collaborative filtering with {n_components} components, {n_products:,} products, {len(ratings_data):,} ratings"
    )
    db.add(metadata)
    db.commit()

    print(f"   Metadata saved to database")

    elapsed = time.time() - start_time
    print(f"\nSVD Model Training Complete ({elapsed:.2f} seconds)")

    return n_products, len(ratings_data), rmse


def retrain_tfidf_kmeans_model(db, models_dir: Path, version: str):
    """
    Retrain TF-IDF + KMeans content-based filtering model from database products.

    Queries Products table using PascalCase columns.

    Args:
        db: Database session
        models_dir: Directory to save models
        version: Version string (timestamp)

    Returns:
        Tuple of (n_products, n_clusters)
    """
    print("\n" + "=" * 60)
    print("STEP 2: Retraining TF-IDF + KMeans Content-Based Model")
    print("=" * 60)

    start_time = time.time()

    # Query all products
    print("\n1. Querying products from database...")
    products = db.query(Product).all()
    print(f"   Fetched {len(products):,} products")

    if len(products) < 20:
        raise ValueError("Insufficient products data (need at least 20 products)")

    # Extract descriptions using PascalCase column names
    print("\n2. Preparing product descriptions...")
    product_ids = [p.Id for p in products]
    descriptions = []
    for p in products:
        text = p.Name or ""
        if p.Description:
            text += " " + p.Description
        descriptions.append(text)

    print(f"   Prepared {len(descriptions):,} descriptions")

    # TF-IDF vectorization
    print("\n3. Applying TF-IDF vectorization...")
    max_features = min(5000, len(descriptions) * 10)
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=2,
        max_df=0.8
    )
    tfidf_matrix = tfidf.fit_transform(descriptions)
    print(f"   TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"   Vocabulary size: {len(tfidf.get_feature_names_out())}")

    # KMeans clustering
    print("\n4. Applying KMeans clustering...")
    n_clusters = min(20, max(5, len(products) // 100))
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    print(f"   Created {n_clusters} clusters")

    # Print cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"   Cluster distribution:")
    for cluster_id, count in zip(unique, counts):
        print(f"      Cluster {cluster_id}: {count} products")

    # Update ProductEmbedding table
    print("\n5. Updating ProductEmbedding table...")

    # Delete old embeddings
    db.query(ProductEmbedding).delete()

    # Insert new embeddings
    embeddings = []
    for i, product_id in enumerate(product_ids):
        embedding = ProductEmbedding(
            product_id=product_id,
            cluster_id=int(cluster_labels[i])
        )
        embeddings.append(embedding)

    db.bulk_save_objects(embeddings)
    db.commit()

    print(f"   Updated {len(embeddings):,} product embeddings")

    # Save models
    print("\n6. Saving models...")
    recommendation_dir = models_dir / "recommendation"
    recommendation_dir.mkdir(parents=True, exist_ok=True)

    # Save with versioning
    tfidf_path = recommendation_dir / f"tfidf_vectorizer_{version}.pkl"
    kmeans_path = recommendation_dir / f"kmeans_model_{version}.pkl"

    joblib.dump(tfidf, tfidf_path)
    joblib.dump(kmeans, kmeans_path)

    # Also save as current (for API to load)
    joblib.dump(tfidf, recommendation_dir / "tfidf_vectorizer.pkl")
    joblib.dump(kmeans, recommendation_dir / "kmeans_model.pkl")

    print(f"   Saved TF-IDF vectorizer: {tfidf_path.name}")
    print(f"   Saved KMeans model: {kmeans_path.name}")

    # Save metadata to database
    print("\n7. Saving metadata to database...")

    # Deactivate previous TF-IDF/KMeans models
    db.query(ModelMetadata).filter(
        ModelMetadata.model_type == 'tfidf_kmeans',
        ModelMetadata.is_active == True
    ).update({'is_active': False})

    metadata = ModelMetadata(
        model_type='tfidf_kmeans',
        version=version,
        file_path=str(tfidf_path),
        training_date=datetime.now(),
        n_clusters=n_clusters,
        max_features=max_features,
        total_products=len(products),
        training_duration_seconds=int(time.time() - start_time),
        is_active=True,
        notes=f"Content-based filtering with {n_clusters} clusters, {max_features} features, {len(products):,} products"
    )
    db.add(metadata)
    db.commit()

    print(f"   Metadata saved to database")

    elapsed = time.time() - start_time
    print(f"\nTF-IDF + KMeans Model Training Complete ({elapsed:.2f} seconds)")

    return len(products), n_clusters


def main():
    """Main entry point for model retraining"""
    print("=" * 60)
    print("ML Model Retraining Script")
    print("Retraining all models from PostgreSQL database")
    print("=" * 60)

    overall_start = time.time()

    # Create version string (timestamp)
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nModel Version: {version}")

    # Get models directory
    models_dir = Path(config.MODELS_DIR)
    print(f"Models Directory: {models_dir}")

    # Create database session
    db = SessionLocal()

    try:
        # Step 1: Retrain SVD model
        svd_products, svd_ratings, svd_rmse = retrain_svd_model(db, models_dir, version)

        # Step 2: Retrain TF-IDF + KMeans model
        content_products, content_clusters = retrain_tfidf_kmeans_model(db, models_dir, version)

        # Success summary
        elapsed = time.time() - overall_start
        print("\n" + "=" * 60)
        print("Model Retraining Complete!")
        print("=" * 60)
        print(f"  Total time:           {elapsed:.2f} seconds")
        print(f"  Model version:        {version}")
        print(f"\n  SVD Model:")
        print(f"    Products:           {svd_products:,}")
        print(f"    Ratings:            {svd_ratings:,}")
        print(f"    RMSE:               {svd_rmse:.4f}")
        print(f"\n  TF-IDF + KMeans:")
        print(f"    Products:           {content_products:,}")
        print(f"    Clusters:           {content_clusters}")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"Error during model retraining: {e}")
        print("=" * 60)
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()
