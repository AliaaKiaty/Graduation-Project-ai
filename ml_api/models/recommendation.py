"""
Recommendation inference functions (Database-backed version)
Implements three recommendation approaches querying the real .NET backend database
Uses hybrid architecture: ML models from files, product metadata from database
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func
from .loader import ModelManager
from ..database import SessionLocal
from .db_models import Product, UserInteraction, ProductEmbedding, ProductCategory


class RecommendationEngine:
    """
    Database-backed recommendation engine with three approaches:
    - Popularity-based recommendations (from database UserInteraction table)
    - SVD collaborative filtering (hybrid: correlation matrix from file, product data from DB)
    - TF-IDF + KMeans content-based filtering (hybrid: models from file, products from DB)
    """

    def __init__(self):
        self.manager = ModelManager()

    def get_popular_items(
        self,
        top_n: int = 10,
        category_id: Optional[int] = None,
        db: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """
        Get most popular items based on rating counts from database.

        Args:
            top_n: Number of top items to return
            category_id: Optional category filter
            db: Database session (created if not provided)

        Returns:
            List of popular items with product details and rating counts
        """
        should_close_db = False
        if db is None:
            db = SessionLocal()
            should_close_db = True

        try:
            # Build query: count ratings per product (only where Rating IS NOT NULL)
            query = (
                db.query(
                    Product.Id,
                    Product.NameEn,
                    Product.NameAr,
                    Product.ImageUrl,
                    Product.Price,
                    func.count(UserInteraction.Id).label('rating_count'),
                    func.avg(UserInteraction.Rating).label('avg_rating'),
                    ProductCategory.NameEn.label('category_name_en'),
                    ProductCategory.NameAr.label('category_name_ar'),
                )
                .join(UserInteraction, Product.Id == UserInteraction.ProductID)
                .outerjoin(ProductCategory, Product.CategoryId == ProductCategory.Id)
                .filter(UserInteraction.Rating.isnot(None))
            )

            # Apply category filter if provided
            if category_id is not None:
                query = query.filter(Product.CategoryId == category_id)

            # Group by product and order by rating count
            query = (
                query.group_by(
                    Product.Id, Product.NameEn, Product.NameAr,
                    Product.ImageUrl, Product.Price,
                    ProductCategory.NameEn, ProductCategory.NameAr
                )
                .order_by(func.count(UserInteraction.Id).desc())
                .limit(top_n)
            )

            results = query.all()

            # Format response
            recommendations = [
                {
                    "product_id": row.Id,
                    "product_name": row.NameAr or row.NameEn or "",
                    "image_url": row.ImageUrl,
                    "price": float(row.Price) if row.Price else None,
                    "rating_count": row.rating_count,
                    "average_rating": float(row.avg_rating) if row.avg_rating else None,
                    "category_name": row.category_name_ar or row.category_name_en,
                    "rank": idx + 1
                }
                for idx, row in enumerate(results)
            ]

            return recommendations

        finally:
            if should_close_db:
                db.close()

    def get_collaborative_recommendations(
        self,
        product_id: int,
        top_n: int = 10,
        min_correlation: float = 0.0,
        category_id: Optional[int] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Get collaborative filtering recommendations using SVD and correlation matrix.
        Hybrid approach: correlation matrix from file, product metadata from database.

        Args:
            product_id: Product.Id (integer) to get recommendations for
            top_n: Number of recommendations to return
            min_correlation: Minimum correlation threshold (0.0 to 1.0)
            category_id: Optional category filter
            db: Database session

        Returns:
            Dictionary with input product info and list of recommended items

        Raises:
            ValueError: If required models are not loaded
            KeyError: If product_id is not found
        """
        should_close_db = False
        if db is None:
            db = SessionLocal()
            should_close_db = True

        try:
            # Load correlation matrix and product ID mapping from files
            correlation_matrix = self.manager.get_model("correlation_matrix")
            product_names = self.manager.get_model("product_names")  # Maps Product.Id to index

            if correlation_matrix is None or product_names is None:
                raise ValueError("Collaborative filtering models not loaded")

            # Get input product from database
            input_product = db.query(Product).filter(
                Product.Id == product_id,
                (Product.IsDeleted == False) | (Product.IsDeleted == None)  # noqa: E712
            ).first()
            if not input_product:
                raise KeyError(f"Product with ID {product_id} not found")

            # Find product index in correlation matrix (by integer ID)
            try:
                if isinstance(product_names, list):
                    product_idx = product_names.index(product_id)
                else:
                    product_idx = list(product_names).index(product_id)
            except ValueError:
                raise KeyError(f"Product ID {product_id} not found in trained model. Model may need retraining.")

            # Get correlations for this product
            correlations = correlation_matrix[product_idx]

            # Sort by correlation in descending order
            sorted_indices = np.argsort(correlations)[::-1]

            # Get product IDs from sorted indices
            recommended_ids = []
            for idx in sorted_indices:
                if idx == product_idx:  # Skip self
                    continue
                correlation_score = float(correlations[idx])
                if correlation_score < min_correlation:
                    continue
                recommended_ids.append((product_names[idx], correlation_score))
                if len(recommended_ids) >= top_n * 3:  # Fetch extra for filtering
                    break

            # Fetch product details from database
            ids = [pid for pid, _ in recommended_ids]
            query = (
                db.query(Product, ProductCategory.NameAr.label('category_name'))
                .outerjoin(ProductCategory, Product.CategoryId == ProductCategory.Id)
                .filter(Product.Id.in_(ids))
            )

            # Apply category filter if provided
            if category_id is not None:
                query = query.filter(Product.CategoryId == category_id)

            products = query.all()

            # Build product map: id -> (product, category_name)
            product_map = {p.Product.Id: (p.Product, p.category_name) for p in products}

            # Build final recommendations
            recommendations = []
            for pid, correlation_score in recommended_ids:
                if pid in product_map:
                    product, category_name = product_map[pid]
                    recommendations.append({
                        "product_id": product.Id,
                        "product_name": product.NameAr or product.NameEn or "",
                        "image_url": product.ImageUrl,
                        "price": float(product.Price) if product.Price else None,
                        "correlation_score": round(correlation_score, 4),
                        "category_name": category_name,
                        "rank": len(recommendations) + 1
                    })
                if len(recommendations) >= top_n:
                    break

            return {
                "input_product_id": input_product.Id,
                "input_product_name": input_product.NameAr or input_product.NameEn or "",
                "recommendations": recommendations
            }

        finally:
            if should_close_db:
                db.close()

    def get_content_based_recommendations(
        self,
        search_query: str,
        top_n: int = 10,
        category_id: Optional[int] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Get content-based recommendations using TF-IDF + KMeans clustering.
        Hybrid approach: TF-IDF/KMeans from files, product metadata from database.

        Args:
            search_query: Search query text
            top_n: Number of recommendations to return
            category_id: Optional category filter
            db: Database session

        Returns:
            Dictionary with predicted_cluster, cluster_keywords, and recommendations

        Raises:
            ValueError: If required models are not loaded
        """
        should_close_db = False
        if db is None:
            db = SessionLocal()
            should_close_db = True

        try:
            # Load TF-IDF and KMeans models from files
            tfidf_vectorizer = self.manager.get_model("tfidf_vectorizer")
            kmeans_model = self.manager.get_model("kmeans_model")

            if tfidf_vectorizer is None or kmeans_model is None:
                raise ValueError("Content-based filtering models not loaded")

            # Vectorize the search query
            query_vector = tfidf_vectorizer.transform([search_query])

            # Predict cluster
            predicted_cluster = int(kmeans_model.predict(query_vector)[0])

            # Get cluster keywords (top terms)
            cluster_keywords = self._get_cluster_keywords(
                kmeans_model,
                tfidf_vectorizer,
                predicted_cluster,
                top_k=7
            )

            # Fetch products from this cluster (via ProductEmbedding table)
            query = (
                db.query(Product, ProductCategory.NameAr.label('category_name'))
                .join(ProductEmbedding, Product.Id == ProductEmbedding.product_id)
                .outerjoin(ProductCategory, Product.CategoryId == ProductCategory.Id)
                .filter(ProductEmbedding.cluster_id == predicted_cluster)
            )

            # Apply category filter if provided
            if category_id is not None:
                query = query.filter(Product.CategoryId == category_id)

            # Limit results
            products = query.limit(top_n).all()

            # Format recommendations
            recommendations = []
            for idx, row in enumerate(products):
                product = row.Product
                category_name = row.category_name
                desc = product.DescriptionEn or product.DescriptionAr or ""
                recommendations.append({
                    "product_id": product.Id,
                    "product_name": product.NameAr or product.NameEn or "",
                    "product_description": desc[:200] if desc else None,
                    "image_url": product.ImageUrl,
                    "category_name": category_name,
                    "rank": idx + 1
                })

            # Count total products in cluster
            total_in_cluster = (
                db.query(func.count(ProductEmbedding.id))
                .join(Product, Product.Id == ProductEmbedding.product_id)
                .filter(ProductEmbedding.cluster_id == predicted_cluster)
            )
            if category_id is not None:
                total_in_cluster = total_in_cluster.filter(Product.CategoryId == category_id)
            total_in_cluster = total_in_cluster.scalar()

            return {
                "search_query": search_query,
                "predicted_cluster": predicted_cluster,
                "cluster_keywords": cluster_keywords,
                "recommendations": recommendations,
                "total_results": total_in_cluster
            }

        finally:
            if should_close_db:
                db.close()

    def _get_cluster_keywords(
        self,
        kmeans_model: Any,
        tfidf_vectorizer: Any,
        cluster_id: int,
        top_k: int = 7
    ) -> List[str]:
        """
        Extract top keywords for a cluster.

        Args:
            kmeans_model: Fitted KMeans model
            tfidf_vectorizer: Fitted TF-IDF vectorizer
            cluster_id: Cluster ID
            top_k: Number of top keywords to return

        Returns:
            List of top keywords for the cluster
        """
        try:
            # Get cluster center
            cluster_center = kmeans_model.cluster_centers_[cluster_id]

            # Get feature names
            feature_names = tfidf_vectorizer.get_feature_names_out()

            # Get top feature indices
            top_indices = np.argsort(cluster_center)[::-1][:top_k]

            # Get keywords
            keywords = [feature_names[idx] for idx in top_indices]

            return keywords
        except Exception:
            return []
