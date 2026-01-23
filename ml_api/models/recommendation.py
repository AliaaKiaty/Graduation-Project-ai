"""
Recommendation inference functions
Implements three recommendation approaches: popularity, collaborative filtering, and content-based
"""
from typing import List, Dict, Any
import numpy as np
from .loader import ModelManager


class RecommendationEngine:
    """
    Recommendation engine with three approaches:
    - Popularity-based recommendations
    - SVD collaborative filtering
    - TF-IDF + KMeans content-based filtering
    """

    def __init__(self):
        self.manager = ModelManager()

    def get_popular_items(self, dataset: str = "english", top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get most popular items based on rating counts.

        Args:
            dataset: Dataset to use ("english" or "arabic")
            top_n: Number of top items to return

        Returns:
            List of popular items with product_id and rating_count

        Raises:
            ValueError: If popularity counts are not loaded
        """
        # Load dataset-specific model
        model_key = f"popularity_counts_{dataset}"
        popularity_counts = self.manager.get_model(model_key)

        # Fall back to default if dataset-specific not available
        if popularity_counts is None:
            popularity_counts = self.manager.get_model("popularity_counts")

        if popularity_counts is None:
            raise ValueError(f"Popularity counts not loaded for dataset '{dataset}'")

        # Get top N items
        # Assuming popularity_counts is a pandas Series or dict
        if hasattr(popularity_counts, 'head'):
            # pandas Series
            top_items = popularity_counts.head(top_n)
            results = [
                {
                    "product_id": str(product_id),
                    "rating_count": int(count),
                    "rank": idx + 1
                }
                for idx, (product_id, count) in enumerate(top_items.items())
            ]
        else:
            # dict or other mapping
            sorted_items = sorted(
                popularity_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            results = [
                {
                    "product_id": str(product_id),
                    "rating_count": int(count),
                    "rank": idx + 1
                }
                for idx, (product_id, count) in enumerate(sorted_items)
            ]

        return results

    def get_collaborative_recommendations(
        self,
        product_id: str,
        dataset: str = "english",
        top_n: int = 10,
        min_correlation: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Get collaborative filtering recommendations using SVD and correlation matrix.

        Args:
            product_id: Product ID to get recommendations for
            dataset: Dataset to use ("english" or "arabic")
            top_n: Number of recommendations to return
            min_correlation: Minimum correlation threshold (0.0 to 1.0)

        Returns:
            List of recommended items with product_id and correlation_score

        Raises:
            ValueError: If required models are not loaded
            KeyError: If product_id is not found in the dataset
        """
        # Load dataset-specific models
        correlation_key = f"correlation_matrix_{dataset}"
        product_names_key = f"product_names_{dataset}"

        correlation_matrix = self.manager.get_model(correlation_key)
        product_names = self.manager.get_model(product_names_key)

        # Fall back to default if dataset-specific not available
        if correlation_matrix is None:
            correlation_matrix = self.manager.get_model("correlation_matrix")
        if product_names is None:
            product_names = self.manager.get_model("product_names")

        if correlation_matrix is None or product_names is None:
            raise ValueError(f"Collaborative filtering models not loaded for dataset '{dataset}'")

        # Find product index
        try:
            if isinstance(product_names, list):
                product_idx = product_names.index(product_id)
            else:
                product_idx = list(product_names).index(product_id)
        except (ValueError, AttributeError):
            raise KeyError(f"Product '{product_id}' not found in dataset")

        # Get correlations for this product
        correlations = correlation_matrix[product_idx]

        # Use adaptive top-N approach (instead of fixed threshold)
        # Sort by correlation in descending order
        sorted_indices = np.argsort(correlations)[::-1]

        recommendations = []
        for idx in sorted_indices:
            # Skip the product itself
            if idx == product_idx:
                continue

            correlation_score = float(correlations[idx])

            # Apply minimum correlation filter
            if correlation_score < min_correlation:
                continue

            recommendations.append({
                "product_id": str(product_names[idx]),
                "correlation_score": round(correlation_score, 4),
                "rank": len(recommendations) + 1
            })

            if len(recommendations) >= top_n:
                break

        return recommendations

    def get_content_based_recommendations(
        self,
        search_query: str,
        dataset: str = "english",
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Get content-based recommendations using TF-IDF + KMeans clustering.

        Args:
            search_query: Search query text
            dataset: Dataset to use ("english" or "arabic")
            top_n: Number of recommendations to return

        Returns:
            Dictionary with predicted_cluster, cluster_keywords, and recommendations

        Raises:
            ValueError: If required models are not loaded
        """
        # Load dataset-specific models
        tfidf_key = f"tfidf_vectorizer_{dataset}"
        kmeans_key = f"kmeans_model_{dataset}"
        products_key = f"products_by_cluster_{dataset}"

        tfidf_vectorizer = self.manager.get_model(tfidf_key)
        kmeans_model = self.manager.get_model(kmeans_key)
        products_by_cluster = self.manager.get_model(products_key)

        # Fall back to default if dataset-specific not available
        if tfidf_vectorizer is None:
            tfidf_vectorizer = self.manager.get_model("tfidf_vectorizer")
        if kmeans_model is None:
            kmeans_model = self.manager.get_model("kmeans_model")
        if products_by_cluster is None:
            products_by_cluster = self.manager.get_model("products_by_cluster")

        if tfidf_vectorizer is None or kmeans_model is None:
            raise ValueError(f"Content-based filtering models not loaded for dataset '{dataset}'")

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

        # Get products from this cluster
        if products_by_cluster is not None:
            cluster_products = products_by_cluster.get(predicted_cluster, [])
        else:
            # If products_by_cluster not available, return empty list
            cluster_products = []

        # Calculate distances to cluster center for ranking
        cluster_center = kmeans_model.cluster_centers_[predicted_cluster]
        query_distance = float(
            np.linalg.norm(query_vector.toarray() - cluster_center)
        )

        # Format recommendations
        recommendations = []
        for idx, product in enumerate(cluster_products[:top_n]):
            if isinstance(product, dict):
                product_id = product.get("product_id", str(product))
                description = product.get("description", "")
            else:
                product_id = str(product)
                description = ""

            recommendations.append({
                "product_id": product_id,
                "product_description": description[:200] if description else None,
                "rank": idx + 1
            })

        return {
            "search_query": search_query,
            "predicted_cluster": predicted_cluster,
            "cluster_keywords": cluster_keywords,
            "recommendations": recommendations,
            "total_results": len(cluster_products)
        }

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
