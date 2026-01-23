"""
Image inference functions
Implements feature extraction, similarity search, and classification
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import io
from .loader import ModelManager

# Try to import keras preprocessing - will be available when tensorflow is installed
try:
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
    HAS_KERAS_PREPROCESS = True
except ImportError:
    HAS_KERAS_PREPROCESS = False
    resnet_preprocess = None


class ImageEngine:
    """
    Image processing engine with two main features:
    - Visual similarity search using ResNet50 features + KNN
    - Butterfly image classification
    """

    def __init__(self):
        self.manager = ModelManager()
        self.target_size = (224, 224)  # Standard ImageNet size

    def preprocess_image(self, image_bytes: bytes, for_resnet: bool = True) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image_bytes: Raw image bytes from uploaded file
            for_resnet: If True, apply ResNet50 preprocessing (caffe mode).
                       If False, use simple [0,1] normalization.

        Returns:
            Preprocessed image array ready for model input

        Raises:
            ValueError: If image cannot be processed
        """
        try:
            # Open image from bytes
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if needed (handle RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize to target size
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)

            # Convert to numpy array
            img_array = np.array(image, dtype=np.float32)

            # Add batch dimension first (required for preprocess_input)
            img_array = np.expand_dims(img_array, axis=0)

            # Apply appropriate preprocessing
            if for_resnet and HAS_KERAS_PREPROCESS and resnet_preprocess is not None:
                # ResNet50 expects "caffe" preprocessing: BGR conversion + mean subtraction
                # Input should be in [0, 255] range, preprocess_input handles the rest
                img_array = resnet_preprocess(img_array)
            else:
                # Fallback: simple [0, 1] normalization
                img_array = img_array / 255.0

            return img_array

        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {str(e)}")

    def extract_features(self, image_bytes: bytes) -> np.ndarray:
        """
        Extract 2048-dimensional feature vector from image using ResNet50.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Feature vector (2048,)

        Raises:
            ValueError: If feature extractor not loaded or processing fails
        """
        # Get feature extractor model
        feature_extractor = self.manager.get_model("resnet50_feature_extractor")
        if feature_extractor is None:
            raise ValueError("ResNet50 feature extractor not loaded")

        # Preprocess image with ResNet50 preprocessing
        img_array = self.preprocess_image(image_bytes, for_resnet=True)

        # Extract features
        features = feature_extractor.predict(img_array, verbose=0)

        # Flatten to 1D if needed
        features = features.flatten()

        return features

    def find_similar_images(
        self,
        image_bytes: bytes,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar images using KNN on feature vectors.

        Args:
            image_bytes: Raw image bytes to search for
            top_k: Number of similar images to return

        Returns:
            List of similar images with filenames and distances

        Raises:
            ValueError: If required models not loaded
        """
        # Get required models
        knn_model = self.manager.get_model("knn_model")
        feature_list = self.manager.get_model("feature_list")
        filenames = self.manager.get_model("filenames")

        if knn_model is None or feature_list is None:
            raise ValueError("KNN similarity search models not loaded")

        # Extract features from query image
        query_features = self.extract_features(image_bytes)

        # Reshape for KNN input (needs 2D array)
        query_features = query_features.reshape(1, -1)

        # Find k nearest neighbors
        distances, indices = knn_model.kneighbors(query_features, n_neighbors=top_k)

        # Format results
        results = []
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), 1):
            result = {
                "rank": rank,
                "similarity_score": float(1.0 / (1.0 + distance)),  # Convert distance to similarity
                "distance": float(distance)
            }

            # Add filename if available
            if filenames is not None:
                if isinstance(filenames, list) and idx < len(filenames):
                    result["filename"] = str(filenames[idx])
                elif isinstance(filenames, np.ndarray) and idx < len(filenames):
                    result["filename"] = str(filenames[idx])
                else:
                    result["filename"] = f"image_{idx}"
            else:
                result["filename"] = f"image_{idx}"

            results.append(result)

        return results

    def classify_image(self, image_bytes: bytes, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Classify butterfly image and return top-K predictions.

        Args:
            image_bytes: Raw image bytes
            top_k: Number of top predictions to return

        Returns:
            List of predictions with class names and confidence scores

        Raises:
            ValueError: If classifier not loaded
        """
        # Get classifier model
        classifier = self.manager.get_model("butterfly_classifier")
        class_names = self.manager.get_model("class_names")

        if classifier is None:
            raise ValueError("Butterfly classifier not loaded")

        # Preprocess image (use ResNet preprocessing since classifier is based on ResNet)
        img_array = self.preprocess_image(image_bytes, for_resnet=True)

        # Get predictions
        predictions = classifier.predict(img_array, verbose=0)

        # Get top-K predictions
        top_indices = np.argsort(predictions[0])[::-1][:top_k]

        # Format results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            confidence = float(predictions[0][idx])

            result = {
                "rank": rank,
                "class_id": int(idx),  # Always include class_id
                "confidence": round(confidence, 4)
            }

            # Add class name if available
            if class_names is not None:
                if isinstance(class_names, list) and idx < len(class_names):
                    result["class_name"] = str(class_names[idx])
                elif isinstance(class_names, dict):
                    result["class_name"] = class_names.get(idx, f"class_{idx}")
                else:
                    result["class_name"] = f"class_{idx}"
            else:
                result["class_name"] = f"class_{idx}"

            results.append(result)

        return results

    def get_supported_formats(self) -> List[str]:
        """Get list of supported image formats."""
        return ["image/jpeg", "image/jpg", "image/png", "image/webp"]

    def validate_content_type(self, content_type: str) -> Tuple[bool, Optional[str]]:
        """
        Validate image content type.

        Args:
            content_type: MIME type of the file

        Returns:
            Tuple of (is_valid, error_message)
        """
        supported_types = self.get_supported_formats()
        if content_type not in supported_types:
            return False, f"Unsupported file type '{content_type}'. Supported: {', '.join(supported_types)}"
        return True, None

    def validate_file_size(self, file_size: int, max_size_mb: int = 10) -> Tuple[bool, Optional[str]]:
        """
        Validate file size.

        Args:
            file_size: Size of the file in bytes
            max_size_mb: Maximum allowed size in MB

        Returns:
            Tuple of (is_valid, error_message)
        """
        max_size_bytes = max_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            return False, f"File too large. Maximum size: {max_size_mb}MB"
        return True, None
