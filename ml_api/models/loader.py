"""
Model Manager - Singleton class for loading and managing ML models
"""
import threading
import gc
import copy
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import joblib

from .. import config


class ModelManager:
    """
    Singleton class to manage ML model loading and unloading.

    Implements lazy loading for heavy models (chatbot) and eager loading
    for lighter models (recommendation, image).
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._models: Dict[str, Any] = {}
        self._loading_status: Dict[str, Dict[str, str]] = {
            "recommendation": {
                "popularity": "not_loaded",
                "collaborative": "not_loaded",
                "content_based": "not_loaded"
            },
            "image": {
                "feature_extractor": "not_loaded",
                "classifier": "not_loaded",
                "knn_index": "not_loaded"
            },
            "chatbot": {
                "llama_base": "not_loaded",
                "lora_adapter": "not_loaded"
            }
        }
        self._initialized = True

    def load_recommendation_models(self) -> None:
        """
        Eagerly load all recommendation models on startup.
        Loads popularity counts, SVD model, correlation matrix, TF-IDF, and KMeans.
        """
        try:
            # Load popularity counts
            if config.POPULARITY_COUNTS_PATH.exists():
                self._models["popularity_counts"] = joblib.load(config.POPULARITY_COUNTS_PATH)
                self._loading_status["recommendation"]["popularity"] = "loaded"

            # Load SVD collaborative filtering models
            if config.SVD_MODEL_PATH.exists():
                self._models["svd_model"] = joblib.load(config.SVD_MODEL_PATH)
                self._loading_status["recommendation"]["collaborative"] = "loaded"

            if config.CORRELATION_MATRIX_PATH.exists():
                self._models["correlation_matrix"] = np.load(config.CORRELATION_MATRIX_PATH)

            if config.PRODUCT_NAMES_PATH.exists():
                self._models["product_names"] = joblib.load(config.PRODUCT_NAMES_PATH)

            # Load content-based models
            if config.TFIDF_VECTORIZER_PATH.exists():
                self._models["tfidf_vectorizer"] = joblib.load(config.TFIDF_VECTORIZER_PATH)
                self._loading_status["recommendation"]["content_based"] = "loaded"

            if config.KMEANS_MODEL_PATH.exists():
                self._models["kmeans_model"] = joblib.load(config.KMEANS_MODEL_PATH)

            if config.PRODUCTS_BY_CLUSTER_PATH.exists():
                self._models["products_by_cluster"] = joblib.load(config.PRODUCTS_BY_CLUSTER_PATH)

            # Load Arabic stop words if available
            if config.ARABIC_STOP_WORDS_PATH.exists():
                self._models["arabic_stop_words"] = joblib.load(config.ARABIC_STOP_WORDS_PATH)

        except Exception as e:
            print(f"Error loading recommendation models: {e}")
            # Set status to error for failed models
            for key in self._loading_status["recommendation"]:
                if self._loading_status["recommendation"][key] == "not_loaded":
                    self._loading_status["recommendation"][key] = f"error: {str(e)}"

    def load_image_models(self) -> None:
        """
        Eagerly load image models on startup.
        Loads ResNet50 feature extractor, butterfly classifier, and KNN index.
        """
        try:
            # Import TensorFlow here to avoid loading if not needed
            import tensorflow as tf
            from sklearn.neighbors import NearestNeighbors

            # Load ResNet50 feature extractor
            if config.RESNET50_PATH.exists():
                self._models["resnet50"] = tf.keras.models.load_model(config.RESNET50_PATH)
                self._loading_status["image"]["feature_extractor"] = "loaded"

            # Load butterfly classifier
            if config.BUTTERFLY_CLASSIFIER_PATH.exists():
                self._models["butterfly_classifier"] = tf.keras.models.load_model(
                    config.BUTTERFLY_CLASSIFIER_PATH
                )
                self._loading_status["image"]["classifier"] = "loaded"

            # Load KNN index and feature list
            if config.FEATURE_LIST_PATH.exists() and config.FILENAMES_PATH.exists():
                feature_list = np.load(config.FEATURE_LIST_PATH)
                filenames = joblib.load(config.FILENAMES_PATH)

                # Build KNN index
                neighbors = NearestNeighbors(
                    n_neighbors=5,
                    algorithm='ball_tree',
                    metric='euclidean'
                )
                neighbors.fit(feature_list)

                self._models["knn_neighbors"] = neighbors
                self._models["feature_list"] = feature_list
                self._models["image_filenames"] = filenames
                self._loading_status["image"]["knn_index"] = "loaded"

            # Load PCA model if available
            if config.PCA_MODEL_PATH.exists():
                self._models["pca_model"] = joblib.load(config.PCA_MODEL_PATH)

        except Exception as e:
            print(f"Error loading image models: {e}")
            for key in self._loading_status["image"]:
                if self._loading_status["image"][key] == "not_loaded":
                    self._loading_status["image"][key] = f"error: {str(e)}"

    def get_chatbot_model(self) -> Optional[Tuple[Any, Any]]:
        """
        Lazy load chatbot model on first request.
        Loads Llama 3 8B with 4-bit quantization and LoRA adapters.

        Returns:
            Tuple of (model, tokenizer) if successful, None if failed
        """
        # Check if already loaded
        if "chatbot_model" in self._models:
            return self._models["chatbot_model"], self._models["chatbot_tokenizer"]

        # Check GPU memory before loading
        if not self._check_gpu_memory(required_gb=5.0):
            print("Insufficient GPU memory for chatbot model")
            self._loading_status["chatbot"]["llama_base"] = "error: insufficient GPU memory"
            return None

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import PeftModel

            # Configure 4-bit quantization
            compute_dtype = torch.float16
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.LLAMA_BASE_MODEL,
                token=config.HF_TOKEN
            )

            # Load base model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                config.LLAMA_BASE_MODEL,
                quantization_config=quant_config,
                device_map="auto",
                token=config.HF_TOKEN
            )
            # Enable cache for faster inference (disable for training)
            model.config.use_cache = True
            model.config.pretraining_tp = 1

            self._loading_status["chatbot"]["llama_base"] = "loaded"

            # Load LoRA adapters if available
            if Path(config.LORA_ADAPTER_PATH).exists():
                model = PeftModel.from_pretrained(model, config.LORA_ADAPTER_PATH)
                self._loading_status["chatbot"]["lora_adapter"] = "loaded"
            else:
                print(f"LoRA adapter not found at {config.LORA_ADAPTER_PATH}")
                self._loading_status["chatbot"]["lora_adapter"] = "not_found"

            # Set padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            self._models["chatbot_model"] = model
            self._models["chatbot_tokenizer"] = tokenizer

            return model, tokenizer

        except Exception as e:
            print(f"Error loading chatbot model: {e}")
            self._loading_status["chatbot"]["llama_base"] = f"error: {str(e)}"
            return None

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a specific model to free memory.

        Args:
            model_name: Name of the model to unload

        Returns:
            True if model was unloaded, False if model wasn't loaded
        """
        if model_name not in self._models:
            return False

        del self._models[model_name]
        gc.collect()

        # Clear GPU cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Update loading status
        status_mapping = {
            "popularity_counts": ("recommendation", "popularity"),
            "svd_model": ("recommendation", "collaborative"),
            "correlation_matrix": ("recommendation", "collaborative"),
            "product_names": ("recommendation", "collaborative"),
            "tfidf_vectorizer": ("recommendation", "content_based"),
            "kmeans_model": ("recommendation", "content_based"),
            "products_by_cluster": ("recommendation", "content_based"),
            "resnet50": ("image", "feature_extractor"),
            "butterfly_classifier": ("image", "classifier"),
            "knn_neighbors": ("image", "knn_index"),
            "chatbot_model": ("chatbot", "llama_base"),
            "chatbot_tokenizer": ("chatbot", "llama_base"),
        }

        if model_name in status_mapping:
            category, key = status_mapping[model_name]
            self._loading_status[category][key] = "unloaded"

        return True

    def get_model(self, model_name: str) -> Optional[Any]:
        """
        Get a loaded model by name.

        Args:
            model_name: Name of the model to retrieve

        Returns:
            Model object if loaded, None otherwise
        """
        return self._models.get(model_name)

    def is_model_loaded(self, model_name: str) -> bool:
        """
        Check if a specific model is loaded.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model is loaded, False otherwise
        """
        return model_name in self._models

    def get_status(self) -> Dict[str, Dict[str, str]]:
        """
        Get current loading status of all models.

        Returns:
            Deep copy of dictionary with model loading status
        """
        return copy.deepcopy(self._loading_status)

    def _check_gpu_memory(self, required_gb: float = 5.0) -> bool:
        """
        Check if sufficient GPU memory is available.

        Args:
            required_gb: Required GPU memory in GB

        Returns:
            True if sufficient memory available, False otherwise
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return False

            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated_memory = torch.cuda.memory_allocated() / 1e9
            available_memory = total_memory - allocated_memory

            return available_memory >= required_gb

        except ImportError:
            return False

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """
        Get GPU memory information.

        Returns:
            Dictionary with GPU memory stats
        """
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "available": True,
                    "memory_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                    "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
                    "memory_cached_gb": round(torch.cuda.memory_reserved() / 1e9, 2)
                }
        except ImportError:
            pass

        return {
            "available": False,
            "memory_used_gb": 0,
            "memory_total_gb": 0
        }
