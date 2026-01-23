"""
Configuration settings for ML API
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# JWT Settings
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "change-this-secret-key-in-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Database Settings
DATABASE_PATH = os.environ.get("DATABASE_PATH", str(BASE_DIR / "data" / "users.db"))
# Convert Windows backslashes to forward slashes for SQLite URL
DATABASE_URL = f"sqlite:///{DATABASE_PATH.replace(chr(92), '/')}"

# Model Paths
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(BASE_DIR / "saved_models")))

# Recommendation Model Paths
RECOMMENDATION_DIR = MODELS_DIR / "recommendation"
POPULARITY_COUNTS_PATH = RECOMMENDATION_DIR / "popularity_counts.pkl"
SVD_MODEL_PATH = RECOMMENDATION_DIR / "svd_model.pkl"
CORRELATION_MATRIX_PATH = RECOMMENDATION_DIR / "correlation_matrix.npy"
PRODUCT_NAMES_PATH = RECOMMENDATION_DIR / "product_names.pkl"
TFIDF_VECTORIZER_PATH = RECOMMENDATION_DIR / "tfidf_vectorizer.pkl"
KMEANS_MODEL_PATH = RECOMMENDATION_DIR / "kmeans_model.pkl"
PRODUCTS_BY_CLUSTER_PATH = RECOMMENDATION_DIR / "products_by_cluster.pkl"
ARABIC_STOP_WORDS_PATH = RECOMMENDATION_DIR / "arabic_stop_words.pkl"

# Image Model Paths
IMAGE_DIR = MODELS_DIR / "image"
RESNET50_PATH = IMAGE_DIR / "resnet50_feature_extractor.keras"
BUTTERFLY_CLASSIFIER_PATH = IMAGE_DIR / "butterfly_classifier_final.keras"
FEATURE_LIST_PATH = IMAGE_DIR / "feature_list.npy"
FILENAMES_PATH = IMAGE_DIR / "filenames.pkl"
PCA_MODEL_PATH = IMAGE_DIR / "pca_model.pkl"

# Chatbot Model Paths
CHATBOT_DIR = MODELS_DIR / "chatbot"
LLAMA_BASE_MODEL = os.environ.get("LLAMA_BASE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
LORA_ADAPTER_PATH = os.environ.get("LORA_ADAPTER_PATH", str(CHATBOT_DIR / "llama-3-8B-Arabic"))

# HuggingFace Token
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# API Settings
API_TITLE = "ML Models API"
API_DESCRIPTION = "Machine Learning Models API for Recommendation, Image Similarity, and Arabic Chatbot"
API_VERSION = "1.0.0"
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

# File Upload Settings
MAX_UPLOAD_SIZE_MB = 10
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg"}

# Rate Limiting
RATE_LIMIT_LOGIN = "10/minute"
RATE_LIMIT_GENERAL = "100/minute"

# Butterfly Classification Classes
BUTTERFLY_CLASSES = [
    'Danaus plexippus',
    'Heliconius charitonius',
    'Heliconius erato',
    'Junonia coenia',
    'Lycaena phlaeas',
    'Nymphalis antiopa',
    'Papilio cresphontes',
    'Pieris rapae',
    'Vanessa atalanta',
    'Vanessa cardui'
]

# Admin User (for initialization)
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@api.local")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "change_this_password")
