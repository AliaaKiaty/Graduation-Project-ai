# AI-Powered E-Commerce Platform — ML Backend

A graduation project implementing a full machine learning backend for an Arabic e-commerce platform. The system combines product recommendations, image similarity search, and an Arabic AI chatbot, exposed via a production-ready REST API deployed on Google Cloud Run.

---

## Features

| Feature | Technology |
|---|---|
| Popularity-based recommendations | PostgreSQL query (rating counts & averages) |
| Collaborative filtering | TruncatedSVD + correlation matrix |
| Content-based search | TF-IDF + KMeans clustering |
| Image similarity search | ResNet50 feature extraction + KNN (Ball Tree) |
| Image classification | Fine-tuned ResNet50 (butterfly dataset, 10 classes) |
| Arabic AI chatbot | Z.AI API (GLM — no GPU required) |
| Authentication | External JWT validation (from .NET backend) |
| Database | PostgreSQL (shared schema with .NET backend) |

---

## System Architecture

```
.NET Backend ──JWT──► ML API (Cloud Run)
                           │
                ┌──────────┼──────────────┐
                │          │              │
          Recommendation  Image         Chat
          Engine          Similarity    (Z.AI API)
          (SVD + TF-IDF)  (ResNet50)    (GLM-4.7-flash)
                │
          PostgreSQL (Amazon RDS)
```

The ML API runs as a stateless container on **Google Cloud Run**. Model files are baked into the Docker image. The database is shared with the .NET backend — the ML API reads product/interaction data and owns `product_embeddings` and `model_metadata` tables.

---

## Project Structure

```
Aliaa Graduation Project/
│
├── Recommendation_System_LV.ipynb      # Recommendation system experiments
├── Object Detection.ipynb              # Image similarity & classification experiments
├── llama_3_fine_tuning_on_arabic_chatbott.ipynb  # Llama 3 fine-tuning (Colab)
│
├── ml_api/                             # Production FastAPI backend
│   ├── main.py                         # App entry point & lifespan
│   ├── config.py                       # All environment variable config
│   ├── database.py                     # PostgreSQL engine & table init
│   ├── limiter.py                      # Rate limiting setup
│   │
│   ├── api/                            # Route handlers
│   │   ├── recommendation.py           # /recommend/* endpoints
│   │   ├── image.py                    # /image/* endpoints
│   │   ├── chat.py                     # /chat/* endpoints
│   │   └── admin.py                    # /admin/* endpoints
│   │
│   ├── models/                         # ML model engines
│   │   ├── loader.py                   # Singleton model manager
│   │   ├── recommendation.py           # SVD + TF-IDF inference
│   │   ├── image.py                    # ResNet50 + KNN inference
│   │   └── chatbot.py                  # Z.AI API client
│   │
│   ├── auth/                           # JWT validation (external)
│   ├── schemas/                        # Pydantic request/response models
│   ├── scripts/                        # DB migration & model retraining
│   │
│   ├── saved_models/
│   │   ├── recommendation/             # correlation_matrix.npy, tfidf_vectorizer.pkl, etc.
│   │   └── image/                      # resnet50_feature_extractor.keras, feature_list.npy, etc.
│   │
│   ├── Dockerfile.cloudrun             # Production Dockerfile (two-stage, no GPU)
│   ├── Dockerfile.cloudrun.dockerignore
│   ├── cloudbuild.yaml                 # Google Cloud Build pipeline
│   ├── requirements-cloudrun.txt       # Slim production dependencies
│   └── DEPLOYMENT.md                   # Full deployment guide
│
├── content/                            # Training datasets
│   ├── product_descriptions.csv
│   └── CompanyReviews.csv
│
└── ratings_Beauty.csv                  # Amazon Beauty ratings (training data)
```

---

## ML Models

### 1. Recommendation System

Three complementary approaches:

**Popularity-based** — ranks products by total rating count and average rating from the database. Best for new users (cold start).

**Collaborative Filtering (SVD)** — builds a user-item utility matrix, applies `TruncatedSVD` (n=10 components), and computes item-item correlations. Recommends items correlated with a given product.

**Content-based (TF-IDF + KMeans)** — vectorises product descriptions with TF-IDF, clusters with KMeans, and retrieves products from the same cluster as a search query. Supports both English and Arabic text.

Evaluation metrics: RMSE, Precision@K, Recall@K, NDCG@K, Coverage.

### 2. Image Similarity Search

- **Feature extraction**: ResNet50 pretrained on ImageNet (`include_top=False, pooling='max'`) → 2048-dimensional vectors
- **Dimensionality reduction**: PCA → 100 dimensions
- **Similarity search**: KNN with Ball Tree (Euclidean distance), k=5
- **Classification extension**: fine-tuned ResNet50 on Leeds Butterfly dataset (10 classes, two-phase training with data augmentation)

### 3. Arabic Chatbot

Powered by the **Z.AI API** (GLM model family). No GPU or local model files required — calls the hosted API at runtime.

- System prompt configured as an Arabic e-commerce assistant
- Responds in Arabic regardless of input language
- Model switchable via `ZAI_MODEL` environment variable (no rebuild needed)

---

## API Endpoints

**Base URL:** `https://ml-api-727549809675.me-central1.run.app`

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/health` | — | Service health & model status |
| GET | `/docs` | — | Swagger UI |
| POST | `/recommend/popular` | Optional | Top products by ratings |
| POST | `/recommend/collaborative` | Optional | SVD item-based recommendations |
| POST | `/recommend/content-based` | Optional | TF-IDF keyword search |
| GET | `/recommend/products` | Optional | List/search products |
| GET | `/recommend/products/{id}` | Optional | Single product detail |
| POST | `/recommend/products` | Admin | Create product |
| PATCH | `/recommend/products/{id}` | Admin | Update product |
| DELETE | `/recommend/products/{id}` | Admin | Soft-delete product |
| GET | `/recommend/categories` | Optional | List categories |
| POST | `/image/similar` | Required | Find similar images (upload) |
| POST | `/image/classify` | Required | Classify image (upload) |
| POST | `/chat/message` | Required | Arabic chatbot response |
| GET | `/chat/status` | Required | Chatbot readiness |
| POST | `/admin/migrate` | Admin | Create DB tables |
| POST | `/admin/retrain` | Admin | Retrain ML models |
| GET | `/admin/models` | Admin | List model versions |

### Example Requests

**Popular products:**
```bash
curl -X POST https://ml-api-727549809675.me-central1.run.app/recommend/popular \
  -H "Content-Type: application/json" \
  -d '{"top_n": 10}'
```

**Content-based search:**
```bash
curl -X POST https://ml-api-727549809675.me-central1.run.app/recommend/content-based \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"search_query": "wireless headphones", "top_n": 5}'
```

**Arabic chatbot:**
```bash
curl -X POST https://ml-api-727549809675.me-central1.run.app/chat/message \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"message": "ما هي أفضل منتجات العناية بالبشرة؟", "max_tokens": 512}'
```

**Image similarity:**
```bash
curl -X POST https://ml-api-727549809675.me-central1.run.app/image/similar \
  -H "Authorization: Bearer <token>" \
  -F "file=@product.jpg" \
  -F "top_k=5"
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | PostgreSQL URL (`postgres://` prefix supported) |
| `ZAI_API_KEY` | Yes | Z.AI API key for the chatbot |
| `ZAI_MODEL` | No | GLM model name (default: `glm-4.5`) |
| `BYPASS_AUTH` | Dev | Set `true` to skip JWT validation |
| `JWT_SECRET_KEY` | Production | Shared secret with .NET backend |
| `JWT_ISSUER` | Optional | Expected token issuer |
| `ALLOWED_ORIGINS` | Optional | CORS origins, comma-separated |

---

## Running Locally

```bash
# Install dependencies
pip install -r ml_api/requirements-cloudrun.txt

# Set environment variables
export DATABASE_URL="postgresql://user:pass@localhost:5432/ml_db"
export ZAI_API_KEY="your-zai-key"
export BYPASS_AUTH="true"

# Run the API
cd ml_api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/docs` for the interactive Swagger UI.

On first run, call `POST /admin/migrate` to create all database tables.

---

## Deployment

The API is deployed on **Google Cloud Run** (region: `me-central1`). See [`ml_api/DEPLOYMENT.md`](ml_api/DEPLOYMENT.md) for the full guide, including:

- First-time GCP setup (Artifact Registry, IAM)
- Standard build & deploy workflow
- How to update model files
- Switching the chatbot model without a rebuild
- Production security checklist

**Quick redeploy after a code change:**
```bash
# From ml_api/ directory
gcloud builds submit . --config=cloudbuild.yaml --ignore-file=Dockerfile.cloudrun.dockerignore --project=project-caba28d9-9df4-468d-87f

gcloud run deploy ml-api \
  --image=me-central1-docker.pkg.dev/project-caba28d9-9df4-468d-87f/ml-api/ml-api:latest \
  --region=me-central1 --project=project-caba28d9-9df4-468d-87f
```

Upload size is ~130KB (model files stay in the registry — no re-upload needed).

---

## Notebooks

| Notebook | Purpose | Platform |
|---|---|---|
| `Recommendation_System_LV.ipynb` | Trains and evaluates all three recommendation models | Local / Colab |
| `Object Detection.ipynb` | Trains ResNet50 feature extractor, KNN index, and butterfly classifier | Local / Colab |
| `llama_3_fine_tuning_on_arabic_chatbott.ipynb` | Fine-tunes Llama 3 8B on Arabic instruction dataset using LoRA | Google Colab Pro (GPU) |

---

## Database Schema

Tables managed by the **.NET backend** (ML API reads only):

| Table | Description |
|---|---|
| `Products` | Product catalogue (name, price, image, description, category) |
| `ProductCategories` | Category hierarchy |
| `UserInteraction` | Ratings, favourites, reviews |

Tables **owned by the ML API**:

| Table | Description |
|---|---|
| `product_embeddings` | KMeans cluster assignment per product (content-based filtering) |
| `model_metadata` | ML model versions, training metrics, active flags |

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn + Gunicorn |
| ML libraries | TensorFlow 2.20, Keras 3.12, scikit-learn 1.7 |
| Database ORM | SQLAlchemy 2.0 + psycopg2 |
| HTTP client | httpx |
| Authentication | python-jose (JWT validation) |
| Rate limiting | slowapi |
| Chatbot API | Z.AI (GLM family) |
| Container | Docker (python:3.10-slim) |
| CI/CD | Google Cloud Build |
| Hosting | Google Cloud Run (me-central1) |
| Registry | Google Artifact Registry |
| Database | Amazon RDS PostgreSQL (eu-west-1) |
