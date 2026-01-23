# ML API Backend Implementation Plan

## Overview
Deploy three ML models (Recommendation System, Image Similarity, Arabic Chatbot) as a FastAPI backend on Google Colab with ngrok tunneling and JWT authentication.

---

## Project Structure

```
ml_api/
├── main.py                      # FastAPI app entry point
├── config.py                    # Environment configuration
├── database.py                  # SQLite setup
├── requirements.txt             # Dependencies
│
├── auth/
│   ├── __init__.py
│   ├── models.py                # User SQLAlchemy model
│   ├── schemas.py               # Auth Pydantic schemas
│   ├── security.py              # JWT + bcrypt utilities
│   └── router.py                # Auth endpoints
│
├── models/
│   ├── __init__.py
│   ├── loader.py                # Model manager (singleton)
│   ├── recommendation.py        # All 3 recommendation approaches
│   ├── image.py                 # ResNet50 + KNN + Classifier
│   └── chatbot.py               # Llama 3 inference
│
├── api/
│   ├── __init__.py
│   ├── recommendation.py        # /recommend/* routes
│   ├── image.py                 # /image/* routes
│   └── chat.py                  # /chat/* routes
│
├── schemas/
│   ├── __init__.py
│   ├── recommendation.py
│   ├── image.py
│   └── chat.py
│
└── saved_models/                # Directory for model artifacts
    ├── popularity_counts.pkl
    ├── svd_model.pkl
    ├── correlation_matrix.npy
    ├── product_names.pkl
    ├── tfidf_vectorizer.pkl
    ├── kmeans_model.pkl
    ├── products_by_cluster.pkl
    ├── feature_list.npy
    ├── filenames.pkl
    └── llama-3-8B-Arabic/       # LoRA adapters
```

---

## API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/login` | Login with username/password, returns JWT |
| POST | `/auth/refresh` | Refresh access token |
| GET | `/auth/me` | Get current user info |

### Recommendation
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/recommend/popular` | Get popular products (query: dataset, top_n) |
| POST | `/recommend/collaborative` | SVD-based recommendations for a product |
| POST | `/recommend/content-based` | TF-IDF+KMeans search by query text |

### Image
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/image/similar` | Find similar images (upload file) |
| POST | `/image/classify` | Classify butterfly image |

### Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat/message` | Arabic chatbot inference |

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API and model status |

---

## Database Schema

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    is_admin BOOLEAN DEFAULT TRUE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Model Loading Strategy

| Model | Size | Strategy | Reason |
|-------|------|----------|--------|
| Popularity counts | ~1MB | Eager | Fast, small |
| SVD + Correlation | ~10MB | Eager | Required for CF |
| TF-IDF + KMeans | ~10MB | Eager | Required for content |
| ResNet50 | ~94MB | Eager | Shared for image endpoints |
| Butterfly classifier | ~100MB | Eager | GPU efficient |
| KNN index | ~75MB | Eager | Fast retrieval |
| Llama 3 8B (4-bit) | ~4.6GB | Lazy | High memory, load on demand |

---

## Security Measures

### 1. JWT Authentication
- Access token: 30 min expiry
- Refresh token: 7 days expiry
- HS256 algorithm

### 2. Password Security
- bcrypt hashing (cost factor 12)
- No plaintext storage

### 3. Input Validation
- Pydantic schemas for all endpoints
- File size limit: 10MB
- Allowed image types: JPEG, PNG

### 4. Rate Limiting
- Login: 10 requests/minute
- General: 100 requests/minute

---

## Dependencies (requirements.txt)

```
# Core
fastapi==0.109.0
uvicorn[standard]==0.27.0
pyngrok==7.0.0

# Authentication
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# Database
sqlalchemy==2.0.25
aiosqlite==0.19.0

# Rate Limiting
slowapi==0.1.9

# ML - Shared
numpy==1.26.3
pandas==2.1.4
scikit-learn==1.4.0
pillow==10.2.0
joblib==1.3.2

# ML - Image
tensorflow==2.15.0

# ML - NLP/LLM
torch==2.1.2
transformers==4.36.2
peft==0.7.1
bitsandbytes==0.41.3
accelerate==0.25.0

# Utilities
python-dotenv==1.0.0
nest-asyncio==1.6.0
```

---

## Implementation Tasks

### Phase 1: Core Infrastructure
1. Create project folder structure (`ml_api/`)
2. Create `config.py` - environment variables and settings
3. Create `database.py` - SQLite connection and session management
4. Create `requirements.txt` - all dependencies
5. Create `main.py` - FastAPI app with middleware

### Phase 2: Authentication System
6. Create `auth/models.py` - User SQLAlchemy model
7. Create `auth/schemas.py` - Login, Token, User Pydantic schemas
8. Create `auth/security.py` - JWT creation/validation, password hashing
9. Create `auth/router.py` - /auth/login, /auth/refresh, /auth/me endpoints
10. Create admin user initialization script

### Phase 3: Model Loader
11. Create `models/loader.py` - Singleton ModelManager class
12. Implement eager loading for recommendation models
13. Implement eager loading for image models
14. Implement lazy loading for LLM with GPU memory check

### Phase 4: Recommendation Service
15. Extract model artifacts from `Recommendation_System_LV.ipynb`:
    - Popularity counts
    - SVD model + correlation matrix
    - TF-IDF vectorizer + KMeans model
16. Create `models/recommendation.py` - inference functions
17. Create `schemas/recommendation.py` - request/response schemas
18. Create `api/recommendation.py` - API routes

### Phase 5: Image Service
19. Create `models/image.py` - feature extraction + KNN + classification
20. Create `schemas/image.py` - request/response schemas
21. Create `api/image.py` - API routes with file upload

### Phase 6: Chat Service
22. Create `models/chatbot.py` - Llama 3 with LoRA inference
23. Create `schemas/chat.py` - request/response schemas
24. Create `api/chat.py` - API routes

### Phase 7: Integration & Deployment
25. Create `deploy_ml_api.ipynb` - Colab deployment notebook
26. Add ngrok tunnel setup
27. Add model loading cells
28. Add testing cells

### Phase 8: Testing & Documentation
29. Test all authentication endpoints
30. Test all recommendation endpoints
31. Test all image endpoints
32. Test chat endpoint
33. Update CLAUDE.md with API documentation

---

## Verification Steps

### 1. Test Authentication
```bash
# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your_password"}'

# Use token for protected endpoints
curl http://localhost:8000/health \
  -H "Authorization: Bearer <token>"
```

### 2. Test Recommendation Endpoints
```bash
# Popular products
curl "http://localhost:8000/recommend/popular?top_n=5" \
  -H "Authorization: Bearer <token>"

# Collaborative filtering
curl -X POST http://localhost:8000/recommend/collaborative \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"product_id": "B001MA0QY2", "top_n": 5}'

# Content-based
curl -X POST http://localhost:8000/recommend/content-based \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"search_query": "cutting tool", "top_n": 5}'
```

### 3. Test Image Endpoints
```bash
# Similar images
curl -X POST http://localhost:8000/image/similar \
  -H "Authorization: Bearer <token>" \
  -F "file=@test_image.jpg"

# Classification
curl -X POST http://localhost:8000/image/classify \
  -H "Authorization: Bearer <token>" \
  -F "file=@butterfly.jpg"
```

### 4. Test Chat Endpoint
```bash
curl -X POST http://localhost:8000/chat/message \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"message": "ما هي عاصمة مصر؟"}'
```

### 5. Colab Integration Test
- Run all cells in deploy_ml_api.ipynb
- Verify ngrok URL is accessible
- Test all endpoints via public URL

---

## Files Reference

### Files to Create
| File | Description |
|------|-------------|
| `ml_api/main.py` | FastAPI application entry point |
| `ml_api/config.py` | Configuration and environment variables |
| `ml_api/database.py` | SQLite database setup |
| `ml_api/requirements.txt` | Python dependencies |
| `ml_api/auth/*.py` | Authentication module (5 files) |
| `ml_api/models/*.py` | Model loaders (5 files) |
| `ml_api/api/*.py` | API routers (4 files) |
| `ml_api/schemas/*.py` | Pydantic schemas (4 files) |
| `deploy_ml_api.ipynb` | Colab deployment notebook |

### Existing Files to Reference
| File | Purpose |
|------|---------|
| `Recommendation_System_LV.ipynb` | Extract recommendation model artifacts |
| `Object Detection.ipynb` | Reference image model loading |
| `llama_3_fine_tuning_on_arabic_chatbott.ipynb` | Reference LLM inference code |
| `resnet50_feature_extractor.keras` | Pre-saved ResNet50 model |

---

## Notes

### GPU Memory Management (16GB GPU)
- ResNet50 + Classifier + KNN: ~500MB
- Llama 3 8B (4-bit): ~4.6GB
- Total when all loaded: ~5.1GB
- Remaining for inference buffers: ~10.9GB

### Colab Session Tips
- Use Colab Pro for longer sessions
- Save model artifacts to Google Drive
- Use ngrok authtoken for stable tunnels
- Implement heartbeat to prevent session timeout
