# ML API Implementation Tasks

## Overview
Checklist of tasks to implement the ML API backend. Check off tasks as you complete them.

---

## Phase 1: Core Infrastructure ✅

- [x] **Task 1.1:** Create `ml_api/` folder structure
  ```
  ml_api/
  ├── auth/
  ├── models/
  ├── api/
  ├── schemas/
  └── saved_models/
  ```

- [x] **Task 1.2:** Create `ml_api/config.py`
  - JWT secret key
  - Token expiration settings
  - Database path
  - Model paths
  - HuggingFace token

- [x] **Task 1.3:** Create `ml_api/database.py`
  - SQLAlchemy engine setup
  - Session management
  - Base model class

- [x] **Task 1.4:** Create `ml_api/requirements.txt`
  - FastAPI, uvicorn, pyngrok
  - python-jose, passlib, bcrypt
  - SQLAlchemy
  - TensorFlow, PyTorch
  - transformers, peft, bitsandbytes
  - scikit-learn, numpy, pandas

- [x] **Task 1.5:** Create `ml_api/main.py`
  - FastAPI app instance
  - CORS middleware
  - Rate limiting middleware
  - Include all routers
  - Health check endpoint

---

## Phase 2: Authentication System ✅

- [x] **Task 2.1:** Create `ml_api/auth/__init__.py`
  - Export router and dependencies

- [x] **Task 2.2:** Create `ml_api/auth/models.py`
  - User SQLAlchemy model
  - Fields: id, username, email, hashed_password, is_admin, is_active, created_at

- [x] **Task 2.3:** Create `ml_api/auth/schemas.py`
  - LoginRequest (username, password)
  - TokenResponse (access_token, refresh_token, token_type, expires_in)
  - UserResponse (id, username, email, is_admin, is_active, created_at)

- [x] **Task 2.4:** Create `ml_api/auth/security.py`
  - `verify_password(plain, hashed)` - bcrypt verification
  - `get_password_hash(password)` - bcrypt hashing
  - `create_access_token(data, expires_delta)` - JWT creation
  - `create_refresh_token(data)` - JWT refresh token
  - `decode_token(token)` - JWT validation

- [x] **Task 2.5:** Create `ml_api/auth/router.py`
  - `POST /auth/login` - authenticate and return tokens
  - `POST /auth/refresh` - refresh access token
  - `GET /auth/me` - get current user info
  - `get_current_user` dependency

- [x] **Task 2.6:** Create admin user initialization
  - Script to create default admin account
  - Hash password with bcrypt
  - Auto-initialization on startup

---

## Phase 3: Model Loader ✅

- [x] **Task 3.1:** Create `ml_api/models/__init__.py`
  - Export ModelManager

- [x] **Task 3.2:** Create `ml_api/models/loader.py`
  - Singleton ModelManager class
  - `load_recommendation_models()` - eager load
  - `load_image_models()` - eager load
  - `get_chatbot_model()` - lazy load
  - `unload_model(name)` - free memory
  - `get_status()` - model loading status
  - GPU memory management utilities

---

## Phase 4: Recommendation Service ✅ (Reviewed)

- [ ] **Task 4.1:** Extract model artifacts from notebook (deferred)
  - Run `Recommendation_System_LV.ipynb` to save:
    - `popularity_counts.pkl`
    - `svd_model.pkl`
    - `correlation_matrix.npy`
    - `product_names.pkl`
    - `tfidf_vectorizer.pkl`
    - `kmeans_model.pkl`
    - `products_by_cluster.pkl`
  - **Note:** Will be done during deployment/testing phase

- [x] **Task 4.2:** Create `ml_api/models/recommendation.py`
  - `get_popular_items(dataset, top_n)` - popularity based
  - `get_collaborative_recommendations(product_id, top_n)` - SVD based
  - `get_content_based_recommendations(query, top_n)` - TF-IDF + KMeans
  - Load models from pickle/numpy files

- [x] **Task 4.3:** Create `ml_api/schemas/recommendation.py`
  - PopularRequest/Response
  - CollaborativeRequest/Response
  - ContentBasedRequest/Response
  - RecommendationItem schema

- [x] **Task 4.4:** Create `ml_api/api/__init__.py`
  - Export routers

- [x] **Task 4.5:** Create `ml_api/api/recommendation.py`
  - `GET /recommend/popular`
  - `POST /recommend/collaborative`
  - `POST /recommend/content-based`
  - Require authentication
  - Input validation

---

## Phase 5: Image Service ✅ (Reviewed)

- [x] **Task 5.1:** Create `ml_api/models/image.py`
  - Load ResNet50 feature extractor
  - Load butterfly classifier
  - Build KNN index from feature list
  - `extract_features(image)` - get 2048-dim vector
  - `find_similar_images(image, top_k)` - KNN search
  - `classify_image(image)` - butterfly classification
  - Image preprocessing utilities

- [x] **Task 5.2:** Create `ml_api/schemas/image.py`
  - SimilarImagesResponse
  - ClassificationResponse
  - SimilarImageItem schema
  - PredictionItem schema

- [x] **Task 5.3:** Create `ml_api/api/image.py`
  - `POST /image/similar` - file upload, return similar images
  - `POST /image/classify` - file upload, return classification
  - Require authentication
  - File validation (size, type)

---

## Phase 6: Chat Service ✅

- [x] **Task 6.1:** Create `ml_api/models/chatbot.py`
  - Load Llama 3 8B with 4-bit quantization
  - Load LoRA adapters
  - `generate_response(message, max_tokens, temperature)`
  - Chat template formatting
  - GPU memory check before loading

- [x] **Task 6.2:** Create `ml_api/schemas/chat.py`
  - ChatRequest (message, max_tokens, temperature)
  - ChatResponse (input_message, response, model, tokens_generated, generation_time_ms)

- [x] **Task 6.3:** Create `ml_api/api/chat.py`
  - `POST /chat/message` - generate chat response
  - `GET /chat/status` - check model status
  - `POST /chat/load` - explicitly load model
  - `POST /chat/unload` - unload model to free memory
  - Require authentication
  - Handle model not loaded (503)

---

## Phase 7: Colab Deployment ✅

- [x] **Task 7.1:** Create `deploy_ml_api.ipynb`
  - Cell 1: Check GPU and runtime
  - Cell 2: Install dependencies
  - Cell 3: Mount Google Drive
  - Cell 4: Configure secrets (ngrok, HF, JWT, admin)
  - Cell 5: Setup ngrok
  - Cell 6: Upload/clone ML API code
  - Cell 7: Upload model files
  - Cell 8: Configure model paths
  - Cell 9: HuggingFace login
  - Cell 10: Start API server
  - Cell 11: Test all endpoints
  - Cell 12: Save models to Drive
  - Cell 13: Keep session alive

- [x] **Task 7.2:** Add ngrok tunnel setup
  - Install pyngrok
  - Set auth token from Colab Secrets
  - Create public URL
  - Display URL with docs and credentials

- [x] **Task 7.3:** Add Google Drive integration
  - Mount drive
  - Create persistent project directory
  - Copy model files from/to Drive
  - Save database to Drive

---

## Phase 8: Testing & Documentation ✅

- [x] **Task 8.1:** Test authentication endpoints (`ml_api/tests/test_auth.py`)
  - Login with valid credentials
  - Login with invalid credentials (username/password)
  - Refresh token (valid/invalid)
  - Access protected endpoint without token
  - Access protected endpoint with expired/malformed token

- [x] **Task 8.2:** Test recommendation endpoints (`ml_api/tests/test_recommendation.py`)
  - Get popular products (English/Arabic)
  - Get collaborative recommendations
  - Get content-based recommendations (English/Arabic)
  - Test with invalid/missing parameters

- [x] **Task 8.3:** Test image endpoints (`ml_api/tests/test_image.py`)
  - Upload valid image for similarity (JPEG/PNG)
  - Upload valid image for classification
  - Upload invalid file type
  - Upload file exceeding size limit
  - Test with various top_k parameters

- [x] **Task 8.4:** Test chat endpoint (`ml_api/tests/test_chat.py`)
  - Send Arabic message
  - Test with different temperatures
  - Test max_tokens limits
  - Test model status/load/unload endpoints

- [x] **Task 8.5:** Update CLAUDE.md
  - Added ML API Backend section
  - Added all API endpoint documentation
  - Added deployment instructions (local & Colab)
  - Added environment variables reference
  - Added test running instructions
  - Added model files reference

---

## Task Summary

| Phase | Tasks | Status |
|-------|-------|--------|
| 1. Core Infrastructure | 5 tasks | ✅ Complete |
| 2. Authentication | 6 tasks | ✅ Complete |
| 3. Model Loader | 2 tasks | ✅ Complete |
| 4. Recommendation | 5 tasks | ✅ Complete |
| 5. Image | 3 tasks | ✅ Complete |
| 6. Chat | 3 tasks | ✅ Complete |
| 7. Colab Deployment | 3 tasks | ✅ Complete |
| 8. Testing & Docs | 5 tasks | ✅ Complete |
| **Total** | **32 tasks** | **✅ All Complete** |

---

## Quick Start Order

For fastest path to working API:

1. Tasks 1.1-1.5 (Core infrastructure)
2. Tasks 2.1-2.6 (Authentication - needed for all endpoints)
3. Tasks 3.1-3.2 (Model loader)
4. Tasks 4.1-4.5 (Recommendation - simplest to test)
5. Tasks 5.1-5.3 (Image)
6. Tasks 6.1-6.3 (Chat - most complex, do last)
7. Tasks 7.1-7.3 (Deployment)
8. Tasks 8.1-8.5 (Testing)

---

## Dependencies Between Tasks

```
1.1 Folder Structure
 └── All other tasks

1.2 Config ─────────┐
1.3 Database ───────┼──> 2.x Auth tasks
1.4 Requirements ───┘

2.x Auth tasks ────────> All API tasks (4.5, 5.3, 6.3)

3.x Model Loader ──────> 4.2, 5.1, 6.1

4.1 Extract Models ────> 4.2 Recommendation Models

7.x Deployment ────────> 8.x Testing
```
