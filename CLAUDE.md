# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning graduation project containing three Jupyter notebooks implementing different AI/ML systems:

1. **Recommendation System** (`Recommendation_System_LV.ipynb`)
2. **Image Similarity Search** (`Object Detection.ipynb`)
3. **Arabic Chatbot Fine-tuning** (`llama_3_fine_tuning_on_arabic_chatbott.ipynb`)

---

## Notebook 1: Recommendation System (`Recommendation_System_LV.ipynb`)

### Purpose
Multi-approach product recommendation system supporting both English and Arabic data.

### Architecture

#### Part I: Popularity-Based Recommendations
- Simple counting of ratings per product/company
- Best for cold-start scenarios with new users
- Dataset: Amazon Beauty ratings (2M+ ratings)

#### Part II: Collaborative Filtering (SVD-based)
- Creates user-item utility matrix (sparse)
- Applies TruncatedSVD for dimensionality reduction (n_components=10)
- Computes correlation matrix between items
- Uses adaptive threshold for recommendations (top-N by correlation)
- **Evaluation Metrics**: RMSE, Precision@K, Recall@K, NDCG@K, Coverage

#### Part III: Content-Based Filtering (TF-IDF + KMeans)
- TF-IDF vectorization of product descriptions
- KMeans clustering with elbow method for optimal k
- Predicts cluster for search queries
- Extended to Arabic text with custom stop words list

### Data Sources
- `ratings_Beauty.csv` - Amazon product ratings
- `content/product_descriptions.csv` - Home Depot product descriptions
- `content/CompanyReviews.csv` - Arabic company reviews

### Key Functions
```python
# Popularity
popular_products = df.groupby('ProductId')['Rating'].count()

# SVD Collaborative Filtering
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(utility_matrix.T)
correlation_matrix = np.corrcoef(decomposed_matrix)

# Content-Based
vectorizer = TfidfVectorizer(stop_words='english')
kmeans = KMeans(n_clusters=10, init='k-means++')

# Evaluation Metrics
precision_at_k(actual, predicted, k)
recall_at_k(actual, predicted, k)
ndcg_at_k(actual, predicted, k)
```

### Recent Improvements
- Added comprehensive evaluation metrics (Precision@K, Recall@K, NDCG@K, Coverage)
- Fixed Arabic recommendation threshold issue (now uses adaptive top-N approach)
- Added cross-platform zip extraction (works on Windows and Linux)
- Added elbow method for optimal cluster selection

---

## Notebook 2: Image Similarity Search (`Object Detection.ipynb`)

### Purpose
Visual similarity search using deep learning feature extraction, plus butterfly classification.

### Architecture

#### Feature Extraction
- **Model**: ResNet50 pretrained on ImageNet
- **Configuration**: `include_top=False`, `pooling='max'`
- **Output**: 2048-dimensional feature vectors per image

#### Similarity Search
- **Method**: K-Nearest Neighbors with Ball Tree algorithm
- **Metric**: Euclidean distance
- **k**: 5 nearest neighbors

#### Dimensionality Reduction
- PCA reduces 2048 -> 100 dimensions
- Maintains search quality while improving speed

#### Classification Extension (Enhanced)
- Two-phase training: freeze base, then fine-tune top layers
- Data augmentation for training
- Early stopping and model checkpointing
- Evaluated on Leeds Butterfly dataset (10 classes)

#### Flask API
- `/health` - Health check endpoint
- `/predict` - Classification or retrieval endpoint
- Configurable via environment variables
- Proper error handling and input validation

### Data Sources
- Caltech-101 (9,144 images, 102 classes)
- Hymenoptera (ants/bees) dataset
- Leeds Butterfly dataset (832 images, 10 classes)

### Saved Models
- `resnet50_feature_extractor.keras` - Base feature extractor
- `butterfly_classifier_final.keras` - Trained classifier
- `best_butterfly_model.keras` - Best checkpoint

### Key Code Patterns
```python
# Feature extraction
model = ResNet50(weights='imagenet', include_top=False, pooling='max')
features = model.predict(datagen, steps=num_epochs)

# KNN search
neighbors = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric='euclidean')
neighbors.fit(feature_list)
distances, indices = neighbors.kneighbors(query_features)

# Two-phase training
# Phase 1: Train classification head only
for layer in base_model.layers:
    layer.trainable = False
history = model.fit(train_gen, epochs=10, validation_data=test_gen)

# Phase 2: Fine-tune top layers
for layer in base_model.layers[-30:]:
    layer.trainable = True
model.compile(optimizer=Adam(lr=0.0001), ...)
```

### Recent Improvements
- Added proper two-phase training loop with callbacks
- Added data augmentation (rotation, shift, flip, zoom)
- Fixed Flask API with environment variable configuration
- Added health check endpoint
- Added cross-platform zip extraction

---

## Notebook 3: Arabic Chatbot Fine-tuning (`llama_3_fine_tuning_on_arabic_chatbott.ipynb`)

### Purpose
Fine-tune Llama 3 8B for Arabic instruction-following using LoRA/PEFT.

### Architecture

#### Base Model
- **Model**: `meta-llama/Meta-Llama-3-8B-Instruct`
- **Quantization**: 4-bit NF4 via bitsandbytes
- **Total params**: 4.54B (quantized to ~4.5GB)

#### LoRA Configuration
```python
LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,  # rank
    bias="none",
    task_type="CAUSAL_LM"
)
# Trainable: 3.4M params (0.07% of total)
```

#### Training Setup (Enhanced)
- **Trainer**: SFTTrainer (Supervised Fine-Tuning)
- **Optimizer**: paged_adamw_32bit
- **Learning rate**: 2e-5 with cosine scheduler
- **Batch size**: 2 (with gradient accumulation)
- **Epochs**: 1
- **Precision**: FP16
- **Evaluation**: Every 200 steps with early stopping
- **Train/Test Split**: 90/10

### Dataset
- Arabic Instruct Chatbot Dataset (52,002 instruction-output pairs)
- Average instruction length: ~48 chars
- Average output length: ~233 chars
- Using 10,000 samples with train/test split

### Key Code
```python
# Quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Train/test split
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# SFTTrainer setup (fixed)
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_args,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# Inference
def single_inference(question):
    messages = [{"role": "system", "content": "اجب علي الاتي بالعربي فقط."}]
    messages.append({"role": "user", "content": question})
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    outputs = model.generate(input_ids, max_new_tokens=256, temperature=0.4)
```

### Recent Fixes (Critical)
- Fixed SyntaxError: removed duplicate `max_seq_length` argument
- Added train/test split for proper evaluation
- Added evaluation strategy during training
- Fixed pyarrow dependency conflict (commented out problematic downgrade)
- Added cross-platform zip extraction

---

## Environment Setup

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (16GB+ VRAM recommended for LLM fine-tuning)
- Google Colab Pro recommended for LLM notebook

### Installation
```bash
pip install -r requirements.txt
```

### Kaggle Setup
```bash
# Place kaggle.json in ~/.kaggle/
kaggle datasets download -d skillsmuggler/amazon-ratings
kaggle datasets download -d fahdseddik/arabic-company-reviews
kaggle datasets download -d imbikramsaha/caltech-101
kaggle datasets download -d omgits0mar/arabic-instruct-chatbot-dataset
```

### HuggingFace Setup (for LLM notebook)
```python
from huggingface_hub import login
login(token="your_token")  # Requires access to Llama 3
```

---

## Common Issues & Solutions

### Windows Compatibility (Fixed)
All notebooks now use cross-platform zip extraction:
```python
import zipfile
with zipfile.ZipFile('file.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
```

### Memory Issues
- Reduce batch size for LLM training
- Use gradient checkpointing
- Enable 8-bit or 4-bit quantization

### Arabic Text
- Ensure UTF-8 encoding for all files
- Use proper Arabic tokenizers
- Handle right-to-left text display

---

## Changes Summary

| Notebook | Change | Status |
|----------|--------|--------|
| LLM | Fixed duplicate `max_seq_length` SyntaxError | Done |
| LLM | Added train/test split | Done |
| LLM | Fixed pyarrow dependency conflict | Done |
| LLM | Added evaluation during training | Done |
| Image | Added proper two-phase training loop | Done |
| Image | Added data augmentation | Done |
| Image | Fixed Flask API with env vars | Done |
| Recommendation | Added Precision@K, Recall@K, NDCG@K | Done |
| Recommendation | Fixed Arabic threshold (adaptive top-N) | Done |
| All | Cross-platform zip extraction | Done |

---

## File Structure
```
Aliaa Graduation Project/
├── Recommendation_System_LV.ipynb    # Recommendation systems
├── Object Detection.ipynb             # Image similarity search
├── llama_3_fine_tuning_on_arabic_chatbott.ipynb  # LLM fine-tuning
├── deploy_ml_api.ipynb                # Colab deployment notebook
├── ratings_Beauty.csv                 # Amazon ratings data
├── resnet50_feature_extractor.keras   # Saved ResNet50 model
├── requirements.txt                   # Python dependencies
├── content/                           # Data directory
│   ├── product_descriptions.csv
│   ├── CompanyReviews.csv
│   ├── hymenoptera_data/
│   └── leedsbutterfly/
├── ml_api/                            # FastAPI backend
│   ├── main.py                        # App entry point
│   ├── config.py                      # Configuration
│   ├── database.py                    # SQLite setup
│   ├── auth/                          # Authentication module
│   ├── models/                        # ML model loaders
│   ├── api/                           # API routers
│   ├── schemas/                       # Pydantic schemas
│   └── tests/                         # Test suite
└── CLAUDE.md                          # This file
```

---

## Future Improvements

| Notebook | Suggested Improvement |
|----------|----------------------|
| Recommendation | Use specialized library (surprise, implicit) |
| Recommendation | Implement hybrid recommendation |
| Image | Use FAISS for large-scale similarity search |
| Image | Try other backbones (EfficientNet, ViT) |
| LLM | Use full 52K dataset for training |
| LLM | Add Arabic-specific evaluation benchmarks |
| LLM | Increase LoRA rank (r=16 or r=32) |

---

## ML API Backend (`ml_api/`)

FastAPI backend exposing all ML models via REST API with JWT authentication.

### Quick Start

```bash
# Install dependencies
pip install -r ml_api/requirements.txt

# Run locally
cd ml_api
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or deploy to Colab
# Open deploy_ml_api.ipynb in Google Colab
```

### API Endpoints

#### Authentication (`/auth`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/login` | Login with username/password, returns JWT tokens |
| POST | `/auth/refresh` | Refresh access token using refresh token |
| GET | `/auth/me` | Get current user info (requires auth) |

**Login Request:**
```json
{
  "username": "admin",
  "password": "your_password"
}
```

**Login Response:**
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### Recommendation (`/recommend`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/recommend/popular` | Get popular products |
| POST | `/recommend/collaborative` | SVD-based recommendations |
| POST | `/recommend/content-based` | TF-IDF + KMeans search |

**Popular Products:**
```bash
GET /recommend/popular?dataset=english&top_n=10
```

**Collaborative Filtering:**
```json
POST /recommend/collaborative
{
  "product_id": "B001MA0QY2",
  "top_n": 5
}
```

**Content-Based Search:**
```json
POST /recommend/content-based
{
  "search_query": "cutting tool",
  "top_n": 5,
  "language": "english"
}
```

#### Image (`/image`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/image/similar` | Find similar images (file upload) |
| POST | `/image/classify` | Classify butterfly image (file upload) |

**Similar Images:**
```bash
curl -X POST http://localhost:8000/image/similar \
  -H "Authorization: Bearer <token>" \
  -F "file=@image.jpg" \
  -F "top_k=5"
```

**Classification:**
```bash
curl -X POST http://localhost:8000/image/classify \
  -H "Authorization: Bearer <token>" \
  -F "file=@butterfly.jpg" \
  -F "top_k=3"
```

#### Chat (`/chat`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat/message` | Generate Arabic chatbot response |
| GET | `/chat/status` | Check model loading status |
| POST | `/chat/load` | Explicitly load model |
| POST | `/chat/unload` | Unload model to free GPU memory |

**Chat Message:**
```json
POST /chat/message
{
  "message": "ما هي عاصمة مصر؟",
  "max_tokens": 256,
  "temperature": 0.4
}
```

**Response:**
```json
{
  "input_message": "ما هي عاصمة مصر؟",
  "response": "عاصمة مصر هي القاهرة...",
  "model": "llama-3-8b-arabic",
  "tokens_generated": 45,
  "generation_time_ms": 1250
}
```

#### Health (`/health`)

```bash
GET /health
```

Returns API status, model loading status, and GPU info.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `JWT_SECRET_KEY` | Secret for JWT tokens | (required) |
| `ADMIN_USERNAME` | Admin username | `admin` |
| `ADMIN_PASSWORD` | Admin password | (required) |
| `HF_TOKEN` | HuggingFace token for Llama 3 | (optional) |
| `MODELS_DIR` | Path to saved models | `./saved_models` |
| `DATABASE_PATH` | SQLite database path | `./data/users.db` |

### Deployment to Google Colab

1. Open `deploy_ml_api.ipynb` in Google Colab
2. Set runtime to GPU (Runtime > Change runtime type > GPU)
3. Add secrets in Colab sidebar:
   - `NGROK_AUTH_TOKEN`: Your ngrok auth token
   - `HF_TOKEN`: HuggingFace token (for chatbot)
   - `JWT_SECRET`: Secret key for JWT
   - `ADMIN_PASSWORD`: Admin password
4. Run all cells sequentially
5. Access API via the ngrok public URL

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest ml_api/tests/ -v

# Run with coverage
pytest ml_api/tests/ -v --cov=ml_api --cov-report=html

# Run specific test file
pytest ml_api/tests/test_auth.py -v
```

### Model Files Required

Place these files in `ml_api/saved_models/`:

**Recommendation** (`saved_models/recommendation/`):
- `popularity_counts.pkl`
- `svd_model.pkl`
- `correlation_matrix.npy`
- `product_names.pkl`
- `tfidf_vectorizer.pkl`
- `kmeans_model.pkl`
- `products_by_cluster.pkl`

**Image** (`saved_models/image/`):
- `resnet50_feature_extractor.keras`
- `butterfly_classifier_final.keras`
- `feature_list.npy`
- `filenames.pkl`

**Chatbot** (`saved_models/chatbot/`):
- `llama-3-8B-Arabic/` (LoRA adapter directory)

### Security Features

- JWT authentication with access/refresh tokens
- bcrypt password hashing
- Rate limiting (10/min for login, 100/min for general)
- File size limits (10MB max)
- Input validation via Pydantic schemas
