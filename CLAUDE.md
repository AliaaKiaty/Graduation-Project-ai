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
- Recommends items with correlation > 0.90 threshold
- **Evaluation**: RMSE on test set = 0.957

#### Part III: Content-Based Filtering (TF-IDF + KMeans)
- TF-IDF vectorization of product descriptions
- KMeans clustering (k=10) to group similar products
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
```

### Issues Found
1. **Cell 90**: Empty recommendations for 'talbat' - threshold too high (0.2) given only 12 companies
2. **Cell 110**: Comprehensive refactored code with elbow method for optimal k, but duplicates earlier work
3. Arabic stop words list generates tokenization warnings

### Recommendations
1. **Use matrix factorization libraries**: Consider `surprise` or `implicit` for better collaborative filtering
2. **Add evaluation metrics**: Implement Precision@K, Recall@K, NDCG alongside RMSE
3. **Handle cold start better**: Hybrid approach combining content + collaborative
4. **Arabic NLP**: Use `camel-tools` or `pyarabic` for proper Arabic text preprocessing
5. **Optimize SVD components**: Use explained variance ratio to choose optimal n_components
6. **Add cross-validation**: Current train/test split is single-fold

---

## Notebook 2: Image Similarity Search (`Object Detection.ipynb`)

### Purpose
Visual similarity search using deep learning feature extraction.

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

#### Classification Extension
- Adds Dense layers on top of ResNet50 for butterfly classification (10 classes)
- Evaluated on Leeds Butterfly dataset

#### Flask API (Template)
- `/predict` endpoint for classification or retrieval
- Supports both tasks via configuration

### Data Sources
- Caltech-101 (9,144 images, 102 classes)
- Hymenoptera (ants/bees) dataset
- Leeds Butterfly dataset (832 images, 10 classes)

### Saved Models
- `resnet50_feature_extractor.keras` - Base feature extractor

### Key Code Patterns
```python
# Feature extraction
model = ResNet50(weights='imagenet', include_top=False, pooling='max')
features = model.predict(datagen, steps=num_epochs)

# KNN search
neighbors = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric='euclidean')
neighbors.fit(feature_list)
distances, indices = neighbors.kneighbors(query_features)

# PCA compression
pca = PCA(n_components=100)
compressed = pca.fit_transform(feature_list)
```

### Issues Found
1. **Naming**: Notebook titled "Object Detection" but implements image similarity/retrieval
2. **Cell 26**: Compiles feature extractor with classification loss (unnecessary for feature extraction)
3. **Cell 28**: Classification model evaluated without training (random weights on Dense layers)
4. **Cell 31**: Flask API has hardcoded placeholder paths
5. `unzip` commands fail on Windows

### Recommendations
1. **Rename notebook**: "Image_Similarity_Search.ipynb" more accurately describes functionality
2. **Add FAISS**: For large-scale similarity search, replace sklearn KNN with FAISS
3. **Try other backbones**: EfficientNet, ViT may provide better features
4. **Implement proper training**: The classification extension needs actual training loop
5. **Add evaluation metrics**: mAP, Precision@K for retrieval quality
6. **Persist features**: Save extracted features to avoid recomputation
7. **Fix Flask API**:
   - Add proper error handling
   - Use environment variables for paths
   - Add input validation
8. **Add data augmentation**: For the classification task

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

#### Training Setup
- **Trainer**: SFTTrainer (Supervised Fine-Tuning)
- **Optimizer**: paged_adamw_32bit
- **Learning rate**: 2e-5 with cosine scheduler
- **Batch size**: 2 (with gradient accumulation)
- **Epochs**: 1
- **Precision**: FP16

### Dataset
- Arabic Instruct Chatbot Dataset (52,002 instruction-output pairs)
- Average instruction length: ~48 chars
- Average output length: ~233 chars
- Only 10,000 samples used for training

### Key Code
```python
# Quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Inference
def single_inference(question):
    messages = [{"role": "system", "content": "اجب علي الاتي بالعربي فقط."}]
    messages.append({"role": "user", "content": question})
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    outputs = model.generate(input_ids, max_new_tokens=256, temperature=0.4)
```

### Issues Found (Critical)
1. **Cell 30**: SyntaxError - `max_seq_length` argument repeated twice
2. **Cell 30**: `test_dataset` undefined - no train/test split performed
3. **Cell 32-34**: Training and saving fail due to Cell 30 error
4. **Cell 4**: Downgrades pyarrow causing dependency conflict with datasets
5. Memory-efficient SDP disabled but may be needed for stability

### Fixed Code for Cell 30
```python
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset,
    peft_config=peft_args,
    dataset_text_field="text",
    max_seq_length=512,  # Single value, increased for Arabic
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)
```

### Recommendations
1. **Fix critical bugs**:
   - Remove duplicate `max_seq_length`
   - Add train/test split before creating trainer
2. **Increase training data**: Use full 52K dataset, not just 10K
3. **Add evaluation**:
   - Split data for validation
   - Implement perplexity, BLEU/ROUGE metrics
4. **Optimize sequence length**:
   - Instruction: 128 tokens may truncate
   - Output: 256 tokens is reasonable
   - Consider dynamic padding
5. **Improve LoRA config**:
   - Try r=16 or r=32 for better performance
   - Target more modules (q_proj, k_proj, v_proj, o_proj)
6. **Add checkpointing**: Save intermediate checkpoints during training
7. **Implement proper chat template**: Current system prompt is minimal
8. **Add Arabic-specific evaluation**: Use Arabic NLU benchmarks

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

### Windows Compatibility
- Replace `unzip` with Python's `zipfile` module
- Replace `mkdir -p` with `os.makedirs(exist_ok=True)`
- Use raw strings or forward slashes for paths

### Memory Issues
- Reduce batch size for LLM training
- Use gradient checkpointing
- Enable 8-bit or 4-bit quantization

### Arabic Text
- Ensure UTF-8 encoding for all files
- Use proper Arabic tokenizers
- Handle right-to-left text display

---

## Suggested Improvements Summary

| Notebook | Priority | Improvement |
|----------|----------|-------------|
| Recommendation | High | Add proper evaluation metrics (Precision@K, NDCG) |
| Recommendation | Medium | Use specialized library (surprise, implicit) |
| Recommendation | Medium | Implement hybrid recommendation |
| Image Similarity | High | Rename to match actual functionality |
| Image Similarity | High | Add proper training for classification |
| Image Similarity | Medium | Use FAISS for scalable search |
| LLM Fine-tuning | Critical | Fix SyntaxError in trainer setup |
| LLM Fine-tuning | Critical | Add train/test split |
| LLM Fine-tuning | High | Use full dataset (52K samples) |
| LLM Fine-tuning | High | Add evaluation metrics |

---

## File Structure
```
Aliaa Graduation Project/
├── Recommendation_System_LV.ipynb    # Recommendation systems
├── Object Detection.ipynb             # Image similarity search
├── llama_3_fine_tuning_on_arabic_chatbott.ipynb  # LLM fine-tuning
├── ratings_Beauty.csv                 # Amazon ratings data
├── resnet50_feature_extractor.keras   # Saved ResNet50 model
├── requirements.txt                   # Python dependencies
├── content/                           # Data directory
│   ├── product_descriptions.csv
│   ├── CompanyReviews.csv
│   ├── hymenoptera_data/
│   └── leedsbutterfly/
└── CLAUDE.md                          # This file
```
