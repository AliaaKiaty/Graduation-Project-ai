# ML API — Cloud Run Deployment Guide

## Overview

The ML API is deployed as a containerised FastAPI service on **Google Cloud Run** (region: `me-central1`, Qatar).
The Docker image is built via **Google Cloud Build** and stored in **Artifact Registry**.
The database is **Amazon RDS PostgreSQL** (eu-west-1).
The Arabic chatbot is powered by the **Z.AI API** (GLM model, no GPU required).

---

## Infrastructure at a Glance

| Resource | Value |
|---|---|
| GCP Project ID | `project-caba28d9-9df4-468d-87f` |
| GCP Project Number | `727549809675` |
| Cloud Run Service | `ml-api` |
| Region | `me-central1` (Qatar) |
| Service URL | `https://ml-api-727549809675.me-central1.run.app` |
| Artifact Registry Image | `me-central1-docker.pkg.dev/project-caba28d9-9df4-468d-87f/ml-api/ml-api:latest` |
| Artifact Registry Repo | `ml-api` |
| Cloud Build SA | `727549809675-compute@developer.gserviceaccount.com` |
| Database | Amazon RDS PostgreSQL (`eu-west-1`) |
| Chatbot API | Z.AI (`https://api.z.ai/api/paas/v4`) |

---

## Environment Variables (Cloud Run)

Set via `gcloud run services update --update-env-vars` or the GCP Console.

| Variable | Required | Description | Example |
|---|---|---|---|
| `DATABASE_URL` | Yes | PostgreSQL connection string (supports `postgres://` prefix) | `postgres://user:pass@host:5432/db` |
| `BYPASS_AUTH` | Dev only | Skip JWT validation; dev user gets admin role | `true` |
| `JWT_SECRET_KEY` | Production | Secret key for validating JWTs from the .NET backend | `your-secret` |
| `JWT_ISSUER` | Optional | Expected issuer in JWT tokens | `https://your-dotnet-app.com` |
| `JWT_AUDIENCE` | Optional | Expected audience in JWT tokens | `ml-api` |
| `JWT_ROLE_CLAIM` | Optional | JWT claim containing roles (default: `role`) | `role` |
| `JWT_ADMIN_ROLE` | Optional | Role value that grants admin access (default: `Admin`) | `Admin` |
| `ZAI_API_KEY` | Yes | Z.AI API key for the Arabic chatbot | `abc123.XYZ` |
| `ZAI_MODEL` | Optional | Z.AI model name (default: `glm-4.5`) | `glm-4.7-flash` |
| `MODELS_DIR` | Optional | Path to saved model files inside the container | `/app/ml_api/saved_models` |
| `ALLOWED_ORIGINS` | Optional | CORS allowed origins, comma-separated (default: `*`) | `https://myapp.com` |
| `DB_POOL_SIZE` | Optional | SQLAlchemy pool size (default: `5`) | `5` |
| `DB_MAX_OVERFLOW` | Optional | SQLAlchemy max overflow (default: `10`) | `10` |

> **Note:** `ZAI_MODEL` can be changed at any time without a rebuild — just update the env var and Cloud Run deploys a new revision in ~30 seconds.

---

## Key Files

```
ml_api/
├── Dockerfile.cloudrun              # Cloud Run Dockerfile (no GPU, models from registry)
├── Dockerfile.cloudrun.dockerignore # Upload filter — excludes saved_models/ (copied from registry)
├── cloudbuild.yaml                  # Cloud Build pipeline
├── requirements-cloudrun.txt        # Slim dependencies (no PyTorch/LLM)
├── config.py                        # All env var configuration
├── database.py                      # DB engine + init_db() (creates all tables on startup)
├── main.py                          # FastAPI app + lifespan startup
├── models/chatbot.py                # Z.AI chatbot engine
└── api/admin.py                     # Admin endpoints including /admin/migrate
```

### How the Build Works

The `Dockerfile.cloudrun` uses a **two-stage build**:

1. **`model-cache` stage** — pulls the previous `:latest` image from Artifact Registry
2. **`python:3.10-slim` stage** — fresh Python base; code is copied from the local source upload; model files are copied from `model-cache`

This means **model files (380MB) never need to be re-uploaded** — only the Python code (~130KB) is uploaded each build.

```
cloudbuild.yaml steps:
  1. bash: cp Dockerfile.cloudrun.dockerignore .dockerignore
  2. docker pull :latest          ← fetches previous image (model-cache)
  3. docker build -f Dockerfile.cloudrun -t :latest .
```

---

## Standard Deployment Workflow

### 1. Code-only change (chatbot, API logic, config, etc.)

```bash
# From ml_api/ directory
gcloud builds submit . \
  --config=cloudbuild.yaml \
  --ignore-file=Dockerfile.cloudrun.dockerignore \
  --project=project-caba28d9-9df4-468d-87f

# Then deploy
gcloud run deploy ml-api \
  --image=me-central1-docker.pkg.dev/project-caba28d9-9df4-468d-87f/ml-api/ml-api:latest \
  --region=me-central1 \
  --project=project-caba28d9-9df4-468d-87f
```

Upload size: ~130KB. Build time: ~5 minutes.

### 2. Change only an environment variable (no rebuild needed)

```bash
gcloud run services update ml-api \
  --region=me-central1 \
  --project=project-caba28d9-9df4-468d-87f \
  --update-env-vars=VARIABLE_NAME=value
```

Examples:
```bash
# Switch chatbot model
--update-env-vars=ZAI_MODEL=glm-4.7-flash

# Enable/disable auth bypass
--update-env-vars=BYPASS_AUTH=false

# Update DB URL
--update-env-vars=DATABASE_URL=postgres://user:pass@host:5432/db

# Set multiple vars at once
--update-env-vars=ZAI_MODEL=glm-4.7-flash,BYPASS_AUTH=false
```

### 3. Update model files (recommendation/image models retrained)

If the saved model files in `saved_models/` change, the two-stage build won't pick them up (it copies from the previous registry image). You must do a **full upload build** once:

```bash
# Step 1: Temporarily allow saved_models in the upload
# Edit Dockerfile.cloudrun.dockerignore — comment out the saved_models/ line

# Step 2: Submit (this will upload ~380MB, takes ~15–20 min)
gcloud builds submit . \
  --config=cloudbuild.yaml \
  --ignore-file=Dockerfile.cloudrun.dockerignore \
  --project=project-caba28d9-9df4-468d-87f

# Step 3: Restore Dockerfile.cloudrun.dockerignore (re-add saved_models/)
# Step 4: Deploy as normal
```

After this one-time upload, subsequent code builds will pull models from the new registry image automatically.

---

## First-Time Setup (New Environment)

### Prerequisites

- `gcloud` CLI installed and authenticated
- Access to GCP project `project-caba28d9-9df4-468d-87f`
- PostgreSQL database URL

### Step 1 — Authenticate gcloud

```bash
gcloud auth login
gcloud config set project project-caba28d9-9df4-468d-87f
```

> **Windows note:** Run `gcloud` commands via PowerShell, not bash — gcloud can crash in Git Bash/WSL on Windows.

### Step 2 — Ensure Artifact Registry repository exists

```bash
gcloud artifacts repositories create ml-api \
  --repository-format=docker \
  --location=me-central1 \
  --project=project-caba28d9-9df4-468d-87f
```

### Step 3 — Grant Cloud Build SA permission to push images

```bash
gcloud artifacts repositories add-iam-policy-binding ml-api \
  --location=me-central1 \
  --member='serviceAccount:727549809675-compute@developer.gserviceaccount.com' \
  --role='roles/artifactregistry.writer' \
  --project=project-caba28d9-9df4-468d-87f
```

### Step 4 — First build (includes model files upload)

For the very first build, `saved_models/` must be temporarily included in the upload (since there is no previous registry image to copy from).

```bash
# Temporarily comment out saved_models/ in Dockerfile.cloudrun.dockerignore

gcloud builds submit . \
  --config=cloudbuild.yaml \
  --ignore-file=Dockerfile.cloudrun.dockerignore \
  --project=project-caba28d9-9df4-468d-87f

# Restore Dockerfile.cloudrun.dockerignore
```

### Step 5 — Deploy to Cloud Run

```bash
gcloud run deploy ml-api \
  --image=me-central1-docker.pkg.dev/project-caba28d9-9df4-468d-87f/ml-api/ml-api:latest \
  --region=me-central1 \
  --platform=managed \
  --allow-unauthenticated \
  --memory=4Gi \
  --cpu=2 \
  --timeout=300 \
  --max-instances=3 \
  --set-env-vars=DATABASE_URL=<your_db_url>,BYPASS_AUTH=true,ZAI_API_KEY=<your_key>,ZAI_MODEL=glm-4.7-flash \
  --project=project-caba28d9-9df4-468d-87f
```

### Step 6 — Create database tables

On first deploy, call the migration endpoint to create all tables:

```bash
curl -X POST https://ml-api-727549809675.me-central1.run.app/admin/migrate
```

Expected response:
```json
{
  "status": "ok",
  "error": null,
  "tables_exist": {
    "ProductCategories": true,
    "Products": true,
    "UserInteraction": true,
    "product_embeddings": true,
    "model_metadata": true
  }
}
```

---

## Database

- Provider: Amazon RDS PostgreSQL (eu-west-1)
- Schema is shared with the .NET backend — the ML API treats `Products`, `ProductCategories`, `UserInteraction` as read/write tables
- ML-owned tables: `product_embeddings`, `model_metadata`
- `init_db()` runs on every startup via `lifespan()` in `main.py` — uses `checkfirst=True` so existing tables are never altered
- The `/admin/migrate` endpoint (`POST /admin/migrate`) can be called manually if `init_db()` fails silently

---

## Chatbot Configuration

The chatbot uses the **Z.AI API** (no GPU, no local model files needed).

| Setting | Value |
|---|---|
| API Base | `https://api.z.ai/api/paas/v4` |
| Endpoint | `/chat/completions` |
| Default model | `glm-4.5` (overridden by `ZAI_MODEL` env var) |
| Current model | `glm-4.7-flash` |
| Thinking mode | Disabled (`"thinking": {"type": "disabled"}`) |
| Max tokens | Up to 4096 |

**System prompt** (Arabic e-commerce assistant):
> أنت مساعد ذكي ومفيد لمنصة تسوق إلكترونية. تساعد المستخدمين في البحث عن المنتجات، تقديم التوصيات، الإجابة عن استفسارات الشحن والطلبات، ومقارنة المنتجات. تجيب دائماً باللغة العربية.

**Switch model without rebuild:**
```bash
gcloud run services update ml-api \
  --region=me-central1 \
  --update-env-vars=ZAI_MODEL=glm-4.7-flash
```

---

## Production Checklist

Before going live (disabling `BYPASS_AUTH`):

- [ ] Set `BYPASS_AUTH=false`
- [ ] Set a strong `JWT_SECRET_KEY` matching the .NET backend
- [ ] Set `JWT_ISSUER` to match the .NET backend's token issuer
- [ ] Confirm `ALLOWED_ORIGINS` is restricted to your frontend domain
- [ ] Verify `/admin/migrate` was called and all tables exist
- [ ] Test `/health` returns `"status": "healthy"`
- [ ] Test `/recommend/popular` returns `200`
- [ ] Test `/chat/message` returns an Arabic response

---

## Useful Commands Reference

```bash
# View current service config & env vars
gcloud run services describe ml-api --region=me-central1 --project=project-caba28d9-9df4-468d-87f

# List recent builds
gcloud builds list --project=project-caba28d9-9df4-468d-87f --limit=5

# View Cloud Run revisions
gcloud run revisions list --service=ml-api --region=me-central1 --project=project-caba28d9-9df4-468d-87f

# Roll back to a previous revision
gcloud run services update-traffic ml-api \
  --to-revisions=ml-api-00010-z4p=100 \
  --region=me-central1 \
  --project=project-caba28d9-9df4-468d-87f

# View logs (requires network access to logging.googleapis.com)
gcloud run services logs read ml-api --region=me-central1 --project=project-caba28d9-9df4-468d-87f

# Update multiple env vars at once
gcloud run services update ml-api \
  --region=me-central1 \
  --project=project-caba28d9-9df4-468d-87f \
  --update-env-vars=KEY1=val1,KEY2=val2
```

---

## API Endpoints Quick Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/health` | None | Service health + model status |
| POST | `/recommend/popular` | Optional | Popular products from DB |
| POST | `/recommend/collaborative` | Optional | SVD-based recommendations |
| POST | `/recommend/content-based` | Optional | TF-IDF + KMeans search |
| GET | `/recommend/products` | Optional | List/search products |
| POST | `/image/similar` | Required | Find similar images |
| POST | `/image/classify` | Required | Classify butterfly image |
| POST | `/chat/message` | Required | Arabic chatbot (Z.AI) |
| GET | `/chat/status` | Required | Chatbot readiness |
| POST | `/admin/migrate` | Admin | Create DB tables |
| POST | `/admin/retrain` | Admin | Retrain ML models |
| GET | `/admin/models` | Admin | List model metadata |
| GET | `/docs` | None | Swagger UI |

> With `BYPASS_AUTH=true`, all endpoints are accessible without a token.
> With `BYPASS_AUTH=false`, "Required" endpoints need `Authorization: Bearer <JWT>` from the .NET backend.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `relation "Products" does not exist` | DB tables not created | Call `POST /admin/migrate` |
| `/chat/message` returns 503 | `ZAI_API_KEY` not set | Set env var on Cloud Run |
| Z.AI error 1113 / 429 | Insufficient account balance | Top up Z.AI account credits |
| Z.AI error 1211 / 400 | Invalid model name | Update `ZAI_MODEL` to a valid model |
| Chat response is empty | GLM thinking model cutting off | Increase `max_tokens`; thinking mode is disabled by default |
| Build upload stuck / very slow | `saved_models/` included in upload | Ensure `saved_models/` is in `Dockerfile.cloudrun.dockerignore` |
| `gcloud` crashes in bash on Windows | Known gcloud/WSL issue | Use PowerShell instead of bash |
| `denied: Permission ... artifactregistry` | Cloud Build SA lacks push rights | Re-run Step 3 of First-Time Setup |
