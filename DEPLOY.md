# Deployment Guide: Cloud Run + Heroku Postgres

## Architecture

```
[.NET Backend] ──┐
                  ├──▶ [Heroku PostgreSQL] (shared database)
[ML API on Cloud Run] ─┘
```

- **Heroku Postgres**: Shared database for both .NET backend and ML API
- **Google Cloud Run**: Hosts the ML API (FastAPI), models baked into Docker image
- **Chatbot**: Returns 503 on Cloud Run (no GPU) — works only on Colab deployment

---

## Part 1: Heroku PostgreSQL Setup

### 1.1 Create the Heroku App (Database Only)

```bash
# Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli
heroku login

# Create a new app (just for the database)
heroku create your-app-name-db

# Add PostgreSQL addon (free Mini plan or Basic for production)
heroku addons:create heroku-postgresql:essential-0 --app your-app-name-db
```

> **Plans**: `essential-0` ($5/mo, 10K rows, 1GB), `essential-1` ($9/mo, 10M rows, 4GB), `essential-2` ($15/mo, 10M rows, 8GB)

### 1.2 Get the Database URL

```bash
heroku config:get DATABASE_URL --app your-app-name-db
```

This returns something like:
```
postgres://username:password@host:5432/dbname
```

**Important**: Save this URL — you'll need it for both the .NET backend and Cloud Run.

### 1.3 Connect Locally to Verify

```bash
# Test connection (requires psql installed)
heroku pg:psql --app your-app-name-db
```

### 1.4 Database Tables

The .NET backend creates these tables via EF Core migrations:

| Table | Owner | Description |
|-------|-------|-------------|
| `ProductCategories` | .NET | Product categories |
| `Products` | .NET | Product catalog |
| `UserInteraction` | .NET | Ratings, reviews, favourites |
| `product_embeddings` | ML API | Cluster assignments (auto-created) |
| `model_metadata` | ML API | Model versioning (auto-created) |

The ML API auto-creates its own tables (`product_embeddings`, `model_metadata`) on startup. You just need the .NET backend to create its tables first.

### 1.5 Useful Heroku DB Commands

```bash
# Check database info
heroku pg:info --app your-app-name-db

# View table sizes
heroku pg:psql --app your-app-name-db -c "\dt+"

# Create a backup
heroku pg:backups:capture --app your-app-name-db

# Reset database (DESTRUCTIVE)
heroku pg:reset --app your-app-name-db --confirm your-app-name-db
```

---

## Part 2: Google Cloud Run Deployment

### 2.1 Prerequisites

```bash
# Install Google Cloud CLI: https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com
```

### 2.2 Deploy with the Script

```bash
bash deploy-cloudrun.sh YOUR_PROJECT_ID
```

Default region is `me-central1` (Middle East). To change:
```bash
bash deploy-cloudrun.sh YOUR_PROJECT_ID europe-west1 ml-api
```

### 2.3 Deploy Step-by-Step (Manual)

If you prefer manual deployment:

#### Step A: Swap .dockerignore

The default `.dockerignore` excludes model files. Swap it temporarily:

```bash
cd ml_api
cp .dockerignore .dockerignore.bak
cp .dockerignore.cloudrun .dockerignore
```

#### Step B: Build and Push Image

```bash
# Option 1: Cloud Build (recommended — builds on Google's servers)
gcloud builds submit . \
    --project=YOUR_PROJECT_ID \
    --tag=gcr.io/YOUR_PROJECT_ID/ml-api \
    --timeout=1800 \
    --dockerfile=Dockerfile.cloudrun

# Option 2: Build locally and push
docker build -f Dockerfile.cloudrun -t gcr.io/YOUR_PROJECT_ID/ml-api .
docker push gcr.io/YOUR_PROJECT_ID/ml-api
```

#### Step C: Deploy to Cloud Run

```bash
gcloud run deploy ml-api \
    --project=YOUR_PROJECT_ID \
    --region=me-central1 \
    --image=gcr.io/YOUR_PROJECT_ID/ml-api \
    --platform=managed \
    --memory=4Gi \
    --cpu=2 \
    --min-instances=0 \
    --max-instances=3 \
    --timeout=60 \
    --concurrency=10 \
    --port=8000 \
    --allow-unauthenticated \
    --startup-cpu-boost \
    --set-env-vars="BYPASS_AUTH=false,DATABASE_URL=postgres://USER:PASS@HOST:5432/DB"
```

#### Step D: Restore .dockerignore

```bash
mv .dockerignore.bak .dockerignore
```

### 2.4 Set Environment Variables

After deployment, configure secrets:

```bash
# Required: Heroku DATABASE_URL (replace postgres:// with postgresql://)
# Heroku gives: postgres://user:pass@host:5432/db
# SQLAlchemy needs: postgresql://user:pass@host:5432/db
gcloud run services update ml-api \
    --region=me-central1 \
    --set-env-vars="\
DATABASE_URL=postgresql://USER:PASS@HOST:5432/DBNAME,\
JWT_SECRET_KEY=your-jwt-secret-must-match-dotnet-backend,\
JWT_ALGORITHMS=HS256,\
JWT_ISSUER=your-dotnet-issuer,\
JWT_AUDIENCE=your-dotnet-audience,\
BYPASS_AUTH=false"
```

> **BYPASS_AUTH=true** skips JWT validation entirely — useful for testing without the .NET backend.

**Full environment variable reference:**

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | Heroku Postgres URL (use `postgresql://` prefix) |
| `JWT_SECRET_KEY` | Yes | Must match .NET backend's JWT signing key |
| `JWT_ALGORITHMS` | No | Default: `HS256` |
| `JWT_ISSUER` | No | Expected JWT issuer claim |
| `JWT_AUDIENCE` | No | Expected JWT audience claim |
| `JWT_ADMIN_ROLE` | No | Default: `Admin` |
| `BYPASS_AUTH` | No | Default: `false`. Set `true` to skip auth for testing |
| `ALLOWED_ORIGINS` | No | Default: `*`. Comma-separated CORS origins |

### 2.5 Verify Deployment

```bash
# Get service URL
gcloud run services describe ml-api \
    --region=me-central1 \
    --format="value(status.url)"

# Test health
curl https://YOUR-SERVICE-URL/health

# Test recommendations (with BYPASS_AUTH=true)
curl -X POST https://YOUR-SERVICE-URL/recommend/popular \
    -H "Content-Type: application/json" \
    -d '{"top_n": 5}'
```

Expected health response:
```json
{
  "status": "healthy",
  "models": {
    "recommendation": {
      "popularity": "loaded",
      "collaborative": "loaded",
      "content_based": "loaded"
    },
    "image": {
      "feature_extractor": "loaded",
      "classifier": "loaded",
      "knn_index": "loaded"
    },
    "chatbot": {
      "llama_base": "not_loaded",
      "lora_adapter": "not_loaded"
    }
  }
}
```

---

## Part 3: Connecting .NET Backend to Heroku Postgres

In your .NET backend's `appsettings.json` or environment variables:

```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Host=HOST;Port=5432;Database=DBNAME;Username=USER;Password=PASS;SSL Mode=Require;Trust Server Certificate=true"
  }
}
```

Or as an environment variable:
```
ConnectionStrings__DefaultConnection=Host=HOST;Port=5432;Database=DBNAME;Username=USER;Password=PASS;SSL Mode=Require;Trust Server Certificate=true
```

> Heroku Postgres requires SSL. Add `SSL Mode=Require;Trust Server Certificate=true` to the .NET connection string.

---

## Part 4: Cloud Run Configuration Details

| Setting | Value | Reason |
|---------|-------|--------|
| Memory | 4 GiB | TensorFlow + ResNet50 + feature vectors in RAM |
| CPU | 2 | TensorFlow inference uses multiple cores |
| Min instances | 0 | Scale to zero (saves cost when idle) |
| Max instances | 3 | Limits cost |
| Timeout | 60s | Image inference takes a few seconds |
| Port | 8000 | Matches uvicorn/gunicorn config |
| Concurrency | 10 | TensorFlow is not fully thread-safe |
| Startup probe | /health | Already implemented in the API |

**Estimated costs (idle most of the time):**
- Cloud Run: ~$0 with scale-to-zero (pay per request)
- Heroku Postgres Essential-0: $5/month

---

## Part 5: Troubleshooting

### Cold starts are slow (~15-30 seconds)
Cloud Run loads TensorFlow + ResNet50 on cold start. To reduce:
- Set `--min-instances=1` (keeps one instance warm, costs ~$50/month)
- Or accept the cold start for a demo project

### "Database connection refused"
- Verify the `DATABASE_URL` uses `postgresql://` not `postgres://`
- Heroku Postgres requires SSL — Cloud Run handles this automatically via psycopg2

### "Model not loaded" errors
- Check logs: `gcloud run services logs read ml-api --region=me-central1`
- Ensure models are baked into the image (check `.dockerignore.cloudrun` was used during build)

### Image too large / build timeout
- The Docker image is ~2-3 GB (TensorFlow is large)
- Use `--timeout=1800` for Cloud Build
- If using `docker push`, ensure good upload speed

### Chatbot returns 503
This is expected — Cloud Run has no GPU. The chatbot endpoint gracefully returns:
```json
{"detail": "Chatbot model not available (requires GPU)"}
```

### View Cloud Run logs
```bash
gcloud run services logs read ml-api --region=me-central1 --limit=50
```

---

## Quick Reference

```bash
# === Heroku DB ===
heroku pg:psql --app your-app-name-db          # Connect to DB
heroku pg:info --app your-app-name-db           # DB stats
heroku config:get DATABASE_URL --app your-app   # Get URL

# === Cloud Run ===
bash deploy-cloudrun.sh YOUR_PROJECT_ID         # Full deploy
gcloud run services logs read ml-api --region=me-central1  # Logs
gcloud run services describe ml-api --region=me-central1   # Status
gcloud run services update ml-api --region=me-central1 \
    --set-env-vars="KEY=VALUE"                  # Update env vars
```
