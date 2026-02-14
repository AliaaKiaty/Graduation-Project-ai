#!/usr/bin/env bash
# Deploy ML API to Google Cloud Run
#
# Usage:
#   bash deploy-cloudrun.sh <GCP_PROJECT_ID> [REGION] [SERVICE_NAME]
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - Docker or gcloud builds submit access
#   - Required env vars set in Cloud Run (or pass via --set-env-vars)

set -euo pipefail

# --- Arguments ---
PROJECT_ID="${1:?Usage: bash deploy-cloudrun.sh <GCP_PROJECT_ID> [REGION] [SERVICE_NAME]}"
REGION="${2:-me-central1}"
SERVICE_NAME="${3:-ml-api}"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_API_DIR="${SCRIPT_DIR}/ml_api"
ORIGINAL_DOCKERIGNORE="${ML_API_DIR}/.dockerignore"
CLOUDRUN_DOCKERIGNORE="${ML_API_DIR}/.dockerignore.cloudrun"
BACKUP_DOCKERIGNORE="${ML_API_DIR}/.dockerignore.bak"

# --- Cleanup trap: always restore original .dockerignore ---
cleanup() {
    if [ -f "${BACKUP_DOCKERIGNORE}" ]; then
        echo "Restoring original .dockerignore..."
        mv "${BACKUP_DOCKERIGNORE}" "${ORIGINAL_DOCKERIGNORE}"
    fi
}
trap cleanup EXIT

echo "=========================================="
echo "  Deploying ML API to Cloud Run"
echo "=========================================="
echo "  Project:  ${PROJECT_ID}"
echo "  Region:   ${REGION}"
echo "  Service:  ${SERVICE_NAME}"
echo "  Image:    ${IMAGE}"
echo "=========================================="

# --- Step 1: Swap .dockerignore ---
echo ""
echo "[1/4] Swapping .dockerignore for Cloud Run build..."
if [ -f "${ORIGINAL_DOCKERIGNORE}" ]; then
    cp "${ORIGINAL_DOCKERIGNORE}" "${BACKUP_DOCKERIGNORE}"
fi
cp "${CLOUDRUN_DOCKERIGNORE}" "${ORIGINAL_DOCKERIGNORE}"

# --- Step 2: Build with Cloud Build ---
echo ""
echo "[2/4] Building Docker image with Cloud Build..."
gcloud builds submit "${ML_API_DIR}" \
    --project="${PROJECT_ID}" \
    --tag="${IMAGE}" \
    --timeout=1800 \
    --machine-type=e2-highcpu-8 \
    --dockerfile="Dockerfile.cloudrun"

# --- Step 3: Deploy to Cloud Run ---
echo ""
echo "[3/4] Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --image="${IMAGE}" \
    --platform=managed \
    --memory=4Gi \
    --cpu=2 \
    --min-instances=0 \
    --max-instances=3 \
    --timeout=60 \
    --concurrency=10 \
    --port=8000 \
    --set-env-vars="BYPASS_AUTH=false,DATABASE_URL=sqlite:///./ml_api.db" \
    --allow-unauthenticated \
    --startup-cpu-boost

# --- Step 4: Print service URL ---
echo ""
echo "[4/4] Deployment complete!"
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --format="value(status.url)")

echo ""
echo "=========================================="
echo "  Service URL: ${SERVICE_URL}"
echo "=========================================="
echo ""
echo "Verify with:"
echo "  curl ${SERVICE_URL}/health"
echo ""
echo "IMPORTANT: Set secrets via Cloud Run console or CLI:"
echo "  gcloud run services update ${SERVICE_NAME} \\"
echo "    --region=${REGION} \\"
echo "    --set-env-vars=JWT_SECRET_KEY=<your-secret> \\"
echo "    --set-env-vars=DATABASE_URL=<your-db-url>"
