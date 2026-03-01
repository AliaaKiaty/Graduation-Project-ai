# Replace with your actual project ID
PROJECT_ID="project-caba28d9-9df4-468d-87f"
SERVICE_ACCOUNT="727549809675-compute@developer.gserviceaccount.com"

# Grant Storage Object Viewer role to the Cloud Build service account
gcloud projects add-iam-policy-binding $PROJECT_ID `
    --member="serviceAccount:$SERVICE_ACCOUNT" `
    --role="roles/storage.objectViewer"