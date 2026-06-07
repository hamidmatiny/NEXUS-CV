output "cloud_run_url" {
  description = "Cloud Run service URL"
  value       = google_cloud_run_v2_service.serving.uri
}

output "models_bucket" {
  description = "GCS bucket for model artifacts"
  value       = google_storage_bucket.models.name
}

output "mlflow_bucket" {
  description = "GCS bucket for MLflow artifacts"
  value       = google_storage_bucket.mlflow.name
}

output "artifact_registry" {
  description = "Artifact Registry repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.nexus.repository_id}"
}

output "service_account_email" {
  description = "Cloud Run service account email (minimal IAM)"
  value       = google_service_account.serving.email
}
