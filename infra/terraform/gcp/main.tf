# NEXUS-CV GCP infrastructure — Cloud Run serving with least-privilege IAM.
#
# Service account permissions (documented):
#   - roles/storage.objectViewer  on models + mlflow buckets (read artifacts)
#   - roles/artifactregistry.reader (pull container images)
#   - roles/logging.logWriter (Cloud Logging)
#   - roles/cloudtrace.agent (optional tracing)

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_service_account" "serving" {
  account_id   = "nexus-cv-serving"
  display_name = "NEXUS-CV Cloud Run Serving"
}

resource "google_storage_bucket" "models" {
  name                        = "${var.project_id}-nexus-cv-models"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false
}

resource "google_storage_bucket" "mlflow" {
  name                        = "${var.project_id}-nexus-cv-mlflow"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false
}

resource "google_artifact_registry_repository" "nexus" {
  location      = var.region
  repository_id = "nexus-cv"
  format        = "DOCKER"
  description   = "NEXUS-CV container images"
}

resource "google_storage_bucket_iam_member" "models_reader" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.serving.email}"
}

resource "google_storage_bucket_iam_member" "mlflow_reader" {
  bucket = google_storage_bucket.mlflow.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.serving.email}"
}

resource "google_project_iam_member" "artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.serving.email}"
}

resource "google_project_iam_member" "log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.serving.email}"
}

resource "google_cloud_run_v2_service" "serving" {
  name     = var.service_name
  location = var.region

  template {
    service_account = google_service_account.serving.email

    scaling {
      min_instance_count = 1
      max_instance_count = 10
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.nexus.repository_id}/nexus-cv:${var.image_tag}"

      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
      }

      ports {
        container_port = 8000
      }

      env {
        name  = "MLFLOW_TRACKING_URI"
        value = "https://mlflow.example.com"
      }
    }

    max_instance_request_concurrency = 80
  }

  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
}

resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  name     = google_cloud_run_v2_service.serving.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}
