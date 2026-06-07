output "ecs_cluster_name" {
  value = aws_ecs_cluster.nexus.name
}

output "ecs_service_name" {
  value = aws_ecs_service.serving.name
}

output "alb_dns_name" {
  value = aws_lb.nexus.dns_name
}

output "models_bucket" {
  value = aws_s3_bucket.models.bucket
}

output "mlflow_bucket" {
  value = aws_s3_bucket.mlflow.bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.nexus.repository_url
}
