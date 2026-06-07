variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "ecr_image_uri" {
  description = "Full ECR image URI for nexus-cv serving"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID for ECS service"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for ECS tasks and ALB"
  type        = list(string)
}

variable "service_name" {
  description = "ECS service name"
  type        = string
  default     = "nexus-cv-serving"
}
