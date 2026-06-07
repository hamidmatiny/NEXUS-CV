terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

resource "aws_ecs_cluster" "nexus" {
  name = "nexus-cv-cluster"
}

resource "aws_ecr_repository" "nexus" {
  name                 = "nexus-cv"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}

resource "aws_s3_bucket" "models" {
  bucket = "nexus-cv-models-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket" "mlflow" {
  bucket = "nexus-cv-mlflow-${data.aws_caller_identity.current.account_id}"
}

data "aws_caller_identity" "current" {}

resource "aws_lb" "nexus" {
  name               = "nexus-cv-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.subnet_ids
}

resource "aws_security_group" "alb" {
  name        = "nexus-cv-alb-sg"
  description = "ALB security group for NEXUS-CV"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_lb_target_group" "serving" {
  name        = "nexus-cv-serving-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    matcher             = "200"
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.nexus.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.serving.arn
  }
}

resource "aws_ecs_task_definition" "serving" {
  family                   = "nexus-cv-serving"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "2048"
  memory                   = "4096"
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name  = "serving"
      image = var.ecr_image_uri
      portMappings = [{ containerPort = 8000, protocol = "tcp" }]
      essential = true
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/nexus-cv-serving"
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "serving"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

resource "aws_cloudwatch_log_group" "serving" {
  name              = "/ecs/nexus-cv-serving"
  retention_in_days = 14
}

resource "aws_iam_role" "ecs_execution" {
  name = "nexus-cv-ecs-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task" {
  name = "nexus-cv-ecs-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "s3_read" {
  name = "nexus-cv-s3-read"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Resource = [
        aws_s3_bucket.models.arn,
        "${aws_s3_bucket.models.arn}/*",
        aws_s3_bucket.mlflow.arn,
        "${aws_s3_bucket.mlflow.arn}/*",
      ]
    }]
  })
}

resource "aws_ecs_service" "serving" {
  name            = var.service_name
  cluster         = aws_ecs_cluster.nexus.id
  task_definition = aws_ecs_task_definition.serving.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    assign_public_ip = true
    security_groups  = [aws_security_group.alb.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.serving.arn
    container_name   = "serving"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.http]
}
