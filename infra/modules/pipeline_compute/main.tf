# Runs the pipeline container (pipeline/Dockerfile) as ECS Fargate tasks. Each task pulls the part of
# the S3 lake it needs into /lake, runs one flow and pushes the results (pipeline/…/remote.py).
# Step Functions starts the tasks; nothing here runs on its own.

locals {
  container_name = "pipeline"
}

resource "aws_ecr_repository" "pipeline" {
  name                 = "${var.name_prefix}-pipeline"
  image_tag_mutability = "MUTABLE" # CI moves the `latest` tag; every image is also tagged with its commit
  force_delete         = var.force_destroy

  image_scanning_configuration {
    scan_on_push = true
  }
  tags = var.tags
}

resource "aws_ecr_lifecycle_policy" "pipeline" {
  repository = aws_ecr_repository.pipeline.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the last ${var.images_to_keep} images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = var.images_to_keep
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_ecs_cluster" "this" {
  name = var.name_prefix
  setting {
    name  = "containerInsights"
    value = var.container_insights ? "enabled" : "disabled"
  }
  tags = var.tags
}

resource "aws_cloudwatch_log_group" "pipeline" {
  name              = "/vocalocate/${var.environment}/pipeline"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}

# --- Roles ------------------------------------------------------------------------------------

data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Used by ECS itself to pull the image and write logs.
resource "aws_iam_role" "execution" {
  name               = "${var.name_prefix}-pipeline-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
  tags               = var.tags
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Used by the pipeline code: read raw data, read and write the lake. Nothing else.
resource "aws_iam_role" "task" {
  name               = "${var.name_prefix}-pipeline-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
  tags               = var.tags
}

data "aws_iam_policy_document" "task" {
  statement {
    sid       = "ListBuckets"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [var.raw_bucket_arn, var.lake_bucket_arn]
  }
  statement {
    sid       = "ReadRaw"
    actions   = ["s3:GetObject"]
    resources = ["${var.raw_bucket_arn}/*"]
  }
  statement {
    sid       = "ReadWriteLake"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"]
    resources = ["${var.lake_bucket_arn}/*"]
  }
}

resource "aws_iam_role_policy" "task" {
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task.json
}

# --- Task definition --------------------------------------------------------------------------

resource "aws_ecs_task_definition" "pipeline" {
  family                   = "${var.name_prefix}-pipeline"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  # The local working copy of the lake (clean audio and features of one source) lives here.
  ephemeral_storage {
    size_in_gib = var.ephemeral_storage_gib
  }

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([{
    name      = local.container_name
    image     = "${aws_ecr_repository.pipeline.repository_url}:${var.image_tag}"
    essential = true
    command   = ["remote", "report"] # Step Functions overrides this per run
    environment = [
      { name = "VOCALOCATE_CONFIG", value = "/app/configs/${var.environment}.toml" },
      { name = "VOCALOCATE_LAKE_URI", value = "s3://${var.lake_bucket_name}" },
      { name = "VOCALOCATE_RAW_URI", value = "s3://${var.raw_bucket_name}" },
      { name = "AWS_REGION", value = var.region },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.pipeline.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "pipeline"
      }
    }
  }])

  tags = var.tags
}
