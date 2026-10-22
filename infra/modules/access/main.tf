# Roles that outside systems assume with short-lived OIDC tokens, so no AWS keys are stored anywhere:
#
#   GitHub Actions  pushes the pipeline image to ECR and runs `terraform plan` (read-only).
#                   `terraform apply` stays a deliberate, manual step by a teammate.
#   Modal           GPU training: reads the warehouse and features from the lake, writes models.
#
# The OIDC providers themselves are account-wide and live in infra/bootstrap.

data "aws_caller_identity" "current" {}

locals {
  account_id      = data.aws_caller_identity.current.account_id
  github_provider = "arn:aws:iam::${local.account_id}:oidc-provider/token.actions.githubusercontent.com"
  modal_provider  = "arn:aws:iam::${local.account_id}:oidc-provider/oidc.modal.com"
}

# --- GitHub Actions -----------------------------------------------------------------------------

data "aws_iam_policy_document" "github_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [local.github_provider]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    # Only workflow jobs that declare `environment: <this environment>` in this repository.
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:environment:${var.environment}"]
    }
  }
}

resource "aws_iam_role" "github" {
  name                 = "${var.name_prefix}-github-actions"
  assume_role_policy   = data.aws_iam_policy_document.github_assume.json
  max_session_duration = 3600
  tags                 = var.tags
}

data "aws_iam_policy_document" "github" {
  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid = "PushPipelineImage"
    actions = [
      "ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage", "ecr:CompleteLayerUpload",
      "ecr:InitiateLayerUpload", "ecr:PutImage", "ecr:UploadLayerPart",
    ]
    resources = [var.ecr_repository_arn]
  }
  # `terraform plan` reads this environment's state and takes its lock file.
  statement {
    sid       = "StateList"
    actions   = ["s3:ListBucket"]
    resources = ["arn:aws:s3:::${var.state_bucket}"]
  }
  statement {
    sid       = "StateObjects"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["arn:aws:s3:::${var.state_bucket}/${var.environment}/*"]
  }
}

resource "aws_iam_role_policy" "github" {
  role   = aws_iam_role.github.id
  policy = data.aws_iam_policy_document.github.json
}

resource "aws_iam_role_policy_attachment" "github_read_only" {
  role       = aws_iam_role.github.name
  policy_arn = "arn:aws:iam::aws:policy/ReadOnlyAccess"
}

# --- Modal --------------------------------------------------------------------------------------

data "aws_iam_policy_document" "modal_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [local.modal_provider]
    }
    condition {
      test     = "StringEquals"
      variable = "oidc.modal.com:aud"
      values   = ["oidc.modal.com"]
    }
    # Only our workspace, the Modal environment for this AWS environment, and our training apps.
    condition {
      test     = "StringLike"
      variable = "oidc.modal.com:sub"
      values = [
        for app in var.modal_apps :
        "modal:workspace_id:${var.modal_workspace_id}:environment_name:${var.modal_environment}:app_name:${app}:*"
      ]
    }
  }
}

resource "aws_iam_role" "modal" {
  name                 = "${var.name_prefix}-modal-training"
  assume_role_policy   = data.aws_iam_policy_document.modal_assume.json
  max_session_duration = 3600
  tags                 = var.tags
}

data "aws_iam_policy_document" "modal" {
  statement {
    sid       = "List"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [var.lake_bucket_arn, var.models_bucket_arn]
  }
  # Training reads the warehouse and the log-mel features; it never sees the raw bucket and cannot
  # change the lake.
  statement {
    sid       = "ReadLake"
    actions   = ["s3:GetObject"]
    resources = ["${var.lake_bucket_arn}/*"]
  }
  statement {
    sid       = "ReadWriteModels"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"]
    resources = ["${var.models_bucket_arn}/*"]
  }
}

resource "aws_iam_role_policy" "modal" {
  role   = aws_iam_role.modal.id
  policy = data.aws_iam_policy_document.modal.json
}
