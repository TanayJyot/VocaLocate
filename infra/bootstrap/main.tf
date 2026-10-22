# One-time, account-wide setup, applied once with local state before anything else:
#
#   * the S3 bucket that holds every other stack's Terraform state (versioned, so a bad apply can
#     be rolled back; locking uses S3 lock files, so no DynamoDB table is needed);
#   * the OIDC identity providers for GitHub Actions and Modal, which each environment's roles
#     trust (an account can only have one provider per URL, so they cannot live in envs/);
#   * a monthly cost budget with email alerts.
#
#   cd infra/bootstrap && terraform init && terraform apply -var 'budget_emails=["you@example.com"]'

terraform {
  required_version = ">= 1.10"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "vocalocate", Environment = "shared", ManagedBy = "terraform" }
  }
}

data "aws_caller_identity" "current" {}

module "state_bucket" {
  source                     = "../modules/bucket"
  name                       = "vocalocate-tfstate-${data.aws_caller_identity.current.account_id}"
  versioning                 = true
  noncurrent_expiration_days = 90
}

data "tls_certificate" "github" {
  url = "https://token.actions.githubusercontent.com/.well-known/openid-configuration"
}

resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.github.certificates[length(data.tls_certificate.github.certificates) - 1].sha1_fingerprint]
}

data "tls_certificate" "modal" {
  url = "https://oidc.modal.com/.well-known/openid-configuration"
}

resource "aws_iam_openid_connect_provider" "modal" {
  url             = "https://oidc.modal.com"
  client_id_list  = ["oidc.modal.com"]
  thumbprint_list = [data.tls_certificate.modal.certificates[length(data.tls_certificate.modal.certificates) - 1].sha1_fingerprint]
}

resource "aws_budgets_budget" "monthly" {
  name         = "vocalocate-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  dynamic "notification" {
    for_each = { ACTUAL = 80, FORECASTED = 100 }
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value
      threshold_type             = "PERCENTAGE"
      notification_type          = notification.key
      subscriber_email_addresses = var.budget_emails
    }
  }
}

variable "region" {
  type    = string
  default = "ca-central-1"
}

variable "monthly_budget_usd" {
  type    = number
  default = 50
}

variable "budget_emails" {
  type = list(string)
}

output "state_bucket" {
  value = module.state_bucket.id
}
