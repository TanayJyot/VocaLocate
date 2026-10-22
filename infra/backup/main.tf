# Backup copies of the prod lake buckets, in a second region (Calgary) and in their own Terraform
# state. Prod replicates every new object here (modules/data_lake), but delete markers are not
# replicated, so destroying prod, or deleting objects in it, leaves these copies intact. This is
# what the disaster-recovery demo restores the data from.
#
#   cd infra/backup
#   terraform init -backend-config="bucket=vocalocate-tfstate-<account id>"
#   terraform apply            # before the first prod apply, which reads these outputs

terraform {
  required_version = ">= 1.10"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }
  backend "s3" {
    key          = "backup/terraform.tfstate"
    region       = "ca-central-1"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "vocalocate", Environment = "backup", ManagedBy = "terraform" }
  }
}

data "aws_caller_identity" "current" {}

module "bucket" {
  source   = "../modules/bucket"
  for_each = toset(["raw", "lake", "models"])

  name       = "vocalocate-backup-${each.key}-${data.aws_caller_identity.current.account_id}"
  versioning = true # required for replication, and lets a restore pick an older version
  # Keep overwritten versions for 90 days, then let them expire.
  noncurrent_expiration_days = 90
}

variable "region" {
  description = "A different region from prod (ca-central-1), so one regional outage cannot take both."
  type        = string
  default     = "ca-west-1"
}

output "bucket_arns" {
  value = { for k, b in module.bucket : k => b.arn }
}

output "bucket_names" {
  value = { for k, b in module.bucket : k => b.id }
}
