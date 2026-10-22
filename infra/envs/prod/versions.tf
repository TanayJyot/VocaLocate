terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }

  # The bucket comes from infra/bootstrap and is passed at init:
  #   terraform init -backend-config="bucket=vocalocate-tfstate-<account id>"
  backend "s3" {
    key          = "prod/terraform.tfstate"
    region       = "ca-central-1"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project     = "vocalocate"
      Environment = "prod"
      ManagedBy   = "terraform"
      Repository  = "TanayJyot/VocaLocate/infra/envs/prod"
    }
  }
}
