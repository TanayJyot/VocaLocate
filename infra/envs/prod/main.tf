# Prod: the data the team trains and reports on. Buckets refuse to be destroyed while they hold
# data, every bucket replicates to the backup stack in another region (infra/backup), and the
# warehouse is rebuilt and checked every Monday.

data "terraform_remote_state" "backup" {
  backend = "s3"
  config = {
    bucket = var.state_bucket
    key    = "backup/terraform.tfstate"
    region = "ca-central-1"
  }
}

module "vocalocate" {
  source = "../../modules/vocalocate_stack"

  environment                = "prod"
  region                     = var.region
  vpc_cidr                   = "10.41.0.0/16"
  force_destroy              = var.force_destroy
  noncurrent_expiration_days = 30
  backup_bucket_arns         = data.terraform_remote_state.backup.outputs.bucket_arns

  task_cpu           = 4096
  task_memory        = 16384
  task_storage_gib   = 100
  log_retention_days = 90
  image_tag          = var.image_tag
  schedules = {
    weekly-warehouse = { expression = "cron(0 6 ? * MON *)", input = {} }
  }

  alert_emails       = var.alert_emails
  state_bucket       = var.state_bucket
  modal_workspace_id = var.modal_workspace_id
  modal_environment  = "main"
}
