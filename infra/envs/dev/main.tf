# Dev: a small, disposable copy of the platform for trying pipeline and infrastructure changes on
# ESC-50 and other small sets. Buckets can be destroyed with their data and nothing is replicated.

module "vocalocate" {
  source = "../../modules/vocalocate_stack"

  environment                = "dev"
  region                     = var.region
  vpc_cidr                   = "10.40.0.0/16"
  force_destroy              = true
  noncurrent_expiration_days = 7

  task_cpu           = 1024
  task_memory        = 4096
  task_storage_gib   = 30
  log_retention_days = 7
  image_tag          = var.image_tag
  schedules          = {}

  alert_emails       = var.alert_emails
  state_bucket       = var.state_bucket
  modal_workspace_id = var.modal_workspace_id
  modal_environment  = "dev"
}
