# Everything one VocaLocate environment needs on AWS. infra/envs/dev and infra/envs/prod call this
# module with different sizes and safety settings, so the two environments cannot drift apart.

data "aws_caller_identity" "current" {}

locals {
  name_prefix = "vocalocate-${var.environment}"
  account_id  = data.aws_caller_identity.current.account_id
  tags        = { Project = "vocalocate", Environment = var.environment }
}

module "network" {
  source     = "../network"
  name       = local.name_prefix
  region     = var.region
  cidr_block = var.vpc_cidr
  tags       = local.tags
}

module "data_lake" {
  source                     = "../data_lake"
  name_prefix                = local.name_prefix
  account_id                 = local.account_id
  force_destroy              = var.force_destroy
  noncurrent_expiration_days = var.noncurrent_expiration_days
  backup_bucket_arns         = var.backup_bucket_arns
  tags                       = local.tags
}

module "pipeline_compute" {
  source                = "../pipeline_compute"
  name_prefix           = local.name_prefix
  environment           = var.environment
  region                = var.region
  raw_bucket_name       = module.data_lake.bucket_names.raw
  raw_bucket_arn        = module.data_lake.bucket_arns.raw
  lake_bucket_name      = module.data_lake.bucket_names.lake
  lake_bucket_arn       = module.data_lake.bucket_arns.lake
  image_tag             = var.image_tag
  cpu                   = var.task_cpu
  memory                = var.task_memory
  ephemeral_storage_gib = var.task_storage_gib
  log_retention_days    = var.log_retention_days
  force_destroy         = var.force_destroy
  tags                  = local.tags
}

module "warehouse_catalog" {
  source           = "../warehouse_catalog"
  name_prefix      = local.name_prefix
  environment      = var.environment
  account_id       = local.account_id
  lake_bucket_name = module.data_lake.bucket_names.lake
  lake_bucket_arn  = module.data_lake.bucket_arns.lake
  tags             = local.tags
}

module "monitoring" {
  source            = "../monitoring"
  name_prefix       = local.name_prefix
  environment       = var.environment
  state_machine_arn = module.orchestration.state_machine_arn
  alert_emails      = var.alert_emails
  tags              = local.tags
}

module "orchestration" {
  source              = "../orchestration"
  name_prefix         = local.name_prefix
  environment         = var.environment
  region              = var.region
  cluster_arn         = module.pipeline_compute.cluster_arn
  task_definition_arn = module.pipeline_compute.task_definition_arn
  container_name      = module.pipeline_compute.container_name
  pass_role_arns      = [module.pipeline_compute.task_role_arn, module.pipeline_compute.execution_role_arn]
  subnet_ids          = module.network.subnet_ids
  security_group_id   = module.network.security_group_id
  raw_bucket_name     = module.data_lake.bucket_names.raw
  alerts_topic_arn    = module.monitoring.topic_arn
  glue_crawler_name   = module.warehouse_catalog.crawler_name
  schedules           = var.schedules
  log_retention_days  = var.log_retention_days
  tags                = local.tags
}

module "access" {
  source             = "../access"
  name_prefix        = local.name_prefix
  environment        = var.environment
  github_repository  = var.github_repository
  state_bucket       = var.state_bucket
  ecr_repository_arn = module.pipeline_compute.ecr_repository_arn
  lake_bucket_arn    = module.data_lake.bucket_arns.lake
  models_bucket_arn  = module.data_lake.bucket_arns.models
  modal_workspace_id = var.modal_workspace_id
  modal_environment  = var.modal_environment
  tags               = local.tags
}
