output "bucket_names" {
  value = module.data_lake.bucket_names
}

output "ecr_repository_url" {
  value = module.pipeline_compute.ecr_repository_url
}

output "state_machine_arn" {
  value = module.orchestration.state_machine_arn
}

output "athena_workgroup" {
  value = module.warehouse_catalog.athena_workgroup
}

output "glue_database" {
  value = module.warehouse_catalog.database_name
}

output "github_role_arn" {
  value = module.access.github_role_arn
}

output "modal_role_arn" {
  value = module.access.modal_role_arn
}

output "modal_secret_command" {
  description = "Creates the Modal secret the training launcher reads (demo/modal/recipe_app.py)."
  value = join(" ", [
    "modal secret create vocalocate-aws-${var.environment} --force",
    "VOCALOCATE_AWS_ROLE_ARN=${module.access.modal_role_arn}",
    "VOCALOCATE_LAKE_URI=s3://${module.data_lake.bucket_names.lake}",
    "VOCALOCATE_MODELS_URI=s3://${module.data_lake.bucket_names.models}",
    "AWS_REGION=${var.region}",
  ])
}
