output "bucket_names" {
  value = module.vocalocate.bucket_names
}

output "ecr_repository_url" {
  value = module.vocalocate.ecr_repository_url
}

output "state_machine_arn" {
  value = module.vocalocate.state_machine_arn
}

output "athena_workgroup" {
  value = module.vocalocate.athena_workgroup
}

output "glue_database" {
  value = module.vocalocate.glue_database
}

output "github_role_arn" {
  description = "Set as the AWS_ROLE_ARN variable of this GitHub environment."
  value       = module.vocalocate.github_role_arn
}

output "modal_secret_command" {
  value = module.vocalocate.modal_secret_command
}
