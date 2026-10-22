output "cluster_arn" {
  value = aws_ecs_cluster.this.arn
}

output "task_definition_arn" {
  value = aws_ecs_task_definition.pipeline.arn
}

output "task_definition_family" {
  value = aws_ecs_task_definition.pipeline.family
}

output "container_name" {
  value = local.container_name
}

output "task_role_arn" {
  value = aws_iam_role.task.arn
}

output "execution_role_arn" {
  value = aws_iam_role.execution.arn
}

output "ecr_repository_url" {
  value = aws_ecr_repository.pipeline.repository_url
}

output "ecr_repository_arn" {
  value = aws_ecr_repository.pipeline.arn
}

output "log_group_name" {
  value = aws_cloudwatch_log_group.pipeline.name
}
