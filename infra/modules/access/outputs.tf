output "github_role_arn" {
  value = aws_iam_role.github.arn
}

output "modal_role_arn" {
  value = aws_iam_role.modal.arn
}
