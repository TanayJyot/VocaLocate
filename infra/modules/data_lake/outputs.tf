output "bucket_names" {
  description = "raw, lake and models bucket names."
  value       = { for k, b in module.bucket : k => b.id }
}

output "bucket_arns" {
  value = { for k, b in module.bucket : k => b.arn }
}
