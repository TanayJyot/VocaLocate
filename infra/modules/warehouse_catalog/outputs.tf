output "database_name" {
  value = aws_glue_catalog_database.lake.name
}

output "crawler_name" {
  value = aws_glue_crawler.lake.name
}

output "athena_workgroup" {
  value = aws_athena_workgroup.lake.name
}

output "results_bucket_arn" {
  value = module.results_bucket.arn
}
