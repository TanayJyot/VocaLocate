# Makes the lake's Parquet tables queryable in AWS: a Glue crawler registers one table per lake
# table folder, and an Athena workgroup runs SQL over them. DuckDB stays the warehouse the team
# and the training code use; this is the managed, serverless view of the same files.

locals {
  # Mirrors TABLE_LAYER in pipeline/src/vocalocate_pipeline/lake.py.
  table_paths = [
    "bronze/tables/raw_files",
    "silver/tables/clips",
    "silver/tables/pairs",
    "silver/tables/queries",
    "silver/tables/sfx_metadata",
    "silver/tables/captions",
    "gold/tables/features",
    "gold/tables/splits",
    "gold/tables/embeddings",
    "gold/tables/hard_negatives",
    "state/library_files",
    "runs",
  ]
  database = replace("${var.name_prefix}_lake", "-", "_")
}

resource "aws_glue_catalog_database" "lake" {
  name        = local.database
  description = "VocaLocate ${var.environment} lake tables (Parquet in s3://${var.lake_bucket_name})"
}

data "aws_iam_policy_document" "glue_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["glue.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "crawler" {
  name               = "${var.name_prefix}-crawler"
  assume_role_policy = data.aws_iam_policy_document.glue_assume.json
  tags               = var.tags
}

resource "aws_iam_role_policy_attachment" "crawler_service" {
  role       = aws_iam_role.crawler.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

resource "aws_iam_role_policy" "crawler_read_lake" {
  role = aws_iam_role.crawler.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["s3:ListBucket"], Resource = var.lake_bucket_arn },
      { Effect = "Allow", Action = ["s3:GetObject"], Resource = "${var.lake_bucket_arn}/*" },
    ]
  })
}

resource "aws_glue_crawler" "lake" {
  name          = "${var.name_prefix}-lake"
  database_name = aws_glue_catalog_database.lake.name
  role          = aws_iam_role.crawler.arn
  schedule      = var.crawler_schedule

  dynamic "s3_target" {
    for_each = local.table_paths
    content {
      path = "s3://${var.lake_bucket_name}/${s3_target.value}/"
    }
  }

  schema_change_policy {
    update_behavior = "UPDATE_IN_DATABASE"
    delete_behavior = "LOG"
  }

  # One table per folder, even though each folder holds one Parquet file per source.
  configuration = jsonencode({
    Version  = 1.0
    Grouping = { TableGroupingPolicy = "CombineCompatibleSchemas" }
  })
  tags = var.tags
}

module "results_bucket" {
  source          = "../bucket"
  name            = "${var.name_prefix}-athena-results-${var.account_id}"
  force_destroy   = true # query results only; nothing to keep
  versioning      = false
  expiration_days = 14
  tags            = var.tags
}

resource "aws_athena_workgroup" "lake" {
  name          = var.name_prefix
  force_destroy = true

  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = true
    bytes_scanned_cutoff_per_query     = var.bytes_scanned_cutoff

    result_configuration {
      output_location = "s3://${module.results_bucket.id}/results/"
      encryption_configuration {
        encryption_option = "SSE_S3"
      }
    }
  }
  tags = var.tags
}
