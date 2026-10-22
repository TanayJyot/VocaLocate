# The VocaLocate data lake for one environment, with the same layout as pipeline/…/lake.py:
#
#   raw     <source>/...                       datasets exactly as downloaded, plus team recordings
#   lake    bronze/tables, silver/, gold/, state/, runs/, warehouse/vocalocate.duckdb
#   models  <source>/recipe-<seed>/...         checkpoints and results from training on Modal
#
# Raw data is the only part that cannot be rebuilt (team recordings), so it is versioned and, in
# prod, replicated to the backup stack along with the lake and the models.

locals {
  buckets = {
    raw = {
      transitions = [{ days = 30, storage_class = "STANDARD_IA" }]
      eventbridge = true
    }
    lake = {
      transitions = []
      eventbridge = false
    }
    models = {
      transitions = [{ days = 60, storage_class = "STANDARD_IA" }]
      eventbridge = false
    }
  }
}

module "bucket" {
  source   = "../bucket"
  for_each = local.buckets

  name                       = "${var.name_prefix}-${each.key}-${var.account_id}"
  force_destroy              = var.force_destroy
  versioning                 = true
  transitions                = each.value.transitions
  noncurrent_expiration_days = var.noncurrent_expiration_days
  eventbridge                = each.value.eventbridge
  tags                       = var.tags
}

# --- Replication to the backup stack (prod) ---------------------------------------------------

data "aws_iam_policy_document" "replication_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["s3.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "replication" {
  count              = length(var.backup_bucket_arns) > 0 ? 1 : 0
  name               = "${var.name_prefix}-replication"
  assume_role_policy = data.aws_iam_policy_document.replication_assume.json
  tags               = var.tags
}

data "aws_iam_policy_document" "replication" {
  count = length(var.backup_bucket_arns) > 0 ? 1 : 0

  statement {
    actions   = ["s3:GetReplicationConfiguration", "s3:ListBucket"]
    resources = [for k in keys(var.backup_bucket_arns) : module.bucket[k].arn]
  }
  statement {
    actions   = ["s3:GetObjectVersionForReplication", "s3:GetObjectVersionAcl", "s3:GetObjectVersionTagging"]
    resources = [for k in keys(var.backup_bucket_arns) : "${module.bucket[k].arn}/*"]
  }
  statement {
    actions   = ["s3:ReplicateObject", "s3:ReplicateDelete", "s3:ReplicateTags"]
    resources = [for arn in values(var.backup_bucket_arns) : "${arn}/*"]
  }
}

resource "aws_iam_role_policy" "replication" {
  count  = length(var.backup_bucket_arns) > 0 ? 1 : 0
  role   = aws_iam_role.replication[0].id
  policy = data.aws_iam_policy_document.replication[0].json
}

resource "aws_s3_bucket_replication_configuration" "backup" {
  for_each = var.backup_bucket_arns
  bucket   = module.bucket[each.key].id
  role     = aws_iam_role.replication[0].arn

  rule {
    id     = "backup"
    status = "Enabled"
    filter {}

    # Deleting the live bucket must not delete the backup: that is the point of the DR demo.
    delete_marker_replication {
      status = "Disabled"
    }

    destination {
      bucket        = each.value
      storage_class = "STANDARD_IA"
    }
  }
}
