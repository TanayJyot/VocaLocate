# Offline checks of the environment stack and its modules: `terraform init && terraform test` in
# infra/modules/vocalocate_stack. The AWS provider is mocked, so `apply` here needs no credentials
# and creates nothing; it only makes computed values (names, rendered JSON) available to check.

mock_provider "aws" {
  mock_data "aws_caller_identity" {
    defaults = { account_id = "123456789012" }
  }
  mock_data "aws_availability_zones" {
    defaults = { names = ["ca-central-1a", "ca-central-1b"] }
  }
  mock_data "aws_iam_policy_document" {
    defaults = { json = "{\"Version\":\"2012-10-17\",\"Statement\":[]}" }
  }
  # Mocked resources get random strings; these attributes are validated as ARNs downstream.
  mock_resource "aws_iam_role" {
    defaults = { arn = "arn:aws:iam::123456789012:role/mock" }
  }
  mock_resource "aws_sns_topic" {
    defaults = { arn = "arn:aws:sns:ca-central-1:123456789012:mock" }
  }
  mock_resource "aws_ecs_cluster" {
    defaults = { arn = "arn:aws:ecs:ca-central-1:123456789012:cluster/mock" }
  }
  mock_resource "aws_ecs_task_definition" {
    defaults = { arn = "arn:aws:ecs:ca-central-1:123456789012:task-definition/mock:1" }
  }
  mock_resource "aws_ecr_repository" {
    defaults = {
      arn            = "arn:aws:ecr:ca-central-1:123456789012:repository/mock"
      repository_url = "123456789012.dkr.ecr.ca-central-1.amazonaws.com/mock"
    }
  }
  mock_resource "aws_cloudwatch_log_group" {
    defaults = { arn = "arn:aws:logs:ca-central-1:123456789012:log-group:mock" }
  }
  mock_resource "aws_sfn_state_machine" {
    defaults = { arn = "arn:aws:states:ca-central-1:123456789012:stateMachine:mock" }
  }
  mock_resource "aws_s3_bucket" {
    defaults = { arn = "arn:aws:s3:::mock" }
  }
}

variables {
  region                     = "ca-central-1"
  vpc_cidr                   = "10.40.0.0/16"
  noncurrent_expiration_days = 7
  task_cpu                   = 1024
  task_memory                = 4096
  task_storage_gib           = 30
  log_retention_days         = 7
  state_bucket               = "vocalocate-tfstate-123456789012"
  modal_workspace_id         = "ac-test"
  modal_environment          = "dev"
}

run "dev_stack" {
  command = apply

  variables {
    environment   = "dev"
    force_destroy = true
  }

  assert {
    condition = output.bucket_names == {
      raw    = "vocalocate-dev-raw-123456789012"
      lake   = "vocalocate-dev-lake-123456789012"
      models = "vocalocate-dev-models-123456789012"
    }
    error_message = "Bucket names must follow vocalocate-<env>-<kind>-<account>."
  }

  assert {
    condition     = strcontains(output.modal_secret_command, "VOCALOCATE_LAKE_URI=s3://vocalocate-dev-lake-123456789012")
    error_message = "The Modal secret must point training at this environment's lake."
  }
}

run "rejects_unknown_environment" {
  command = plan

  variables {
    environment   = "staging"
    force_destroy = true
  }

  expect_failures = [var.environment]
}

run "state_machine" {
  command = apply

  module {
    source = "../orchestration"
  }

  variables {
    name_prefix         = "vocalocate-dev"
    environment         = "dev"
    cluster_arn         = "arn:aws:ecs:ca-central-1:123456789012:cluster/vocalocate-dev"
    task_definition_arn = "arn:aws:ecs:ca-central-1:123456789012:task-definition/vocalocate-dev-pipeline:3"
    container_name      = "pipeline"
    pass_role_arns      = ["arn:aws:iam::123456789012:role/task"]
    subnet_ids          = ["subnet-1", "subnet-2"]
    security_group_id   = "sg-1"
    raw_bucket_name     = "vocalocate-dev-raw-123456789012"
    alerts_topic_arn    = "arn:aws:sns:ca-central-1:123456789012:alerts"
    glue_crawler_name   = "vocalocate-dev-lake"
  }

  assert {
    condition     = jsondecode(aws_sfn_state_machine.pipeline.definition).States.Ingest.Parameters.Overrides.ContainerOverrides[0]["Command.$"] == "States.Array('remote', 'ingest', '--source', $.source)"
    error_message = "Ingest must run `remote ingest --source <input source>`."
  }

  assert {
    condition     = jsondecode(aws_sfn_state_machine.pipeline.definition).States.Report.Next == "Catalog"
    error_message = "The Glue crawl must follow the report."
  }

  assert {
    condition     = jsondecode(aws_cloudwatch_event_rule.upload["team"].event_pattern).detail.object.key == ["team/recordings.csv"]
    error_message = "Team recordings are ingested when recordings.csv is uploaded."
  }
}

run "state_machine_without_catalog" {
  command = apply

  module {
    source = "../orchestration"
  }

  variables {
    name_prefix         = "vocalocate-dev"
    environment         = "dev"
    cluster_arn         = "arn:aws:ecs:ca-central-1:123456789012:cluster/vocalocate-dev"
    task_definition_arn = "arn:aws:ecs:ca-central-1:123456789012:task-definition/vocalocate-dev-pipeline:3"
    container_name      = "pipeline"
    pass_role_arns      = ["arn:aws:iam::123456789012:role/task"]
    subnet_ids          = ["subnet-1"]
    security_group_id   = "sg-1"
    raw_bucket_name     = "raw"
    alerts_topic_arn    = "arn:aws:sns:ca-central-1:123456789012:alerts"
    glue_crawler_name   = null
  }

  assert {
    condition     = !contains(keys(jsondecode(aws_sfn_state_machine.pipeline.definition).States), "Catalog")
    error_message = "No crawler, no Catalog state."
  }
}

run "task_uses_environment_config" {
  command = apply

  module {
    source = "../pipeline_compute"
  }

  variables {
    name_prefix      = "vocalocate-prod"
    environment      = "prod"
    raw_bucket_name  = "raw"
    raw_bucket_arn   = "arn:aws:s3:::raw"
    lake_bucket_name = "lake"
    lake_bucket_arn  = "arn:aws:s3:::lake"
  }

  assert {
    condition = {
      for e in jsondecode(aws_ecs_task_definition.pipeline.container_definitions)[0].environment : e.name => e.value
      } == {
      VOCALOCATE_CONFIG   = "/app/configs/prod.toml"
      VOCALOCATE_LAKE_URI = "s3://lake"
      VOCALOCATE_RAW_URI  = "s3://raw"
      AWS_REGION          = "ca-central-1"
    }
    error_message = "The task must point the pipeline at this environment's config and buckets."
  }
}

run "prod_lake_replicates_to_backup" {
  command = apply

  module {
    source = "../data_lake"
  }

  variables {
    name_prefix = "vocalocate-prod"
    account_id  = "123456789012"
    backup_bucket_arns = {
      raw    = "arn:aws:s3:::vocalocate-backup-raw-123456789012"
      lake   = "arn:aws:s3:::vocalocate-backup-lake-123456789012"
      models = "arn:aws:s3:::vocalocate-backup-models-123456789012"
    }
  }

  assert {
    condition     = length(aws_s3_bucket_replication_configuration.backup) == 3
    error_message = "Every prod bucket must replicate to the backup stack."
  }

  assert {
    condition     = aws_s3_bucket_replication_configuration.backup["raw"].rule[0].delete_marker_replication[0].status == "Disabled"
    error_message = "Deletes in prod must not reach the backup."
  }
}

run "dev_lake_does_not_replicate" {
  command = apply

  module {
    source = "../data_lake"
  }

  variables {
    name_prefix   = "vocalocate-dev"
    account_id    = "123456789012"
    force_destroy = true
  }

  assert {
    condition     = length(aws_s3_bucket_replication_configuration.backup) == 0 && length(aws_iam_role.replication) == 0
    error_message = "Dev must not replicate."
  }
}
