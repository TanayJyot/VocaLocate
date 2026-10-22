# Step Functions runs the pipeline as ECS Fargate tasks, one step per flow stage:
#
#   {"source": "<dataset>"}  ->  Ingest (ingest, clean, transform, rebuild warehouse)  ->  Report  -> Catalog
#   {}                       ->  LoadWarehouse                                          ->  Report  -> Catalog
#
# Any failed step publishes to the alerts topic and fails the execution. Runs start from:
#   * an upload of team/recordings.csv to the raw bucket (EventBridge), after a recording session;
#   * the schedules in var.schedules (EventBridge Scheduler);
#   * by hand: aws stepfunctions start-execution --input '{"source":"vis"}' (new dataset or version).

locals {
  network = {
    AwsvpcConfiguration = {
      Subnets        = var.subnet_ids
      SecurityGroups = [var.security_group_id]
      AssignPublicIp = "ENABLED"
    }
  }

  retry_capacity = [{
    ErrorEquals     = ["ECS.AmazonECSException", "ECS.ServerException"]
    IntervalSeconds = 60
    MaxAttempts     = 2
    BackoffRate     = 2
  }]

  catch_all = [{
    ErrorEquals = ["States.ALL"]
    ResultPath  = "$.error"
    Next        = "NotifyFailure"
  }]

  # One Fargate task running `vocalocate-pipeline -v <command>`.
  run_task = {
    Type     = "Task"
    Resource = "arn:aws:states:::ecs:runTask.sync"
    Parameters = {
      LaunchType           = "FARGATE"
      Cluster              = var.cluster_arn
      TaskDefinition       = var.task_definition_arn
      NetworkConfiguration = local.network
      PropagateTags        = "TASK_DEFINITION"
    }
    ResultPath     = null
    TimeoutSeconds = var.task_timeout_seconds
    Retry          = local.retry_capacity
    Catch          = local.catch_all
  }

  crawl_state = {
    Catalog = {
      Type       = "Task"
      Resource   = "arn:aws:states:::aws-sdk:glue:startCrawler"
      Parameters = { Name = var.glue_crawler_name }
      ResultPath = null
      # A crawl already in progress will pick up the new files anyway.
      Catch = [{ ErrorEquals = ["Glue.CrawlerRunningException"], ResultPath = null, Next = "Done" }]
      Next  = "Done"
    }
  }

  states = merge(
    {
      Route = {
        Type    = "Choice"
        Choices = [{ Variable = "$.source", IsPresent = true, Next = "Ingest" }]
        Default = "LoadWarehouse"
      }
      Ingest = merge(local.run_task, {
        Parameters = merge(local.run_task.Parameters, {
          Overrides = { ContainerOverrides = [{
            Name        = var.container_name
            "Command.$" = "States.Array('remote', 'ingest', '--source', $.source)"
          }] }
        })
        Next = "Report"
      })
      LoadWarehouse = merge(local.run_task, {
        Parameters = merge(local.run_task.Parameters, {
          Overrides = { ContainerOverrides = [{ Name = var.container_name, Command = ["remote", "load-warehouse"] }] }
        })
        Next = "Report"
      })
      # Data-quality summary (v_clip_quality) in the task's log, after every run.
      Report = merge(local.run_task, {
        Parameters = merge(local.run_task.Parameters, {
          Overrides = { ContainerOverrides = [{ Name = var.container_name, Command = ["remote", "report"] }] }
        })
        Next = var.glue_crawler_name == null ? "Done" : "Catalog"
      })
      NotifyFailure = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn    = var.alerts_topic_arn
          Subject     = "VocaLocate ${var.environment} pipeline run failed"
          "Message.$" = "States.JsonToString($)"
        }
        Next = "Failed"
      }
      Failed = { Type = "Fail", Error = "PipelineFailed", Cause = "A pipeline step failed; see the alert and the task logs." }
      Done   = { Type = "Succeed" }
    },
    { for name, state in local.crawl_state : name => state if var.glue_crawler_name != null },
  )
}

resource "aws_cloudwatch_log_group" "state_machine" {
  name              = "/vocalocate/${var.environment}/state-machine"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}

resource "aws_sfn_state_machine" "pipeline" {
  name     = "${var.name_prefix}-pipeline"
  role_arn = aws_iam_role.state_machine.arn
  definition = jsonencode({
    Comment = "VocaLocate ${var.environment} data pipeline"
    StartAt = "Route"
    States  = local.states
  })

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.state_machine.arn}:*"
    include_execution_data = false
    level                  = "ERROR"
  }
  tags = var.tags
}

# --- State machine role -------------------------------------------------------------------------

data "aws_caller_identity" "current" {}

data "aws_iam_policy_document" "states_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "state_machine" {
  name               = "${var.name_prefix}-pipeline-sfn"
  assume_role_policy = data.aws_iam_policy_document.states_assume.json
  tags               = var.tags
}

data "aws_iam_policy_document" "state_machine" {
  statement {
    sid       = "RunPipelineTasks"
    actions   = ["ecs:RunTask"]
    resources = ["${replace(var.task_definition_arn, "/:\\d+$/", "")}:*"]
  }
  statement {
    sid       = "TrackPipelineTasks"
    actions   = ["ecs:StopTask", "ecs:DescribeTasks"]
    resources = ["*"]
    condition {
      test     = "ArnEquals"
      variable = "ecs:cluster"
      values   = [var.cluster_arn]
    }
  }
  statement {
    sid       = "PassTaskRoles"
    actions   = ["iam:PassRole"]
    resources = var.pass_role_arns
  }
  # runTask.sync waits for the task through this managed EventBridge rule.
  statement {
    sid       = "SyncRule"
    actions   = ["events:PutTargets", "events:PutRule", "events:DescribeRule"]
    resources = ["arn:aws:events:${var.region}:${data.aws_caller_identity.current.account_id}:rule/StepFunctionsGetEventsForECSTaskRule"]
  }
  statement {
    sid       = "Alerts"
    actions   = ["sns:Publish"]
    resources = [var.alerts_topic_arn]
  }
  dynamic "statement" {
    for_each = var.glue_crawler_name == null ? [] : [var.glue_crawler_name]
    content {
      sid       = "Catalog"
      actions   = ["glue:StartCrawler"]
      resources = ["arn:aws:glue:${var.region}:${data.aws_caller_identity.current.account_id}:crawler/${statement.value}"]
    }
  }
  statement {
    sid = "Logs"
    actions = [
      "logs:CreateLogDelivery", "logs:GetLogDelivery", "logs:UpdateLogDelivery", "logs:DeleteLogDelivery",
      "logs:ListLogDeliveries", "logs:PutResourcePolicy", "logs:DescribeResourcePolicies", "logs:DescribeLogGroups",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "state_machine" {
  role   = aws_iam_role.state_machine.id
  policy = data.aws_iam_policy_document.state_machine.json
}

# --- Triggers -----------------------------------------------------------------------------------

data "aws_iam_policy_document" "start_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com", "scheduler.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "start" {
  name               = "${var.name_prefix}-pipeline-start"
  assume_role_policy = data.aws_iam_policy_document.start_assume.json
  tags               = var.tags
}

resource "aws_iam_role_policy" "start" {
  role = aws_iam_role.start.id
  policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = "states:StartExecution", Resource = aws_sfn_state_machine.pipeline.arn }]
  })
}

# Each upload trigger: a key in the raw bucket whose creation starts an ingest of that source.
resource "aws_cloudwatch_event_rule" "upload" {
  for_each    = var.upload_triggers
  name        = "${var.name_prefix}-upload-${each.key}"
  description = "Ingest ${each.key} when s3://${var.raw_bucket_name}/${each.value} is uploaded"
  event_pattern = jsonencode({
    source      = ["aws.s3"]
    detail-type = ["Object Created"]
    detail = {
      bucket = { name = [var.raw_bucket_name] }
      object = { key = [each.value] }
    }
  })
  tags = var.tags
}

resource "aws_cloudwatch_event_target" "upload" {
  for_each = var.upload_triggers
  rule     = aws_cloudwatch_event_rule.upload[each.key].name
  arn      = aws_sfn_state_machine.pipeline.arn
  role_arn = aws_iam_role.start.arn
  input    = jsonencode({ source = each.key })
}

resource "aws_scheduler_schedule" "this" {
  for_each                     = var.schedules
  name                         = "${var.name_prefix}-${each.key}"
  schedule_expression          = each.value.expression
  schedule_expression_timezone = "America/Toronto"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_sfn_state_machine.pipeline.arn
    role_arn = aws_iam_role.start.arn
    input    = jsonencode(each.value.input)
  }
}
