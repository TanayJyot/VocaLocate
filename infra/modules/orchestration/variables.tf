variable "name_prefix" {
  type = string
}

variable "environment" {
  type = string
}

variable "region" {
  type = string
}

variable "cluster_arn" {
  type = string
}

variable "task_definition_arn" {
  type = string
}

variable "container_name" {
  type = string
}

variable "pass_role_arns" {
  description = "The pipeline task and execution roles, which the state machine hands to ECS."
  type        = list(string)
}

variable "subnet_ids" {
  type = list(string)
}

variable "security_group_id" {
  type = string
}

variable "raw_bucket_name" {
  type = string
}

variable "alerts_topic_arn" {
  type = string
}

variable "glue_crawler_name" {
  description = "Crawler to start after each run so Athena sees new tables (null to skip)."
  type        = string
  default     = null
}

variable "upload_triggers" {
  description = "Source => raw-bucket key whose upload starts an ingest of that source."
  type        = map(string)
  default     = { team = "team/recordings.csv" }
}

variable "schedules" {
  description = "Name => schedule expression (Toronto time) and state machine input."
  type = map(object({
    expression = string
    input      = map(string)
  }))
  default = {}
}

variable "task_timeout_seconds" {
  type    = number
  default = 4 * 3600
}

variable "log_retention_days" {
  type    = number
  default = 30
}

variable "tags" {
  type    = map(string)
  default = {}
}
