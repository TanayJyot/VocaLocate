variable "environment" {
  description = "dev or prod."
  type        = string

  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "environment must be dev or prod (it also picks pipeline/configs/<environment>.toml)."
  }
}

variable "region" {
  type = string
}

variable "vpc_cidr" {
  type = string
}

variable "force_destroy" {
  description = "Let `terraform destroy` remove buckets and images that still hold data."
  type        = bool
}

variable "noncurrent_expiration_days" {
  type = number
}

variable "backup_bucket_arns" {
  description = "Lake buckets (raw, lake, models) to replicate to the backup stack."
  type        = map(string)
  default     = {}
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "task_cpu" {
  type = number
}

variable "task_memory" {
  type = number
}

variable "task_storage_gib" {
  type = number
}

variable "log_retention_days" {
  type = number
}

variable "schedules" {
  type = map(object({
    expression = string
    input      = map(string)
  }))
  default = {}
}

variable "alert_emails" {
  type    = list(string)
  default = []
}

variable "github_repository" {
  type    = string
  default = "TanayJyot/VocaLocate"
}

variable "state_bucket" {
  type = string
}

variable "modal_workspace_id" {
  type = string
}

variable "modal_environment" {
  type = string
}
