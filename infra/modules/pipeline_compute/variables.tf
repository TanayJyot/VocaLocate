variable "name_prefix" {
  type = string
}

variable "environment" {
  description = "dev or prod; also picks pipeline/configs/<environment>.toml inside the image."
  type        = string
}

variable "region" {
  type = string
}

variable "raw_bucket_name" {
  type = string
}

variable "raw_bucket_arn" {
  type = string
}

variable "lake_bucket_name" {
  type = string
}

variable "lake_bucket_arn" {
  type = string
}

variable "image_tag" {
  description = "Tag of the pipeline image in this environment's ECR repository."
  type        = string
  default     = "latest"
}

variable "cpu" {
  description = "Fargate task CPU units (1024 = 1 vCPU)."
  type        = number
  default     = 2048
}

variable "memory" {
  description = "Fargate task memory in MiB."
  type        = number
  default     = 8192
}

variable "ephemeral_storage_gib" {
  description = "Task disk for the local lake working copy (21-200 GiB)."
  type        = number
  default     = 50

  validation {
    condition     = var.ephemeral_storage_gib >= 21 && var.ephemeral_storage_gib <= 200
    error_message = "Fargate ephemeral storage must be between 21 and 200 GiB."
  }
}

variable "log_retention_days" {
  type    = number
  default = 30
}

variable "container_insights" {
  type    = bool
  default = false
}

variable "images_to_keep" {
  type    = number
  default = 20
}

variable "force_destroy" {
  description = "Let `terraform destroy` delete the ECR repository while it still holds images."
  type        = bool
  default     = false
}

variable "tags" {
  type    = map(string)
  default = {}
}
