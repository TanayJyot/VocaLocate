variable "name_prefix" {
  description = "e.g. vocalocate-prod"
  type        = string
}

variable "account_id" {
  description = "Appended to bucket names to make them globally unique."
  type        = string
}

variable "force_destroy" {
  description = "Let `terraform destroy` delete buckets that still hold data (dev only, normally)."
  type        = bool
  default     = false
}

variable "noncurrent_expiration_days" {
  description = "How long overwritten or deleted object versions are kept."
  type        = number
  default     = 30
}

variable "backup_bucket_arns" {
  description = "Replicate these lake buckets (keys: raw, lake, models) to these backup bucket ARNs."
  type        = map(string)
  default     = {}

  validation {
    condition     = alltrue([for k in keys(var.backup_bucket_arns) : contains(["raw", "lake", "models"], k)])
    error_message = "Keys must be raw, lake or models."
  }
}

variable "tags" {
  type    = map(string)
  default = {}
}
