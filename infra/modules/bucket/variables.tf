variable "name" {
  description = "Globally unique bucket name."
  type        = string
}

variable "force_destroy" {
  description = "Allow `terraform destroy` to delete the bucket while it still holds objects."
  type        = bool
  default     = false
}

variable "versioning" {
  description = "Keep earlier versions of overwritten or deleted objects (required for replication)."
  type        = bool
  default     = true
}

variable "transitions" {
  description = "Move current objects to cheaper storage classes after this many days."
  type = list(object({
    days          = number
    storage_class = string
  }))
  default = []
}

variable "expiration_days" {
  description = "Delete current objects after this many days (null keeps them)."
  type        = number
  default     = null
}

variable "noncurrent_expiration_days" {
  description = "Delete earlier object versions after this many days (null keeps them)."
  type        = number
  default     = null
}

variable "eventbridge" {
  description = "Send this bucket's object events to EventBridge (used to trigger pipeline runs)."
  type        = bool
  default     = false
}

variable "tags" {
  type    = map(string)
  default = {}
}
