variable "name_prefix" {
  type = string
}

variable "environment" {
  type = string
}

variable "account_id" {
  type = string
}

variable "lake_bucket_name" {
  type = string
}

variable "lake_bucket_arn" {
  type = string
}

variable "crawler_schedule" {
  description = "Optional cron() schedule; the pipeline state machine also starts a crawl after each run."
  type        = string
  default     = null
}

variable "bytes_scanned_cutoff" {
  description = "Cancel any Athena query that would scan more than this many bytes."
  type        = number
  default     = 1073741824
}

variable "tags" {
  type    = map(string)
  default = {}
}
