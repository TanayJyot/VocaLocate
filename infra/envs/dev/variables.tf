variable "region" {
  type    = string
  default = "ca-central-1"
}

variable "state_bucket" {
  description = "Terraform state bucket from infra/bootstrap (vocalocate-tfstate-<account id>)."
  type        = string
}

variable "modal_workspace_id" {
  description = "Modal workspace ID (ac-...)."
  type        = string
}

variable "alert_emails" {
  type    = list(string)
  default = []
}

variable "image_tag" {
  description = "Commit SHA of the pipeline image to run; the pipeline-image workflow pushes one per commit."
  type        = string

  validation {
    condition     = can(regex("^[0-9a-f]{40}$", var.image_tag))
    error_message = "image_tag must be a full commit SHA, so every run can be traced to its code."
  }
}
