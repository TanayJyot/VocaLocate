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
  description = "Pipeline image tag to run; CI pushes one per commit plus `latest`."
  type        = string
  default     = "latest"
}

variable "force_destroy" {
  description = "Only for the disaster-recovery demo: apply with true, then destroy (see infra/README.md)."
  type        = bool
  default     = false
}
