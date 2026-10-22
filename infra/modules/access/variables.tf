variable "name_prefix" {
  type = string
}

variable "environment" {
  type = string
}

variable "github_repository" {
  description = "owner/name of the repository whose workflows may assume the CI role."
  type        = string
}

variable "state_bucket" {
  description = "Terraform state bucket from infra/bootstrap."
  type        = string
}

variable "ecr_repository_arn" {
  type = string
}

variable "lake_bucket_arn" {
  type = string
}

variable "models_bucket_arn" {
  type = string
}

variable "modal_workspace_id" {
  description = "Modal workspace ID (ac-...), shown by `modal profile current` or in the Modal dashboard."
  type        = string
}

variable "modal_environment" {
  description = "Modal environment whose apps may assume the training role."
  type        = string
  default     = "main"
}

variable "modal_apps" {
  description = "Modal app names allowed to assume the training role."
  type        = list(string)
  default     = ["vocalocate-recipe"]
}

variable "tags" {
  type    = map(string)
  default = {}
}
