variable "name_prefix" {
  type = string
}

variable "environment" {
  type = string
}

variable "state_machine_arn" {
  description = "Pipeline state machine to alarm on."
  type        = string
}

variable "alert_emails" {
  description = "Each address gets a confirmation email from AWS before alerts arrive."
  type        = list(string)
  default     = []
}

variable "tags" {
  type    = map(string)
  default = {}
}
