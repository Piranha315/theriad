terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google",
      version = "~> 6.0",
    }
  }
}


variable "project_id" {
  type = string
}
variable "billing_account" {
  type = string
}
variable "pilot_alert_usd" {
  type    = number
  default = 25
  validation {
    condition     = var.pilot_alert_usd > 0 && floor(var.pilot_alert_usd) == var.pilot_alert_usd
    error_message = "Billing alert must be a positive whole-dollar amount."
  }
}
data "google_project" "pilot" {
  project_id = var.project_id
}
resource "google_billing_budget" "pilot" {
  billing_account = trimprefix(var.billing_account, "billingAccounts/")
  display_name    = "Theriad project spending alert"
  budget_filter {
    projects               = ["projects/${data.google_project.pilot.number}"]
    credit_types_treatment = "EXCLUDE_ALL_CREDITS"
  }
  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.pilot_alert_usd)
    }
  }
  threshold_rules { threshold_percent = 0.5 }
  threshold_rules { threshold_percent = 0.9 }
  threshold_rules { threshold_percent = 1.0 }
  depends_on = [google_project_service.services]
}


variable "region" {
  type    = string
  default = "us-central1"
}


variable "image" {
  type = string
}


variable "bucket_name" {
  type = string
}


variable "release_signing_secret" {
  type = string
}


variable "gateway_tokens_secret" {
  type = string
}


variable "openrouter_secret" {
  type    = string
  default = ""
}


variable "artificial_analysis_secret" {
  type    = string
  default = ""
}


variable "suite_object" {
  type    = string
  default = ""
}


variable "identities_object" {
  type    = string
  default = ""
}


variable "suite_sha256" {
  type    = string
  default = ""
}


variable "identities_sha256" {
  type    = string
  default = ""
}

variable "enable_updates" {
  type        = bool
  default     = true
  description = "Create the update job and Scheduler only when reviewed evaluation inputs are ready."
}

moved {
  from = google_cloud_run_v2_job.updater
  to   = google_cloud_run_v2_job.updater[0]
}
moved {
  from = google_cloud_run_v2_job_iam_member.schedule
  to   = google_cloud_run_v2_job_iam_member.schedule[0]
}
moved {
  from = google_cloud_scheduler_job.daily
  to   = google_cloud_scheduler_job.daily[0]
}
moved {
  from = google_service_account.updater
  to   = google_service_account.updater[0]
}
moved {
  from = google_service_account.scheduler
  to   = google_service_account.scheduler[0]
}


variable "scheduler_paused" {
  type    = bool
  default = true
}


variable "authorized_invokers" {
  type = set(string)
  validation {
    condition     = length(var.authorized_invokers) > 0 && alltrue([for identity in var.authorized_invokers : can(regex("^(user|group|serviceAccount):[^ ]+@[^ ]+$", identity))])
    error_message = "Explicit Google identities required; public invokers are prohibited."
  }
}
variable "network" {
  type    = string
  default = null
}
variable "subnetwork" {
  type    = string
  default = null
}
variable "ingress" {
  type    = string
  default = "INGRESS_TRAFFIC_ALL"
  validation {
    condition     = contains(["INGRESS_TRAFFIC_ALL", "INGRESS_TRAFFIC_INTERNAL_ONLY", "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"], var.ingress)
    error_message = "Choose a supported Cloud Run ingress boundary."
  }
}
variable "vpc_egress" {
  type    = string
  default = "ALL_TRAFFIC"
}


provider "google" {
  project = var.project_id
  region  = var.region
}


resource "google_project_service" "services" {
  for_each = toset(concat([
    "run.googleapis.com",
    "firestore.googleapis.com",
    "storage.googleapis.com",
    "secretmanager.googleapis.com",
    "iamcredentials.googleapis.com",
    "aiplatform.googleapis.com",
    "billingbudgets.googleapis.com"
  ], var.enable_updates ? ["cloudscheduler.googleapis.com"] : []))
  service            = each.key
  disable_on_destroy = false
}


resource "google_service_account" "gateway" {
  account_id = "theriad-gateway"
}


resource "google_service_account" "updater" {
  count      = var.enable_updates ? 1 : 0
  account_id = "theriad-updater"
}


resource "google_service_account" "scheduler" {
  count      = var.enable_updates ? 1 : 0
  account_id = "theriad-scheduler"
}
resource "google_service_account" "bridge" {
  account_id = "theriad-bridge"
}
resource "google_service_account_iam_member" "bridge_tokens" {
  for_each           = var.authorized_invokers
  service_account_id = google_service_account.bridge.name
  role               = "roles/iam.serviceAccountOpenIdTokenCreator"
  member             = each.value
}


resource "google_storage_bucket" "records" {
  name                        = var.bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false

  lifecycle_rule {
    condition {
      age = 1
      matches_prefix = [
        "ephemeral/",
      ]
    }


    action {
      type = "Delete"
    }
  }
}


resource "google_firestore_field" "expiry" {
  project    = var.project_id
  database   = google_firestore_database.gateway.name
  collection = "theriad_gateway"
  field      = "expires_at"

  ttl_config {

  }


  index_config {

  }
}


resource "google_firestore_database" "gateway" {
  project     = var.project_id
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"
  depends_on = [
    google_project_service.services,
  ]
}


resource "google_project_iam_member" "database" {
  for_each = merge({
    gateway = google_service_account.gateway.email,
  }, var.enable_updates ? { updater = google_service_account.updater[0].email } : {})
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${each.value}"
}


resource "google_storage_bucket_iam_member" "objects" {
  for_each = merge({
    gateway = google_service_account.gateway.email,
  }, var.enable_updates ? { updater = google_service_account.updater[0].email } : {})
  bucket = google_storage_bucket.records.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${each.value}"
}


resource "google_project_iam_member" "vertex" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.gateway.email}"
}

resource "google_project_iam_member" "updater_vertex" {
  count   = var.enable_updates ? 1 : 0
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.updater[0].email}"
}


resource "google_secret_manager_secret_iam_member" "access" {
  for_each = merge({
    gateway_signing = {
      secret  = var.release_signing_secret,
      account = google_service_account.gateway.email,
    },
    gateway_tokens = {
      secret  = var.gateway_tokens_secret,
      account = google_service_account.gateway.email,
    },
    }, var.openrouter_secret != "" ? { gateway_openrouter = {
      secret  = var.openrouter_secret,
      account = google_service_account.gateway.email,
    } } : {}, var.enable_updates ? {
    updater_signing = {
      secret  = var.release_signing_secret,
      account = google_service_account.updater[0].email,
    },
    updater_openrouter = {
      secret  = var.openrouter_secret,
      account = google_service_account.updater[0].email,
    },
    updater_analysis = {
      secret  = var.artificial_analysis_secret,
      account = google_service_account.updater[0].email,
    },
  } : {})
  project   = var.project_id
  secret_id = each.value.secret
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${each.value.account}"
}


resource "google_cloud_run_v2_service" "gateway" {
  name     = "theriad-gateway"
  location = var.region
  ingress  = var.ingress

  template {
    service_account = google_service_account.gateway.email
    dynamic "vpc_access" {
      for_each = var.network == null ? [] : [1]
      content {
        egress = var.vpc_egress
        network_interfaces {
          network    = var.network
          subnetwork = var.subnetwork
          tags       = ["theriad-gateway"]
        }
      }
    }
    timeout                          = "600s"
    max_instance_request_concurrency = 8

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }


    containers {
      image = var.image

      resources {
        limits = {
          cpu    = "2",
          memory = "2Gi",
        }
        cpu_idle = true
      }


      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }


      env {
        name  = "THERIAD_GATEWAY_BUCKET"
        value = google_storage_bucket.records.name
      }


      dynamic "env" {
        for_each = merge({
          THERIAD_RELEASE_SIGNING_KEY = var.release_signing_secret,
          THERIAD_GATEWAY_TOKENS      = var.gateway_tokens_secret,
        }, var.openrouter_secret != "" ? { OPENROUTER_API_KEY = var.openrouter_secret } : {})

        content {
          name = env.key

          value_source {
            secret_key_ref {
              secret  = env.value
              version = "latest"
            }
          }
        }
      }


      startup_probe {
        http_get {
          path = "/health"
        }

        period_seconds = 10
      }
    }
  }

  depends_on = [
    google_project_service.services,
    google_secret_manager_secret_iam_member.access,
  ]
}


resource "google_cloud_run_v2_job" "updater" {
  count    = var.enable_updates ? 1 : 0
  name     = "theriad-update"
  location = var.region

  template {
    template {
      service_account = google_service_account.updater[0].email
      dynamic "vpc_access" {
        for_each = var.network == null ? [] : [1]
        content {
          egress = var.vpc_egress
          network_interfaces {
            network    = var.network
            subnetwork = var.subnetwork
            tags       = ["theriad-updater"]
          }
        }
      }
      timeout     = "3600s"
      max_retries = 0

      containers {
        image = var.image
        command = [
          "python",
          "-m",
          "theriad.gateway.cloud_job",
        ]

        env {
          name  = "GOOGLE_CLOUD_PROJECT"
          value = var.project_id
        }


        env {
          name  = "THERIAD_GATEWAY_BUCKET"
          value = google_storage_bucket.records.name
        }


        env {
          name  = "THERIAD_SUITE_OBJECT"
          value = var.suite_object
        }


        env {
          name  = "THERIAD_IDENTITIES_OBJECT"
          value = var.identities_object
        }


        env {
          name  = "THERIAD_SUITE_SHA256"
          value = var.suite_sha256
        }


        env {
          name  = "THERIAD_IDENTITIES_SHA256"
          value = var.identities_sha256
        }


        dynamic "env" {
          for_each = {
            THERIAD_RELEASE_SIGNING_KEY = var.release_signing_secret,
            OPENROUTER_API_KEY          = var.openrouter_secret,
            ARTIFICIAL_ANALYSIS_API_KEY = var.artificial_analysis_secret,
          }

          content {
            name = env.key

            value_source {
              secret_key_ref {
                secret  = env.value
                version = "latest"
              }
            }
          }
        }
      }
    }
  }

  depends_on = [
    google_project_service.services,
    google_secret_manager_secret_iam_member.access,
  ]
}


resource "google_cloud_run_v2_job_iam_member" "schedule" {
  count    = var.enable_updates ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_job.updater[0].name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler[0].email}"
}


resource "google_cloud_scheduler_job" "daily" {
  count     = var.enable_updates ? 1 : 0
  name      = "theriad-update-daily"
  region    = var.region
  schedule  = "0 3 * * *"
  time_zone = "America/New_York"
  paused    = var.scheduler_paused

  http_target {
    uri         = "https://run.googleapis.com/v2/projects/${var.project_id}/locations/${var.region}/jobs/${google_cloud_run_v2_job.updater[0].name}:run"
    http_method = "POST"

    oauth_token {
      service_account_email = google_service_account.scheduler[0].email
    }
  }
}


resource "google_cloud_run_v2_service_iam_member" "invokers" {
  for_each = setunion(var.authorized_invokers, toset(["serviceAccount:${google_service_account.bridge.email}"]))
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.gateway.name
  role     = "roles/run.invoker"
  member   = each.value
}


output "gateway_url" {
  value = google_cloud_run_v2_service.gateway.uri
}
output "bridge_service_account" {
  value = google_service_account.bridge.email
}
