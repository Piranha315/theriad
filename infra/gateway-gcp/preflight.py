"""Read-only project/credit preflight. Never creates resources or prints tokens."""

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def gcloud(*args):
    result = subprocess.run(
        ["gcloud", *args, "--format=json"], check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout)


def validate_credit(record, project, now=None):
    now = now or datetime.now(timezone.utc)
    if record.get("project_id") != project or not record.get("billing_account"):
        raise ValueError("credit record must identify project and billing account")
    balance = float(record.get("remaining_usd", 0))
    if not math.isfinite(balance) or balance <= 0:
        raise ValueError("positive verified credit balance is required")
    expiry = datetime.fromisoformat(record["expires_at"].replace("Z", "+00:00"))
    verified = datetime.fromisoformat(record["verified_at"].replace("Z", "+00:00"))
    if expiry <= now or not 0 <= (now - verified).total_seconds() <= 86400:
        raise ValueError("credit is expired or verification is older than 24 hours")
    required = {"Cloud Run", "Vertex AI", "Cloud Storage", "Firestore", "Cloud Scheduler"}
    if not required.issubset(set(record.get("eligible_services", []))):
        raise ValueError("credit eligibility must cover every pilot service")
    if not record.get("evidence_reference"):
        raise ValueError("credit balance and eligibility require a billing evidence reference")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--credit-record", required=True)
    args = parser.parse_args()
    record = json.loads(Path(args.credit_record).read_text())
    validate_credit(record, args.project)
    project = gcloud("projects", "describe", args.project)
    billing = gcloud("billing", "projects", "describe", args.project)
    if project.get("lifecycleState") != "ACTIVE" or not billing.get("billingEnabled"):
        raise ValueError("project must be active with billing enabled")
    if billing["billingAccountName"].split("/")[-1] != record["billing_account"].split("/")[-1]:
        raise ValueError("credit and project billing accounts differ")
    # Credit eligibility is verified from supplied billing evidence; gcloud does
    # not expose a universal promotional-credit balance API.
    print(
        json.dumps(
            {
                "project": args.project,
                "billing_enabled": True,
                "credit_record_valid": True,
                "resources_created": False,
            }
        )
    )


if __name__ == "__main__":
    main()
