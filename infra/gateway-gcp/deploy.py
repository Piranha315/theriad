"""Preflight-gated Terraform deployment; no automatic credit assumptions."""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from preflight import gcloud, validate_credit


def check_inputs(inputs, credit):
    project = inputs["project_id"]
    validate_credit(credit, project)
    if inputs.get("billing_account") != credit["billing_account"]:
        raise ValueError("deployment and credit billing accounts differ")
    identities = inputs.get("authorized_invokers", [])
    if not identities or any(
        not i.startswith(("user:", "serviceAccount:", "group:")) or "@" not in i for i in identities
    ):
        raise ValueError("explicit authorized Google identities are required")
    if bool(inputs.get("network")) != bool(inputs.get("subnetwork")):
        raise ValueError("preexisting network and subnetwork must be supplied together")
    required = [
        "image",
        "bucket_name",
        "release_signing_secret",
        "gateway_tokens_secret",
    ]
    if inputs.get("enable_updates", True):
        required += [
            "openrouter_secret", "artificial_analysis_secret", "suite_object", "identities_object",
            "suite_sha256", "identities_sha256"
        ]
        for field in ("suite_sha256", "identities_sha256"):
            if not re.fullmatch(r"[a-f0-9]{64}", inputs.get(field, "")):
                raise ValueError("approved evaluation input must be pinned by SHA-256: " + field)
    for field in required:
        if not inputs.get(field):
            raise ValueError("missing deployment input: " + field)
    if not re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", inputs["image"]):
        raise ValueError("deployment image must be pinned by SHA-256 digest")
    if not inputs["scheduler_paused"]:
        raise ValueError(
            "initial deployment must keep the worker paused until bootstrap and cloud acceptance pass"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--credit-record", required=True)
    parser.add_argument("--terraform", default="terraform")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    inputs = json.loads(Path(args.inputs).read_text())
    credit = json.loads(Path(args.credit_record).read_text())
    check_inputs(inputs, credit)
    if not shutil.which("gcloud") or not shutil.which(args.terraform):
        raise ValueError("gcloud and Terraform are required before deployment")
    project = gcloud("projects", "describe", inputs["project_id"])
    billing = gcloud("billing", "projects", "describe", inputs["project_id"])
    if project.get("lifecycleState") != "ACTIVE" or not billing.get("billingEnabled"):
        raise ValueError("active, billing-enabled project required")
    if billing["billingAccountName"].split("/")[-1] != credit["billing_account"].split("/")[-1]:
        raise ValueError("project is attached to a different billing account")
    secret_fields = ["release_signing_secret", "gateway_tokens_secret"]
    if inputs.get("openrouter_secret"):
        secret_fields.append("openrouter_secret")
    if inputs.get("enable_updates", True):
        secret_fields.append("artificial_analysis_secret")
    for secret in {inputs[k] for k in secret_fields}:
        gcloud("secrets", "describe", secret, "--project", inputs["project_id"])
    directory = Path(__file__).resolve().parent
    # Only identifiers/configuration enter Terraform; secret values never do.
    config = dict(inputs)
    path = directory / "pilot.auto.tfvars.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(config, handle)
    subprocess.run([args.terraform, f"-chdir={directory}", "init"], check=True)
    plan = directory / "pilot.tfplan"
    subprocess.run([args.terraform, f"-chdir={directory}", "plan", "-out=" + str(plan)], check=True)
    if args.apply:
        subprocess.run([args.terraform, f"-chdir={directory}", "apply", str(plan)], check=True)
    print(
        json.dumps(
            {
                "project": inputs["project_id"],
                "credit_record_validated": True,
                "applied": args.apply,
                "scheduler_paused": True,
            }
        )
    )


if __name__ == "__main__":
    main()
