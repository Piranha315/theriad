"""Use-specific engineering admission; never changes research corpus clearance.

Preparation is outcome-blind. A separate paid authorization is required before
collection. The source package and original exposure/group identities stay intact.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from .episode_study import Episode, FAMILIES, digest, validate_episodes

PROFILE = "engineering_canary"
VERSION = "router-engineering-admission-v1"
CORPUS_SHA256 = "00d7731121beb103980af0f1efbf239899e62b4ada05468f5b32422ef30bf60d"
REVIEW_HASHES = frozenset({
    "3ab67c1a6c74e1944497444f19e0052eabaa542190a94a65a9d4910f32f92d51",
    "0526acfb9f104b0f22437d266bfbff82bf62493675901ad16293660930f47c51",
})
SPLITS = ("pilot", "train", "calibration", "threshold_selection", "smoke")
# Explicit family names avoid dependence on the research FAMILIES ordering.
ALLOCATIONS = {
    "pilot": dict(chat=1, coding=1, search=1, tools=1, agents=1),
    "train": dict(chat=20, coding=18, search=18, tools=2, agents=2),
    "calibration": dict(chat=10, coding=9, search=9, tools=1, agents=1),
    "threshold_selection": dict(chat=12, coding=12, search=13, tools=1, agents=2),
    "smoke": dict(chat=7, coding=4, search=8, tools=1, agents=1),
}


def _validate_review(row: Episode) -> None:
    review = row.fixture.get("semantic_review", {})
    if (review.get("review_complete") is not True
            or review.get("status") != "reviewed_research_only"
            or review.get("review_hash") not in REVIEW_HASHES
            or review.get("disposition") not in {"retained_research_only", "development_only_prior_overlap"}):
        raise ValueError("engineering admission requires completed retained semantic review")
    if row.split not in SPLITS:
        raise ValueError("engineering admission cannot include research test or unknown exposure stages")
    if review["disposition"] == "development_only_prior_overlap" and row.split not in {"pilot", "train"}:
        raise ValueError("prior exposure cannot enter calibration, selection or smoke")


def _select(rows: Sequence[Episode], *, seed: int = 731) -> list[Episode]:
    validate_episodes(rows)
    grouped = defaultdict(list)
    for row in rows:
        _validate_review(row)
        grouped[row.split, row.family, row.group_id].append(row)
    selected = []
    for split in SPLITS:
        for family in FAMILIES:
            group_ids = [group for stage, fam, group in grouped if (stage, fam) == (split, family)]
            group_ids.sort(key=lambda group: (digest([VERSION, seed, split, family, group]), group))
            count = ALLOCATIONS[split][family]
            if len(group_ids) < count:
                raise ValueError(f"engineering allocation deficient: {family}/{split}")
            for group in group_ids[:count]:
                selected.append(min(grouped[split, family, group], key=lambda row: row.episode_id))
    validate_episodes(selected)
    return selected


def prepare_admission(corpus_path: Path, *, seed: int = 731) -> tuple[list[Episode], dict]:
    """Pin the reviewed v9 corpus and select existing groups without outcomes."""
    raw = corpus_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CORPUS_SHA256:
        raise ValueError("engineering admission requires the exact reviewed v9 corpus")
    # Do not instantiate or export test episodes. Their source bytes are only
    # covered by the already-published whole-package identity.
    records = (json.loads(line) for line in raw.splitlines() if line.strip())
    rows = [Episode(**record) for record in records if record.get("split") != "test"]
    diagnostic = [row for row in rows if row.split == "smoke"
                  and row.fixture.get("semantic_review", {}).get("disposition") == "development_only_prior_overlap"]
    rows = [row for row in rows if row not in diagnostic]
    selected = _select(rows, seed=seed)
    admission = {
        "version": VERSION, "profile": PROFILE,
        "source_corpus_sha256": CORPUS_SHA256,
        "review_hashes": sorted(REVIEW_HASHES), "seed": seed,
        "allocations": ALLOCATIONS,
        "diagnostic_only_prior_overlap": [{"episode_id": row.episode_id, "group_id": row.group_id,
                                            "split": row.split, "fixture_hash": row.content_hash}
                                           for row in diagnostic],
        "scope": "engineering fitting and one-shot smoke; internal text-chat eligibility only",
        "research_collection_clearance": False, "paid_calls_authorized": False,
        "independent_external_review": False,
        "limitations": ["same-author review", "public benchmark contamination possible",
                        "short deterministic fixtures are not representative customer conversations"],
        "rows": [{"episode_id": row.episode_id, "group_id": row.group_id,
                  "family": row.family, "split": row.split,
                  "fixture_hash": row.content_hash} for row in selected],
    }
    return selected, admission


def verify_admission(directory: Path, rows: Sequence[Episode], contract: dict) -> None:
    admission = json.loads((directory / "engineering-admission.json").read_text())
    if (contract.get("profile") != PROFILE
            or digest(admission) != contract.get("engineering_admission_hash")
            or admission.get("version") != VERSION
            or admission.get("source_corpus_sha256") != CORPUS_SHA256
            or admission.get("research_collection_clearance") is not False):
        raise ValueError("engineering admission changed or has an unsupported scope")
    indexed = {item["episode_id"]: item for item in admission["rows"]}
    if len(indexed) != len(admission["rows"]):
        raise ValueError("duplicate engineering admission row")
    for row in rows:
        _validate_review(row)
        actual = {"episode_id": row.episode_id, "group_id": row.group_id,
                  "family": row.family, "split": row.split, "fixture_hash": row.content_hash}
        if indexed.get(row.episode_id) != actual:
            raise ValueError("episode does not match engineering admission")


def require_paid_authorization(directory: Path, contract: dict, *, phase: str) -> None:
    """Require a separately recorded approval bound to this exact paid scope.

    This function verifies an approval record, it does not grant approval.
    The command executor must already have explicit user authorization to write
    the record. No environment switch or rehearsal marker bypasses this guard.
    """
    path = directory / f"authorization-{phase}.json"
    if not path.is_file():
        raise ValueError(f"explicit paid {phase} authorization is missing; no provider calls permitted")
    approved = json.loads(path.read_text())
    if (approved.get("contract_hash") != digest(contract)
            or approved.get("phase") != phase
            or approved.get("paid_calls_authorized") is not True
            or not isinstance(approved.get("user_authorization_reference"), str)
            or not approved["user_authorization_reference"].strip()
            or approved.get("budget_proposal_hash") != contract.get("budget_proposal_hash")
            or approved.get("budget_config") != contract.get("budget_config")
            or approved.get("protocol_manifest_hash") != contract.get("protocol_manifest_hash")
            or not contract.get("protocol_manifest_hash")):
        raise ValueError("paid authorization does not match frozen contract and budget")


def authored_provenance(row: Episode) -> dict:
    metadata = row.fixture.get("source_metadata", {})
    if row.source == "newly_authored_20260911":
        _validate_review(row)
        source_id = metadata.get("source_id")
        source_sha = metadata.get("source_file_sha256")
        if not source_id or not isinstance(source_sha, str) or len(source_sha) != 64:
            raise ValueError("authored extension needs pinned source identity")
        return {**metadata, "license_name": "user-authorized newly authored study tasks",
                "license_status": "approved",
                "provenance_url": f"repo:data/router-v2-authored-20260911/specs-v4/{'coding_tasks.py' if row.family == 'coding' else 'workspace_tasks.py'}#task={source_id}&sha256={source_sha}"}
    return metadata


def execution_protocol_record(directory: Path, contract: dict, protocol: dict) -> str:
    """An approved source repair preserves the original contract and ledger.

    Approval is separate from the original spend grant; this is never an
    environment bypass. Data, model settings and stage caps remain bound to
    the original contract. Each execution snapshot is retained independently.
    """
    original = contract.get('protocol_manifest_hash')
    current = protocol.get('manifest_hash')
    if current == original and original:
        return 'execution-protocol.json'
    path = directory / 'source-amendment.json'
    if not path.is_file():
        raise ValueError('engineering source/environment changed; approved source amendment required')
    amendment = json.loads(path.read_text())
    caps = contract.get('budget_config', {}).get('stage_caps_usd', {})
    phase_cap = sum(caps.get(stage, 0) for stage in ('pilot', 'development'))
    if (amendment.get('approved') is not True
            or amendment.get('contract_hash') != digest(contract)
            or amendment.get('original_protocol_hash') != original
            or amendment.get('replacement_protocol_hash') != current
            or phase_cap <= 0 or amendment.get('approved_phase_a_cap_usd') != phase_cap
            or not isinstance(amendment.get('user_authorization_reference'), str)
            or not amendment['user_authorization_reference'].strip()):
        raise ValueError('source amendment does not match the approved contract and current source')
    return f'execution-protocol-{current}.json'


def apply_billing_reconciliation(directory: Path, attempts: dict, raw_events: list) -> dict | None:
    """Read an auditable provider charge correction without rewriting raw events."""
    path = directory / 'billing-reconciliation.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    contract = json.loads((directory / 'contract.json').read_text())
    metadata_bytes = (directory / 'cancelled-generation-metadata.json').read_bytes()
    metadata = json.loads(metadata_bytes)
    originals = [e for e in raw_events if e.get('event') == 'attempt_finished'
                 and e.get('attempt_id') == record.get('attempt_id')]
    if (record.get('contract_hash') != digest(contract) or len(originals) != 1
            or digest(originals[0]) != record.get('original_attempt_terminal_hash')
            or originals[0].get('cost_usd') is not None
            or hashlib.sha256(metadata_bytes).hexdigest() != record.get('provider_metadata_sha256')
            or metadata.get('id') != record.get('generation_id')
            or metadata.get('total_cost') != record.get('cancelled_request_cost_usd')):
        raise ValueError('billing reconciliation does not match immutable provider evidence')
    amount = metadata['total_cost']
    import math
    if type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
        raise ValueError('invalid reconciled provider charge')
    attempts[record['attempt_id']].update(cost_usd=amount, billed_cost_usd=amount,
        cost_known=True, cost_basis='provider_generation_metadata',
        generation_id=metadata['id'], original_cost_usd=None)
    return record


def engineering_price_bounds(provider_pricing: dict, *, max_input_tokens: int | None = None) -> dict:
    """Bound published token/cache tiers while keeping nominal routing prices.

    Add cache-write charge to prompt price conservatively, rather than assume
    the provider's write rate replaces the base rate. Native paid add-ons are
    separately forbidden by the request guard for this profile.
    """
    import math
    if max_input_tokens is not None and (type(max_input_tokens) is not int or max_input_tokens <= 0):
        raise ValueError("pricing input-token limit must be a positive integer")
    tiers = [provider_pricing]
    for tier in provider_pricing.get("overrides", []):
        if not isinstance(tier, dict):
            raise ValueError("invalid provider pricing tier")
        minimum = tier.get("min_prompt_tokens", 0)
        if type(minimum) is not int or minimum < 0:
            raise ValueError("invalid pricing tier threshold")
        if max_input_tokens is None or minimum <= max_input_tokens:
            tiers.append(tier)
    maxima = {"prompt": 0.0, "completion": 0.0, "input_cache_write": 0.0}
    for tier in tiers:
        if not isinstance(tier, dict):
            raise ValueError("invalid provider pricing tier")
        for key in maxima:
            raw = tier.get(key, provider_pricing.get(key, 0))
            if isinstance(raw, bool):
                raise ValueError("invalid provider token price")
            value = float(raw)
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid provider token price")
            maxima[key] = max(maxima[key], value)
    return {"reservation_input_per_1m": (maxima["prompt"] + maxima["input_cache_write"]) * 1e6,
            "reservation_output_per_1m": maxima["completion"] * 1e6,
            "require_provider_billing": True, "forbid_paid_addons": True,
            "reservation_input_token_limit": max_input_tokens}
