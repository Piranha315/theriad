"""Small, reproducible product-episode seed corpus with semantic group splits.

Fifty distinct templates are a wiring/calibration seed, not a powered accuracy
benchmark. Repetitions increase trials, never the independent sample size.
Public-reference search snapshots are explicitly labeled authored summaries;
URLs identify the source of facts, and collection never depends on the web.
The runner must expose fixture inputs only, keeping grader answers and expected
workspace state private until the policy's complete episode terminates.
"""
from __future__ import annotations

import copy
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .episode_study import Episode, validate_episodes

FIXTURE_VERSION = "balanced-product-seed-v1"
GRADER_VERSION = "deterministic-product-grading-v1"
SPLIT_ORDER = ("pilot", "train", "train", "train", "train", "validation", "validation", "test", "test", "smoke")
FOUR_WAY_SPLIT_ORDER = ("pilot", "train", "train", "train", "calibration", "calibration", "threshold_selection", "threshold_selection", "test", "smoke")


@dataclass(frozen=True)
class _Template:
    name: str
    prompt: str
    fixture: dict[str, Any]
    grader: dict[str, Any]
    tags: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ("text",)
    source: str = "newly_authored"


def known_template_lineage(row: Episode) -> tuple[str, ...]:
    """Expose declared lineage and the previously generated variant relationship.

    Changing numbers or function names does not create independent tasks. This
    catches the historical expansion even when its per-variant template IDs
    differ. New authored tasks still require a semantic independence review;
    this registry is a guard against known relationships, not an inference of
    independence from differing strings.
    """
    lineage: list[str] = []
    if any(re.search(r"(?:^|\.)authored_variant_\d+(?:\.|$)", identity)
           for identity in (row.template_id, row.group_id, row.episode_id)):
        lineage.append(f"known-variant-family:{row.family}")
    for field, prefix in (("lineage_id", "lineage"), ("parent_template_id", "template")):
        value = row.fixture.get(field)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"fixture {field} must be a nonempty string")
            lineage.append(f"{prefix}:{value}")
    semantic = row.fixture.get("semantic_lineage_ids", [])
    if not isinstance(semantic, list) or any(not isinstance(x, str) or not x.strip() for x in semantic):
        raise ValueError("semantic lineage must be a list of nonempty identifiers")
    lineage.extend(f"semantic:{value}" for value in semantic)
    return tuple(lineage)


def require_collection_ready(rows: Sequence[Episode]) -> None:
    """Honor acquisition/review quarantine before credentials or reservations.

    This is a guard for explicitly marked candidate corpora, not a certificate
    that historical unmarked authored fixtures are independently sampled.
    """
    pending = []
    completed_but_quarantined = []
    for row in rows:
        review = row.fixture.get("semantic_review")
        metadata = row.fixture.get("source_metadata", {})
        candidate = bool(metadata.get("independence_status")) or review is not None
        if not candidate:
            continue
        reviewed = (isinstance(review, dict)
                    and review.get("collection_ready") is True
                    and review.get("status") == "reviewed_for_collection"
                    and isinstance(review.get("review_hash"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", review["review_hash"])
                    and bool(review.get("reviewer")))
        if not reviewed:
            pending.append(row.episode_id)
            if isinstance(review, dict) and review.get("review_complete") is True:
                completed_but_quarantined.append(row.episode_id)
    if pending:
        if len(completed_but_quarantined) == len(pending):
            raise ValueError(f"semantic review complete but collection clearance missing for {len(pending)} candidate episodes; collection is quarantined")
        raise ValueError(f"semantic review incomplete for {len(pending)} candidate episodes; collection is quarantined")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2) + "\n"


def _chat() -> list[_Template]:
    messages = [
        {"role": "user", "content": "The release is on Thursday. Mira owns tests; Joel owns docs."},
        {"role": "assistant", "content": "I have the release assignments."},
        {"role": "user", "content": "Move the release to Friday and swap the owners. Keep the rest."},
    ]
    inventory = "\n".join(f"asset-{i:03d}: custodian=Team-{i % 7}; state=active" for i in range(180))
    ledger = "\n".join(f"case-{i:03d}: status=queued; owner=O{i % 9}" for i in range(170))
    return [
        _Template("unit_conversion", "A pump supplies 0.75 liters per minute for 2 hours. Return only the total number of milliliters.", {}, {"kind": "numeric", "answer": 90000}),
        _Template("conversation_revision", 'Use the conversation to return exactly a JSON object with keys "release_day", "tests_owner", and "docs_owner".', {"messages": messages}, {"kind": "json", "answer": {"release_day": "Friday", "tests_owner": "Joel", "docs_owner": "Mira"}}, ("multiturn",)),
        _Template("receipt_visual_total", "Read the attached receipt. Return only its TOTAL amount as a decimal number, excluding any currency symbol.", {"image": {"text": "HARBOR CAFE\nTea          3.25\nSandwich     7.50\nTax          0.86\nTOTAL       11.61\nCash        20.00\nChange       8.39", "fixture_origin": "authored_receipt"}}, {"kind": "numeric", "answer": 11.61}, ("vision", "receipt"), ("text", "vision")),
        _Template("structured_extraction", 'Convert this note to JSON using only keys "contact", "due", "priority". Use null for missing fields. Note: "Contact Sana. Priority is high. No due date has been set."', {}, {"kind": "json", "answer": {"contact": "Sana", "due": None, "priority": "high"}}),
        _Template("constraint_filtering", 'Return a JSON array of product codes whose stock is positive, price is under 20, and color is blue, sorted by code. Rows: C, stock 4, price 19, blue; A, stock 0, price 8, blue; B, stock 2, price 12, red; D, stock 1, price 20, blue; E, stock 3, price 9, blue.', {}, {"kind": "json", "answer": ["C", "E"]}),
        _Template("long_context_override", f"The following inventory is chronological. Later notices override earlier entries.\n{inventory}\nNOTICE: asset-041 is retired and its custodian is Archives.\nNOTICE: asset-109 is transferred to Team-8 but remains active.\nReturn a JSON object mapping asset-041 and asset-109 to their final custodian strings.", {}, {"kind": "json", "answer": {"asset-041": "Archives", "asset-109": "Team-8"}}, ("long_context",)),
        _Template("logical_scheduling", 'Three talks A, B, C occupy slots 1, 2, 3 exactly once. A is before C. B is not first. C is not second. Return the talk order as a JSON array of letters.', {}, {"kind": "json", "answer": ["A", "B", "C"]}),
        _Template("conversation_negation", 'Using the conversation, return a JSON object with "allowed" and "forbidden" arrays sorted alphabetically.', {"messages": [{"role": "user", "content": "Allow CSV and JSON exports. Forbid XML."}, {"role": "assistant", "content": "CSV and JSON are allowed; XML is forbidden."}, {"role": "user", "content": "Correction: forbid CSV, allow XML, and keep JSON allowed."}]}, {"kind": "json", "answer": {"allowed": ["JSON", "XML"], "forbidden": ["CSV"]}}, ("multiturn",)),
        _Template("schedule_visual_duration", "Use the attached schedule to find the elapsed duration of route B, in minutes. Departure and arrival are on the same day. Return only an integer.", {"image": {"text": "SHUTTLE SCHEDULE\nRoute  Departure  Arrival\nA      07:00      07:45\nB      07:35      09:10\nC      08:20      09:00", "fixture_origin": "authored_schedule"}}, {"kind": "numeric", "answer": 95}, ("vision", "table_reasoning"), ("text", "vision")),
        _Template("long_context_missing_fact", f"Use only this case log.\n{ledger}\nReturn exactly UNKNOWN if the owner of case-999 is not provided; otherwise return the owner.", {}, {"kind": "exact", "answer": "UNKNOWN"}, ("long_context", "abstention")),
    ]


def _code(name: str, specification: str, entry: str, tests: str) -> _Template:
    return _Template(name, f"Write Python 3 code defining {entry}. {specification} Use only the standard library. Return only the code, without explanations or Markdown fences.", {"entry_point": entry.split("(", 1)[0], "tests": tests}, {"kind": "code"}, ("executable", "edge_cases"), ("text", "code"))


def _coding() -> list[_Template]:
    return [
        _code("stable_unique", "Accept a list of integers and return unique values in order of first appearance without changing the input list.", "stable_unique(values)", "a=[3,1,3,2,1]; assert stable_unique(a)==[3,1,2]; assert a==[3,1,3,2,1]\nassert stable_unique([])==[]\nassert stable_unique([-1,0,-1])==[-1,0]"),
        _code("merge_intervals", "Accept unsorted closed integer intervals as pairs; merge overlapping or touching endpoints (not adjacent integers), return sorted lists [start,end], and leave the input unchanged.", "merge_intervals(intervals)", "a=[[5,7],[1,3],[3,5],[10,10]]; assert merge_intervals(a)==[[1,7],[10,10]]; assert a==[[5,7],[1,3],[3,5],[10,10]]\nassert merge_intervals([])==[]\nassert merge_intervals([[1,2],[3,4]])==[[1,2],[3,4]]"),
        _code("run_length_codec", "Implement encode_runs(text), returning a list of (character, count) tuples for consecutive Unicode characters; an empty string returns an empty list.", "encode_runs(text)", "assert encode_runs('aaabbcaa')==[('a',3),('b',2),('c',1),('a',2)]\nassert encode_runs('')==[]\nassert encode_runs('éé🙂🙂🙂')==[('é',2),('🙂',3)]"),
        _code("signed_csv_sum", "Parse comma-separated integers, ignoring whitespace and empty fields, and return their sum. Empty input returns 0. Raise ValueError for any nonempty field that is not a signed base-10 integer; decimals and underscores are invalid.", "sum_csv(text)", "assert sum_csv(' 1, -2,, +3 ')==2\nassert sum_csv(' , , ')==0\nfor bad in ('1.0','2,a','1_000'):\n    try: sum_csv(bad)\n    except ValueError: pass\n    else: raise AssertionError(bad)"),
        _code("nested_lookup", "Follow a list of dictionary keys through a nested dictionary. Return default if a key is missing or traversal meets a non-dictionary. An empty path returns the original object. Present None values remain None.", "get_path(data, keys, default=None)", "d={'a':{'b':None},'x':2}; assert get_path(d,['a','b'],'fallback') is None\nassert get_path(d,['x','y'],9)==9\nassert get_path(d,[]) is d\nassert get_path(d,['missing'],0)==0"),
        _code("rolling_average", "Return the arithmetic mean of each full consecutive window of size k. Raise ValueError when k is not a positive integer (booleans are invalid). Return [] if k exceeds input length; do not mutate the input.", "rolling_average(values, k)", "assert rolling_average([1,2,3,4],2)==[1.5,2.5,3.5]\nassert rolling_average([],2)==[]\nassert rolling_average([2],1)==[2.0]\nfor k in (0,-1,1.5,True):\n    try: rolling_average([1,2],k)\n    except ValueError: pass\n    else: raise AssertionError(k)"),
        _code("escaped_delimiter", r"Split text on commas unless a comma is escaped by a backslash. Backslash escapes the next character of any kind and is removed. Preserve a final unmatched backslash literally. Keep empty fields.", "split_escaped(text)", r"assert split_escaped('a,b,,c')==['a','b','','c']" + "\n" + r"assert split_escaped(r'a\,b,c')==['a,b','c']" + "\nassert split_escaped('')==['']\nassert split_escaped('abc'+chr(92))==['abc'+chr(92)]"),
        _code("topological_layers", "Accept a dict mapping each node to its prerequisite node names. Include prerequisite-only nodes. Return alphabetically sorted layers of nodes that can run together, consuming all nodes with no unmet prerequisite each round. Raise ValueError for a cycle.", "topological_layers(dependencies)", "assert topological_layers({'build':['lint','test'],'test':['setup'],'lint':['setup']})==[['setup'],['lint','test'],['build']]\nassert topological_layers({})==[]\ntry: topological_layers({'a':['b'],'b':['a']})\nexcept ValueError: pass\nelse: raise AssertionError('cycle accepted')"),
        _code("decimal_money", "Accept a sequence of decimal currency strings, sum using decimal arithmetic, round the final sum to two places using ROUND_HALF_UP, and return a string with exactly two decimal places. An empty sequence returns '0.00'.", "total_money(amounts)", "assert total_money(['0.10','0.20'])=='0.30'\nassert total_money(['1.005'])=='1.01'\nassert total_money(['-1.005'])=='-1.01'\nassert total_money([])=='0.00'\nassert total_money(['0.004','0.004'])=='0.01'"),
        _code("bounded_binary_search", "Return the first index i such that sorted values[i] >= target, or len(values) if none. Handle duplicates and empty inputs. Implement binary search without importing bisect.", "lower_bound(values, target)", "assert lower_bound([1,2,2,4],2)==1\nassert lower_bound([],9)==0\nassert lower_bound([1,2],3)==2\nassert lower_bound([1,2],0)==0\nassert lower_bound([2,2,2],2)==0"),
    ]


def _workspace(name: str, prompt: str, files: dict[str, str], *, expected_json: dict[str, Any] | None = None, expected_files: dict[str, str] | None = None) -> _Template:
    fixture: dict[str, Any] = {"files": files}
    if expected_json:
        fixture["expected_json"] = expected_json
    retained_files = {path: content for path, content in files.items() if path not in (expected_json or {}) and path not in (expected_files or {})}
    retained_files.update(expected_files or {})
    if retained_files:
        fixture["expected_files"] = retained_files
    return _Template(name, prompt, fixture, {"kind": "workspace"}, ("resettable_workspace", "terminal_state"), ("text", "tools"))


def _tools() -> list[_Template]:
    return [
        _workspace("csv_row_filter", "Read inventory.csv and write available.json as a JSON array of item names with qty greater than zero, preserving CSV order.", {"inventory.csv": "item,qty\npen,0\nbook,3\nclip,-1\ntape,2\n"}, expected_json={"available.json": ["book", "tape"]}),
        _workspace("json_configuration_patch", "Read config.json. Change only retries to 4 and logging.level to warning; preserve all other values. Save config.json.", {"config.json": _json({"retries": 1, "logging": {"level": "debug", "json": True}, "region": "west"})}, expected_json={"config.json": {"retries": 4, "logging": {"level": "warning", "json": True}, "region": "west"}}),
        _workspace("directory_manifest", "List only .txt input files recursively under notes. Write manifest.json as a sorted JSON array of their relative paths from the workspace root. Exclude other extensions.", {"notes/b.txt": "B\n", "notes/a.txt": "A\n", "notes/nested/c.txt": "C\n", "notes/ignore.csv": "x\n"}, expected_json={"manifest.json": ["notes/a.txt", "notes/b.txt", "notes/nested/c.txt"]}),
        _workspace("escaped_path_write", "Read the file whose literal relative path is drafts/Quarter 1 [final].txt. Convert its text to uppercase, preserve the newline, and write exports/Quarter 1.txt.", {"drafts/Quarter 1 [final].txt": "alpha & beta\n"}, expected_files={"exports/Quarter 1.txt": "ALPHA & BETA\n"}),
        _workspace("integer_ledger_total", "Sum signed integer amounts from ledger.json and write total.json as an object with sum and count. Do not modify ledger.json.", {"ledger.json": _json([120, -20, 5, 0, -7])}, expected_json={"total.json": {"sum": 98, "count": 5}, "ledger.json": [120, -20, 5, 0, -7]}),
        _workspace("markdown_heading_index", "Read README.md and write headings.json with a JSON array containing the text of only level-two headings (exactly two # followed by a space), in document order. Ignore headings inside fenced code blocks.", {"README.md": "# Guide\n## Setup\nText\n```md\n## Not a heading\n```\n### Detail\n## Usage\n"}, expected_json={"headings.json": ["Setup", "Usage"]}),
        _workspace("timestamp_latest_record", "Read events.json. For each device, select the record with greatest integer timestamp. Write latest.json mapping device to its selected value; break equal timestamps by taking the later input record.", {"events.json": _json([{"device": "a", "timestamp": 2, "value": "old"}, {"device": "b", "timestamp": 4, "value": "ok"}, {"device": "a", "timestamp": 3, "value": "new"}, {"device": "b", "timestamp": 4, "value": "updated"}])}, expected_json={"latest.json": {"a": "new", "b": "updated"}}),
        _workspace("csv_join", "Join orders.csv to customers.csv by customer_id. Write joined.json as order-id-sorted objects containing order_id and customer_name. Use null for unknown customers.", {"orders.csv": "order_id,customer_id\n3,z\n1,b\n2,a\n", "customers.csv": "customer_id,name\na,Ada\nb,Bo\n"}, expected_json={"joined.json": [{"order_id": 1, "customer_name": "Bo"}, {"order_id": 2, "customer_name": "Ada"}, {"order_id": 3, "customer_name": None}]}),
        _workspace("atomic_text_substitution", "Read template.txt and replace every literal ${NAME} with Mira. Leave $NAME and ${OTHER} untouched. Write rendered.txt.", {"template.txt": "Hello ${NAME}. $NAME ${OTHER} ${NAME}\n"}, expected_files={"rendered.txt": "Hello Mira. $NAME ${OTHER} Mira\n"}),
        _workspace("json_validation_partition", "Read input.json. Records are valid only if id is a positive integer (not boolean) and email is a string containing exactly one @ with nonempty sides. Write validation.json with valid_ids and invalid_indices in original order; indices are zero-based.", {"input.json": _json([{"id": 1, "email": "a@b"}, {"id": True, "email": "x@y"}, {"id": 2, "email": "@b"}, {"id": 3, "email": "c@d"}])}, expected_json={"validation.json": {"valid_ids": [1, 3], "invalid_indices": [1, 2]}}),
    ]


def _evidence(name: str, question: str, documents: list[tuple[str, str, str]], answer: dict[str, Any], urls: list[str]) -> _Template:
    # Fixed facts plus citation URLs give this small seed an auditable rubric;
    # keyword presence alone would accept answers that contradict the evidence.
    shape = {key: ("boolean" if isinstance(value, bool) else "number" if isinstance(value, (int, float)) else "string") for key, value in answer.items()}
    shape["citations"] = "array of supporting source URL strings"
    prompt = question + " Use only the frozen evidence. Return only one JSON object with exactly these fields and value types: " + json.dumps(shape, sort_keys=True) + "."
    return _Template(name, prompt, {"documents": [{"title": title, "url": url, "text": text} for title, url, text in documents], "snapshot_kind": "authored_summary_of_public_reference", "network_required": False}, {"kind": "json", "answer": {**answer, "citations": urls}}, ("frozen_evidence", "public_reference", "citation_support", "structured_facts"), ("text", "search"), "public_reference_with_authored_question")


def _search() -> list[_Template]:
    python = "https://docs.python.org/3/library/stdtypes.html#dict"
    sqlite = "https://www.sqlite.org/foreignkeys.html"
    http = "https://www.rfc-editor.org/rfc/rfc9110.html#name-304-not-modified"
    json_url = "https://www.rfc-editor.org/rfc/rfc8259.html#section-4"
    toml = "https://toml.io/en/v1.0.0#string"
    git = "https://git-scm.com/docs/gitignore"
    css = "https://www.w3.org/TR/css-sizing-3/#box-sizing"
    csv = "https://www.rfc-editor.org/rfc/rfc4180.html#section-2"
    robots = "https://www.rfc-editor.org/rfc/rfc9309.html#section-2.2.2"
    semver = "https://semver.org/spec/v2.0.0.html"
    return [
        _evidence("python_dict_order", "In Python 3.7+, what iteration order does a dict preserve, and does replacing an existing value move its key?", [("Python dictionary reference", python, "Insertion order is guaranteed in Python 3.7 and later. Updating an existing key does not change its position. Deleting and reinserting a key puts it at the end."), ("Set ordering distractor", "https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset", "A set is an unordered collection of distinct hashable objects.")], {"preserves_insertion_order": True, "updating_moves_key": False, "reinserting_moves_to_end": True}, [python]),
        _evidence("sqlite_foreign_key_scope", "Where must SQLite foreign-key enforcement be enabled, and can its PRAGMA switch take effect in the middle of a transaction?", [("SQLite foreign-key enforcement", sqlite, "Applications should enable foreign keys with PRAGMA foreign_keys = ON for each database connection. Changing enforcement within a multi-statement transaction has no effect.")], {"enabled_per_connection": True, "changes_inside_transaction_take_effect": False}, [sqlite]),
        _evidence("http_conditional_not_modified", "What does HTTP status 304 say about a conditional GET representation, and does the response contain representation content?", [("HTTP 304", http, "304 Not Modified indicates that a conditional GET or HEAD need not transfer the selected representation because it has not changed. A 304 response cannot contain content or trailers.")], {"status": 304, "representation_transfer_needed": False, "response_content_allowed": False}, [http]),
        _evidence("json_duplicate_keys", "Are duplicate JSON object member names recommended, and is parser behavior interoperably predictable when duplicates occur?", [("JSON object member names", json_url, "Names within an object SHOULD be unique. If names are not unique, receiving software behavior is unpredictable; implementations may report the last pair, report an error, or expose all pairs.")], {"unique_names_recommended": True, "duplicate_behavior_predictable": False}, [json_url]),
        _evidence("toml_literal_strings", "In TOML 1.0, which single-line quote form should preserve a Windows path's backslashes without escaping, and what is the limiting quote character?", [("TOML strings", toml, "Literal strings are enclosed by single quotes and do not support escaping. They preserve content literally but cannot contain an unescaped single quote. Basic double-quoted strings process backslash escape sequences.")], {"literal_single_quotes_preserve_backslashes": True, "single_quote_allowed_inside_single_line_literal": False}, [toml]),
        _evidence("gitignore_tracked_files", "Will adding a tracked file's name to .gitignore make Git stop tracking it, and what command removes only its index entry?", [("Git ignore patterns", git, "Ignore patterns apply to intentionally untracked files. Files already tracked by Git are unaffected. To stop tracking a file while retaining the working-tree copy, use git rm --cached.")], {"ignore_pattern_stops_tracking": False, "index_only_removal_command": "git rm --cached"}, [git]),
        _evidence("css_border_box", "If a CSS box has width 100px, padding 10px on each side, border 2px on each side, and box-sizing:border-box, what is its content width?", [("CSS box sizing", css, "With border-box, specified width sizes the border box. Content size subtracts padding and borders; margins remain outside. The content box cannot have negative size.")], {"content_width_px": 76, "margin_included_in_width": False}, [css]),
        _evidence("csv_embedded_quote", "According to RFC 4180, how should a field containing both a comma and a double quote be escaped? Give the encoding of: Ada said \"hi\", today", [("CSV field escaping", csv, "Fields containing commas, line breaks, or double quotes should be enclosed in double quotes. Each double quote within an enclosed field must be escaped by a second double quote.")], {"encoded_field": '"Ada said ""hi"", today"'}, [csv]),
        _evidence("robots_rule_precedence", "For a robots.txt path that matches both Allow and Disallow rules, which matching rule wins, and what happens if their matched lengths are equivalent?", [("Robots matching rules", robots, "The most specific matching rule is the rule with the most octets. If an Allow rule and a Disallow rule are equivalent, the Allow rule SHOULD be used. Path matching is case-sensitive.")], {"longest_matching_rule_wins": True, "allow_wins_equal_length": True, "path_case_sensitive": True}, [robots]),
        _evidence("semantic_version_compatibility", "Under SemVer 2.0.0 for stable releases, which version component changes for a backward-incompatible public API change, and what special caveat applies to 0.y.z?", [("Semantic Versioning", semver, "For version X.Y.Z with X greater than zero, incompatible public API changes require incrementing the major version. Major version zero is for initial development; anything may change, and the public API should not be considered stable.")], {"stable_incompatible_change_component": "major", "zero_major_api_stable": False}, [semver]),
    ]


def _agents() -> list[_Template]:
    return [
        _workspace("inventory_replenishment_project", "Complete the inventory project: read data/stock.csv and policy.json; calculate max(0,target-stock) for active items only, omit zero orders, and write reports/orders.json sorted by sku. Also write reports/summary.txt exactly as 'units: N\n' using the total units ordered.", {"data/stock.csv": "sku,stock,active\nA,3,true\nB,12,true\nC,0,false\nD,1,true\n", "policy.json": _json({"target": 8})}, expected_json={"reports/orders.json": [{"sku": "A", "units": 5}, {"sku": "D", "units": 7}]}, expected_files={"reports/summary.txt": "units: 12\n"}),
        _workspace("invoice_reconciliation_project", "Reconcile invoices with payments by invoice ID using integer cents. Write output/balances.json sorted by invoice ID with balance_cents, retaining overpayments as negative. Write output/unmatched.json containing payment IDs whose invoice is unknown. Preserve input files.", {"data/invoices.json": _json([{"id": "I2", "cents": 500}, {"id": "I1", "cents": 1000}]), "data/payments.json": _json([{"id": "P1", "invoice": "I1", "cents": 300}, {"id": "P2", "invoice": "I2", "cents": 600}, {"id": "P3", "invoice": "I9", "cents": 25}])}, expected_json={"output/balances.json": [{"id": "I1", "balance_cents": 700}, {"id": "I2", "balance_cents": -100}], "output/unmatched.json": ["P3"]}),
        _workspace("documentation_link_audit", "Audit local Markdown links in docs/index.md relative to that file; ignore https URLs and strip #fragments before checking existence. Write reports/links.json with sorted existing and missing target arrays. Write reports/status.txt as PASS or FAIL followed by a newline, FAIL if any target is missing.", {"docs/index.md": "[Guide](guide.md#intro)\n[Missing](missing.md)\n[Web](https://example.org)\n[Spec](specs/api.md)\n", "docs/guide.md": "# Intro\n", "docs/specs/api.md": "# API\n"}, expected_json={"reports/links.json": {"existing": ["guide.md", "specs/api.md"], "missing": ["missing.md"]}}, expected_files={"reports/status.txt": "FAIL\n"}),
        _workspace("record_migration_project", "Migrate data/users-v1.json to data/users-v2.json: id stays unchanged; name becomes trimmed display_name; enabled defaults to true only when absent (preserve false); remove all legacy_note fields. Sort users by id. Update schema.json version to 2, preserving its other fields.", {"data/users-v1.json": _json([{"id": 2, "name": " Bo ", "enabled": False, "legacy_note": "old"}, {"id": 1, "name": " Ada "}]), "schema.json": _json({"version": 1, "owner": "ops"})}, expected_json={"data/users-v2.json": [{"id": 1, "display_name": "Ada", "enabled": True}, {"id": 2, "display_name": "Bo", "enabled": False}], "schema.json": {"version": 2, "owner": "ops"}}),
        _workspace("incident_log_report", "Analyze logs/app.jsonl. Count ERROR events by service, count malformed JSON lines, and count WARN events separately. Write report.json with error_by_service, malformed_lines, and warning_count. Write affected.txt with alphabetically sorted services that had ERROR events, one per line.", {"logs/app.jsonl": '{"service":"api","level":"ERROR"}\nnot json\n{"service":"worker","level":"WARN"}\n{"service":"api","level":"ERROR"}\n{"service":"worker","level":"ERROR"}\n'}, expected_json={"report.json": {"error_by_service": {"api": 2, "worker": 1}, "malformed_lines": 1, "warning_count": 1}}, expected_files={"affected.txt": "api\nworker\n"}),
        _workspace("release_manifest_project", "Prepare release metadata from project.toml and CHANGELOG.md. Write dist/release.json with name and version from TOML plus notes: the bullet texts only under the changelog heading matching that version, without '- '. Write VERSION with the version and newline. Do not include older release notes.", {"project.toml": '[project]\nname = "lantern"\nversion = "2.1.0"\n', "CHANGELOG.md": "# Changes\n## 2.1.0\n- Fix Unicode export\n- Add retry limit\n## 2.0.0\n- Old change\n"}, expected_json={"dist/release.json": {"name": "lantern", "version": "2.1.0", "notes": ["Fix Unicode export", "Add retry limit"]}}, expected_files={"VERSION": "2.1.0\n"}),
        _workspace("contact_deduplication_project", "Merge contacts from incoming/a.json and incoming/b.json by trimmed lowercase email. Process a then b; later nonempty names replace earlier names, empty names do not. Write clean/contacts.json sorted by email with name and email only. Write clean/stats.json with input_count and unique_count.", {"incoming/a.json": _json([{"email": " A@EXAMPLE.ORG ", "name": "Ada"}, {"email": "b@example.org", "name": "Bo"}]), "incoming/b.json": _json([{"email": "a@example.org", "name": "Ada L"}, {"email": "b@example.org", "name": ""}, {"email": "c@example.org", "name": "Cy"}])}, expected_json={"clean/contacts.json": [{"email": "a@example.org", "name": "Ada L"}, {"email": "b@example.org", "name": "Bo"}, {"email": "c@example.org", "name": "Cy"}], "clean/stats.json": {"input_count": 5, "unique_count": 3}}),
        _workspace("deployment_readiness_project", "Check services.json against release.json. A service is ready only when its version equals the required release version and healthy_replicas is at least its minimum_replicas. Write reports/readiness.json mapping each service name to a boolean. Write reports/blocked.txt listing the names of unready services alphabetically, one per line. Preserve inputs.", {"services.json": _json([{"name": "api", "version": "3.2", "healthy_replicas": 2, "minimum_replicas": 2}, {"name": "worker", "version": "3.1", "healthy_replicas": 4, "minimum_replicas": 2}, {"name": "cache", "version": "3.2", "healthy_replicas": 0, "minimum_replicas": 1}]), "release.json": _json({"version": "3.2"})}, expected_json={"reports/readiness.json": {"api": True, "worker": False, "cache": False}}, expected_files={"reports/blocked.txt": "cache\nworker\n"}),
        _workspace("multi_currency_sales_report", "Read data/sales.csv and rates.json. Convert each sale to USD cents by multiplying its integer cents by that currency's integer rate; exclude refunded rows. Write reports/totals.json mapping customer to total USD cents, and reports/ranking.json as customer names sorted by descending total, then name for ties.", {"data/sales.csv": "customer,cents,currency,refunded\nAda,200,EUR,false\nBo,300,USD,false\nAda,100,USD,true\nCy,150,EUR,false\n", "rates.json": _json({"EUR": 2, "USD": 1})}, expected_json={"reports/totals.json": {"Ada": 400, "Bo": 300, "Cy": 300}, "reports/ranking.json": ["Ada", "Bo", "Cy"]}),
        _workspace("configuration_overlay_project", "Combine config/base.json with config/prod.json using recursive dictionary merging: overlay scalar/list values replace base values; explicit null also replaces. Write build/config.json. Write build/keys.txt listing final top-level keys alphabetically, one per line. Preserve both input files.", {"config/base.json": _json({"server": {"port": 80, "host": "localhost"}, "flags": ["debug"], "cache": True}), "config/prod.json": _json({"server": {"port": 443}, "flags": [], "cache": None, "region": "east"})}, expected_json={"build/config.json": {"server": {"port": 443, "host": "localhost"}, "flags": [], "cache": None, "region": "east"}}, expected_files={"build/keys.txt": "cache\nflags\nregion\nserver\n"}),
    ]


def build_seed_corpus(*, repeats: int = 1, splits: Sequence[str] | None = None) -> list[Episode]:
    """Build 50 semantic groups; each family has 1/4/2/2/1 split allocation.

    Optional ``repeats`` creates identical input trials inside the same group.
    Optional ``splits`` filters *after* the immutable template allocation, so a
    caller can materialize only development fixtures without opening test data.
    Expanding independent sample size requires adding new semantic templates
    and a new version, never renaming the same task or changing its split.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    allowed = set(SPLIT_ORDER)
    requested = allowed if splits is None else set(splits)
    if not requested or not requested <= allowed:
        raise ValueError("splits must contain known nonempty split names")
    families = {"chat": _chat(), "coding": _coding(), "tools": _tools(), "search": _search(), "agents": _agents()}
    rows: list[Episode] = []
    for family, templates in families.items():
        for template, split in zip(templates, SPLIT_ORDER, strict=True):
            if split not in requested:
                continue
            identity = f"{family}.{template.name}"
            for trial in range(repeats):
                rows.append(Episode(
                    episode_id=f"{identity}.trial-{trial + 1}", group_id=identity,
                    family=family, split=split, prompt=template.prompt,
                    fixture=copy.deepcopy(template.fixture), grader=copy.deepcopy(template.grader),
                    capabilities=template.capabilities,
                    tags=template.tags + ("seed_corpus", "underpowered", "repeated_trial" if repeats > 1 else "single_trial"),
                    fixture_version=FIXTURE_VERSION, grader_version=GRADER_VERSION,
                    source=template.source, template_id=identity,
                    max_calls={"chat": 8, "coding": 8, "tools": 6, "search": 2, "agents": 12}[family],
                    max_output_tokens=4096 if family == "coding" else 2048,
                    max_input_tokens=24000, max_seconds=180 if family in {"chat", "coding", "search"} else 300,
                ))
    validate_episodes(rows)
    return rows


def build_four_way_seed_corpus(*, repeats: int = 1, splits: Sequence[str] | None = None) -> list[Episode]:
    """Build the underpowered 50-template seed with four disjoint study splits.

    Model-fit support cannot be manufactured by renaming or parameterizing a
    template. Expansion requires reviewed independent tasks and a new version.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    allowed = set(FOUR_WAY_SPLIT_ORDER)
    requested = allowed if splits is None else set(splits)
    if not requested or not requested <= allowed:
        raise ValueError("splits must contain known four-way split names")
    families = {"chat": _chat(), "coding": _coding(), "tools": _tools(), "search": _search(), "agents": _agents()}
    rows: list[Episode] = []
    for family, templates in families.items():
        for template, split in zip(templates, FOUR_WAY_SPLIT_ORDER, strict=True):
            if split not in requested:
                continue
            identity = f"{family}.{template.name}"
            for trial in range(repeats):
                rows.append(Episode(
                    episode_id=f"{identity}.trial-{trial + 1}", group_id=identity,
                    family=family, split=split, prompt=template.prompt,
                    fixture=copy.deepcopy(template.fixture), grader=copy.deepcopy(template.grader),
                    capabilities=template.capabilities,
                    tags=template.tags + ("seed_corpus_v2", "underpowered", "repeated_trial" if repeats > 1 else "single_trial"),
                    fixture_version=FIXTURE_VERSION + "-four-way-seed-v2", grader_version=GRADER_VERSION,
                    source=template.source, template_id=identity,
                    max_calls={"chat": 8, "coding": 8, "tools": 6, "search": 2, "agents": 12}[family],
                    max_output_tokens=4096 if family == "coding" else 2048,
                    max_input_tokens=24000, max_seconds=180 if family in {"chat", "coding", "search"} else 300,
                ))
    validate_episodes(rows)
    return rows


def corpus_manifest(rows: Sequence[Episode]) -> dict[str, Any]:
    """Describe independent evidence without representing this seed as powered."""
    result = validate_episodes(rows)
    return {
        **result,
        "fixture_versions": sorted({row.fixture_version for row in rows}),
        "fixture_version": FIXTURE_VERSION,
        "independence_scope": "authored template groups; semantic review required for expansion",
        "purpose": "reproducible authored implementation corpus; empirical power is not claimed",
        "powered_for_joint_acceptance": False,
        "public_content": "frozen_authored_summaries_of_public_references_with_source_urls",
        "live_route_evidence": False,
        "imported_public_benchmark_items": 0,
        "browser_environment_coverage": "not_covered_project_agent_fixtures_only",
        "grading_scope": "objective_structured_tasks_not_open_ended_quality",
        "independent_groups_by_family": {
            family: len({row.group_id for row in rows if row.family == family})
            for family in ("chat", "coding", "tools", "search", "agents")
        },
    }
