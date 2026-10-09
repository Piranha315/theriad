#!/usr/bin/env python3
"""Train the Router V2 surface baseline from grouped JSONL outcomes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.v2_training import (  # noqa: E402
    StudyRow,
    assert_study_row_contract,
    train_surface_artifact,
)

DEFAULT_GENERAL_MODELS = (
    "nvidia/nemotron-3-super-120b-a12b:free",
    "xiaomi/mimo-v2.5",
    "openai/gpt-5.6-luna",
    "x-ai/grok-4.6",
)
DEFAULT_CODE_MODELS = (
    "poolside/laguna-xs-2.1:free",
    "deepseek/deepseek-v4-flash-0731",
    "openai/gpt-5.6-luna",
    "x-ai/grok-4.6",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Grouped study JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Router artifact JSON")
    parser.add_argument("--general-fallback", default="openai/gpt-5.6-luna")
    parser.add_argument("--code-fallback", default="openai/gpt-5.6-luna")
    parser.add_argument("--general-model", action="append")
    parser.add_argument("--code-model", action="append")
    args = parser.parse_args()

    rows: list[StudyRow] = []
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(StudyRow.from_dict(json.loads(line)))
                except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise SystemExit(f"{args.input}:{line_number}: {exc}") from exc
    try:
        assert_study_row_contract(rows)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    artifact = train_surface_artifact(
        rows,
        fallback_models={"general": args.general_fallback, "code": args.code_fallback},
        active_models_by_lane={
            "general": args.general_model or DEFAULT_GENERAL_MODELS,
            "code": args.code_model or DEFAULT_CODE_MODELS,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "router_version": artifact["router_version"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
