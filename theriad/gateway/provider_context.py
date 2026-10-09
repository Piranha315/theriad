"""Opaque Google tool state, bound to the issuing deployment and exact calls."""

from __future__ import annotations

import copy
import hashlib

from .contracts import GatewayError

CONTEXT_KIND = "theriad_google_tool_context_v1"
CONTEXT_ID_PREFIX = "rs_theriad_"


def merge_extra_content(target, incoming):
    if incoming is None:
        return
    if not isinstance(incoming, dict):
        raise GatewayError("upstream_protocol_error", "Malformed provider context", 502)
    for key, value in incoming.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            merge_extra_content(target[key], value)
        elif key in target and target[key] != value:
            raise GatewayError("upstream_protocol_error", "Conflicting provider context", 502)
        else:
            target[key] = copy.deepcopy(value)


def arguments_hash(arguments):
    return hashlib.sha256(arguments.encode()).hexdigest()


def bind_calls(calls, binding):
    return {
        call["call_id"]: {
            **binding,
            "name": call["name"],
            "arguments_hash": arguments_hash(call["arguments"]),
            "extra_content": copy.deepcopy(call["extra_content"]),
        }
        for call in calls
        if call.get("extra_content")
    }


def restore_calls(messages, bindings):
    for message in messages:
        for call in message.get("tool_calls") or []:
            context = bindings.get(call["id"])
            if not context:
                continue
            function = call["function"]
            if (
                function["name"] != context["name"]
                or arguments_hash(function["arguments"]) != context["arguments_hash"]
            ):
                raise GatewayError(
                    "provider_context_mismatch", "Provider context does not match tool call", 409
                )
            call["extra_content"] = copy.deepcopy(context["extra_content"])
