from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel

from .browser_agent import BrowserAgentStartRequest, STORE
from .intent import ChatIntent, classify_chat_intent
from .tool_broker import ToolBroker
from theriad.context_layer import SecondBrainService, no_results_answer


class ChatRequest(BaseModel):
    message: str
    project_root: str | None = None


class ChatResponse(BaseModel):
    message_id: str
    content: str
    intent: ChatIntent
    agent_run_id: str | None = None
    status: str
    trace: list[dict[str, Any]] = []


# Compatibility facade: chat now creates the same session used by the product
# browser API instead of a chat-only stub runtime.
browser_runtime = STORE
agent_debug_trace: dict[str, dict[str, Any]] = {}
app = FastAPI(title="Theriad Chat Agents")


def handle_chat_message(message: str, project_root: str | None = None) -> ChatResponse:
    parsed, context_results = SecondBrainService().recall_for_message(message)
    if parsed.requiresRetrieval:
        message_id = f"chat-{uuid4().hex}"
        if not context_results:
            return ChatResponse(
                message_id=message_id,
                content=no_results_answer(parsed),
                intent=ChatIntent.NORMAL_CHAT,
                status="completed",
                trace=[
                    {
                        "type": "second_brain.search",
                        "query": parsed.rewrittenQuery,
                        "sources": parsed.sources,
                        "results": 0,
                    }
                ],
            )
        content = _format_context_recall_answer(context_results)
        return ChatResponse(
            message_id=message_id,
            content=content,
            intent=ChatIntent.NORMAL_CHAT,
            status="completed",
            trace=[
                {
                    "type": "second_brain.search",
                    "query": parsed.rewrittenQuery,
                    "sources": parsed.sources,
                    "results": len(context_results),
                    "memory_ids": [result.id for result in context_results],
                }
            ],
        )

    intent = classify_chat_intent(message)
    message_id = f"chat-{uuid4().hex}"
    if intent == ChatIntent.NORMAL_CHAT:
        return ChatResponse(message_id=message_id, content="Handled as normal chat.", intent=intent, status="completed")

    if intent in (ChatIntent.BROWSER_AGENT, ChatIntent.MULTI_TOOL_AGENT):
        session = STORE.start(BrowserAgentStartRequest(goal=message))
        agent_debug_trace[session.session_id] = {"session": session, "trace": session.activity_log}
        return ChatResponse(
            message_id=message_id,
            content="Browser agent run created from chat.",
            intent=intent,
            agent_run_id=session.session_id,
            status=session.status.value,
            trace=[{"status": item.status, "message": item.message, "tool": item.tool} for item in session.activity_log],
        )

    root = Path(project_root or ".").resolve()
    broker = ToolBroker(root, terminal_mode="mock")
    result = broker.call("terminal.run_mock", command=_suggest_terminal_command(message))
    run_id = f"terminal-{uuid4().hex}"
    trace = [{"status": "Running command", "tool": result.tool, "command": result.metadata["command"], "mock": True}]
    agent_debug_trace[run_id] = {"trace": trace, "tools": broker.log}
    return ChatResponse(
        message_id=message_id,
        content=result.output,
        intent=intent,
        agent_run_id=run_id,
        status="waiting_for_approval" if result.requires_approval else "completed",
        trace=trace,
    )


def _suggest_terminal_command(message: str) -> str:
    text = message.lower()
    if "git status" in text:
        return "git status"
    if "npm build" in text or "build" in text:
        return "npm run build"
    if "pytest" in text or "run tests" in text:
        return "python -m pytest"
    return "pwd"


def _format_context_recall_answer(results: list[Any]) -> str:
    lines = ["Here’s what I found in imported context:"]
    for result in results:
        provenance = result.source.provider or result.source.type
        if result.source.sourceFileName:
            provenance = f"{provenance} / {result.source.sourceFileName}"
        lines.append(f"- {result.title}: {result.summary} ({provenance})")
    return "\n".join(lines)


@app.post("/api/chat/send", response_model=ChatResponse)
def chat_send(request: ChatRequest) -> ChatResponse:
    return handle_chat_message(request.message, request.project_root)


@app.post("/api/chat/agent", response_model=ChatResponse)
def chat_agent(request: ChatRequest) -> ChatResponse:
    return handle_chat_message(request.message, request.project_root)


@app.get("/api/agent/{agent_run_id}/trace")
def agent_trace(agent_run_id: str) -> dict[str, Any]:
    return agent_debug_trace.get(agent_run_id, {"trace": []})
