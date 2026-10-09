from enum import StrEnum


class ChatIntent(StrEnum):
    NORMAL_CHAT = "normal_chat"
    BROWSER_AGENT = "browser_agent"
    TERMINAL_AGENT = "terminal_agent"
    MULTI_TOOL_AGENT = "multi_tool_agent"

BROWSER_CUES = (
    "go to", "open website", "visit", "browse", "click", "webpage",
    "summarize this site", "inspect page",
)
TERMINAL_CUES = (
    "run tests", "run command", "pytest", "npm build", "git status",
    "inspect repo", "fix failing test",
)


def classify_chat_intent(prompt: str) -> ChatIntent:
    text = prompt.lower()
    browser = any(cue in text for cue in BROWSER_CUES)
    terminal = any(cue in text for cue in TERMINAL_CUES)
    if browser and terminal:
        return ChatIntent.MULTI_TOOL_AGENT
    if browser:
        return ChatIntent.BROWSER_AGENT
    if terminal:
        return ChatIntent.TERMINAL_AGENT
    return ChatIntent.NORMAL_CHAT
