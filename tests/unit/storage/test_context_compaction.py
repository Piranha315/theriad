from theriad.storage.context import build_context


def test_old_history_is_summarized_while_recent_turns_stay_verbatim():
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"turn-{index} " * 180}
        for index in range(18)
    ]

    result = build_context(messages=messages, max_context_tokens=1_500)

    assert result.history_summarized is True
    assert result.messages_summarized > 0
    assert result.recent_messages_kept >= 2
    assert "Earlier conversation, compressed locally" in result.text
    assert "turn-17" in result.text


def test_default_context_preserves_large_file_without_application_cap():
    result = build_context(file_texts=[("notes.md", "head " * 2_000 + "tail-marker")])

    assert result.truncated is False
    assert result.budget_tokens is None
    assert "tail-marker" in result.text
