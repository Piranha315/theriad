"""The artifact preview must run JavaScript without escaping its sandbox.

Scripts used to be gated on the "Tailwind CDN" checkbox in two places at once
(the iframe sandbox attribute and the CSP script-src). With Tailwind off there
was no script-src at all, so `default-src 'none'` blocked every script and a
generated app rendered its <noscript> fallback instead of running.
"""
from pathlib import Path

CHAT = Path("theriad-ui/src/Chat.jsx").read_text(encoding="utf-8")
THERIAD = Path("theriad-ui/src/Theriad.jsx").read_text(encoding="utf-8")
PREVIEW = Path("theriad-ui/src/previewArtifact.js").read_text(encoding="utf-8")


def test_scripts_are_not_gated_on_the_tailwind_toggle():
    """Tailwind is a styling choice; it must not double as a JS master switch."""
    for source in (CHAT, THERIAD):
        assert "tailwindEnabled ? 'allow-scripts' : ''" not in source


def test_every_preview_iframe_allows_scripts():
    assert CHAT.count("const sandbox = 'allow-scripts';") == 2
    assert THERIAD.count("const sandbox = 'allow-scripts';") == 1


def _csp_branches() -> list[str]:
    """Return the two CSP template literals, ignoring surrounding comments."""
    return [
        line for line in PREVIEW.splitlines()
        if "default-src 'none'" in line
        and "`" in line
        and not line.lstrip().startswith("//")
    ]


def test_csp_allows_inline_scripts_in_both_branches():
    """The non-Tailwind branch needs its own script-src, not just Tailwind's."""
    branches = _csp_branches()
    assert len(branches) == 2, branches
    for branch in branches:
        assert "script-src 'unsafe-inline'" in branch, branch


def test_sandbox_never_grants_same_origin():
    """allow-scripts + allow-same-origin lets a frame remove its own sandbox.

    Checks the sandbox values actually assigned, so a comment naming the
    dangerous token does not trip the assertion.
    """
    for source in (CHAT, THERIAD):
        for line in source.splitlines():
            if "const sandbox =" in line or "sandbox=\"" in line:
                assert "allow-same-origin" not in line, line


def test_preview_still_blocks_network_exfiltration():
    branches = _csp_branches()
    assert len(branches) == 2, branches
    for branch in branches:
        assert "connect-src 'none'" in branch, branch


def test_toggling_tailwind_remounts_the_inspector_iframe():
    """A sandbox attribute only applies at load, so the key must change."""
    assert "key={`${preview?.sourceKey}_${tailwindEnabled}`}" in CHAT
