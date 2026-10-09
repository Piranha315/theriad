from pathlib import Path


THERIAD = Path("theriad-ui/src/Theriad.jsx").read_text(encoding="utf-8")
AUTH = Path("theriad-ui/src/auth.js").read_text(encoding="utf-8")


def test_email_code_challenge_gates_cached_guest_shell():
    """Google email verification must render even while a guest is cached."""
    assert "if (authModal?.type === 'mfa')" in THERIAD
    assert "if (!user && authModal?.type === 'mfa')" not in THERIAD


def test_email_code_challenge_has_one_modal_render_path():
    """Competing modal paths can lose the OAuth-return challenge on remount."""
    assert THERIAD.count("mfa={authModal}") == 1


def test_registered_session_wins_over_persisted_google_session():
    """A returning account must not receive a new Google code on every reload."""
    existing_session_guard = "if (existing && !existing.is_guest) return existing;"
    assert existing_session_guard in AUTH
    assert AUTH.index(existing_session_guard) < AUTH.index("if (supabase) {")
