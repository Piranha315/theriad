"""Backend-managed provider settings and API-key storage.

Anonymous local requests use ``~/.theriad/secrets.json`` for compatibility.
Requests carrying a Theriad identity context use a separate file under
``~/.theriad/secrets-users/`` so one account cannot read another account's
saved BYOK settings. Environment variables take precedence over file-backed
values, and public status helpers expose only a masked suffix -- never the
full credential.

Provider credentials are encrypted individually before they are written. A
hosted deployment should inject ``THERIAD_SECRET_ENCRYPTION_KEY`` from its
managed secret store; local development creates a user-only Fernet key beside
the settings directory so old installations can migrate without a manual
secret rewrite. Legacy plaintext values remain readable for migration, but
new writes never put provider credentials in the JSON file as plaintext.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import threading
from pathlib import Path
from typing import Literal, TypedDict, cast

from theriad.core.logging.logger import get_logger

logger = get_logger("settings.secrets")

ProviderName = Literal["openrouter", "openai", "anthropic"]
AssistantProvider = Literal["theriad", "openai", "anthropic"]

SUPPORTED_PROVIDERS: tuple[ProviderName, ...] = (
    "openrouter",
    "openai",
    "anthropic",
)
SUPPORTED_ASSISTANT_PROVIDERS: tuple[AssistantProvider, ...] = (
    "theriad",
    "openai",
    "anthropic",
)

_THERIAD_DIR = Path.home() / ".theriad"
_SECRETS_FILE = _THERIAD_DIR / "secrets.json"
_SECRETS_LOCK = threading.RLock()

_PROVIDER_SETTINGS: dict[ProviderName, tuple[str, str]] = {
    "openrouter": ("OPENROUTER_API_KEY", "openrouter_api_key"),
    "openai": ("OPENAI_API_KEY", "openai_api_key"),
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic_api_key"),
}
_ACTIVE_ASSISTANT_PROVIDER = "active_assistant_provider"
_DEFAULT_ASSISTANT_PROVIDER: AssistantProvider = "theriad"
_MASK_CHAR = "•"
_VISIBLE_SUFFIX_LEN = 4
_MAX_API_KEY_LENGTH = 8192
_SCOPE_UNSET = object()
_ENCRYPTED_VALUE_PREFIX = "enc:v1:"


class SecretStorageError(ValueError):
    """Raised when the configured local/managed encryption boundary is unavailable."""


class KeyStatus(TypedDict):
    configured: bool
    source: str  # "env" | "local_settings" | "missing"
    masked_key: str | None


class ProviderSettingsStatus(TypedDict):
    active_provider: AssistantProvider
    providers: dict[ProviderName, KeyStatus]


# -- Internal file helpers ----------------------------------------------------


def _request_scope_user_id() -> str | None:
    """Read the request-local identity without importing auth at module load."""
    try:
        from theriad.api.auth_context import current_auth_context
    except ImportError:  # pragma: no cover - defensive for standalone imports
        return None
    context = current_auth_context.get()
    return context.user_id if context is not None else None


def _secrets_file(scope_id: str | None | object = _SCOPE_UNSET) -> Path:
    """Return the global or per-user secret file for the current request."""
    if scope_id is _SCOPE_UNSET:
        scope_id = _request_scope_user_id()
    if isinstance(scope_id, str) and scope_id:
        digest = hashlib.sha256(scope_id.encode("utf-8")).hexdigest()[:32]
        return _THERIAD_DIR / "secrets-users" / f"{digest}.json"
    return _SECRETS_FILE


def _ensure_dir(directory: Path | None = None) -> None:
    """Create a settings directory and enforce user-only permissions."""
    directory = directory or _THERIAD_DIR
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(directory, stat.S_IRWXU)
    except OSError:
        # Some filesystems do not expose POSIX permissions.  The subsequent
        # write may still succeed; callers receive that write error if it does not.
        pass


def _fernet():
    """Return the Fernet cipher backed by the injected or local key."""
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject.toml
        raise SecretStorageError(
            "Encrypted secret storage is unavailable on this server."
        ) from exc

    configured = os.environ.get("THERIAD_SECRET_ENCRYPTION_KEY", "").strip()
    if configured:
        try:
            return Fernet(configured.encode("ascii"))
        except (ValueError, UnicodeEncodeError) as exc:
            raise SecretStorageError("THERIAD_SECRET_ENCRYPTION_KEY is invalid.") from exc

    # Local-first fallback: keep the key user-readable only. Hosted operations
    # should inject the same key from a managed secret store so a new worker
    # can decrypt existing account settings without copying a key file around.
    key_path = _THERIAD_DIR / "secret-store.key"
    _ensure_dir(key_path.parent)
    try:
        key = key_path.read_bytes().strip()
    except FileNotFoundError:
        key = Fernet.generate_key()
        fd = -1
        try:
            fd = os.open(
                key_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                stat.S_IRUSR | stat.S_IWUSR,
            )
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(key)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
        except FileExistsError:
            key = key_path.read_bytes().strip()
        except OSError as exc:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise SecretStorageError(
                "Encrypted secret storage could not initialize its local key."
            ) from exc
    try:
        os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)
        return Fernet(key)
    except (OSError, ValueError) as exc:
        raise SecretStorageError("Encrypted secret storage has an invalid local key.") from exc


def _encrypt_secret_value(value: str) -> str:
    if value.startswith(_ENCRYPTED_VALUE_PREFIX):
        return value
    encrypted = _fernet().encrypt(value.encode("utf-8")).decode("ascii")
    return _ENCRYPTED_VALUE_PREFIX + encrypted


def _decrypt_secret_value(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if not value.startswith(_ENCRYPTED_VALUE_PREFIX):
        # Read legacy plaintext values so an upgrade does not silently lose a
        # user's local key. The next save rewrites the value encrypted.
        return value
    try:
        encrypted = value.removeprefix(_ENCRYPTED_VALUE_PREFIX).encode("ascii")
        return _fernet().decrypt(encrypted).decode("utf-8").strip() or None
    except Exception:
        # Never return an undecipherable credential or surface cipher details.
        return None


def _read_secrets(path: Path | None = None) -> dict[str, object]:
    """Read the settings file, returning an empty mapping on malformed input."""
    path = path or _secrets_file()
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _write_secrets(data: dict[str, object], path: Path | None = None) -> None:
    """Atomically write settings with user-only file permissions."""
    path = path or _secrets_file()
    _ensure_dir(path.parent)
    fd = -1
    tmp_path: Path | None = None
    try:
        fd, raw_tmp_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        tmp_path = Path(raw_tmp_path)
        # mkstemp creates the file as 0600; enforce it explicitly for clarity
        # and for platforms with unusual default permission behavior.
        os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1  # fdopen owns the descriptor from this point onward.
            json.dump(data, handle)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        tmp_path.replace(path)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


# -- Validation and masking ---------------------------------------------------


def _normalize_provider(provider: str) -> ProviderName:
    normalized = provider.strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    return cast(ProviderName, normalized)


def _normalize_assistant_provider(provider: str) -> AssistantProvider:
    normalized = provider.strip().lower()
    # Accept the implementation-provider name as a compatibility alias while
    # keeping the user-facing preference canonical as "theriad".
    if normalized == "openrouter":
        normalized = "theriad"
    if normalized not in SUPPORTED_ASSISTANT_PROVIDERS:
        raise ValueError(f"Unsupported assistant provider: {provider}")
    return cast(AssistantProvider, normalized)


def _normalize_api_key(key: str) -> str:
    if not isinstance(key, str):
        raise ValueError("API key must be a string")
    normalized = key.strip()
    if not normalized:
        raise ValueError("API key must not be empty")
    if len(normalized) > _MAX_API_KEY_LENGTH:
        raise ValueError("API key is too long")
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in normalized):
        raise ValueError("API key must not contain whitespace or control characters")
    return normalized


def _mask_key(key: str) -> str:
    """Return a masked key with only its final four characters visible."""
    if not key:
        return ""
    if len(key) <= _VISIBLE_SUFFIX_LEN:
        return _MASK_CHAR * len(key)
    suffix = key[-_VISIBLE_SUFFIX_LEN:] if len(key) >= _VISIBLE_SUFFIX_LEN else key
    prefix_len = max(0, len(key) - len(suffix))
    return _MASK_CHAR * min(prefix_len, 12) + suffix


# -- Generic provider key API -------------------------------------------------


def _env_api_key(env_name: str) -> str:
    """Read a provider key from the environment, repairing one common corruption.

    macOS `security find-generic-password -w` prints a secret as hex when it holds a
    non-printable byte (typically a trailing newline), so a login script that exports
    its output hands us the hex of the key instead of the key. Left alone, that value
    shadows a correct .env entry and every provider call fails with a 401. A value that
    is entirely hex and decodes to text starting with "sk-" is decoded and stripped.
    """
    value = os.environ.get(env_name, "").strip()
    if len(value) >= 12 and len(value) % 2 == 0 and value.lower().startswith("736b2d"):
        try:
            decoded = bytes.fromhex(value).decode("utf-8").strip()
        except ValueError:
            return value
        if decoded.startswith("sk-") and decoded.isprintable():
            logger.warning(
                "provider key in environment was hex-encoded; decoded it",
                extra={"env": env_name},
            )
            return decoded
    return value


def get_provider_api_key(provider: str) -> str | None:
    """Resolve a provider key from its environment variable, then local settings."""
    provider_name = _normalize_provider(provider)
    env_name, file_key_name = _PROVIDER_SETTINGS[provider_name]

    env_key = _env_api_key(env_name)
    if env_key:
        return env_key

    with _SECRETS_LOCK:
        secrets = _read_secrets()
    return _decrypt_secret_value(secrets.get(file_key_name, ""))


def save_provider_api_key(
    provider: str, key: str, *, scope_id: str | None | object = _SCOPE_UNSET
) -> None:
    """Save one provider key without modifying other provider settings."""
    provider_name = _normalize_provider(provider)
    normalized_key = _normalize_api_key(key)
    _, file_key_name = _PROVIDER_SETTINGS[provider_name]

    path = _secrets_file(scope_id)
    with _SECRETS_LOCK:
        secrets = _read_secrets(path)
        secrets[file_key_name] = _encrypt_secret_value(normalized_key)
        _write_secrets(secrets, path)
    logger.info("provider API key saved to local settings", extra={"provider": provider_name})


def delete_provider_api_key(
    provider: str, *, scope_id: str | None | object = _SCOPE_UNSET
) -> None:
    """Delete one file-backed provider key, preserving all other settings."""
    provider_name = _normalize_provider(provider)
    _, file_key_name = _PROVIDER_SETTINGS[provider_name]

    path = _secrets_file(scope_id)
    with _SECRETS_LOCK:
        secrets = _read_secrets(path)
        if file_key_name not in secrets:
            return
        del secrets[file_key_name]
        _write_secrets(secrets, path)
    logger.info("provider API key removed from local settings", extra={"provider": provider_name})


def get_provider_key_status(provider: str) -> KeyStatus:
    """Return masked configuration status for one provider."""
    provider_name = _normalize_provider(provider)
    env_name, file_key_name = _PROVIDER_SETTINGS[provider_name]

    env_key = _env_api_key(env_name)
    if env_key:
        return KeyStatus(configured=True, source="env", masked_key=_mask_key(env_key))

    with _SECRETS_LOCK:
        secrets = _read_secrets()
    file_key = _decrypt_secret_value(secrets.get(file_key_name, ""))
    if file_key:
        return KeyStatus(
            configured=True,
            source="local_settings",
            masked_key=_mask_key(file_key),
        )

    return KeyStatus(configured=False, source="missing", masked_key=None)


def get_all_provider_key_statuses() -> dict[ProviderName, KeyStatus]:
    """Return masked statuses for all supported key providers."""
    return {provider: get_provider_key_status(provider) for provider in SUPPORTED_PROVIDERS}


# -- Active assistant preference ---------------------------------------------


def get_active_assistant_provider() -> AssistantProvider:
    """Return the selected assistant API, defaulting safely to Theriad."""
    with _SECRETS_LOCK:
        value = _read_secrets().get(_ACTIVE_ASSISTANT_PROVIDER, _DEFAULT_ASSISTANT_PROVIDER)
    if not isinstance(value, str):
        return _DEFAULT_ASSISTANT_PROVIDER
    try:
        return _normalize_assistant_provider(value)
    except ValueError:
        return _DEFAULT_ASSISTANT_PROVIDER


def set_active_provider(provider: str) -> AssistantProvider:
    """Persist and return the canonical active assistant provider."""
    provider_name = _normalize_assistant_provider(provider)
    with _SECRETS_LOCK:
        secrets = _read_secrets()
        secrets[_ACTIVE_ASSISTANT_PROVIDER] = provider_name
        _write_secrets(secrets)
    logger.info("active assistant provider updated", extra={"provider": provider_name})
    return provider_name


def save_active_assistant_provider(provider: str) -> AssistantProvider:
    """Backward-compatible descriptive alias for :func:`set_active_provider`."""
    return set_active_provider(provider)


def get_provider_settings_status() -> ProviderSettingsStatus:
    """Return the aggregate response shape consumed by provider settings UI."""
    return ProviderSettingsStatus(
        active_provider=get_active_assistant_provider(),
        providers=get_all_provider_key_statuses(),
    )


# -- Compatibility wrappers ---------------------------------------------------


# ── Browser backend preference ───────────────────────────────────────────────

_BROWSER_BACKEND_KEY = "browser_action_backend"
_VALID_SAVED_BACKENDS = frozenset({"opensandbox"})


def get_browser_backend() -> str:
    """Return the saved browser backend from local settings, or '' if not set."""
    with _SECRETS_LOCK:
        value = _read_secrets().get(_BROWSER_BACKEND_KEY, "")
    if isinstance(value, str) and value.strip().lower() in _VALID_SAVED_BACKENDS:
        return value.strip().lower()
    return ""


def save_browser_backend(backend: str, browserbase_api_key: str | None = None) -> None:
    """Persist the canonical OpenSandbox browser backend choice.

    ``browserbase_api_key`` remains an ignored compatibility argument so old
    callers do not accidentally write a deprecated provider secret.
    """
    clean = backend.strip().lower()
    if clean not in _VALID_SAVED_BACKENDS:
        raise ValueError(
            f"Unsupported browser backend: {backend!r}. "
            f"Choose from: {', '.join(sorted(_VALID_SAVED_BACKENDS))}"
        )
    with _SECRETS_LOCK:
        secrets = _read_secrets()
        secrets[_BROWSER_BACKEND_KEY] = clean
        _write_secrets(secrets)
    logger.info("browser backend saved to local settings", extra={"backend": clean})


def get_browserbase_api_key() -> str | None:
    """Deprecated compatibility accessor; Browserbase is no longer supported."""
    return None


# ── Compatibility wrappers ───────────────────────────────────────────────────


def get_openrouter_api_key() -> str | None:
    return get_provider_api_key("openrouter")


def save_openrouter_api_key(key: str) -> None:
    save_provider_api_key("openrouter", key)


def delete_openrouter_api_key() -> None:
    delete_provider_api_key("openrouter")


def get_openrouter_key_status() -> KeyStatus:
    return get_provider_key_status("openrouter")


def get_openai_api_key() -> str | None:
    return get_provider_api_key("openai")


def save_openai_api_key(key: str) -> None:
    save_provider_api_key("openai", key)


def delete_openai_api_key() -> None:
    delete_provider_api_key("openai")


def get_openai_key_status() -> KeyStatus:
    return get_provider_key_status("openai")


def get_anthropic_api_key() -> str | None:
    return get_provider_api_key("anthropic")


def save_anthropic_api_key(key: str) -> None:
    save_provider_api_key("anthropic", key)


def delete_anthropic_api_key() -> None:
    delete_provider_api_key("anthropic")


def get_anthropic_key_status() -> KeyStatus:
    return get_provider_key_status("anthropic")
