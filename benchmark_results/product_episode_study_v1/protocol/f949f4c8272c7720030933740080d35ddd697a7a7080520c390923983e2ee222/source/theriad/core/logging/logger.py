"""
Structured logging for Theriad.

Decision: Use stdlib logging + JSON formatter rather than loguru/structlog.
Rationale: zero extra dependency for a logging concern; keeps the package lean.
Tradeoff: slightly more verbose setup; acceptable given Principle 7 (simplicity).
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        extra = {k: v for k, v in record.__dict__.items()
                 if k not in logging.LogRecord.__dict__ and not k.startswith("_")}
        if extra:
            payload["extra"] = extra
        return json.dumps(payload)


_configured = False


def configure_logging(level: str = "INFO", fmt: str = "json", file: str | None = None) -> None:
    """Configure root theriad logger."""
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("theriad")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False

    handler: logging.Handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))

    root.addHandler(handler)

    if file:
        fh = logging.FileHandler(file)
        fh.setFormatter(handler.formatter)
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'theriad' namespace."""
    return logging.getLogger(f"theriad.{name}")
