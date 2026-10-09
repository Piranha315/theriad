"""Browser observation safety checks."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

CAPTCHA_MARKERS = (
    "i'm not a robot",
    "recaptcha",
    "our systems have detected unusual traffic",
    "unusual traffic from your computer network",
)


@dataclass(frozen=True)
class BrowserObservation:
    current_url: str
    page_title: str
    visible_text: str
    screenshot: str | None = None


def is_captcha_observation(observation: BrowserObservation) -> bool:
    """Return True for known CAPTCHA text or Google's challenge URL."""

    parsed = urlparse(observation.current_url)
    hostname = (parsed.hostname or "").lower()
    google_sorry = (
        parsed.path.rstrip("/").lower() == "/sorry/index"
        and (hostname == "google.com" or ".google." in hostname)
    )
    text = f"{observation.page_title}\n{observation.visible_text}".casefold()
    return google_sorry or any(marker in text for marker in CAPTCHA_MARKERS)
