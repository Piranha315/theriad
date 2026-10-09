/**
 * Long-output regression fixture for layout overflow testing.
 *
 * Paste this into the "Model Output" panel (or use in Storybook / manual QA)
 * to verify the layout does not break.
 *
 * Criteria:
 *   - Output panel stays inside its card — no bleed past the right edge.
 *   - Code blocks scroll horizontally inside the panel.
 *   - Sidebar and main shell remain fixed / properly aligned.
 *   - No white background visible below or beside the output.
 *   - Normal short outputs look unchanged.
 */

export const LONG_OUTPUT_FIXTURE = `# Async URL Status Checker

Here is a production-ready implementation with error handling, retries, and a CLI.

---

## 1. Core implementation

\`\`\`python
import asyncio
import httpx
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UrlResult:
    url: str
    status_code: Optional[int] = None
    error: Optional[str] = None
    latency_ms: float = 0.0


async def check_url(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    retries: int = 2,
) -> UrlResult:
    """Check a single URL and return its HTTP status code."""
    for attempt in range(retries + 1):
        try:
            import time
            t0 = time.monotonic()
            resp = await client.get(url, follow_redirects=True, timeout=timeout)
            latency = (time.monotonic() - t0) * 1000
            return UrlResult(url=url, status_code=resp.status_code, latency_ms=latency)
        except httpx.TimeoutException:
            if attempt == retries:
                return UrlResult(url=url, error=f"timeout after {timeout}s")
        except httpx.RequestError as exc:
            if attempt == retries:
                return UrlResult(url=url, error=str(exc))
        await asyncio.sleep(0.1 * (attempt + 1))
    return UrlResult(url=url, error="unreachable")


async def check_urls(urls: list[str], *, concurrency: int = 20) -> list[UrlResult]:
    """Check a list of URLs concurrently, bounded by concurrency."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(url: str) -> UrlResult:
        async with semaphore:
            return await check_url(client, url)

    async with httpx.AsyncClient() as client:
        tasks = [_bounded(url) for url in urls]
        return await asyncio.gather(*tasks)
\`\`\`

---

## 2. CLI wrapper

\`\`\`python
#!/usr/bin/env python3
"""
url-checker — async HTTP status checker

Usage:
    python url_checker.py https://example.com https://httpbin.org/status/404 ...
    cat urls.txt | xargs python url_checker.py
"""
import asyncio
import sys
from url_checker import check_urls


def main() -> None:
    urls = sys.argv[1:]
    if not urls:
        print("Usage: url_checker.py <url> [<url> ...]", file=sys.stderr)
        sys.exit(1)

    results = asyncio.run(check_urls(urls))

    ok = sum(1 for r in results if r.status_code and r.status_code < 400)
    print(f"Checked {len(results)} URLs — {ok} OK, {len(results) - ok} failed\\n")

    for r in sorted(results, key=lambda x: x.url):
        if r.error:
            print(f"  ERR  {r.url}  ({r.error})")
        else:
            icon = "OK " if r.status_code < 400 else "ERR"
            print(f"  {icon}  {r.status_code}  {r.latency_ms:6.0f}ms  {r.url}")


if __name__ == "__main__":
    main()
\`\`\`

---

## 3. Configuration file

\`\`\`json
{
  "concurrency": 50,
  "timeout_seconds": 15,
  "retries": 3,
  "output": {
    "format": "jsonl",
    "path": "/var/log/url-checker/results-2026-06-30T00:00:00Z.jsonl",
    "include_headers": false,
    "include_body": false
  },
  "alerts": {
    "on_status_gte": 400,
    "webhook_url": "https://hooks.example.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
  }
}
\`\`\`

---

## 4. Example output (JSONL)

Each line is one result:

\`\`\`jsonl
{"url":"https://example.com","status_code":200,"error":null,"latency_ms":142.3}
{"url":"https://httpbin.org/status/404","status_code":404,"error":null,"latency_ms":388.1}
{"url":"https://this-domain-definitely-does-not-exist-abc123xyz.com/some/very/deep/path/that/goes/on/and/on/forever","status_code":null,"error":"timeout after 15s","latency_ms":15002.0}
{"url":"https://cdn.example.com/assets/app.a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6.min.js","status_code":200,"error":null,"latency_ms":55.2}
\`\`\`

---

## 5. File paths used by this tool

| Purpose | Path |
|---|---|
| Config | \`/etc/url-checker/config.json\` |
| Results log | \`/var/log/url-checker/results-YYYY-MM-DD.jsonl\` |
| PID file | \`/run/url-checker/url-checker.pid\` |
| Virtual env | \`/opt/url-checker/venv/lib/python3.12/site-packages/httpx/__version__.py\` |

---

## 6. Shell one-liner

\`\`\`bash
python url_checker.py $(cat /tmp/urls.txt | tr '\\n' ' ') --concurrency=100 --timeout=10 --retries=2 2>&1 | tee /var/log/url-checker/run-$(date +%Y%m%dT%H%M%S).log
\`\`\`

---

## 7. Long unbroken string test

The following base64-encoded token is intentionally long and contains no spaces or word-break characters. It must not cause horizontal page overflow:

\`\`\`
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c3ItMTIzNDU2Nzg5MCIsIm5hbWUiOiJQdXJhbiBWZXJtYSIsImlhdCI6MTcxOTcyNjQwMCwiZXhwIjoxNzUxMjYyNDAwLCJzY29wZXMiOlsicmVhZCIsIndyaXRlIiwiYWRtaW4iXX0.AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
\`\`\`

---

## 8. Wide environment table

\`\`\`
VARIABLE                                         VALUE
OPENROUTER_API_KEY                               sk-or-v1-••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••1234
THERIAD_METRICS_PATH                               /Users/puranvs/.theriad/metrics.jsonl
PYTHONPATH                                       /Users/puranvs/Claude/Projects/Theriad:/Users/puranvs/.local/lib/python3.12/site-packages
VIRTUAL_ENV                                      /Users/puranvs/Claude/Projects/Theriad/.venv
PATH                                             /Users/puranvs/Claude/Projects/Theriad/.venv/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
\`\`\`

---

## 9. Error scenarios

- **Timeout**: URL did not respond within the configured timeout. Check network connectivity and increase \`timeout_seconds\` if the host is legitimately slow.
- **DNS failure**: The domain could not be resolved. Verify the URL is correct and that your DNS resolver is reachable.
- **SSL error**: Certificate validation failed. Pass \`verify=False\` to \`httpx.AsyncClient\` for internal hosts with self-signed certificates (not recommended for production).
- **Connection refused**: The server is down or the port is closed. Check that the service is running on the expected host:port.
- **Too many redirects**: The URL redirects more than \`httpx\`'s default limit (20). Use \`max_redirects=5\` to detect redirect loops.

---

## 10. Performance notes

With \`concurrency=50\` and a \`timeout=10s\`, you can expect:

- ~500 URLs/second on a fast network
- ~50 URLs/second against real production endpoints with variable latency
- Memory: approximately 1–2 KB per in-flight request

For large URL lists (>100k entries), consider chunking the input with \`itertools.islice\` and writing results to disk incrementally to avoid accumulating all results in RAM.
`;

/**
 * QA checklist for manual verification.
 *
 * Open the Route screen, run a prompt, then replace execResult.output with
 * LONG_OUTPUT_FIXTURE to simulate a long response. Check each item:
 */
export const QA_CHECKLIST = [
  'Model Output card does not expand past the right edge of the viewport',
  'Code blocks (``` fences) scroll horizontally inside the panel',
  'Long unbroken strings (base64, JWT) break at the card boundary — no horizontal page scroll',
  'Sidebar stays left-aligned and fixed',
  'No white background visible below the content or to the right',
  'Normal short outputs look unchanged (revert to a short output and re-check)',
  'Dark background extends below the fold when output is very long',
  'Copy Output button still works correctly',
];
