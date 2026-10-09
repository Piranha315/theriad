"""Scheduled job entry point: pinned suite/identity objects, no writable container state."""

import asyncio
import hashlib
import json
import os

from .cli import runtime
from .worker import UpdateWorker


def main():
    engine = runtime()
    bucket = engine.store.bucket

    def pinned(kind):
        data = bucket.blob(os.environ[f"THERIAD_{kind}_OBJECT"]).download_as_bytes()
        if hashlib.sha256(data).hexdigest() != os.environ[f"THERIAD_{kind}_SHA256"]:
            raise ValueError(f"{kind} content hash does not match configured input")
        return json.loads(data)

    suite, mappings = pinned("SUITE"), pinned("IDENTITIES")
    report = asyncio.run(
        UpdateWorker(engine.releases, engine.transport, engine.accounting_store).run(
            suite, mappings
        )
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
