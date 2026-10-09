"""Private fixture-request capture for the isolated pilot, never the service entrypoint."""

import json
import os
import uuid
from pathlib import Path

from theriad.gateway.app import create_app

app = create_app()


@app.middleware("http")
async def capture(request, call_next):
    if request.method == "POST" and request.url.path in {
        "/v1/responses",
        "/v1/chat/completions",
        "/v1/responses/compact",
    }:
        directory = Path(os.environ["THERIAD_PILOT_CAPTURE"])
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        body = await request.body()
        target = directory / (uuid.uuid4().hex + ".json")
        target.write_text(json.dumps({"path": request.url.path, "body": json.loads(body)}))
        target.chmod(0o600)
    return await call_next(request)
