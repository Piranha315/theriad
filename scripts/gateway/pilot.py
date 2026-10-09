"""Real, isolated harness trial. Every inference attempt uses the shared pilot ledger.

Run with the repository Python; Hermes uses its installed Python. No global
harness config is written. Fixture files, logs and evidence stay in --output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

from theriad.gateway.contracts import Deployment
from theriad.gateway.releases import Releases, digest
from theriad.gateway.store import LocalStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", choices=["codex", "hermes", "sdk"], required=True)
    parser.add_argument("--model", help="Default: tested Qwen SDK/Hermes or DeepSeek Codex pilot")
    parser.add_argument("--hosting-provider", help="Exact catalog host tag for this isolated trial")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--ledger",
        default=os.environ.get("THERIAD_EVALUATION_DB", "~/.theriad/gateway/state.sqlite"),
    )
    parser.add_argument("--hermes-root", default="~/.hermes/hermes-agent")
    parser.add_argument("--virtual", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", help="Verify a persisted Codex CLI session"
    )
    parser.add_argument(
        "--compact-at", type=int, help="Explicit Codex automatic compaction threshold"
    )
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high"])
    parser.add_argument(
        "--native-responses",
        action="store_true",
        help="Probe an explicitly opted-in upstream Responses deployment",
    )
    args = parser.parse_args()
    args.model = args.model or (
        "deepseek/deepseek-v4-flash-0731" if args.harness == "codex" else "qwen/qwen3-coder"
    )
    args.reasoning_effort = args.reasoning_effort or ("low" if args.harness == "codex" else "none")
    if (args.resume or args.compact_at is not None) and args.harness != "codex":
        parser.error("resume/compaction options require Codex")
    if args.compact_at is not None and args.compact_at <= 0:
        parser.error("compaction threshold must be positive")
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixture = output / "fixture"
    fixture.mkdir(exist_ok=True)
    (fixture / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (fixture / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n    assert add(-1, 1) == 0\n"
    )
    original_tests = (fixture / "test_calc.py").read_text()
    (fixture / "AGENTS.md").write_text(
        "Work only in this fixture directory. Do not access credentials or files outside it.\n"
    )
    provider_key = os.environ.get("OPENROUTER_API_KEY") or dotenv_values(root / ".env").get(
        "OPENROUTER_API_KEY"
    )
    if not provider_key:
        parser.error("OPENROUTER_API_KEY is unavailable")
    catalog = httpx.get("https://openrouter.ai/api/v1/models", timeout=30).json()
    row = next(m for m in catalog["data"] if m["id"] == args.model)
    from theriad.gateway.catalog import initial_endpoint

    endpoints = httpx.get(
        "https://openrouter.ai/api/v1/models/" + row["id"] + "/endpoints", timeout=30
    ).json()["data"]
    if args.hosting_provider:
        endpoints = {
            **endpoints,
            "endpoints": [
                endpoint
                for endpoint in endpoints["endpoints"]
                if endpoint.get("tag") == args.hosting_provider
            ],
        }
    host = initial_endpoint(row["id"], endpoints)
    deployment = Deployment(
        id="pilot-"
        + digest(
            {
                "model": row["id"],
                "revision": row["canonical_slug"],
                "host": host["tag"],
                "variant": host["quantization"],
            }
        )[:20],
        model=row["id"],
        revision=row["canonical_slug"],
        provider="openrouter",
        endpoint="https://openrouter.ai/api/v1",
        region="provider-managed",
        tiers=["cheap_code"],
        context_window=host["context_length"],
        hosting_provider=host["tag"],
        supported_parameters=host["supported_parameters"],
        serving_variant=host["quantization"],
        max_output_tokens=host["max_completion_tokens"],
        max_input_tokens=host.get("max_prompt_tokens"),
        tools=True,
        schemas="response_format" in host["supported_parameters"],
        reasoning="reasoning" in host["supported_parameters"],
        native_responses=args.native_responses,
        input_price=float(host["pricing"]["prompt"]) * 1e6,
        output_price=float(host["pricing"]["completion"]) * 1e6,
        priced_at=time.time(),
        credential_env="OPENROUTER_API_KEY",
    )
    store = LocalStore(output / "gateway-state.sqlite")
    ledger = LocalStore(args.ledger)
    signer = store.get("pilot/signer")
    if not signer:
        signer = secrets.token_urlsafe(48)
        store.put("pilot/signer", signer)
    releases = Releases(store, signer)
    release = releases.install(
        {
            "deployments": [deployment.model_dump()],
            "policy": {
                "allowed_endpoints": [deployment.endpoint],
                "allowed_regions": [deployment.region],
            },
            "source": "isolated-harness-pilot",
        }
    )
    releases.activate(release)
    token = secrets.token_urlsafe(32)
    principal = "eval:" + args.harness + "-" + secrets.token_hex(8)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}/v1"
    env = {
        **os.environ,
        "PYTHONPATH": str(root),
        "OPENROUTER_API_KEY": provider_key,
        "THERIAD_GATEWAY_DB": str(store.path),
        "THERIAD_EVALUATION_DB": str(ledger.path),
        "THERIAD_RELEASE_SIGNING_KEY": signer,
        "THERIAD_GATEWAY_TOKENS": json.dumps(
            {hashlib.sha256(token.encode()).hexdigest(): principal}
        ),
        "THERIAD_GATEWAY_STORE": "local",
        "THERIAD_PILOT_CAPTURE": str(output / "requests"),
    }
    model = "theriad/agent" if args.virtual else deployment.model
    prompt = "Read calc.py and test_calc.py, fix add() so both tests pass, then run python -m pytest -q. Work only here. Finish with the observed test result."
    started = time.time()
    evidence = {
        "harness": args.harness,
        "release": release,
        "deployment": deployment.model_dump(),
        "virtual": args.virtual,
        "principal": principal,
        "started_at": started,
    }
    with (output / "gateway.log").open("w") as log:
        gateway = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "scripts.gateway.capture_app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=root,
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            for _ in range(100):
                try:
                    if httpx.get(url.removesuffix("/v1") + "/ready", timeout=1).is_success:
                        break
                except httpx.HTTPError:
                    pass
                if gateway.poll() is not None:
                    raise RuntimeError("Gateway startup failed; inspect gateway.log")
                time.sleep(0.1)
            client_env = {
                **{
                    k: v
                    for k, v in os.environ.items()
                    if k in {"PATH", "HOME", "LANG", "TMPDIR", "TERM", "SYSTEMROOT"}
                },
                "THERIAD_API_KEY": token,
                "THERIAD_BASE_URL": url,
                "THERIAD_MODEL": model,
            }
            client_env.pop("OPENROUTER_API_KEY", None)
            if args.harness == "codex":
                codex_home = output / "codex-home"
                codex_home.mkdir(exist_ok=True, mode=0o700)
                client_env["CODEX_HOME"] = str(codex_home)
                command = [
                    "codex",
                    "exec",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "--json",
                    "-s",
                    "workspace-write",
                    "-C",
                    str(fixture),
                    "-m",
                    model,
                    "-c",
                    'model_provider="theriad"',
                    "-c",
                    'model_providers.theriad.name="Theriad"',
                    "-c",
                    f'model_providers.theriad.base_url="{url}"',
                    "-c",
                    'model_providers.theriad.env_key="THERIAD_API_KEY"',
                    "-c",
                    'model_providers.theriad.wire_api="responses"',
                    "-c",
                    "model_providers.theriad.supports_websockets=false",
                    "-c",
                    'model_reasoning_summary="none"',
                    "-c",
                    f'model_reasoning_effort="{args.reasoning_effort}"',
                    "-c",
                    "features.apply_patch_freeform=false",
                    "-c",
                    'approval_policy="never"',
                    "-c",
                    "model_context_window=64000",
                    "-c",
                    'web_search="disabled"',
                    "-c",
                    "features.apps=false",
                    "-c",
                    "features.plugins=false",
                    "-c",
                    "features.multi_agent=false",
                    "-c",
                    "features.multi_agent_v2=false",
                    prompt,
                ]
                if args.resume:
                    command.remove("--ephemeral")
                if args.compact_at is not None:
                    command[-1:-1] = ["-c", f"model_auto_compact_token_limit={args.compact_at}"]
            else:
                hermes = Path(args.hermes_root).expanduser()
                home = output / "hermes-home"
                home.mkdir(exist_ok=True)
                (home / "config.yaml").write_text(
                    "model:\n  context_length: 64000\ncompression:\n  enabled: false\nterminal:\n  backend: local\n"
                )
                client_env.update(
                    HERMES_HOME=str(home), TERMINAL_CWD=str(fixture), PYTHONPATH=str(hermes)
                )
                if args.harness == "hermes":
                    code = (
                        "import os,json; from run_agent import AIAgent; a=AIAgent(base_url=os.environ['THERIAD_BASE_URL'],api_key=os.environ['THERIAD_API_KEY'],provider='custom',api_mode='chat_completions',model=os.environ['THERIAD_MODEL'],enabled_toolsets=['terminal','file'],max_iterations=10,max_tokens=2048,quiet_mode=True,skip_context_files=True,skip_memory=True,skip_background_review=True,load_soul_identity=False); print(json.dumps(a.run_conversation("
                        + repr(prompt)
                        + "),default=str))"
                    )
                else:
                    code = "import os,json; from openai import OpenAI; c=OpenAI(base_url=os.environ['THERIAD_BASE_URL'],api_key=os.environ['THERIAD_API_KEY']); print(c.chat.completions.create(model=os.environ['THERIAD_MODEL'],messages=[{'role':'user','content':'Return exactly SDK_OK'}],max_tokens=32).model_dump_json())"
                command = [str(hermes / "venv/bin/python"), "-c", code]
            with (output / "harness.log").open("w") as client_log:
                result = subprocess.run(
                    command,
                    cwd=fixture,
                    env=client_env,
                    stdout=client_log,
                    stderr=client_log,
                    timeout=600,
                )
            evidence["harness_exit"] = result.returncode
            if args.resume and result.returncode == 0:
                events = [
                    json.loads(line)
                    for line in (output / "harness.log").read_text().splitlines()
                    if line.startswith("{")
                ]
                session = next(
                    event["thread_id"] for event in events if event["type"] == "thread.started"
                )
                overrides = [
                    value
                    for index, value in enumerate(command)
                    if index > 0 and command[index - 1] == "-c"
                ]
                resumed = [
                    "codex",
                    "exec",
                    "resume",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--skip-git-repo-check",
                    "--json",
                    "-m",
                    model,
                    "-c",
                    'sandbox_mode="workspace-write"',
                ]
                for override in overrides:
                    resumed.extend(["-c", override])
                resumed.extend(
                    [
                        session,
                        "Continue this session: read calc.py and test_calc.py, run python -m pytest -q again, and report the observed result. Do not change the files.",
                    ]
                )
                with (output / "resume.log").open("w") as resume_log:
                    resumed_result = subprocess.run(
                        resumed,
                        cwd=fixture,
                        env=client_env,
                        stdout=resume_log,
                        stderr=resume_log,
                        timeout=600,
                    )
                evidence.update(resume_exit=resumed_result.returncode, resumed_session=session)
            if args.harness == "sdk":
                evidence["sdk_passed"] = (
                    result.returncode == 0 and "SDK_OK" in (output / "harness.log").read_text()
                )
            if args.harness != "sdk":
                evidence["tests_preserved"] = (
                    fixture / "test_calc.py"
                ).read_text() == original_tests
                # Revalidate against the caller's oracle even if the agent edited tests.
                (fixture / "test_calc.py").write_text(original_tests)
                validation = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q"],
                    cwd=fixture,
                    capture_output=True,
                    text=True,
                )
                evidence.update(
                    fixture_tests_passed=validation.returncode == 0,
                    fixture_test_output=validation.stdout + validation.stderr,
                )
        except Exception as exc:
            evidence["error"] = str(exc)
            raise
        finally:
            gateway.terminate()
            try:
                gateway.wait(timeout=10)
            except subprocess.TimeoutExpired:
                gateway.kill()
            with store.connection() as connection:
                records = connection.execute(
                    "SELECT key,value FROM records WHERE key LIKE ?", (f"episode/{principal}/%",)
                ).fetchall()
            episodes = [json.loads(value) for _, value in records]
            evidence.update(
                finished_at=time.time(),
                episodes=episodes,
                completed_requests=len(episodes),
                known_cost_usd=sum(e["cost_usd"] for e in episodes if e["cost_known"]),
                unknown_cost_requests=sum(not e["cost_known"] for e in episodes),
                remote_compaction_requests=sum(
                    json.loads(path.read_text())["path"] == "/v1/responses/compact"
                    for path in (output / "requests").glob("*.json")
                ),
            )
            (output / "evidence.json").write_text(json.dumps(evidence, indent=2))
            print(
                json.dumps(
                    {
                        k: evidence[k]
                        for k in [
                            "harness",
                            "completed_requests",
                            "known_cost_usd",
                            "unknown_cost_requests",
                        ]
                    }
                    | {"evidence": str(output / "evidence.json")}
                )
            )

    if (
        evidence.get("harness_exit") != 0
        or evidence.get("resume_exit", 0) != 0
        or not evidence.get("sdk_passed", evidence.get("fixture_tests_passed", False))
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
