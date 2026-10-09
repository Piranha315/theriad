#!/usr/bin/env python3
"""Verify real isolation before executing pinned public coding references."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theriad.routing.sandbox import default_policy, run_isolated, run_python_tests, runtime_details, sandbox_available
from theriad.routing.episode_study import atomic_json


def verify_boundary(runtime):
    policy = default_policy(runtime=str(runtime))
    results = {}
    with tempfile.TemporaryDirectory(prefix='theriad-boundary-task-') as task, tempfile.TemporaryDirectory(prefix='theriad-boundary-outside-') as outside:
        task = Path(task); outside = Path(outside)
        sentinel = outside/'sentinel.txt'; sentinel.write_text('public harmless isolation sentinel')
        listener = socket.socket(); listener.bind(('127.0.0.1', 0)); listener.listen(1)
        address = listener.getsockname()
        probes = {
            'private_write': "from pathlib import Path; Path('ok').write_text('ok'); assert Path('ok').read_text() == 'ok'",
            'outside_read_denied': f"from pathlib import Path\ntry: Path({str(sentinel)!r}).read_text()\nexcept PermissionError: pass\nelse: raise AssertionError('outside read allowed')",
            'outside_write_denied': f"from pathlib import Path\ntry: Path({str(outside/'write.txt')!r}).write_text('no')\nexcept PermissionError: pass\nelse: raise AssertionError('outside write allowed')",
            'network_denied': f"import socket\ns=socket.socket()\ns.settimeout(1)\ntry: s.connect({address!r})\nexcept PermissionError: pass\nelse: raise AssertionError('network allowed')",
            'fork_creation_blocked': "import os\ntry: pid=os.fork()\nexcept (PermissionError, BlockingIOError): pass\nelse:\n if pid == 0: os._exit(0)\n os.waitpid(pid,0)\n raise AssertionError('fork allowed')",
            'environment_scrubbed': "import os; assert 'THERIAD_ISOLATION_TEST_SENTINEL' not in os.environ",
        }
        old = os.environ.get('THERIAD_ISOLATION_TEST_SENTINEL')
        os.environ['THERIAD_ISOLATION_TEST_SENTINEL'] = 'public-non-secret-test-value'
        try:
            for name, code in probes.items():
                result = run_isolated((policy.runtime_allowlist[0], '-c', code), task_directory=task, policy=policy)
                results[name] = result.to_dict()
            result = run_isolated((policy.runtime_allowlist[0], '-c', 'while True: pass'), task_directory=task, policy=replace(policy, wall_clock_seconds=.2))
            results['timeout_enforced'] = {**result.to_dict(), 'probe_passed': result.timed_out}
        finally:
            listener.close()
            if old is None: os.environ.pop('THERIAD_ISOLATION_TEST_SENTINEL', None)
            else: os.environ['THERIAD_ISOLATION_TEST_SENTINEL'] = old
        passed = all(results[name]['status'] == 'passed' for name in probes) and results['timeout_enforced']['probe_passed']
        return {'passed': passed, 'probes': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('refusing to overwrite verification evidence')
    report = {'sandbox_available': sandbox_available(), 'runtime': runtime_details(args.runtime), 'model_calls': 0}
    report['boundary'] = verify_boundary(args.runtime)
    if not report['boundary']['passed']:
        atomic_json(args.output, report); raise RuntimeError('isolation probe failed; reference execution refused')
    package = args.directory
    rows = [json.loads(x) for x in (package/'new-episodes.jsonl').read_text().splitlines()]
    refs = {x['episode_id']:x['reference'] for x in map(json.loads, (package/'reference-answers.jsonl').read_text().splitlines())}
    policy = default_policy(runtime=str(args.runtime))
    grades = []
    for row in rows:
        if row['family'] != 'coding': continue
        fixture = row['fixture']
        result = run_python_tests(refs[row['episode_id']], fixture['tests'], entry_point=fixture['entry_point'], policy=policy)
        wrong = run_python_tests('def '+fixture['entry_point']+'(*args, **kwargs): raise AssertionError("deliberately wrong")', fixture['tests'], entry_point=fixture['entry_point'], policy=policy)
        grades.append({'episode_id': row['episode_id'], 'reference': result.to_dict(), 'negative_control': wrong.to_dict()})
    report.update({'input_sha256': hashlib.sha256((package/'new-episodes.jsonl').read_bytes()).hexdigest(),
        'coding_rows': len(grades), 'reference_passes': sum(x['reference']['status']=='passed' for x in grades),
        'negative_controls_rejected': sum(x['negative_control']['status']=='failed' for x in grades), 'grades': grades})
    atomic_json(args.output, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('grades','runtime','boundary')}, indent=2))
    if report['reference_passes'] != len(grades) or report['negative_controls_rejected'] != len(grades):
        raise SystemExit(1)

if __name__ == '__main__': main()
