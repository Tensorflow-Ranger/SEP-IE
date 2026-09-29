"""Execute a Python function (as source text) against BigCodeBench-style
`unittest.TestCase` test suites (the `test` field in data/raw/orchid_bcb.jsonl
/ orchid_bcb_expand.jsonl -- always a class named `TestCases`), sandboxed via
subprocess with a timeout. Analogous to heval_run_tests.py, but for BCB's
unittest format instead of HumanEval's {input, output} function-call cases.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT_S = 30

_DRIVER_TEMPLATE = '''\
{code}

{test}

import unittest, json as _json
if __name__ == "__main__":
    _loader = unittest.TestLoader()
    _suite = _loader.loadTestsFromTestCase(TestCases)
    _runner = unittest.TextTestRunner(verbosity=0, stream=open("/dev/null", "w"))
    _result = _runner.run(_suite)
    print(_json.dumps({{"n_total": _result.testsRun,
                        "n_fail": len(_result.failures) + len(_result.errors)}}))
'''


def run_all(code: str, test: str, py_bin: str = sys.executable, timeout_s: int = TIMEOUT_S) -> dict:
    n_declared = test.count("def test_")
    driver = _DRIVER_TEMPLATE.format(code=code, test=test)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(driver)
        path = f.name
    try:
        proc = subprocess.run([py_bin, path], capture_output=True, text=True, timeout=timeout_s)
        if proc.returncode != 0:
            return {"n_pass": 0, "n_total": n_declared, "msg": f"DRIVER_CRASH: {proc.stderr[-500:]}"}
        line = next((l for l in proc.stdout.strip().splitlines()[::-1] if l.strip()), None)
        if line is None:
            return {"n_pass": 0, "n_total": n_declared, "msg": "NO_OUTPUT"}
        out = json.loads(line)
        return {"n_pass": out["n_total"] - out["n_fail"], "n_total": out["n_total"], "msg": ""}
    except subprocess.TimeoutExpired:
        return {"n_pass": 0, "n_total": n_declared, "msg": "TIMEOUT"}
    finally:
        Path(path).unlink(missing_ok=True)
