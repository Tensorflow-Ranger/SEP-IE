"""Execute a Python function (as source text) against Orchid-HEval-style test
cases -- {"input": "<args repr>", "output": "<expected repr>", "relation": "=="}
-- sandboxed via subprocess with a timeout. Analogous to v10_run_tests.py, but
for function-call assertions instead of stdin/stdout programs.

Only handles relation == "==" -- confirmed to be the only relation type present
across the 32 held-out test-set problems (data/v10_code_quality/held_out_orchid32.json);
raises on anything else rather than silently mishandling it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT_S = 15  # whole test suite, not per-test -- all cases run in one subprocess

_DRIVER_TEMPLATE = '''\
import json

{code}

def _safe_literal(text):
    # some Orchid-HEval "output" fields store a raw string's *content* rather than a
    # proper Python literal repr -- e.g. output="YES" (unquoted) instead of "'YES'" --
    # so a plain eval() raises NameError/SyntaxError treating it as code. Fall back to
    # the literal text as a plain string, which is what those cases actually mean.
    try:
        return eval(text)
    except (NameError, SyntaxError):
        return text

test_cases = json.loads({test_cases_json!r})
fn = globals().get({entry_point!r})
for i, tc in enumerate(test_cases):
    result = {{"idx": i}}
    try:
        if fn is None:
            raise NameError("entry_point " + {entry_point!r} + " not defined")
        args = eval("(" + tc["input"] + ",)")
        expected = _safe_literal(tc["output"])
        got = fn(*args)
        ok = got == expected
        result["pass"] = ok
        result["msg"] = "" if ok else "WRONG_ANSWER: got " + repr(got) + " want " + repr(expected)
    except Exception as e:
        result["pass"] = False
        result["msg"] = "RUNTIME_ERROR: " + type(e).__name__ + ": " + str(e)
    print(json.dumps(result), flush=True)
'''


def _build_driver(code: str, entry_point: str, test_cases: list[dict]) -> str:
    for tc in test_cases:
        assert tc.get("relation") == "==", f"unsupported relation {tc.get('relation')!r} -- only '==' handled"
    return _DRIVER_TEMPLATE.format(
        code=code, entry_point=entry_point, test_cases_json=json.dumps(test_cases),
    )


def run_all(code: str, entry_point: str, test_cases: list[dict],
            py_bin: str = sys.executable, timeout_s: int = TIMEOUT_S) -> dict:
    driver = _build_driver(code, entry_point, test_cases)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(driver)
        path = f.name
    try:
        proc = subprocess.run([py_bin, path], capture_output=True, text=True, timeout=timeout_s)
        if proc.returncode != 0:
            # driver itself crashed (syntax error in candidate code, import error, etc.)
            # -- attribute the failure to every test case rather than raising
            msg = f"DRIVER_CRASH: {proc.stderr[-500:]}"
            results = [{"idx": i, "pass": False, "msg": msg} for i in range(len(test_cases))]
        else:
            results = [json.loads(line) for line in proc.stdout.strip().splitlines() if line.strip()]
            if len(results) != len(test_cases):
                # driver exited early (e.g. an uncaught error mid-loop despite the try/except --
                # shouldn't happen, but don't let it silently under-report)
                got = len(results)
                results += [{"idx": i, "pass": False, "msg": "DRIVER_EXITED_EARLY"}
                            for i in range(got, len(test_cases))]
    except subprocess.TimeoutExpired:
        results = [{"idx": i, "pass": False, "msg": "TIMEOUT"} for i in range(len(test_cases))]
    finally:
        Path(path).unlink(missing_ok=True)
    n_pass = sum(r["pass"] for r in results)
    return {"n_pass": n_pass, "n_total": len(test_cases), "results": results}
