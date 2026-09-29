"""Execute a Python solution (as source text) against a list of {input, output}
stdin/stdout test cases, sandboxed via subprocess with a timeout. Returns
per-test pass/fail plus aggregate pass count -- used both to validate our
own hand-written reference solutions and to score Qwen-generated code.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT_S = 10


def normalize(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.strip().split("\n"))


def run_one(code: str, stdin: str, expected: str, py_bin: str = sys.executable) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            [py_bin, path],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
        if proc.returncode != 0:
            return False, f"RUNTIME_ERROR: {proc.stderr[-500:]}"
        ok = normalize(proc.stdout) == normalize(expected)
        return ok, "" if ok else f"WRONG_ANSWER: got {proc.stdout[:200]!r} want {expected[:200]!r}"
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        Path(path).unlink(missing_ok=True)


def run_all(code: str, test_cases: list[dict], py_bin: str = sys.executable) -> dict:
    results = []
    for i, tc in enumerate(test_cases):
        ok, msg = run_one(code, tc["input"], tc["output"], py_bin=py_bin)
        results.append({"idx": i, "pass": ok, "msg": msg})
    n_pass = sum(r["pass"] for r in results)
    return {"n_pass": n_pass, "n_total": len(test_cases), "results": results}


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--held_out", default=str(Path(__file__).resolve().parents[1] / "data/v10_code_quality/held_out_3.json"))
    ap.add_argument("--solutions", required=True, help="JSON file: {amb_id: python_source}")
    args = ap.parse_args()

    data = json.load(open(args.held_out))
    sols = json.load(open(args.solutions))
    for amb_id, code in sols.items():
        tcs = data[amb_id]["test_cases"]
        res = run_all(code, tcs)
        print(f"{amb_id}: {res['n_pass']}/{res['n_total']} passed")
        for r in res["results"]:
            if not r["pass"]:
                print(f"  test {r['idx']}: {r['msg']}")
