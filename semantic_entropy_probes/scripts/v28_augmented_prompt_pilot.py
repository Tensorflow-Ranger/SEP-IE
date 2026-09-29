"""v28 pilot: does reinstating bare parameter TYPES + Requirements + Example (but NOT full
parameter descriptions) into the ambiguity-injected orchid_bcb_expand prompt reduce the
"trivial fail" rate (codegen bugs unrelated to the labeled ambiguity, e.g. BigCodeBench/484's
int-vs-datetime crash) without re-leaking the labeled ambiguity itself?

Motivation (this session's conversation): full Parameters-block reinstatement re-leaks the
deleted vagueness phrase verbatim in 18.5% of rows (params text was written by the SAME
docstring the ambiguity edit deleted a phrase from); Examples reinstatement leaks another
10% (23.1% combined). Bare type tokens alone (no descriptive text) hit only 8/399 (2.0%)
because the deleted phrase is essentially never literally "int"/"float"/"list"/etc.

Pipeline (mirrors v21_code_vs_tests_eval.py's DeepSeek-codegen path exactly, so results are
directly comparable to data/v21_code_vs_tests/results_deepseek_codegen_fixed_env.json's
round1_n_pass/round1_n_total baseline for the SAME 145 orchid_bcb_expand ids):
  1. augmented_spec = {Semantic,Vagueness}-Expand_prompt + bare param types + Requirements +
     Example(s), pulled from doc_struct.
  2. leak flag (vagueness rows only): does the deleted phrase reappear in the Example text?
     (types are excluded from this channel by construction -- see 2.0% above)
  3. codegen via the same DeepSeek v3.2 Bedrock call + CODE_INSTRUCTION_BCB template as v21.
  4. run against the real `test` field via v21_bcb_run_tests.run_all.
  5. compare n_pass/n_total against the cached v21 baseline for the same id.

Resumable: one cache file per group id under data/v28_augmented_pilot/codegen/.
Usage: python v28_augmented_prompt_pilot.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from coding_agent import extract_code_block  # noqa: E402
from v21_bcb_run_tests import run_all as run_bcb_tests  # noqa: E402

BEDROCK_MODEL_ID = "deepseek.v3.2"
BEDROCK_REGION = "ap-northeast-1"
N_WORKERS = 8
MAX_RETRIES = 3

RAW_BCB_EXPAND = ROOT / "data" / "raw" / "orchid_bcb_expand.jsonl"
BASELINE_PATH = ROOT / "data" / "v21_code_vs_tests" / "results_deepseek_codegen_fixed_env.json"

OUT_DIR = ROOT / "data" / "v28_augmented_pilot"
CODEGEN_DIR = OUT_DIR / "codegen"
RESULTS_PATH = OUT_DIR / "results.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CODEGEN_DIR.mkdir(parents=True, exist_ok=True)

CODE_INSTRUCTION_BCB = (
    "Complete the following Python function. Here is the existing code (imports and function "
    "signature):\n```python\n{code_prompt}\n```\n\nHere is the function's full specification "
    "(it may include a clarification not shown in the code above):\n<spec>\n{spec}\n</spec>\n\n"
    "Output ONLY the complete function definition -- starting from the imports and signature "
    "given above, with your implementation added -- inside a single ```python ... ``` block. "
    "No explanation, no usage examples, no extra text."
)


def load_aws_env() -> None:
    env = {}
    for line in open(ROOT.parent / ".env"):
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = env["AWS"]


def safe_name(rid: str) -> str:
    return rid.replace("/", "__")


def call_deepseek_code(rt, user_prompt: str, max_tokens: int = 2000) -> str:
    body = json.dumps({
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": max_tokens, "temperature": 0.3,
    })
    delay = 5
    for attempt in range(8):
        try:
            resp = rt.invoke_model(modelId=BEDROCK_MODEL_ID, body=body)
            payload = json.loads(resp["body"].read())
            return payload["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            if attempt == 7:
                raise
            msg = str(e)
            delay = min(delay * 2, 120)
            time.sleep(delay if ("Throttling" in msg or "TooManyRequests" in msg) else min(delay, 20))


PARAM_TYPE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]+)\)")
REMOVED_PHRASE_RE = re.compile(r"[Rr]emov(?:ing|ed) [\"“]([^\"”]+)[\"”]")


def extract_bare_types(params: list[str]) -> dict[str, str]:
    out = {}
    for p in params:
        m = PARAM_TYPE_RE.match(p.strip())
        if m:
            name, typ = m.group(1), m.group(2).split(",")[0].strip()
            out[name] = typ
    return out


def build_augmented_spec(row: dict, amb_type: str) -> tuple[str, bool | None]:
    cap = {"semantic": "Semantic", "vagueness": "Vagueness"}[amb_type]
    base = row.get(f"{cap}-Expand_prompt")
    try:
        ds = json.loads(row["doc_struct"])
    except Exception:
        ds = {}
    types = extract_bare_types(ds.get("params", []))
    reqs = ds.get("reqs", [])
    examples = ds.get("examples", [])

    parts = [base]
    if types:
        parts.append("Parameter types:\n" + "\n".join(f"- {k}: {v}" for k, v in types.items()))
    if reqs:
        parts.append("Requirements:\n" + ", ".join(reqs))
    if examples:
        parts.append("Example:\n" + "\n".join(examples))
    augmented = "\n\n".join(parts)

    leaked = None
    if amb_type == "vagueness":
        expl = row.get("Vagueness-Expand_ambiguity_explanation", "")
        m = REMOVED_PHRASE_RE.search(expl)
        if m:
            phrase = m.group(1).lower()
            leaked = phrase in " ".join(examples).lower() or any(phrase == v.lower() for v in types.values())
    return augmented, leaked


def main() -> None:
    load_aws_env()

    baseline = {r["id"]: r for r in json.load(open(BASELINE_PATH))}
    target_ids = [gid for gid, r in baseline.items() if r["source"] == "orchid_bcb_expand"]
    print(f"[v28] {len(target_ids)} orchid_bcb_expand groups to re-run with augmented prompts", flush=True)

    bcb_by_task = {r["task_id"]: r for r in (json.loads(l) for l in open(RAW_BCB_EXPAND))}

    jobs = []
    for gid in target_ids:
        # id format: bcbexpand_BigCodeBench/{n}_{semantic|vagueness}
        body = gid[len("bcbexpand_"):]
        task_id, amb_type = body.rsplit("_", 1)
        row = bcb_by_task[task_id]
        augmented, leaked = build_augmented_spec(row, amb_type)
        jobs.append({
            "id": gid, "task_id": task_id, "amb_type": amb_type, "leaked": leaked,
            "code_prompt": row["code_prompt"], "test": row["test"], "spec": augmented,
        })

    def worker(job):
        cache_path = CODEGEN_DIR / f"{safe_name(job['id'])}.txt"
        if cache_path.exists():
            raw = cache_path.read_text()
        else:
            rt = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
            user_prompt = CODE_INSTRUCTION_BCB.format(code_prompt=job["code_prompt"], spec=job["spec"])
            raw = call_deepseek_code(rt, user_prompt)
            cache_path.write_text(raw)
        code = extract_code_block(raw)
        res = run_bcb_tests(code, job["test"])
        return {
            "id": job["id"], "amb_type": job["amb_type"], "leaked": job["leaked"],
            "n_pass": res["n_pass"], "n_total": res["n_total"], "msg": res.get("msg", ""),
        }

    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(worker, j): j["id"] for j in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if n % 10 == 0 or n == len(jobs):
                print(f"[v28] {n}/{len(jobs)} done", flush=True)

    RESULTS_PATH.write_text(json.dumps(results, indent=1))
    print(f"[v28] wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
