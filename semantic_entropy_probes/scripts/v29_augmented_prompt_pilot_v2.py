"""v29: corrected version of v28's augmented-prompt pilot.

v28's bug (found while scoping this to orchid_bcb): it augmented the RAW ambiguity-injected
prompt (no clarification), but baseline's round1_n_pass is generated from
`round1_final_prompt` = base_prompt + "\\n\\nClarification: {round1_answer}" (v21's own
CONCAT_TEMPLATE). So v28 wasn't "same prompt + augmentation" vs "same prompt, no
augmentation" -- it removed the round-1 clarification AND added the augmentation at once.

This version reuses the ACTUAL prompt text the original v21 run used, recovered from
data/v21_code_vs_tests/fresh_round2_hits/{safe_id}.picked.json's cached "prompt2" field
(only written for hit1=1 / unrecoverable-miss groups, i.e. the ones that went through FRESH
round-2 elicitation rather than branch-chaining) -- NO GPU/probe re-scoring needed, since
that file already IS the exact round1+round2 text the original run generated code from:
    prompt2 = CONCAT_TEMPLATE(CONCAT_TEMPLATE(base_prompt, round1_answer), round2_answer)
so prompt1 = prompt2.rsplit("\\n\\nClarification: ", 1)[0] recovers the exact round-1 text.

Chained-miss groups (hit1=0, chained to a pre-built wrong-candidate branch) are NOT
recoverable this way without re-running the trained SupCon probe on GPU (this shell has no
torch), so they're excluded from this comparison -- scope is the hit1=1 subset only (96/145
orchid_bcb_expand, 11/21 orchid_bcb), which is exactly the "does forcing an unforced 2nd
clarification on an already-correct answer help/hurt" arm from FINDINGS.md's v21 section,
not the "can round-2 recover a miss" arm.

For each recoverable group: build augmented_prompt{1,2} = prompt{1,2} + bare param types +
Requirements + Example (pulled from the source's own doc_struct), codegen both rounds via
the same DeepSeek v3.2 Bedrock call v21 used, run against the real test suite, compare
round1/round2 n_pass/n_total against the SAME ids' baseline numbers in
results_deepseek_codegen_fixed_env.json.

Resumable: one cache file per group id per round under data/v29_augmented_pilot/codegen/.
Usage: python v29_augmented_prompt_pilot_v2.py
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

RAW_BCB = ROOT / "data" / "raw" / "orchid_bcb.jsonl"
RAW_BCB_EXPAND = ROOT / "data" / "raw" / "orchid_bcb_expand.jsonl"
BASELINE_PATH = ROOT / "data" / "v21_code_vs_tests" / "results_deepseek_codegen_fixed_env.json"
FRESH_R2_DIR = ROOT / "data" / "v21_code_vs_tests" / "fresh_round2_hits"

OUT_DIR = ROOT / "data" / "v29_augmented_pilot"
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


def extract_bare_types(params: list[str]) -> dict[str, str]:
    out = {}
    for p in params:
        m = PARAM_TYPE_RE.match(p.strip())
        if m:
            name, typ = m.group(1), m.group(2).split(",")[0].strip()
            out[name] = typ
    return out


def build_augmentation_suffix(row: dict) -> str:
    try:
        ds = json.loads(row["doc_struct"])
    except Exception:
        return ""
    types = extract_bare_types(ds.get("params", []))
    reqs = ds.get("reqs", [])
    examples = ds.get("examples", [])
    parts = []
    if types:
        parts.append("Parameter types:\n" + "\n".join(f"- {k}: {v}" for k, v in types.items()))
    if reqs:
        parts.append("Requirements:\n" + ", ".join(reqs))
    if examples:
        parts.append("Example:\n" + "\n".join(examples))
    return ("\n\n" + "\n\n".join(parts)) if parts else ""


def load_recoverable(source: str, baseline: dict, raw_by_task: dict, prefix: str) -> list[dict]:
    jobs = []
    for gid, r in baseline.items():
        if r["source"] != source:
            continue
        picked_path = FRESH_R2_DIR / f"{safe_name(gid)}.picked.json"
        if not picked_path.exists():
            continue
        prompt2 = json.loads(picked_path.read_text())["prompt2"]
        if prompt2 is None:
            continue
        prompt1 = prompt2.rsplit("\n\nClarification: ", 1)[0]
        task_id = gid[len(prefix):].rsplit("_", 1)[0]
        row = raw_by_task[task_id]
        suffix = build_augmentation_suffix(row)
        jobs.append({
            "id": gid, "source": source, "task_id": task_id,
            "code_prompt": row["code_prompt"], "test": row["test"],
            "prompt1_aug": prompt1 + suffix, "prompt2_aug": prompt2 + suffix,
        })
    return jobs


def codegen_and_test(job: dict, round_tag: str, spec: str) -> dict:
    cache_path = CODEGEN_DIR / f"{safe_name(job['id'])}.{round_tag}.txt"
    if cache_path.exists():
        raw = cache_path.read_text()
    else:
        rt = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        user_prompt = CODE_INSTRUCTION_BCB.format(code_prompt=job["code_prompt"], spec=spec)
        raw = call_deepseek_code(rt, user_prompt)
        cache_path.write_text(raw)
    code = extract_code_block(raw)
    res = run_bcb_tests(code, job["test"])
    return {"n_pass": res["n_pass"], "n_total": res["n_total"], "msg": res.get("msg", "")}


def main() -> None:
    load_aws_env()
    baseline = {r["id"]: r for r in json.load(open(BASELINE_PATH))}
    bcb_by_task = {r["task_id"]: r for r in (json.loads(l) for l in open(RAW_BCB))}
    bcbx_by_task = {r["task_id"]: r for r in (json.loads(l) for l in open(RAW_BCB_EXPAND))}

    jobs = (
        load_recoverable("orchid_bcb", baseline, bcb_by_task, "bcb_")
        + load_recoverable("orchid_bcb_expand", baseline, bcbx_by_task, "bcbexpand_")
    )
    print(f"[v29] {len(jobs)} recoverable groups "
          f"({sum(1 for j in jobs if j['source']=='orchid_bcb')} orchid_bcb, "
          f"{sum(1 for j in jobs if j['source']=='orchid_bcb_expand')} orchid_bcb_expand)", flush=True)

    def worker(job):
        r1 = codegen_and_test(job, "r1", job["prompt1_aug"])
        r2 = codegen_and_test(job, "r2", job["prompt2_aug"])
        return {"id": job["id"], "source": job["source"],
                "round1_n_pass": r1["n_pass"], "round1_n_total": r1["n_total"], "round1_msg": r1["msg"],
                "round2_n_pass": r2["n_pass"], "round2_n_total": r2["n_total"], "round2_msg": r2["msg"]}

    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(worker, j): j["id"] for j in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if n % 10 == 0 or n == len(jobs):
                print(f"[v29] {n}/{len(jobs)} done", flush=True)

    RESULTS_PATH.write_text(json.dumps(results, indent=1))
    print(f"[v29] wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
