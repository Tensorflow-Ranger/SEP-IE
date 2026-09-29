"""v24: sanity check for the user's "something is fishy" concern -- v21/v22 held
codegen constant at Qwen3-8B across every ablation (DeepSeek only ever picked
WHICH clarification to ask, never wrote code). This script has DeepSeek V3.2
itself write the code for a sample of the exact same round-1 prompts Qwen3-8B
failed on, run through the identical test harness, to check whether the low
pass rates are a genuine Qwen3-8B capability ceiling (expected: DeepSeek does
much better) or a harness bug (would show: DeepSeek fails identically).

Usage: python v24_deepseek_codegen_sanity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import v21_code_vs_tests_eval as v21  # noqa: E402

RESULTS_PATH = ROOT / "data" / "v21_code_vs_tests" / "results.json"
CODEGEN_R1_DIR = ROOT / "data" / "v21_code_vs_tests" / "codegen" / "r1"
N_SAMPLE = 20


def call_deepseek_code(rt, user_prompt: str) -> str:
    body = json.dumps({
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": 2000, "temperature": 0.3,
    })
    resp = rt.invoke_model(modelId="deepseek.v3.2", body=body)
    payload = json.loads(resp["body"].read())
    return payload["choices"][0]["message"]["content"]


def main():
    results = {r["id"]: r for r in json.loads(RESULTS_PATH.read_text())}
    fails = [r for r in results.values() if r["source"] == "orchid_bcb_expand" and not r["round1_pass"]]
    sample = fails[:N_SAMPLE]
    print(f"[v24] sampling {len(sample)} orchid_bcb_expand groups Qwen3-8B failed on round-1", flush=True)

    v21.load_aws_env()
    rt = boto3.client("bedrock-runtime", region_name=v21.BEDROCK_REGION)
    lookups = v21.load_raw_lookups()
    chunk_groups = v21.load_chunk_groups()
    # NOTE: uses each group's ORIGINAL (unclarified) prompt, not the exact round-1 clarified
    # prompt Qwen3-8B saw (reconstructing that needs re-deriving probe_1's per-group top1_idx,
    # not persisted in results.json) -- close enough for this sanity check's actual question
    # ("can a stronger coder solve tasks Qwen3-8B failed on"), since the two differ by only one
    # appended clarifying sentence.
    prompt1_by_gid = {g: chunk_groups[g]["prompt"] for g in chunk_groups}

    qwen_pass, deepseek_pass = 0, 0
    for i, r in enumerate(sample, 1):
        gid = r["id"]
        source = r["source"]
        pinfo = v21.problem_info(gid, source, lookups)
        original_prompt = prompt1_by_gid[gid]
        code_user = v21.CODE_INSTRUCTION_BCB.format(code_prompt=pinfo["code_prompt"], spec=original_prompt)

        raw = call_deepseek_code(rt, code_user)
        code = v21.extract_code_block(raw)
        res = v21.run_tests_for(pinfo, raw)
        qwen_res_pass = r["round1_pass"]
        deepseek_res_pass = res["all_pass"]
        qwen_pass += int(qwen_res_pass)
        deepseek_pass += int(deepseek_res_pass)
        print(f"[v24] ({i}/{len(sample)}) {gid}: qwen={r['round1_n_pass']}/{r['round1_n_total']} "
              f"deepseek={res['n_pass']}/{res['n_total']}", flush=True)

    print(f"\n[v24] === SANITY CHECK (n={len(sample)}, all Qwen3-8B round-1 FAILURES) ===")
    print(f"  Qwen3-8B pass rate (should be 0 by construction): {qwen_pass}/{len(sample)}")
    print(f"  DeepSeek V3.2 pass rate on the SAME (unclarified-base) problems: {deepseek_pass}/{len(sample)}")


if __name__ == "__main__":
    main()
