"""Rerun codegen+test for the 18 batch18 qualifying tasks with the coding
agent (Groq openai/gpt-oss-120b, falling back to OpenRouter Qwen3-Coder-480B-
A35B when Groq is exhausted -- see scripts/coding_agent.py), reusing the
already-computed elicit -> oracle -> score -> clarify trajectory from the
original batch18 run (data/v10_code_quality/batch18/round_N/<amb_id>/summary.json)
instead of redoing it.

This is safe because it reads that trajectory's already-computed code_prompt
values straight from the original run's summary.json files rather than
recomputing them -- only the final codegen call (CODE_INSTRUCTION -> Python
solution) differs, so re-running elicitation on the GPU would just waste a
model load + 3 rounds x 18 GPU forward passes to reproduce input text this
script already has on disk. (Note: v10_round_worker.py's local generation now
samples rather than using greedy decoding -- see generate() there -- so a
*fresh* elicitation run would no longer even reproduce that same trajectory;
irrelevant here since this script never re-runs elicitation, only replays it.)
Round 0 (no clarification, raw ambiguous_prompt) is generated fresh here too,
for an apples-to-apples comparison against the old Qwen3-8B-codegen numbers
across all 4 rounds.

No GPU/torch/transformers needed -- pure API calls + subprocess test
execution. Resumable: skips <out_dir>/round_N/<amb_id>/summary.json that
already exist.

History: an earlier attempt with OpenRouter alone hit its 50-prompt/day
account limit after 11/72 calls (round 0 only) -- those partial results live
in data/v10_code_quality/batch18_openrouter_coder/ and are NOT reused here.
A subsequent attempt with Groq alone hit Groq's own 200,000-tokens/day cap
partway through. Since scripts/coding_agent.py now falls back from Groq to
OpenRouter automatically, this OUT_DIR (batch18_groq_gptoss120b) may end up
containing a mix of both providers' results across the 72 calls -- each
summary.json records which one actually answered via "coding_agent_provider".

Usage: python v10_batch18_rerun_coder.py [--workers 6]
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v10_run_tests import run_all  # noqa: E402
from scripts.coding_agent import extract_code_block, generate_code_agent  # noqa: E402

HELD_OUT_PATH = ROOT / "data" / "v10_code_quality" / "held_out_40.json"
QUALIFYING_PATH = ROOT / "data" / "v10_code_quality" / "baseline_sweep_40" / "qualifying.json"
OLD_BATCH_DIR = ROOT / "data" / "v10_code_quality" / "batch18"
OUT_DIR = ROOT / "data" / "v10_code_quality" / "batch18_groq_gptoss120b"


def code_prompt_for_round(held_out: dict, aid: str, r: int) -> str:
    if r == 0:
        return held_out[aid]["ambiguous_prompt"]
    summary = json.load(open(OLD_BATCH_DIR / f"round_{r}" / aid / "summary.json"))
    return summary["code_prompt"]


def run_one(aid: str, r: int, held_out: dict) -> dict:
    out_aid_dir = OUT_DIR / f"round_{r}" / aid
    out_aid_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_aid_dir / "summary.json"
    if summary_path.exists():
        return json.load(open(summary_path))

    problem = code_prompt_for_round(held_out, aid, r)
    t0 = time.time()
    # a couple of extra attempts specifically for the case where extract_code_block
    # can't find anything resembling valid Python (e.g. the model answered in prose
    # with no code block at all) -- distinct from generate_code_agent's own retries,
    # which only cover empty content / HTTP errors, not "non-empty but not code"
    code = ""
    coding_agent_provider = None
    for attempt in range(3):
        raw_code, coding_agent_provider = generate_code_agent(problem)
        code = extract_code_block(raw_code)
        try:
            ast.parse(code)
            break
        except SyntaxError:
            code = ""
    test_res = run_all(code, held_out[aid]["test_cases"]) if code else {
        "n_pass": 0, "n_total": len(held_out[aid]["test_cases"]), "results": [],
    }
    elapsed = time.time() - t0

    (out_aid_dir / "generated_code.py").write_text(code)
    summary = {
        "amb_id": aid, "round": r, "code_prompt": problem,
        "n_pass": test_res["n_pass"], "n_total": test_res["n_total"],
        "coding_agent_provider": coding_agent_provider,
        "elapsed_s": round(elapsed, 1),
    }
    json.dump({**summary, "test_results": test_res["results"]}, open(summary_path, "w"), indent=2)
    print(f"[rerun] round {r} {aid} ({coding_agent_provider}): "
          f"{test_res['n_pass']}/{test_res['n_total']} ({elapsed:.0f}s)", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rounds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()

    held_out = json.load(open(HELD_OUT_PATH))
    amb_ids = json.load(open(QUALIFYING_PATH))
    print(f"[rerun] {len(amb_ids)} amb_ids x rounds {args.rounds}, {args.workers} workers", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {aid: {} for aid in amb_ids}

    jobs = [(aid, r) for r in args.rounds for aid in amb_ids]
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, aid, r, held_out): (aid, r) for aid, r in jobs}
        for fut in cf.as_completed(futs):
            aid, r = futs[fut]
            try:
                summary = fut.result()
            except Exception as e:
                print(f"[rerun] round {r} {aid}: FAILED: {e}", flush=True)
                summary = {"amb_id": aid, "round": r, "error": str(e)}
            all_results[aid][r] = summary

    json.dump(all_results, open(OUT_DIR / "all_amb_ids_summary.json", "w"), indent=2)

    print("\n[rerun] pass rate by round (new coder agent):", flush=True)
    for r in sorted(args.rounds):
        pairs = [all_results[aid].get(r, {}) for aid in amb_ids]
        n_pass = sum(p.get("n_pass", 0) for p in pairs)
        n_total = sum(p.get("n_total", 0) for p in pairs)
        n_ok = sum(1 for p in pairs if p.get("n_total") and p["n_pass"] == p["n_total"])
        print(f"    round {r}: {n_pass}/{n_total} test cases passed "
              f"({n_ok}/{len(amb_ids)} tasks fully solved)", flush=True)

    print("[rerun] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
