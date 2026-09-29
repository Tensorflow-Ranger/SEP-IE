"""Sweep the no-clarification baseline (round 0: code generated directly from
the raw ambiguous_prompt) across the 40 held-out prompts, to find prompts in
the sweet spot for testing whether clarification helps: the coding agent gets
MORE THAN 2 test cases right (so it has the right general approach / isn't
simply incapable of the algorithm) but NOT all of them (so there's a real gap
for clarification to close). Prompts scoring <=2 are excluded too -- those look
like a capability ceiling, not an ambiguity gap, per the amb_task_4 case in the
3-prompt pilot where correct clarification never moved 0/43.

NOTE: codegen here is now the coding agent -- Groq gpt-oss-120b, falling back
to OpenRouter Qwen3-Coder-480B-A35B when Groq is exhausted, same as
v10_round_worker.py/v10_batch_worker.py/v10_baseline_no_clarification.py --
not the local Qwen3-8B this sweep was originally calibrated against. The
qualifying band (2 < n_pass < n_total) is a heuristic tuned for Qwen3-8B's
weaker baseline; with a stronger coding agent, expect fewer/no prompts to land
in that exact band since both replacement models are more likely to either
solve the raw ambiguous prompt outright or miss it in a way unrelated to
ambiguity.

Stops early once `--target_qualify` prompts in that band are found. Resumable:
skips amb_ids already scored in baseline_sweep_40/summary.json from a prior
(possibly interrupted) run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v10_run_tests import run_all  # noqa: E402
from scripts.coding_agent import extract_code_block, generate_code_agent  # noqa: E402

HELD_OUT_PATH = ROOT / "data" / "v10_code_quality" / "held_out_40.json"


def qualifies(n_pass: int, n_total: int) -> bool:
    return n_pass > 2 and n_pass < n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_qualify", type=int, default=20)
    args = ap.parse_args()

    held_out = json.load(open(HELD_OUT_PATH))
    amb_ids = list(held_out.keys())

    out_dir = ROOT / "data" / "v10_code_quality" / "baseline_sweep_40"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        results = json.load(open(summary_path))
        print(f"[v10_sweep40] resuming, {len(results)} amb_ids already scored", flush=True)

    qualifying = [aid for aid, r in results.items() if qualifies(r["n_pass"], r["n_total"])]
    remaining = [aid for aid in amb_ids if aid not in results]

    if len(qualifying) >= args.target_qualify or not remaining:
        print(f"[v10_sweep40] already have {len(qualifying)} qualifying prompts from cache, nothing to do", flush=True)
        json.dump(qualifying, open(out_dir / "qualifying.json", "w"), indent=2)
        print("[v10_sweep40] ALL DONE", flush=True)
        return

    for i, amb_id in enumerate(remaining):
        item = held_out[amb_id]
        t0 = time.time()
        raw_code, coding_agent_provider = generate_code_agent(item["ambiguous_prompt"])
        code = extract_code_block(raw_code)
        test_res = run_all(code, item["test_cases"])
        elapsed = time.time() - t0
        n_pass, n_total = test_res["n_pass"], test_res["n_total"]
        qual = qualifies(n_pass, n_total)
        status = "QUALIFY" if qual else ("PERFECT" if n_pass == n_total else "TOO_HARD")
        print(f"[v10_sweep40] {i+1}/{len(remaining)} {amb_id} ({coding_agent_provider}): {n_pass}/{n_total} "
              f"[{status}] ({elapsed:.0f}s) -- qualifying_so_far={len(qualifying) + (1 if qual else 0)}", flush=True)

        results[amb_id] = {"n_pass": n_pass, "n_total": n_total, "code": code, "coding_agent_provider": coding_agent_provider}
        (out_dir / f"{amb_id}.py").write_text(code)
        if qual:
            qualifying.append(amb_id)

        json.dump(results, open(summary_path, "w"), indent=2)
        json.dump(qualifying, open(out_dir / "qualifying.json", "w"), indent=2)

        if len(qualifying) >= args.target_qualify:
            print(f"[v10_sweep40] reached target of {args.target_qualify} qualifying prompts, stopping early "
                  f"({i+1}/{len(remaining)} of remaining checked)", flush=True)
            break

    print(f"[v10_sweep40] ALL DONE, {len(qualifying)} qualifying prompts out of {len(results)} checked", flush=True)


if __name__ == "__main__":
    main()
