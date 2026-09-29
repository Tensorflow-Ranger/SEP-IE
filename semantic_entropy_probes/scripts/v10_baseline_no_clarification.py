"""Control condition for v10: generate code directly from each held-out
prompt's RAW, unclarified ambiguous_prompt (round 0 -- no elicitation, no
oracle, no clarification at all) and test it, so the round-1/2/3 pass rates
have something to be compared against. Codegen is the coding agent -- Groq
gpt-oss-120b, falling back to OpenRouter Qwen3-Coder-480B-A35B when Groq is
exhausted, same as v10_round_worker.py/v10_batch_worker.py -- so the
comparison uses the same coding agent on both sides (results record which
provider actually answered each call).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v10_run_tests import run_all  # noqa: E402
from scripts.coding_agent import extract_code_block, generate_code_agent  # noqa: E402

HELD_OUT_PATH = ROOT / "data" / "v10_code_quality" / "held_out_3.json"


def main():
    held_out = json.load(open(HELD_OUT_PATH))
    amb_ids = list(held_out.keys())

    results = {}
    for amb_id in amb_ids:
        item = held_out[amb_id]
        t0 = time.time()
        raw_code, coding_agent_provider = generate_code_agent(item["ambiguous_prompt"])
        code = extract_code_block(raw_code)
        out_dir = ROOT / "data" / "v10_code_quality" / amb_id / "round_0_baseline"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "generated_code.py").write_text(code)
        test_res = run_all(code, item["test_cases"])
        print(f"[v10_baseline][{amb_id}] ({coding_agent_provider}) {test_res['n_pass']}/{test_res['n_total']} passed "
              f"({time.time()-t0:.0f}s)", flush=True)
        results[amb_id] = {
            "n_pass": test_res["n_pass"], "n_total": test_res["n_total"], "code": code,
            "coding_agent_provider": coding_agent_provider,
        }
        json.dump(test_res, open(out_dir / "test_results.json", "w"), indent=2)

    json.dump(results, open(ROOT / "data" / "v10_code_quality" / "baseline_no_clarification_summary.json", "w"), indent=2)
    print("[v10_baseline] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
