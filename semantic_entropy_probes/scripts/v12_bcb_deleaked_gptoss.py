"""Same de-leaked selection test as v12_bcb_deleaked_selfrank.py, but using
Groq's gpt-oss-120b (the same elicitation-provider model, via
scripts/coding_agent.py's generate_agent) instead of local Qwen3-8B, to see
whether a larger/different model succeeds where Qwen3-8B still missed all 4
de-leaked groups.

Usage: python v12_bcb_deleaked_gptoss.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.coding_agent import generate_agent  # noqa: E402
from scripts.v12_bcb_deleaked_selfrank import DELEAKED_PROMPTS, SELECT_INSTRUCTION, build_candidates_block, parse_answer  # noqa: E402

JUDGED_PATH = ROOT / "data" / "v12_ranking" / "bcb_judged.jsonl"
OUT_PATH = ROOT / "data" / "v12_ranking" / "gptoss_bcb_deleaked_selfrank_results.json"


def main() -> None:
    judged = {}
    with JUDGED_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                judged[r["id"]] = r

    per_problem = []
    for gid, prompt in DELEAKED_PROMPTS.items():
        row = judged[gid]
        candidates = row["candidates"]
        best_tier = max(c["tier"] for c in candidates)
        valid_idxs = {c["idx"] for c in candidates}

        user_msg = SELECT_INSTRUCTION.format(prompt=prompt, candidates=build_candidates_block(candidates))
        raw, provider = generate_agent(user_msg, max_tokens=500)

        picked = parse_answer(raw, valid_idxs)
        if picked is None:
            hit, note, picked_tier = 0, "unparsed", None
        else:
            picked_tier = next(c["tier"] for c in candidates if c["idx"] == picked)
            hit, note = int(picked_tier == best_tier), "ok"

        per_problem.append({"id": gid, "hit": hit, "picked_idx": picked, "picked_tier": picked_tier,
                             "best_tier": best_tier, "note": note, "provider": provider, "raw_response": raw.strip()})
        mark = "HIT " if hit else "MISS"
        print(f"[gptoss-deleak] {mark} {gid} (via {provider}): picked={picked} note={note}", flush=True)
        print(f"    raw: {raw.strip()[:400]}", flush=True)

    hits = [r["hit"] for r in per_problem]
    print(f"[gptoss-deleak] de-leaked rank-1 hit rate: {sum(hits)}/{len(hits)}", flush=True)

    OUT_PATH.write_text(json.dumps({"per_problem": per_problem, "hit_rate": sum(hits) / len(hits) if hits else None},
                                    indent=2))
    print(f"[gptoss-deleak] saved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
