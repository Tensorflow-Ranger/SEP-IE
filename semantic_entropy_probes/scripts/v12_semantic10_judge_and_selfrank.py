"""Judge (tier-assign, hand-done against semantic10_tiers.json) +
zero-shot selection for the 20-row Semantic scale-up
(10 orchid_heval + 10 orchid_bcb, none previously touched).

Usage: python v12_semantic10_judge_and_selfrank.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v12_bcb_judge_and_selfrank import SELECT_INSTRUCTION, build_candidates_block, parse_answer  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-8B"
CANDIDATES_PATH = ROOT / "data" / "v12_ranking" / "semantic10_candidates.jsonl"
TIERS_PATH = Path("/tmp/claude-1000/-home-ubuntu-SEP-IE/68609f4a-7d53-4b01-bf70-fc84ec45b5a7/scratchpad/semantic10_tiers.json")
JUDGED_OUT_PATH = ROOT / "data" / "v12_ranking" / "semantic10_judged.jsonl"
RESULTS_OUT_PATH = ROOT / "data" / "v12_ranking" / "qwen_semantic10_selfrank_results.json"


def main() -> None:
    tiers_map = json.loads(TIERS_PATH.read_text())

    rows = []
    with CANDIDATES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    judged_rows = []
    skipped = []
    for r in rows:
        t = tiers_map[r["id"]]
        tier2 = set(t["tier2_idx"])
        tier1 = set(t["tier1_idx"])
        if not tier2:
            skipped.append(r["id"])
            continue
        candidates = []
        for i, q in enumerate(r["questions"]):
            tier = 2 if i in tier2 else (1 if i in tier1 else 0)
            candidates.append({"idx": i, "question": q, "tier": tier})
        judged_rows.append({"id": r["id"], "source": r["source"], "prompt": r["prompt"], "candidates": candidates})

    with JUDGED_OUT_PATH.open("w", encoding="utf-8") as f:
        for row in judged_rows:
            f.write(json.dumps({"id": row["id"], "candidates": row["candidates"]}, ensure_ascii=False) + "\n")
    print(f"[sem10-judge] {len(judged_rows)}/{len(rows)} usable (have a tier-2 candidate), "
          f"{len(skipped)} skipped: {skipped}", flush=True)

    if not judged_rows:
        return

    print(f"[sem10-judge] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: "13GiB", "cpu": "40GiB"},
    )
    model.eval()
    print("[sem10-judge] model loaded", flush=True)

    hits, per_problem = [], []
    for n, row in enumerate(judged_rows, 1):
        candidates = row["candidates"]
        best_tier = max(c["tier"] for c in candidates)
        valid_idxs = {c["idx"] for c in candidates}

        user_msg = SELECT_INSTRUCTION.format(prompt=row["prompt"], candidates=build_candidates_block(candidates))
        messages = [{"role": "user", "content": user_msg}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=400, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        picked = parse_answer(gen_text, valid_idxs)
        if picked is None:
            hit, note, picked_tier = 0, "unparsed", None
        else:
            picked_tier = next(c["tier"] for c in candidates if c["idx"] == picked)
            hit, note = int(picked_tier == best_tier), "ok"

        hits.append(hit)
        per_problem.append({"id": row["id"], "source": row["source"], "hit": hit, "picked_idx": picked,
                             "picked_tier": picked_tier, "best_tier": best_tier, "note": note,
                             "raw_response": gen_text.strip()})
        mark = "HIT " if hit else "MISS"
        print(f"[sem10-judge] ({n}/{len(judged_rows)}) {mark} {row['id']}: picked={picked} note={note}", flush=True)

    hit_rate = sum(hits) / len(hits)
    heval_hits = [r["hit"] for r in per_problem if r["source"] == "orchid_heval"]
    bcb_hits = [r["hit"] for r in per_problem if r["source"] == "orchid_bcb"]
    print(f"[sem10-judge] OVERALL rank-1 hit rate: {hit_rate:.3f} ({sum(hits)}/{len(hits)})", flush=True)
    if heval_hits:
        print(f"[sem10-judge] orchid_heval: {sum(heval_hits)}/{len(heval_hits)}", flush=True)
    if bcb_hits:
        print(f"[sem10-judge] orchid_bcb:   {sum(bcb_hits)}/{len(bcb_hits)}", flush=True)

    RESULTS_OUT_PATH.write_text(json.dumps({
        "n_usable": len(judged_rows), "n_skipped": len(skipped), "skipped_ids": skipped,
        "hit_rate": hit_rate,
        "heval_hit_rate": (sum(heval_hits) / len(heval_hits)) if heval_hits else None,
        "bcb_hit_rate": (sum(bcb_hits) / len(bcb_hits)) if bcb_hits else None,
        "per_problem": per_problem,
    }, indent=2))
    print(f"[sem10-judge] saved {RESULTS_OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
