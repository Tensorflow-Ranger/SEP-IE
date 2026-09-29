"""v25: the missing baseline -- how well does DeepSeek V3.2 do with NO
clarification at all (the raw, ambiguous original prompt), on the same
244-group test split every other v21/v22 number was computed on? Answers
the user's question directly: is round-1's ~+11pp-under-DeepSeek-codegen
pass rate actually coming from clarification, or would DeepSeek score
about the same on the unclarified prompt alone?

Usage: python v25_baseline_no_clarification.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import v21_code_vs_tests_eval as v21  # noqa: E402
import v22_selector_ablation as v22  # noqa: E402

OUT_DIR = ROOT / "data" / "v25_baseline_no_clarification"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUT_DIR / "results.json"


def main():
    test_groups, branches_by_group, branches = v22.reconstruct_test_groups()
    chunk_groups = v21.load_chunk_groups()
    lookups = v21.load_raw_lookups()
    src_of = {g: chunk_groups[g]["dataset_source"] for g in chunk_groups}

    scored = [g for g in test_groups if g in chunk_groups]
    print(f"[v25] {len(scored)}/{len(test_groups)} test groups reconstructable", flush=True)

    pinfos = {g: v21.problem_info(g, src_of[g], lookups) for g in scored}
    inputs = [v21.code_instruction(pinfos[g], chunk_groups[g]["prompt"]) for g in scored]
    sane = [v21.make_is_sane(pinfos[g]) for g in scored]

    v21.load_aws_env()
    v21.CODEGEN_CACHE_DIR = OUT_DIR / "codegen_deepseek"
    v21.CODEGEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("[v25] generating code from the RAW unclarified prompt (DeepSeek V3.2) ...", flush=True)
    raw = v21.generate_cached_deepseek(scored, "baseline", inputs, sane)

    results = []
    for g, r in zip(scored, raw):
        res = v21.run_tests_for(pinfos[g], r)
        results.append({"id": g, "source": src_of[g], "n_pass": res["n_pass"], "n_total": res["n_total"],
                        "pass": res["all_pass"]})
        print(f"[v25] {g} [{src_of[g]}]: {res['n_pass']}/{res['n_total']}", flush=True)

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    n = len(results)
    rate = sum(r["pass"] for r in results) / n
    print(f"\n[v25] === NO-CLARIFICATION BASELINE (DeepSeek V3.2 codegen, n={n}) ===")
    print(f"  overall pass rate: {rate:.4f}")
    from collections import defaultdict
    by_source = defaultdict(list)
    for r in results:
        by_source[r["source"]].append(r)
    for src, rows in sorted(by_source.items()):
        p = sum(r["pass"] for r in rows) / len(rows)
        print(f"  {src:20s} (n={len(rows):3d}): {p:.4f}")
    print(f"[v25] saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
