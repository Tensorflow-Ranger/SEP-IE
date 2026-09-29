"""v26: re-run test EXECUTION ONLY (no new LLM calls) against already-generated,
already-cached code, now that 43 previously-missing packages (seaborn, nltk,
lxml, pytz, statsmodels, bs4, cv2, ...) are installed. ~10% of the 244-group
test set (24 orchid_bcb/orchid_bcb_expand groups needing one of these) were
failing 100% of the time purely on ModuleNotFoundError, regardless of
clarification or coding quality -- this corrects that floor.

For each existing results.json, re-runs run_tests_for() against the SAME
cached codegen/{r1,r2}/*.txt files (never regenerated) and overwrites
n_pass/n_total/pass fields, keeping hit1/same_prompt metadata as-is.

Usage: python v26_rerun_tests_fixed_env.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import v21_code_vs_tests_eval as v21  # noqa: E402

N_WORKERS = 16

JOBS = [
    # (results_path, r1_dir, r2_dir)
    ("data/v21_code_vs_tests/results.json",
     "data/v21_code_vs_tests/codegen/r1", "data/v21_code_vs_tests/codegen/r2"),
    ("data/v21_code_vs_tests/results_deepseek_codegen.json",
     "data/v21_code_vs_tests/codegen_deepseek/r1", "data/v21_code_vs_tests/codegen_deepseek/r2"),
    ("data/v22_selector_ablation/qwen_selfrank/results.json",
     "data/v22_selector_ablation/qwen_selfrank/codegen/r1", "data/v22_selector_ablation/qwen_selfrank/codegen/r2"),
    ("data/v22_selector_ablation/qwen_selfrank/results_deepseek_codegen.json",
     "data/v22_selector_ablation/qwen_selfrank/codegen_deepseek/r1",
     "data/v22_selector_ablation/qwen_selfrank/codegen_deepseek/r2"),
    ("data/v22_selector_ablation/deepseek_selfrank/results.json",
     "data/v22_selector_ablation/deepseek_selfrank/codegen/r1",
     "data/v22_selector_ablation/deepseek_selfrank/codegen/r2"),
    ("data/v22_selector_ablation/deepseek_selfrank/results_deepseek_codegen.json",
     "data/v22_selector_ablation/deepseek_selfrank/codegen_deepseek/r1",
     "data/v22_selector_ablation/deepseek_selfrank/codegen_deepseek/r2"),
]


def main():
    lookups = v21.load_raw_lookups()
    for rel_results, rel_r1, rel_r2 in JOBS:
        results_path = ROOT / rel_results
        r1_dir, r2_dir = ROOT / rel_r1, ROOT / rel_r2
        if not results_path.exists():
            print(f"[v26] SKIP (not found): {rel_results}")
            continue
        old = json.loads(results_path.read_text())

        def process(row):
            gid, source = row["id"], row["source"]
            pinfo = v21.problem_info(gid, source, lookups)
            r1_path = r1_dir / f"{v21.safe_name(gid)}.txt"
            r2_path = r2_dir / f"{v21.safe_name(gid)}.txt"
            r1_res = v21.run_tests_for(pinfo, r1_path.read_text()) if r1_path.exists() else None
            if row.get("same_prompt") or not r2_path.exists():
                r2_res = r1_res
            else:
                r2_res = v21.run_tests_for(pinfo, r2_path.read_text())
            new_row = dict(row)
            if r1_res is not None:
                new_row.update({"round1_n_pass": r1_res["n_pass"], "round1_n_total": r1_res["n_total"],
                                 "round1_pass": r1_res["all_pass"]})
            if r2_res is not None:
                new_row.update({"round2_n_pass": r2_res["n_pass"], "round2_n_total": r2_res["n_total"],
                                 "round2_pass": r2_res["all_pass"]})
            return new_row

        new = [None] * len(old)
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(process, row): i for i, row in enumerate(old)}
            done = 0
            for fut in as_completed(futs):
                new[futs[fut]] = fut.result()
                done += 1
                if done % 50 == 0:
                    print(f"[v26]   {rel_results}: {done}/{len(old)} re-tested", flush=True)

        out_path = results_path.with_name(results_path.stem + "_fixed_env.json")
        out_path.write_text(json.dumps(new, indent=2))
        n = len(new)
        old_r1 = sum(r["round1_pass"] for r in old) / n
        new_r1 = sum(r["round1_pass"] for r in new) / n
        old_r2 = sum(r["round2_pass"] for r in old) / n
        new_r2 = sum(r["round2_pass"] for r in new) / n
        print(f"[v26] {rel_results}: round1 {old_r1:.4f} -> {new_r1:.4f}   "
              f"round2 {old_r2:.4f} -> {new_r2:.4f}   saved -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
