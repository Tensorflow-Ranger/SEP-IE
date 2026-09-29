"""v22: same functional-correctness (code-vs-real-tests) pipeline as
v21_code_vs_tests_eval.py, on the IDENTICAL 244-group test split, but with
the SupCon-round-2 probe's SELECTION mechanism replaced by a zero-shot LLM
self-rank (Qwen3-8B or DeepSeek V3.2 picking which candidate question
matters most, via v19_control_qwen_selfrank.py's SELECT_INSTRUCTION) -- the
ablation the user asked for, to see whether the trained probe's ranking is
actually earning its keep over just asking a model "which question matters
most?" directly.

Held constant across every ablation (probe / qwen_selfrank / deepseek_selfrank):
  - the CANDIDATE POOLS themselves (round-1: deepseek_judged.jsonl's elicited
    candidates; round-2 for chainable misses: the pre-built branches'
    elicited_round2.jsonl candidates; round-2 for hits/unrecoverable misses:
    a fresh DeepSeek elicitation on top of THIS selector's own round-1 pick,
    same FRESH_R2_SYSTEM_PROMPT v21 used) -- only WHICH candidate gets
    chosen from each pool differs.
  - the codegen model (Qwen3-8B, same CODE_INSTRUCTION_* templates) and the
    real test-execution harnesses (heval/bcb/stdin-stdout).
Only the selection mechanism varies: probe (v21, already run) vs. this
script's --selector {qwen_selfrank, deepseek_selfrank}.

Round-1 picks:
  - qwen_selfrank: reused verbatim from the existing, already-computed
    data/v19_concat_activations/qwen_selfrank_results.json (covers all 1229
    groups already -- no new model calls needed for round 1).
  - deepseek_selfrank: fresh Bedrock SELECT_INSTRUCTION calls, one per test
    group, cached to data/v22_selector_ablation/deepseek_r1_picks/.

Round-2 picks (both selectors): a fresh self-rank call (Qwen local generate
for qwen_selfrank, Bedrock for deepseek_selfrank) over whichever candidate
pool applies (chained branch or fresh elicitation), using the SAME
SELECT_INSTRUCTION prompt format.

Usage: python v22_selector_ablation.py --selector qwen_selfrank
       python v22_selector_ablation.py --selector deepseek_selfrank
Fully resumable (per-group caches at every stage); run one selector at a
time (both share the GPU for codegen, and running two Qwen-loading
processes at once risks the same NVML/allocator flakiness v21 hit).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import boto3
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v19_control_qwen_selfrank import (  # noqa: E402
    SELECT_INSTRUCTION, build_candidates_block, build_groups, parse_answer,
    load_answers_by_question as v19_load_answers_by_question,
)
import v21_code_vs_tests_eval as v21  # noqa: E402  (reuse problem_info/codegen/test-harness code as-is)

BRANCHES_PATH = ROOT / "data" / "v20_branches_full" / "branches_round2.json"
ELICITED_PATH = ROOT / "data" / "v20_branches_full" / "elicited_round2.jsonl"
ANSWERS_R2_PATH = ROOT / "data" / "v20_branches_full" / "answers_round2.jsonl"
ACT_DIR = ROOT / "data" / "v20_branches_full" / "round2_activations"
QWEN_STAGE1_CACHE = ROOT / "data" / "v19_concat_activations" / "qwen_selfrank_results.json"
V12_JUDGED_PATH = ROOT / "data" / "v12_ranking" / "judged.jsonl"
SEED = 0
TEST_FRAC = 0.2
MODEL_NAME = "Qwen/Qwen3-8B"

OUT_ROOT = ROOT / "data" / "v22_selector_ablation"


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def reconstruct_test_groups():
    branches = json.loads(BRANCHES_PATH.read_text())
    branches_by_group = defaultdict(list)
    for i, b in enumerate(branches):
        if not (ACT_DIR / f"{i}.npz").exists():
            continue
        branches_by_group[b["id"]].append((i, b["removed_question"].strip()))
    groups_with_branches = sorted(branches_by_group.keys())
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(groups_with_branches))
    n_test = max(1, int(len(groups_with_branches) * TEST_FRAC))
    test_groups = [g for g in np.array(groups_with_branches)[perm[:n_test]].tolist()]
    return test_groups, branches_by_group, branches


def call_deepseek_select(rt, prompt: str, candidates_block: str) -> str:
    user_msg = SELECT_INSTRUCTION.format(prompt=prompt, candidates=candidates_block)
    body = json.dumps({
        "messages": [{"role": "user", "content": user_msg}],
        "max_tokens": 400, "temperature": 0.0,
    })
    delay = 5
    for attempt in range(8):
        try:
            resp = rt.invoke_model(modelId="deepseek.v3.2", body=body)
            payload = json.loads(resp["body"].read())
            return payload["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            if attempt == 7:
                raise
            msg = str(e)
            delay = min(delay * 2, 120)
            time.sleep(delay if ("Throttling" in msg or "TooManyRequests" in msg) else min(delay, 20))


def qwen_select(tokenizer, model, prompt: str, candidates_block: str, valid_idxs: set[int]) -> tuple[int | None, str]:
    user_msg = SELECT_INSTRUCTION.format(prompt=prompt, candidates=candidates_block)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = v21.with_retry(lambda: model.generate(
        **inputs, max_new_tokens=400, do_sample=False, pad_token_id=tokenizer.eos_token_id,
    ), log_prefix="[v22-qwen-select] ")
    gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_answer(gen_text, valid_idxs), gen_text


def get_round1_picks(selector: str, test_groups: list[str], groups_cand: dict,
                      tokenizer=None, model=None, rt=None, cache_dir: Path | None = None) -> dict:
    """Returns {gid: {"picked_idx0": int, "hit1": int, "question": str, "tier": int}}."""
    out = {}
    if selector == "qwen_selfrank":
        stage1 = json.loads(QWEN_STAGE1_CACHE.read_text())
        by_id = {r["id"]: r for r in stage1["per_problem"]}
        for gid in test_groups:
            row = by_id.get(gid)
            g = groups_cand[gid]
            if row is None or row.get("note") != "ok" or row.get("picked_idx") is None:
                out[gid] = None
                continue
            idx0 = row["picked_idx"] - 1
            if not (0 <= idx0 < len(g["candidates"])):
                out[gid] = None
                continue
            out[gid] = {"picked_idx0": idx0, "hit1": row["hit"], "question": g["candidates"][idx0]["question"],
                        "tier": g["candidates"][idx0]["tier"]}
        return out

    assert selector == "deepseek_selfrank"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, gid in enumerate(test_groups, 1):
        p = cache_dir / f"{v21.safe_name(gid)}.json"
        if p.exists():
            row = json.loads(p.read_text())
        else:
            g = groups_cand[gid]
            block = build_candidates_block(g["candidates"])
            raw = call_deepseek_select(rt, g["prompt"], block)
            valid = set(range(1, len(g["candidates"]) + 1))
            picked = parse_answer(raw, valid)
            row = {"id": gid, "picked_idx": picked, "raw_response": raw}
            p.write_text(json.dumps(row))
            if i % 20 == 0:
                print(f"[v22-r1-deepseek] {i}/{len(test_groups)} picked", flush=True)
        g = groups_cand[gid]
        picked = row["picked_idx"]
        if picked is None or not (1 <= picked <= len(g["candidates"])):
            out[gid] = None
            continue
        idx0 = picked - 1
        best_tier = max(c["tier"] for c in g["candidates"])
        out[gid] = {"picked_idx0": idx0, "hit1": int(g["candidates"][idx0]["tier"] == best_tier),
                    "question": g["candidates"][idx0]["question"], "tier": g["candidates"][idx0]["tier"]}
    return out


def round2_pick_from_branch(selector, branch, e_candidates, tokenizer, model, rt) -> tuple[int | None, str]:
    block = build_candidates_block([{"question": c["question"]} for c in e_candidates])
    valid = set(range(1, len(e_candidates) + 1))
    if selector == "qwen_selfrank":
        picked, raw = qwen_select(tokenizer, model, branch["branch_prompt"], block, valid)
    else:
        raw = call_deepseek_select(rt, branch["branch_prompt"], block)
        picked = parse_answer(raw, valid)
    return picked, raw


def round2_pick_from_fresh(selector, prompt1: str, cands: list[dict], tokenizer, model, rt) -> tuple[int | None, str]:
    block = build_candidates_block([{"question": c["question"]} for c in cands])
    valid = set(range(1, len(cands) + 1))
    if selector == "qwen_selfrank":
        picked, raw = qwen_select(tokenizer, model, prompt1, block, valid)
    else:
        raw = call_deepseek_select(rt, prompt1, block)
        picked = parse_answer(raw, valid)
    return picked, raw


def fetch_fresh_candidates_for_selector(selector: str, needs_gids: list[str], prompt1_by_gid: dict,
                                         chunk_groups: dict, cache_dir: Path) -> dict[str, list[dict]]:
    """Same elicitation mechanism/prompt as v21's fetch_fresh_round2_candidates, but cached
    per-selector (the round-1 prompt being elicited against can differ from probe_1's pick,
    so it cannot safely share v21's own FRESH_R2_DIR cache)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    v21.load_aws_env()
    todo = [g for g in needs_gids if not (cache_dir / f"{v21.safe_name(g)}.json").exists()]
    print(f"[v22-fresh-r2-{selector}] {len(needs_gids)-len(todo)} cached, {len(todo)} to elicit", flush=True)
    if todo:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def worker(gid):
            try:
                rt = boto3.client("bedrock-runtime", region_name=v21.BEDROCK_REGION)
                g = chunk_groups[gid]
                cands = v21.call_deepseek_elicit_answer(rt, prompt1_by_gid[gid], g["reference_text"], g["extra_context"])
                (cache_dir / f"{v21.safe_name(gid)}.json").write_text(json.dumps({"id": gid, "candidates": cands}))
                return gid, None
            except Exception as e:  # noqa: BLE001
                return gid, str(e)

        with ThreadPoolExecutor(max_workers=v21.N_WORKERS) as ex:
            futs = {ex.submit(worker, gid): gid for gid in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                gid, err = fut.result()
                if err:
                    print(f"[v22-fresh-r2-{selector}] ({i}/{len(todo)}) ERROR {gid}: {err[:150]}", flush=True)
                elif i % 20 == 0:
                    print(f"[v22-fresh-r2-{selector}] ({i}/{len(todo)}) elicited", flush=True)
    out = {}
    for g in needs_gids:
        p = cache_dir / f"{v21.safe_name(g)}.json"
        if p.exists():
            out[g] = json.loads(p.read_text())["candidates"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", required=True, choices=["qwen_selfrank", "deepseek_selfrank"])
    ap.add_argument("--codegen", choices=["qwen", "deepseek"], default="qwen",
                     help="which model writes the code -- independent of --selector, which only "
                          "controls who picks the clarifying question")
    args = ap.parse_args()
    selector = args.selector
    codegen = args.codegen

    out_dir = OUT_ROOT / selector
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / ("results.json" if codegen == "qwen" else "results_deepseek_codegen.json")
    r1_cache_dir = out_dir / "r1_picks"
    r2_branch_cache_dir = out_dir / "r2_branch_picks"
    fresh_cache_dir = out_dir / "fresh_r2_candidates"
    fresh_pick_cache_dir = out_dir / "fresh_r2_picks"
    v21.CODEGEN_CACHE_DIR = out_dir / ("codegen" if codegen == "qwen" else "codegen_deepseek")
    v21.CODEGEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[v22] selector={selector} codegen={codegen}", flush=True)
    test_groups, branches_by_group, branches = reconstruct_test_groups()
    groups_cand = build_groups()
    v12_judged = {r["id"]: r for r in read_jsonl(V12_JUDGED_PATH)}
    chunk_groups = v21.load_chunk_groups()
    lookups = v21.load_raw_lookups()
    elicited_by_idx = {e["branch_idx"]: e for e in read_jsonl(ELICITED_PATH)}
    answers_r2 = {d["branch_idx"]: d for d in read_jsonl(ANSWERS_R2_PATH) if "error" not in d}

    scored_test_groups = [g for g in test_groups if g in groups_cand]
    print(f"[v22] {len(scored_test_groups)}/{len(test_groups)} test groups reconstructable", flush=True)

    need_qwen_model = selector == "qwen_selfrank"
    tokenizer = model = rt = None
    if need_qwen_model:
        print(f"[v22] loading {MODEL_NAME} ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = v21.with_retry(lambda: AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16, device_map="cuda"), log_prefix="[v22-load] ")
        model.eval()
        print("[v22] model loaded", flush=True)
    else:
        v21.load_aws_env()
        rt = boto3.client("bedrock-runtime", region_name=v21.BEDROCK_REGION)

    # --- Round 1 picks ---
    r1 = get_round1_picks(selector, scored_test_groups, groups_cand, tokenizer, model, rt, r1_cache_dir)
    scored_test_groups = [g for g in scored_test_groups if r1.get(g) is not None]
    n1 = len(scored_test_groups)
    hit1_count = sum(r1[g]["hit1"] for g in scored_test_groups)
    print(f"[v22] {selector} round-1 hit rate: {hit1_count}/{n1} = {hit1_count/n1:.4f}", flush=True)

    # --- Determine round-2 source: chained branch (miss w/ matching branch) or fresh elicit ---
    prompt_by_gid = {g: chunk_groups[g]["prompt"] for g in chunk_groups}
    answers_by_gid = {g: v19_load_answers_by_question(g, v12_judged) for g in scored_test_groups}
    prompt1_by_gid = {}
    for g in scored_test_groups:
        ans = answers_by_gid[g].get(r1[g]["question"])
        prompt1_by_gid[g] = v21.CONCAT_TEMPLATE.format(prompt=prompt_by_gid[g], answer=ans)

    chain_branch = {}
    needs_fresh = []
    for g in scored_test_groups:
        if r1[g]["hit1"]:
            needs_fresh.append(g)
            continue
        match = next((idx for idx, bq in branches_by_group[g] if bq == r1[g]["question"]), None)
        if match is None or match not in elicited_by_idx:
            needs_fresh.append(g)
        else:
            chain_branch[g] = match
    print(f"[v22] {len(chain_branch)} chain to an existing branch; {len(needs_fresh)} need fresh round-2", flush=True)

    fresh_cands = fetch_fresh_candidates_for_selector(selector, needs_fresh, prompt1_by_gid, chunk_groups, fresh_cache_dir)

    # --- Round 2 picks ---
    prompt2_by_gid = {}
    fresh_pick_cache_dir.mkdir(parents=True, exist_ok=True)
    r2_branch_cache_dir.mkdir(parents=True, exist_ok=True)
    for i, g in enumerate(scored_test_groups, 1):
        if g in chain_branch:
            cache_p = r2_branch_cache_dir / f"{v21.safe_name(g)}.json"
            branch_idx = chain_branch[g]
            branch = branches[branch_idx]
            e_cands = elicited_by_idx[branch_idx]["candidates"]
            if cache_p.exists():
                picked = json.loads(cache_p.read_text())["picked_idx0"]
            else:
                picked1, raw = round2_pick_from_branch(selector, branch, e_cands, tokenizer, model, rt)
                picked = (picked1 - 1) if picked1 is not None else None
                cache_p.write_text(json.dumps({"id": g, "picked_idx0": picked, "raw": raw}))
            if picked is None:
                prompt2_by_gid[g] = prompt1_by_gid[g]  # self-rank failed to parse -- no round-2 change
                continue
            ans_by_q = {a["question"].strip(): a["answer"] for a in answers_r2.get(branch_idx, {}).get("answers", [])
                        if "question" in a and "answer" in a}
            q = e_cands[picked]["question"].strip()
            ans = ans_by_q.get(q)
            prompt2_by_gid[g] = (v21.CONCAT_TEMPLATE.format(prompt=branch["branch_prompt"], answer=ans)
                                 if ans else prompt1_by_gid[g])
        else:
            cands = fresh_cands.get(g) or []
            cands = [c for c in cands if c.get("question") and c.get("answer")]
            if not cands:
                prompt2_by_gid[g] = prompt1_by_gid[g]
                continue
            if len(cands) < 2:
                prompt2_by_gid[g] = v21.CONCAT_TEMPLATE.format(prompt=prompt1_by_gid[g], answer=cands[0]["answer"])
                continue
            cache_p = fresh_pick_cache_dir / f"{v21.safe_name(g)}.json"
            if cache_p.exists():
                picked = json.loads(cache_p.read_text())["picked_idx0"]
            else:
                picked1, raw = round2_pick_from_fresh(selector, prompt1_by_gid[g], cands, tokenizer, model, rt)
                picked = (picked1 - 1) if picked1 is not None else None
                cache_p.write_text(json.dumps({"id": g, "picked_idx0": picked, "raw": raw}))
            prompt2_by_gid[g] = (v21.CONCAT_TEMPLATE.format(prompt=prompt1_by_gid[g], answer=cands[picked]["answer"])
                                 if picked is not None else prompt1_by_gid[g])
        if i % 30 == 0:
            print(f"[v22-r2-{selector}] {i}/{len(scored_test_groups)} round-2 picks done", flush=True)

    # --- Build codegen tasks (reusing v21's problem_info/code_instruction/is_sane/test harness) ---
    tasks = []
    for g in scored_test_groups:
        source = chunk_groups[g]["dataset_source"]
        pinfo = v21.problem_info(g, source, lookups)
        prompt1, prompt2 = prompt1_by_gid[g], prompt2_by_gid[g]
        tasks.append({"id": g, "source": source, "pinfo": pinfo, "prompt1": prompt1, "prompt2": prompt2,
                      "hit1": r1[g]["hit1"], "same_prompt": prompt1 == prompt2})
    print(f"[v22] {len(tasks)} tasks built", flush=True)

    if not need_qwen_model and codegen == "qwen":
        print(f"[v22] loading {MODEL_NAME} for codegen ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = v21.with_retry(lambda: AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16, device_map="cuda"), log_prefix="[v22-load] ")
        model.eval()
        print("[v22] model loaded", flush=True)
    elif not need_qwen_model and codegen == "deepseek":
        print("[v22] selector and codegen both DeepSeek -- no GPU/Qwen needed for this run", flush=True)

    ids_list = [t["id"] for t in tasks]
    round1_inputs = [v21.code_instruction(t["pinfo"], t["prompt1"]) for t in tasks]
    sane1 = [v21.make_is_sane(t["pinfo"]) for t in tasks]
    print("[v22] generating round-1 code ...", flush=True)
    if codegen == "deepseek":
        raw1 = v21.generate_cached_deepseek(ids_list, "r1", round1_inputs, sane1)
    else:
        raw1 = v21.generate_cached(tokenizer, model, ids_list, "r1", round1_inputs, sane1)

    diff_idxs = [i for i, t in enumerate(tasks) if not t["same_prompt"]]
    diff_ids = [ids_list[i] for i in diff_idxs]
    round2_inputs = [v21.code_instruction(tasks[i]["pinfo"], tasks[i]["prompt2"]) for i in diff_idxs]
    sane2 = [v21.make_is_sane(tasks[i]["pinfo"]) for i in diff_idxs]
    print(f"[v22] generating round-2 code for {len(diff_idxs)} groups ...", flush=True)
    if diff_idxs:
        raw2_partial = (v21.generate_cached_deepseek(diff_ids, "r2", round2_inputs, sane2) if codegen == "deepseek"
                        else v21.generate_cached(tokenizer, model, diff_ids, "r2", round2_inputs, sane2))
    else:
        raw2_partial = []
    raw2 = list(raw1)
    for k, i in enumerate(diff_idxs):
        raw2[i] = raw2_partial[k]

    if model is not None:
        del model
        torch.cuda.empty_cache()

    results = []
    for i, t in enumerate(tasks):
        r1res = v21.run_tests_for(t["pinfo"], raw1[i])
        r2res = r1res if t["same_prompt"] else v21.run_tests_for(t["pinfo"], raw2[i])
        results.append({
            "id": t["id"], "source": t["source"], "hit1": t["hit1"], "same_prompt": t["same_prompt"],
            "round1_n_pass": r1res["n_pass"], "round1_n_total": r1res["n_total"], "round1_pass": r1res["all_pass"],
            "round2_n_pass": r2res["n_pass"], "round2_n_total": r2res["n_total"], "round2_pass": r2res["all_pass"],
        })
        print(f"[v22-{selector}] ({i+1}/{len(tasks)}) {t['id']} [{t['source']}] "
              f"round1={r1res['n_pass']}/{r1res['n_total']} round2={r2res['n_pass']}/{r2res['n_total']}", flush=True)

    results_path.write_text(json.dumps(results, indent=2))

    def summarize(rows, key):
        n = len(rows)
        return (sum(r[key] for r in rows) / n, n) if n else (0.0, 0)

    r1_rate, n = summarize(results, "round1_pass")
    r2_rate, _ = summarize(results, "round2_pass")
    print(f"\n[v22] === {selector} SUMMARY (n={n}) ===", flush=True)
    print(f"  OVERALL: round1-only={r1_rate:.4f}  round1+round2={r2_rate:.4f}", flush=True)
    by_source = defaultdict(list)
    for r in results:
        by_source[r["source"]].append(r)
    for src, rows in sorted(by_source.items()):
        r1s, ns = summarize(rows, "round1_pass")
        r2s, _ = summarize(rows, "round2_pass")
        print(f"  {src:20s} (n={ns:3d}): round1-only={r1s:.4f}  round1+round2={r2s:.4f}", flush=True)
    print(f"[v22] saved -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
