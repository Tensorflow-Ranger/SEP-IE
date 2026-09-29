"""Batched, wave-based version of v10_round_worker.py for running many
held-out prompts through the elicit -> oracle -> score -> clarify -> codegen
-> test loop without reloading Qwen3-8B once per prompt (18 prompts x 3
rounds = 54 GPU-process launches under the per-prompt design, ~90s reload
each) and without 54 separate oracle round-trips. Elicitation/scoring/style-
rewrite still use the local Qwen3-8B; codegen is the coding agent -- Groq
gpt-oss-120b, falling back to OpenRouter Qwen3-Coder-480B-A35B when Groq is
exhausted (see generate_code_agent in scripts/coding_agent.py).

Model is loaded ONCE. All `--amb_ids` are advanced through round r together:
elicitation for every amb_id happens first, all their questions are written
to one round_N/questions.json, and the script blocks on one round_N/answers.json
(keyed by amb_id) before scoring/clarifying/codegen/testing each amb_id and
moving everyone to round r+1. The oracle side (this script does not answer
its own questions) is expected to split the amb_ids into a few groups and
answer each group with an isolated agent grounded only in that group's true
prompts -- but from this script's point of view there is exactly one
combined answers file per round, however many oracle calls produced it.

Per-prompt state (current_prompt for elicitation/scoring, inline_prompt for
codegen) is cumulative across rounds, same semantics as v10_round_worker.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v10_run_tests import run_all  # noqa: E402
from scripts.v10_round_worker import (  # noqa: E402
    ELICIT_QUESTIONS_INSTRUCTION, STYLE_INSTRUCTION,
    RANK1_LAYER, RANK1_PATH, RANK2_LAYER, RANK2_PATH,
    extract_code_block, generate_code_agent, generate_sane, hidden_states_at_layers,
    is_sane_rewrite, parse_numbered, score,
)

MODEL_NAME = "Qwen/Qwen3-8B"


def wait_for_file(path: Path, poll_s: int = 5):
    while not path.exists():
        time.sleep(poll_s)
    time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amb_ids", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max_candidates", type=int, default=12)
    ap.add_argument("--held_out_path", type=str,
                     default=str(ROOT / "data" / "v10_code_quality" / "held_out_40.json"))
    ap.add_argument("--out_dir", type=str,
                     default=str(ROOT / "data" / "v10_code_quality" / "batch18"))
    args = ap.parse_args()

    held_out = json.load(open(args.held_out_path))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v10_batch] {len(args.amb_ids)} amb_ids, {args.rounds} rounds, "
          f"{args.max_candidates} candidates/round", flush=True)
    print(f"[v10_batch] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    rank1 = joblib.load(RANK1_PATH)
    rank2 = joblib.load(RANK2_PATH)
    print(f"[v10_batch] model + probes loaded (rank1_layer={RANK1_LAYER}, rank2_layer={RANK2_LAYER})", flush=True)

    state = {
        aid: {
            "current_prompt": held_out[aid]["ambiguous_prompt"],
            "inline_prompt": held_out[aid]["ambiguous_prompt"],
        }
        for aid in args.amb_ids
    }

    layers = [RANK1_LAYER, RANK2_LAYER]
    all_rounds_summary = {aid: [] for aid in args.amb_ids}

    for r in range(1, args.rounds + 1):
        round_dir = out_dir / f"round_{r}"
        round_dir.mkdir(exist_ok=True)
        t0 = time.time()

        # ---- resume: if every amb_id already has a completed summary for this round
        # (from a prior, possibly-crashed run), skip straight to updating state ----
        already_done = all((round_dir / aid / "summary.json").exists() for aid in args.amb_ids)
        if already_done:
            for aid in args.amb_ids:
                rs = json.load(open(round_dir / aid / "summary.json"))
                state[aid]["current_prompt"] = rs["next_prompt"]
                state[aid]["inline_prompt"] = rs["code_prompt"]
                all_rounds_summary[aid].append(rs)
            print(f"[v10_batch][round {r}] RESUMED from existing per-amb_id summaries, skipping", flush=True)
            continue

        # ---- 1. elicit questions for every amb_id ----
        all_questions = {}
        for i, aid in enumerate(args.amb_ids):
            cp = state[aid]["current_prompt"]
            uc = ELICIT_QUESTIONS_INSTRUCTION.format(original=cp, max_candidates=args.max_candidates)
            raw_q = generate_sane(
                tokenizer, model, uc, max_new_tokens=700,
                is_sane=lambda raw: len(parse_numbered(raw, args.max_candidates)) >= 1,
                log_prefix=f"[v10_batch][round {r}][elicit][{aid}] ",
            )
            qs = parse_numbered(raw_q, args.max_candidates)
            all_questions[aid] = qs
            print(f"[v10_batch][round {r}] elicited {i+1}/{len(args.amb_ids)} ({aid}: {len(qs)} qs)", flush=True)
            if not qs:
                print(f"[v10_batch][round {r}] WARNING: 0 questions for {aid} even after retries, "
                      f"raw={raw_q[:200]!r}", flush=True)

        q_path = round_dir / "questions.json"
        a_path = round_dir / "answers.json"
        json.dump(all_questions, open(q_path, "w"), indent=2)
        print(f"[v10_batch][round {r}] elicitation done for all {len(args.amb_ids)} amb_ids "
              f"({time.time()-t0:.0f}s)", flush=True)

        # ---- 2. block for combined oracle answers ----
        print(f"[v10_batch][round {r}] WAITING_FOR_ORACLE questions_file={q_path} answers_file={a_path}", flush=True)
        wait_for_file(a_path)
        all_answers = json.load(open(a_path))
        for aid in args.amb_ids:
            assert aid in all_answers, f"missing answers for {aid}"
            assert len(all_answers[aid]["answers"]) == len(all_questions[aid]), \
                f"{aid}: answers/questions length mismatch"
        print(f"[v10_batch][round {r}] got oracle answers for all {len(args.amb_ids)} amb_ids", flush=True)

        # ---- 3. score, clarify, codegen, test -- per amb_id ----
        t1 = time.time()
        for i, aid in enumerate(args.amb_ids):
            cp = state[aid]["current_prompt"]
            ip = state[aid]["inline_prompt"]
            qs = all_questions[aid]
            ans = all_answers[aid]["answers"]

            base_vecs = hidden_states_at_layers(tokenizer, model, cp, layers)
            base_p1, base_p2 = score(rank1, rank2, base_vecs)
            base_mean = (base_p1 + base_p2) / 2

            cand_results = []
            for cidx, (q, a) in enumerate(zip(qs, ans)):
                cand_prompt = cp + f"\n\nClarification: {a}"
                vecs = hidden_states_at_layers(tokenizer, model, cand_prompt, layers)
                p1, p2 = score(rank1, rank2, vecs)
                mean_p = (p1 + p2) / 2
                cand_results.append({
                    "idx": cidx, "question": q, "answer": a, "prompt": cand_prompt,
                    "p1": p1, "p2": p2, "mean_p": mean_p, "delta": mean_p - base_mean,
                })
            cand_results.sort(key=lambda c: c["delta"])
            top3 = cand_results[:3]
            best = cand_results[0] if cand_results else None

            coding_agent_provider = None
            if best is None:
                print(f"[v10_batch][round {r}] {aid}: NO CANDIDATES, skipping clarify/codegen", flush=True)
                next_prompt, next_inline, code, test_res = cp, ip, "", {"n_pass": 0, "n_total": 0, "results": []}
            else:
                next_prompt = best["prompt"]
                detail = f"Add this clarification, integrated naturally into the relevant sentence: {best['answer']}"
                rewrite_user = STYLE_INSTRUCTION.format(original=ip, detail=detail)
                next_inline = generate_sane(
                    tokenizer, model, rewrite_user, max_new_tokens=600,
                    is_sane=lambda raw: is_sane_rewrite(raw, ip),
                    log_prefix=f"[v10_batch][round {r}][rewrite][{aid}] ",
                )

                raw_code, coding_agent_provider = generate_code_agent(next_inline)
                code = extract_code_block(raw_code)
                test_res = run_all(code, held_out[aid]["test_cases"])

            round_dir_aid = round_dir / aid
            round_dir_aid.mkdir(exist_ok=True)
            (round_dir_aid / "generated_code.py").write_text(code)

            round_summary = {
                "round": r,
                "baseline_prompt": cp,
                "baseline_mean_p": base_mean,
                "candidates": cand_results,
                "top3": [{"idx": c["idx"], "question": c["question"], "answer": c["answer"], "delta": c["delta"]} for c in top3],
                "chosen": ({"idx": best["idx"], "question": best["question"], "answer": best["answer"], "delta": best["delta"]}
                           if best else None),
                "next_prompt": next_prompt,
                "code_prompt": next_inline,
                "coding_agent_provider": coding_agent_provider,
                "test_n_pass": test_res["n_pass"],
                "test_n_total": test_res["n_total"],
            }
            json.dump(round_summary, open(round_dir_aid / "summary.json", "w"), indent=2)
            all_rounds_summary[aid].append(round_summary)

            print(f"[v10_batch][round {r}] {i+1}/{len(args.amb_ids)} {aid}: "
                  f"pass={test_res['n_pass']}/{test_res['n_total']} "
                  f"top1_delta={best['delta']:+.4f}" if best else "no candidates", flush=True)

            state[aid]["current_prompt"] = next_prompt
            state[aid]["inline_prompt"] = next_inline

        print(f"[v10_batch][round {r}] all clarify/codegen/test done ({time.time()-t1:.0f}s)", flush=True)

    json.dump(all_rounds_summary, open(out_dir / "all_amb_ids_summary.json", "w"), indent=2)
    print("[v10_batch] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
