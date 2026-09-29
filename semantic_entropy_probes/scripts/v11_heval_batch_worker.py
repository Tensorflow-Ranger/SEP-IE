"""v11: same elicit -> oracle -> score -> clarify -> codegen -> test loop as
v10_batch_worker.py, on the 32 Orchid-HEval "easy problems" test set
(data/v10_code_quality/held_out_orchid32.json) instead of AtCoder/LiveCodeBench.

Question ELICITATION now goes through the same Groq-primary/OpenRouter-fallback
API agent used for codegen (scripts/coding_agent.py's generate_agent), instead
of local Qwen3-8B. Rationale: a qualitative review of the first (all-local) run
found that on ~12/32 round-1 "misses" -- where the chosen clarifying question
didn't target the actually-injected ambiguity -- about half the time the
correct question was never even proposed among Qwen3-8B's up-to-8 candidates
at all (a pure elicitation-capability gap, not a probe-scoring failure; see
data/v10_code_quality/heval32/ for that run and its analysis). This run tests
whether a more capable model proposes the right question more often. Probe
SCORING stays on local Qwen3-8B regardless -- the rank1/rank2 probes were
trained on Qwen3-8B's own activations, so scoring literally requires this
model's hidden states; swapping it would invalidate the probes entirely, not
just change which model is "smarter." The inline STYLE_INSTRUCTION rewrite and
CODE_INSTRUCTION codegen also stay local Qwen3-8B, unchanged from the prior run
-- only elicitation moved to the API, to isolate that one variable.

The rank1/rank2 probes used for scoring were trained on a combined dataset
that includes orchid_heval (see models/rank1_lr_layer20_pca100_ccb_heval.json)
-- so scoring these prompts isn't off-distribution the way an entirely new
problem source would be.

Round 0 (no clarification) is IDENTICAL across all these run variants -- no
elicitation involved either way -- so it's reused from the original all-local
run (data/v10_code_quality/heval32/round_0/) rather than regenerated.

Same oracle protocol as v10_batch_worker.py: this script does NOT answer its
own elicited questions. It writes them to round_N/questions.json and blocks on
round_N/answers.json, expected to be written by an isolated agent grounded
ONLY in that problem's `clear_prompt` + `solution` + `missing_detail` (NOT this
pipeline's internal state).

Usage: python v11_heval_batch_worker.py --amb_ids heval_1 heval_104 ... [--rounds 3]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import joblib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.heval_run_tests import run_all  # noqa: E402
from scripts.coding_agent import extract_code_block, generate_agent  # noqa: E402
from scripts.v10_round_worker import (  # noqa: E402
    RANK1_LAYER, RANK1_PATH, RANK2_LAYER, RANK2_PATH,
    generate_sane, hidden_states_at_layers, is_sane_rewrite, parse_numbered, score,
)

MODEL_NAME = "Qwen/Qwen3-8B"  # scoring, rewrite, codegen (elicitation now via API, see generate_agent)
HELD_OUT_PATH = ROOT / "data" / "v10_code_quality" / "held_out_orchid32.json"
# round 0 has no elicitation either way -- reuse the original all-local run's baseline
LOCAL_ELICIT_ROUND0_DIR = ROOT / "data" / "v10_code_quality" / "heval32" / "round_0"

ELICIT_QUESTIONS_INSTRUCTION = (
    "You are reviewing a Python function specification (a function signature plus its "
    "docstring) for specification ambiguity -- places where the docstring under-specifies "
    "behavior, leaving multiple reasonable interpretations (e.g. edge cases, what counts as "
    "a valid/separate case, tie-breaking rules, boundary conditions, argument constraints "
    "left implicit, terminology, etc.).\n\n"
    "Here is the function specification:\n<problem>\n{original}\n</problem>\n\n"
    "List up to {max_candidates} distinct QUESTIONS you would need answered to remove ambiguity "
    "from this specification before implementing it. The questions MUST be diverse -- cover "
    "DIFFERENT kinds of ambiguity (e.g. terminology/what a word or phrase means, the core "
    "behavior or algorithm logic, output format, edge cases, indices/ranges, comparison or "
    "tie-breaking rules, collection semantics), not just variations on the same generic "
    "edge-case theme (type validation, negative numbers, empty input, overflow, etc.). Do not "
    "propose two questions that probe the same underlying ambiguity in slightly different "
    "words. If the specification's core computation, terminology, or behavior is genuinely "
    "unclear, prioritize asking about THAT over generic input-validation edge cases. For EACH "
    "one, output a single line in this exact format:\nN. <one-sentence question>\n\n"
    "Only output the numbered list, nothing else -- no preamble, no explanation. Each item MUST "
    "be phrased as an actual question (ending in '?'), not a statement."
)

STYLE_INSTRUCTION = (
    "You are editing a Python function's docstring specification (concise, precise, in the "
    "style of a real Python docstring -- keeps the same function signature line and overall "
    "structure).\n\n"
    "Here is the ORIGINAL function specification:\n<original>\n{original}\n</original>\n\n"
    "Rewrite this specification to incorporate the following change, INLINE into the existing "
    "docstring prose (edit an existing sentence, or insert one short sentence in the same place "
    "and style a real author would have put it originally). Do NOT append a new bullet point or "
    "trailing note at the end. Do NOT change the function signature or anything else about the "
    "text -- keep the rest word-for-word identical. Output ONLY the full rewritten function "
    "specification (signature + docstring), with no preamble, no explanation, and no markdown "
    "formatting around it.\n\n"
    "Change to make: {detail}"
)

CODE_INSTRUCTION = (
    "Complete the following Python function. Include any necessary imports (the signature line "
    "itself may omit them). Output ONLY the complete function -- the given signature plus your "
    "implementation -- optionally inside a single ```python ... ``` block. No explanation, no "
    "usage examples, no extra text.\n\n"
    "<problem>\n{problem}\n</problem>"
)


def is_sane_code(raw_text: str, entry_point: str) -> bool:
    """Reject output that isn't usable Python defining the required function: must parse,
    and must actually define entry_point (rejects e.g. the model explaining instead of
    coding, or defining a differently-named/wrapped function)."""
    code = extract_code_block(raw_text)
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return f"def {entry_point}" in code


def elicit_questions_via_api(problem: str, max_candidates: int, max_sanity_retries: int = 3) -> tuple[list[str], str]:
    """Elicit clarifying questions via the Groq/OpenRouter API agent instead of local
    Qwen3-8B (see scripts/coding_agent.py's generate_agent -- same provider fallback used
    for codegen). generate_agent already retries per-provider on 429s/empty content; this
    wraps it with an outer retry that checks the *content* is actually usable (>=1
    parseable question), since a well-formed non-empty response can still fail to follow
    the numbered-list format. Returns (questions, provider_used)."""
    uc = ELICIT_QUESTIONS_INSTRUCTION.format(original=problem, max_candidates=max_candidates)
    last_qs, last_provider = [], None
    for _ in range(max_sanity_retries):
        raw, provider = generate_agent(uc, max_tokens=500)
        qs = parse_numbered(raw, max_candidates)
        last_qs, last_provider = qs, provider
        if qs:
            return qs, provider
    return last_qs, last_provider


def wait_for_file(path: Path, poll_s: int = 5):
    while not path.exists():
        time.sleep(poll_s)
    time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amb_ids", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max_candidates", type=int, default=8)
    ap.add_argument("--held_out_path", type=str, default=str(HELD_OUT_PATH))
    ap.add_argument("--out_dir", type=str,
                     default=str(ROOT / "data" / "v10_code_quality" / "heval32_api_elicit"))
    args = ap.parse_args()

    held_out = json.load(open(args.held_out_path))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v11] {len(args.amb_ids)} amb_ids, {args.rounds} rounds, "
          f"{args.max_candidates} candidates/round", flush=True)
    print(f"[v11] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    rank1 = joblib.load(RANK1_PATH)
    rank2 = joblib.load(RANK2_PATH)
    print(f"[v11] model + probes loaded (rank1_layer={RANK1_LAYER}, rank2_layer={RANK2_LAYER})", flush=True)

    state = {
        aid: {
            "current_prompt": held_out[aid]["ambiguous_prompt"],
            "inline_prompt": held_out[aid]["ambiguous_prompt"],
        }
        for aid in args.amb_ids
    }

    layers = [RANK1_LAYER, RANK2_LAYER]
    all_rounds_summary = {aid: [] for aid in args.amb_ids}

    # ---- round 0: no-clarification baseline, straight from ambiguous_prompt ----
    # identical across every run variant (no elicitation involved), so reuse the
    # original all-local run's baseline where available instead of regenerating.
    round0_dir = out_dir / "round_0"
    round0_dir.mkdir(exist_ok=True)
    for i, aid in enumerate(args.amb_ids):
        summary_path = round0_dir / aid / "summary.json"
        if summary_path.exists():
            all_rounds_summary[aid].append(json.load(open(summary_path)))
            continue
        reuse_src = LOCAL_ELICIT_ROUND0_DIR / aid / "summary.json"
        if reuse_src.exists():
            rs = json.load(open(reuse_src))
            (round0_dir / aid).mkdir(exist_ok=True)
            json.dump(rs, open(summary_path, "w"), indent=2)
            all_rounds_summary[aid].append(rs)
            print(f"[v11][round 0] {i+1}/{len(args.amb_ids)} {aid}: reused from {reuse_src}", flush=True)
            continue
        item = held_out[aid]
        code_user = CODE_INSTRUCTION.format(problem=item["ambiguous_prompt"])
        raw_code = generate_sane(
            tokenizer, model, code_user, max_new_tokens=600,
            is_sane=lambda raw: is_sane_code(raw, item["entry_point"]),
            log_prefix=f"[v11][round 0][codegen][{aid}] ",
        )
        code = extract_code_block(raw_code)
        test_res = run_all(code, item["entry_point"], item["test_case"])
        round0_dir_aid = round0_dir / aid
        round0_dir_aid.mkdir(exist_ok=True)
        (round0_dir_aid / "generated_code.py").write_text(code)
        round_summary = {
            "round": 0, "code_prompt": item["ambiguous_prompt"],
            "test_n_pass": test_res["n_pass"], "test_n_total": test_res["n_total"],
        }
        json.dump(round_summary, open(summary_path, "w"), indent=2)
        all_rounds_summary[aid].append(round_summary)
        print(f"[v11][round 0] {i+1}/{len(args.amb_ids)} {aid}: "
              f"pass={test_res['n_pass']}/{test_res['n_total']}", flush=True)

    for r in range(1, args.rounds + 1):
        round_dir = out_dir / f"round_{r}"
        round_dir.mkdir(exist_ok=True)
        t0 = time.time()

        already_done = all((round_dir / aid / "summary.json").exists() for aid in args.amb_ids)
        if already_done:
            for aid in args.amb_ids:
                rs = json.load(open(round_dir / aid / "summary.json"))
                state[aid]["current_prompt"] = rs["next_prompt"]
                state[aid]["inline_prompt"] = rs["code_prompt"]
                all_rounds_summary[aid].append(rs)
            print(f"[v11][round {r}] RESUMED from existing per-amb_id summaries, skipping", flush=True)
            continue

        # ---- 1. elicit questions for every amb_id, via the API agent (Groq -> OpenRouter) ----
        all_questions = {}
        elicit_providers = {}
        for i, aid in enumerate(args.amb_ids):
            cp = state[aid]["current_prompt"]
            qs, provider = elicit_questions_via_api(cp, args.max_candidates)
            all_questions[aid] = qs
            elicit_providers[aid] = provider
            print(f"[v11][round {r}] elicited {i+1}/{len(args.amb_ids)} ({aid}: {len(qs)} qs via {provider})",
                  flush=True)
            if not qs:
                print(f"[v11][round {r}] WARNING: 0 questions for {aid} even after retries", flush=True)

        q_path = round_dir / "questions.json"
        a_path = round_dir / "answers.json"
        json.dump(all_questions, open(q_path, "w"), indent=2)
        print(f"[v11][round {r}] elicitation done for all {len(args.amb_ids)} amb_ids "
              f"({time.time()-t0:.0f}s)", flush=True)

        # ---- 2. block for combined oracle answers ----
        print(f"[v11][round {r}] WAITING_FOR_ORACLE questions_file={q_path} answers_file={a_path}", flush=True)
        wait_for_file(a_path)
        all_answers = json.load(open(a_path))
        for aid in args.amb_ids:
            assert aid in all_answers, f"missing answers for {aid}"
            assert len(all_answers[aid]["answers"]) == len(all_questions[aid]), \
                f"{aid}: answers/questions length mismatch"
        print(f"[v11][round {r}] got oracle answers for all {len(args.amb_ids)} amb_ids", flush=True)

        # ---- 3. score, clarify, codegen, test -- per amb_id ----
        t1 = time.time()
        for i, aid in enumerate(args.amb_ids):
            item = held_out[aid]
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

            if best is None:
                print(f"[v11][round {r}] {aid}: NO CANDIDATES, skipping clarify/codegen", flush=True)
                next_prompt, next_inline = cp, ip
            else:
                next_prompt = best["prompt"]
                detail = f"Add this clarification, integrated naturally into the relevant sentence: {best['answer']}"
                rewrite_user = STYLE_INSTRUCTION.format(original=ip, detail=detail)
                next_inline = generate_sane(
                    tokenizer, model, rewrite_user, max_new_tokens=500,
                    is_sane=lambda raw: is_sane_rewrite(raw, ip),
                    log_prefix=f"[v11][round {r}][rewrite][{aid}] ",
                )

            code_user = CODE_INSTRUCTION.format(problem=next_inline)
            raw_code = generate_sane(
                tokenizer, model, code_user, max_new_tokens=600,
                is_sane=lambda raw: is_sane_code(raw, item["entry_point"]),
                log_prefix=f"[v11][round {r}][codegen][{aid}] ",
            )
            code = extract_code_block(raw_code)
            test_res = run_all(code, item["entry_point"], item["test_case"])

            round_dir_aid = round_dir / aid
            round_dir_aid.mkdir(exist_ok=True)
            (round_dir_aid / "generated_code.py").write_text(code)

            round_summary = {
                "round": r,
                "baseline_prompt": cp,
                "baseline_mean_p": base_mean,
                "elicit_provider": elicit_providers[aid],
                "candidates": cand_results,
                "top3": [{"idx": c["idx"], "question": c["question"], "answer": c["answer"], "delta": c["delta"]} for c in top3],
                "chosen": ({"idx": best["idx"], "question": best["question"], "answer": best["answer"], "delta": best["delta"]}
                           if best else None),
                "next_prompt": next_prompt,
                "code_prompt": next_inline,
                "test_n_pass": test_res["n_pass"],
                "test_n_total": test_res["n_total"],
            }
            json.dump(round_summary, open(round_dir_aid / "summary.json", "w"), indent=2)
            all_rounds_summary[aid].append(round_summary)

            delta_str = f"top1_delta={best['delta']:+.4f}" if best else "no candidates"
            print(f"[v11][round {r}] {i+1}/{len(args.amb_ids)} {aid}: "
                  f"pass={test_res['n_pass']}/{test_res['n_total']} {delta_str}", flush=True)

            state[aid]["current_prompt"] = next_prompt
            state[aid]["inline_prompt"] = next_inline

        print(f"[v11][round {r}] all clarify/codegen/test done ({time.time()-t1:.0f}s)", flush=True)

    json.dump(all_rounds_summary, open(out_dir / "all_amb_ids_summary.json", "w"), indent=2)
    print("[v11] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
