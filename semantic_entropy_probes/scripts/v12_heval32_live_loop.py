"""Live multi-round elicit -> oracle -> score -> clarify -> codegen -> test
loop on a subset of heval32, using the NEW v12 pairwise-ranking probe
(models/rank_probe_v12_pairwise.joblib) instead of the old deployed
rank1/rank2 probes -- the real end-to-end test of whether the new training
objective actually helps pick better clarifying questions in practice, not
just on the static tier-judged eval set.

Reuses v12_extract_activations.py's STYLE_INSTRUCTION for the inline rewrite
(this feeds the probe's hidden-state extraction, so it must match what the
probe was trained on). Question ELICITATION, however, uses its OWN
HumanEval-docstring-specific ELICIT_QUESTIONS_INSTRUCTION defined below --
NOT v12_elicit_candidates.py's competitive-programming-worded version --
since elicitation quality doesn't feed the probe's representation space and
a domain-matched prompt measurably helps the elicitation model ask relevant
questions. (Confirmed this mattered: 4/10 problems in an earlier run of this
script kept missing the actual injected ambiguity in all 3 rounds because
elicitation was using the wrong-genre instruction.) Also elicits 12
candidates per round (not 8), to give the probe a wider pool to rank over.

Unlike v10/v11, only ONE prompt is tracked per problem (the natural-flowing
inline-rewritten prompt) -- no separate raw-concatenation "current_prompt"
is needed since the new probe scores candidates by absolute score
(argmax w.x over the candidate set), not a baseline-relative delta the way
the old probes' i.i.d. binary objective required.

Oracle answering is NOT automated in-process (same protocol as v10/v11):
this script's "elicit" mode writes round_N/questions.json and stops; a
separate agent (haiku, grounded ONLY in each problem's missing_detail /
clear_prompt / solution -- not this script's internal state) must write
round_N/answers.json; then this script's "score" mode picks up from there.

Usage:
  python v12_heval32_live_loop.py --mode elicit --round N --amb_ids ...
  python v12_heval32_live_loop.py --mode score  --round N --amb_ids ...
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.heval_run_tests import run_all  # noqa: E402
from scripts.coding_agent import extract_code_block  # noqa: E402
from scripts.coding_agent import generate_agent  # noqa: E402
from scripts.v12_elicit_candidates import parse_numbered  # noqa: E402
from scripts.v12_extract_activations import (  # noqa: E402
    MODEL_NAME,
    STYLE_INSTRUCTION,
    generate_sane_batch,
    hidden_states_at_layers_batch,
    is_sane_rewrite,
)

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


def elicit_questions_via_api(problem: str, max_candidates: int, max_sanity_retries: int = 3) -> tuple[list[str], str]:
    """HumanEval-docstring-specific elicitation (domain-matched to heval32's actual problem
    style, unlike v12_elicit_candidates.py's competitive-programming-worded version)."""
    uc = ELICIT_QUESTIONS_INSTRUCTION.format(original=problem, max_candidates=max_candidates)
    last_qs, last_provider = [], None
    for _ in range(max_sanity_retries):
        raw, provider = generate_agent(uc, max_tokens=500)
        qs = parse_numbered(raw, max_candidates)
        last_qs, last_provider = qs, provider
        if qs:
            return qs, provider
    return last_qs, last_provider

VARIANT_PATHS = {
    "vagueness": {
        "held_out": ROOT / "data" / "v10_code_quality" / "held_out_orchid32.json",
        "round0_reuse_dir": ROOT / "data" / "v10_code_quality" / "heval32" / "round_0",
        "out_dir": ROOT / "data" / "v12_ranking" / "heval32_live_loop",
    },
    "lexical": {
        "held_out": ROOT / "data" / "v10_code_quality" / "held_out_orchid32_lexical.json",
        # No pre-existing zero-shot baseline for the lexical prompts (they were never
        # run through v10/v11) -- ensure_round0() generates a fresh one when this dir
        # has no matching reuse file, instead of the vagueness path's pure reuse.
        "round0_reuse_dir": None,
        "out_dir": ROOT / "data" / "v12_ranking" / "heval32_lexical_live_loop",
    },
}

HELD_OUT_PATH = VARIANT_PATHS["vagueness"]["held_out"]
ROUND0_REUSE_DIR = VARIANT_PATHS["vagueness"]["round0_reuse_dir"]
OUT_DIR = VARIANT_PATHS["vagueness"]["out_dir"]
STATE_PATH = OUT_DIR / "state.json"
MODEL_JSON_PATH = ROOT / "models" / "rank_probe_v12_pairwise.json"
MODEL_PATH = ROOT / "models" / "rank_probe_v12_pairwise.joblib"

CODE_INSTRUCTION = (
    "Complete the following Python function. Include any necessary imports (the signature line "
    "itself may omit them). Output ONLY the complete function -- the given signature plus your "
    "implementation -- optionally inside a single ```python ... ``` block. No explanation, no "
    "usage examples, no extra text.\n\n"
    "<problem>\n{problem}\n</problem>"
)


def is_sane_code(raw_text: str, entry_point: str) -> bool:
    code = extract_code_block(raw_text)
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return f"def {entry_point}" in code


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def ensure_round0(amb_ids: list[str], held_out: dict, state: dict, round0_reuse_dir: Path | None) -> None:
    """Populate round_0/{aid}/summary.json either by reusing an existing zero-shot
    baseline (vagueness variant, already run under v10/v11) or, when none exists
    (lexical variant: these prompts were never run before), generating a fresh
    zero-shot codegen baseline with the same local Qwen3-8B model used elsewhere
    in this script."""
    round0_dir = OUT_DIR / "round_0"
    round0_dir.mkdir(parents=True, exist_ok=True)
    needs_fresh_baseline = []
    for aid in amb_ids:
        if aid not in state:
            state[aid] = {"prompt": held_out[aid]["ambiguous_prompt"]}
        summary_path = round0_dir / aid / "summary.json"
        if summary_path.exists():
            continue
        (round0_dir / aid).mkdir(exist_ok=True)
        reuse_src = (round0_reuse_dir / aid / "summary.json") if round0_reuse_dir else None
        if reuse_src is not None and reuse_src.exists():
            rs = json.loads(reuse_src.read_text())
            summary_path.write_text(json.dumps(rs, indent=2))
            print(f"[v12-live][round 0] {aid}: reused baseline from {reuse_src}", flush=True)
        else:
            needs_fresh_baseline.append(aid)

    if needs_fresh_baseline:
        print(f"[v12-live][round 0] generating fresh zero-shot baseline for "
              f"{len(needs_fresh_baseline)} problems (no reusable baseline found)", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        for aid in needs_fresh_baseline:
            item = held_out[aid]
            prompt = state[aid]["prompt"]
            code_user = CODE_INSTRUCTION.format(problem=prompt)
            raw_code = generate_sane_batch(
                tokenizer, model, [code_user], max_new_tokens=600,
                is_sane_list=[lambda raw: is_sane_code(raw, item["entry_point"])],
                log_prefix=f"[v12-live][round 0][{aid}] ",
            )[0]
            code = extract_code_block(raw_code)
            test_res = run_all(code, item["entry_point"], item["test_case"])
            (round0_dir / aid / "generated_code.py").write_text(code)
            round_summary = {
                "round": 0, "prompt_before": prompt, "candidates": [], "chosen": None,
                "next_prompt": prompt,
                "test_n_pass": test_res["n_pass"], "test_n_total": test_res["n_total"],
            }
            (round0_dir / aid / "summary.json").write_text(json.dumps(round_summary, indent=2))
            print(f"[v12-live][round 0][{aid}] fresh baseline pass={test_res['n_pass']}/{test_res['n_total']}", flush=True)
        del model
        torch.cuda.empty_cache()

    save_state(state)


def mode_elicit(args, held_out: dict) -> None:
    state = load_state()
    ensure_round0(args.amb_ids, held_out, state, VARIANT_PATHS[args.variant]["round0_reuse_dir"])

    round_dir = OUT_DIR / f"round_{args.round}"
    round_dir.mkdir(parents=True, exist_ok=True)
    q_path = round_dir / "questions.json"
    if q_path.exists():
        print(f"[v12-live][round {args.round}][elicit] {q_path} already exists, skipping", flush=True)
        return

    all_questions = {}
    for i, aid in enumerate(args.amb_ids):
        prompt = state[aid]["prompt"]
        qs, provider = elicit_questions_via_api(prompt, args.max_candidates)
        all_questions[aid] = qs
        print(f"[v12-live][round {args.round}][elicit] {i+1}/{len(args.amb_ids)} {aid}: "
              f"{len(qs)} questions via {provider}", flush=True)
        if not qs:
            print(f"[v12-live][round {args.round}][elicit] WARNING: 0 questions for {aid}", flush=True)

    q_path.write_text(json.dumps(all_questions, indent=2))
    print(f"[v12-live][round {args.round}][elicit] wrote {q_path}", flush=True)
    print(f"[v12-live][round {args.round}] WAITING_FOR_ORACLE answers_file={round_dir / 'answers.json'}", flush=True)


def mode_score(args, held_out: dict) -> None:
    state = load_state()
    round_dir = OUT_DIR / f"round_{args.round}"
    q_path = round_dir / "questions.json"
    a_path = round_dir / "answers.json"
    assert q_path.exists(), f"missing {q_path} -- run --mode elicit first"
    assert a_path.exists(), f"missing {a_path} -- oracle answers not written yet"

    all_questions = json.loads(q_path.read_text())
    all_answers = json.loads(a_path.read_text())
    for aid in args.amb_ids:
        assert aid in all_answers, f"missing answers for {aid}"
        assert len(all_answers[aid]["answers"]) == len(all_questions[aid]), \
            f"{aid}: answers/questions length mismatch"

    meta = json.loads(MODEL_JSON_PATH.read_text())
    layer = meta["layer"]
    bundle = joblib.load(MODEL_PATH)
    pipe, w = bundle["pipe"], bundle["w"]
    print(f"[v12-live][round {args.round}][score] loaded probe (layer={layer})", flush=True)

    print(f"[v12-live][round {args.round}][score] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"[v12-live][round {args.round}][score] model loaded", flush=True)

    for i, aid in enumerate(args.amb_ids):
        item = held_out[aid]
        prompt = state[aid]["prompt"]
        qs = all_questions[aid]
        ans = all_answers[aid]["answers"]
        log_prefix = f"[v12-live][round {args.round}][{aid}] "

        if not qs:
            print(f"{log_prefix}NO CANDIDATES, skipping clarify/codegen", flush=True)
            next_prompt = prompt
            cand_results = []
            best = None
        else:
            rewrite_users = [
                STYLE_INSTRUCTION.format(
                    original=prompt,
                    detail=f"Add this clarification, integrated naturally into the relevant sentence: {a}",
                )
                for a in ans
            ]
            is_sane_list = [(lambda raw: is_sane_rewrite(raw, prompt))] * len(rewrite_users)
            inline_prompts = generate_sane_batch(
                tokenizer, model, rewrite_users, max_new_tokens=500, is_sane_list=is_sane_list,
                log_prefix=log_prefix,
            )
            cand_vecs = hidden_states_at_layers_batch(tokenizer, model, inline_prompts, [layer])[:, 0, :]
            z = pipe.transform(cand_vecs)
            scores = (z @ w).tolist()

            cand_results = [
                {"idx": idx, "question": q, "answer": a, "prompt": p, "score": s}
                for idx, (q, a, p, s) in enumerate(zip(qs, ans, inline_prompts, scores))
            ]
            cand_results.sort(key=lambda c: -c["score"])
            best = cand_results[0]
            next_prompt = best["prompt"]

        code_user = CODE_INSTRUCTION.format(problem=next_prompt)
        raw_code = generate_sane_batch(
            tokenizer, model, [code_user], max_new_tokens=600,
            is_sane_list=[lambda raw: is_sane_code(raw, item["entry_point"])],
            log_prefix=log_prefix,
        )[0]
        code = extract_code_block(raw_code)
        test_res = run_all(code, item["entry_point"], item["test_case"])

        round_dir_aid = round_dir / aid
        round_dir_aid.mkdir(exist_ok=True)
        (round_dir_aid / "generated_code.py").write_text(code)

        round_summary = {
            "round": args.round,
            "prompt_before": prompt,
            "candidates": cand_results,
            "chosen": ({"idx": best["idx"], "question": best["question"], "answer": best["answer"],
                        "score": best["score"]} if best else None),
            "next_prompt": next_prompt,
            "test_n_pass": test_res["n_pass"],
            "test_n_total": test_res["n_total"],
        }
        (round_dir_aid / "summary.json").write_text(json.dumps(round_summary, indent=2))

        score_str = f"chosen_score={best['score']:+.3f}" if best else "no candidates"
        print(f"{log_prefix}pass={test_res['n_pass']}/{test_res['n_total']} {score_str}", flush=True)

        state[aid]["prompt"] = next_prompt

    save_state(state)
    print(f"[v12-live][round {args.round}][score] done", flush=True)


def main() -> None:
    global OUT_DIR, STATE_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["elicit", "score"], required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--amb_ids", nargs="+", required=True)
    ap.add_argument("--max_candidates", type=int, default=12)
    ap.add_argument("--variant", choices=["vagueness", "lexical"], default="vagueness")
    ap.add_argument("--out_dir", type=str, default=None,
                     help="Override default output dir, e.g. for a separate batch of problem ids "
                          "that shouldn't collide with an existing round_N/questions.json.")
    args = ap.parse_args()
    OUT_DIR = Path(args.out_dir) if args.out_dir else VARIANT_PATHS[args.variant]["out_dir"]
    STATE_PATH = OUT_DIR / "state.json"

    held_out = json.loads(VARIANT_PATHS[args.variant]["held_out"].read_text())
    if args.mode == "elicit":
        mode_elicit(args, held_out)
    else:
        mode_score(args, held_out)


if __name__ == "__main__":
    main()
