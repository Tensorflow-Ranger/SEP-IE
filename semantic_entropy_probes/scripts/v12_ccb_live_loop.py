"""Live multi-round elicit -> oracle -> score -> clarify loop on a subset of
held-out ClarifyCodeBench (ccb) problems, using the v12 pairwise-ranking
probe (models/rank_probe_v12_pairwise.joblib) -- the ccb counterpart to
v12_heval32_live_loop.py.

NO codegen or test execution here -- the only thing being measured is
whether the probe ranks the question that targets the real injected
ambiguity (qa_pairs) at #1, same as the corrected heval32 analysis. Codegen
success is a separate, noisier signal (confounded by an 8B/API model's raw
coding ability) that isn't needed to answer that question.

Unlike heval32, ccb problems ARE competitive-programming-style (AtCoder/
LiveCodeBench: terse, formal, stdin/stdout I/O), so this script uses
v12_elicit_candidates.py's ELICIT_QUESTIONS_INSTRUCTION and
v12_extract_activations.py's STYLE_INSTRUCTION UNCHANGED -- both are already
domain-matched here (they were the wrong domain for heval32's HumanEval-
docstring problems, which is why v12_heval32_live_loop.py needed its own
locally-defined instruction instead).

Ground truth for oracle-answering is `qa_pairs` (the real ClarifyCodeBench
annotated Q&A). Every held_out_40.json entry has exactly one qa_pair (one
injected ambiguity), like heval32.

Usage:
  python v12_ccb_live_loop.py --mode elicit --round N --amb_ids ...
  python v12_ccb_live_loop.py --mode score  --round N --amb_ids ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v12_elicit_candidates import elicit_questions_via_api  # noqa: E402
from scripts.v12_extract_activations import (  # noqa: E402
    MODEL_NAME,
    STYLE_INSTRUCTION,
    generate_sane_batch,
    hidden_states_at_layers_batch,
    is_sane_rewrite,
)

HELD_OUT_PATH = ROOT / "data" / "v10_code_quality" / "held_out_40.json"
OUT_DIR = ROOT / "data" / "v12_ranking" / "ccb_live_loop"
STATE_PATH = OUT_DIR / "state.json"
MODEL_JSON_PATH = ROOT / "models" / "rank_probe_v12_pairwise.json"
MODEL_PATH = ROOT / "models" / "rank_probe_v12_pairwise.joblib"


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def ensure_init(amb_ids: list[str], held_out: dict, state: dict) -> None:
    for aid in amb_ids:
        if aid not in state:
            state[aid] = {"prompt": held_out[aid]["ambiguous_prompt"]}
    save_state(state)


def mode_elicit(args, held_out: dict) -> None:
    state = load_state()
    ensure_init(args.amb_ids, held_out, state)

    round_dir = OUT_DIR / f"round_{args.round}"
    round_dir.mkdir(parents=True, exist_ok=True)
    q_path = round_dir / "questions.json"
    if q_path.exists():
        print(f"[v12-ccb-live][round {args.round}][elicit] {q_path} already exists, skipping", flush=True)
        return

    all_questions = {}
    for i, aid in enumerate(args.amb_ids):
        prompt = state[aid]["prompt"]
        qs, provider = elicit_questions_via_api(prompt, args.max_candidates)
        all_questions[aid] = qs
        print(f"[v12-ccb-live][round {args.round}][elicit] {i+1}/{len(args.amb_ids)} {aid}: "
              f"{len(qs)} questions via {provider}", flush=True)
        if not qs:
            print(f"[v12-ccb-live][round {args.round}][elicit] WARNING: 0 questions for {aid}", flush=True)

    q_path.write_text(json.dumps(all_questions, indent=2))
    print(f"[v12-ccb-live][round {args.round}][elicit] wrote {q_path}", flush=True)
    print(f"[v12-ccb-live][round {args.round}] WAITING_FOR_ORACLE answers_file={round_dir / 'answers.json'}", flush=True)


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
    print(f"[v12-ccb-live][round {args.round}][score] loaded probe (layer={layer})", flush=True)

    print(f"[v12-ccb-live][round {args.round}][score] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"[v12-ccb-live][round {args.round}][score] model loaded", flush=True)

    for i, aid in enumerate(args.amb_ids):
        prompt = state[aid]["prompt"]
        qs = all_questions[aid]
        ans = all_answers[aid]["answers"]
        log_prefix = f"[v12-ccb-live][round {args.round}][{aid}] "

        if not qs:
            print(f"{log_prefix}NO CANDIDATES, skipping clarify", flush=True)
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

        round_dir_aid = round_dir / aid
        round_dir_aid.mkdir(exist_ok=True)

        round_summary = {
            "round": args.round,
            "prompt_before": prompt,
            "candidates": cand_results,
            "chosen": ({"idx": best["idx"], "question": best["question"], "answer": best["answer"],
                        "score": best["score"]} if best else None),
            "next_prompt": next_prompt,
        }
        (round_dir_aid / "summary.json").write_text(json.dumps(round_summary, indent=2))

        score_str = f"chosen_score={best['score']:+.3f}" if best else "no candidates"
        print(f"{log_prefix}{score_str}", flush=True)

        state[aid]["prompt"] = next_prompt

    save_state(state)
    print(f"[v12-ccb-live][round {args.round}][score] done", flush=True)


def main() -> None:
    global OUT_DIR, STATE_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["elicit", "score"], required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--amb_ids", nargs="+", required=True)
    ap.add_argument("--max_candidates", type=int, default=12)
    ap.add_argument("--out_dir", type=str, default=None,
                     help="Override default output dir, e.g. for a separate batch of problem ids "
                          "that shouldn't collide with an existing round_N/questions.json.")
    args = ap.parse_args()
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
        STATE_PATH = OUT_DIR / "state.json"

    held_out = json.loads(HELD_OUT_PATH.read_text())
    if args.mode == "elicit":
        mode_elicit(args, held_out)
    else:
        mode_score(args, held_out)


if __name__ == "__main__":
    main()
