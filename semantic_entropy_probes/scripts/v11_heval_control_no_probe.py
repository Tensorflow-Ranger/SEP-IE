"""Control for v11_heval_batch_worker.py: does the probe-based candidate
SELECTION actually add value, or would asking any single question per round
do just as well? Same 32 Orchid-HEval problems, same oracle protocol, same
STYLE_INSTRUCTION inline-rewrite, same CODE_INSTRUCTION codegen with the same
local Qwen3-8B -- the only thing removed is the multi-candidate step: instead
of eliciting up to 8 candidate questions and using rank1/rank2 probe deltas to
pick the one that most reduces predicted ambiguity, this version elicits
exactly ONE question per round and applies it directly, no probe, no scoring,
no choice. No probe models are even loaded.

Round 0 (no clarification) is IDENTICAL between the two conditions -- it's a
straight codegen from the raw ambiguous_prompt, no elicitation or probes
involved either way -- so this script reuses the experimental run's round_0
summaries (data/v10_code_quality/heval32/round_0/) rather than regenerating
them; comparing rounds 1-3 against that same shared baseline is what makes
this a valid control (same starting point, only the middle mechanism differs).

Same oracle protocol as v11_heval_batch_worker.py: this script does NOT
answer its own elicited question. It writes it to round_N/questions.json
(one question per amb_id) and blocks on round_N/answers.json, expected to be
written by an isolated agent grounded ONLY in that problem's `clear_prompt` +
`solution` + `missing_detail`.

Usage: python v11_heval_control_no_probe.py --amb_ids heval_1 heval_104 ... [--rounds 3]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.heval_run_tests import run_all  # noqa: E402
from scripts.coding_agent import extract_code_block  # noqa: E402
from scripts.v10_round_worker import generate_sane, is_sane_rewrite  # noqa: E402
from scripts.v11_heval_batch_worker import (  # noqa: E402
    CODE_INSTRUCTION, STYLE_INSTRUCTION, is_sane_code,
)

MODEL_NAME = "Qwen/Qwen3-8B"
HELD_OUT_PATH = ROOT / "data" / "v10_code_quality" / "held_out_orchid32.json"
EXPERIMENTAL_ROUND0_DIR = ROOT / "data" / "v10_code_quality" / "heval32" / "round_0"

CONTROL_ELICIT_QUESTION_INSTRUCTION = (
    "You are reviewing a Python function specification (a function signature plus its "
    "docstring) for specification ambiguity -- places where the docstring under-specifies "
    "behavior, leaving multiple reasonable interpretations (e.g. edge cases, what counts as "
    "a valid/separate case, tie-breaking rules, boundary conditions, argument constraints "
    "left implicit, terminology, etc.).\n\n"
    "Here is the function specification:\n<problem>\n{original}\n</problem>\n\n"
    "Ask exactly ONE question -- the single most important question -- you would need "
    "answered to remove ambiguity from this specification before implementing it. Output "
    "ONLY that one question, ending in '?'. No preamble, no numbering, no explanation."
)


def is_sane_question(raw_text: str) -> bool:
    text = raw_text.strip()
    return bool(text) and "?" in text and len(text) < 500


def extract_question(raw_text: str) -> str:
    """Take everything up to (and including) the first '?', stripping any leading numbering
    the model adds despite being told not to (e.g. "1. Is X...?")."""
    text = raw_text.strip()
    idx = text.find("?")
    q = text[: idx + 1] if idx != -1 else text
    q = re.sub(r'^\d+[\.\)]\s*', '', q).strip()
    return q


def wait_for_file(path: Path, poll_s: int = 5):
    while not path.exists():
        time.sleep(poll_s)
    time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amb_ids", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--held_out_path", type=str, default=str(HELD_OUT_PATH))
    ap.add_argument("--out_dir", type=str,
                     default=str(ROOT / "data" / "v10_code_quality" / "heval32_control"))
    args = ap.parse_args()

    held_out = json.load(open(args.held_out_path))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v11ctrl] {len(args.amb_ids)} amb_ids, {args.rounds} rounds, "
          f"1 question/round, NO PROBE", flush=True)
    print(f"[v11ctrl] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print("[v11ctrl] model loaded (no probes needed for this control)", flush=True)

    state = {
        aid: {
            "current_prompt": held_out[aid]["ambiguous_prompt"],
            "inline_prompt": held_out[aid]["ambiguous_prompt"],
        }
        for aid in args.amb_ids
    }
    all_rounds_summary = {aid: [] for aid in args.amb_ids}

    # ---- round 0: reuse the experimental run's identical no-clarification baseline ----
    round0_dir = out_dir / "round_0"
    round0_dir.mkdir(exist_ok=True)
    for aid in args.amb_ids:
        src = EXPERIMENTAL_ROUND0_DIR / aid / "summary.json"
        assert src.exists(), f"missing experimental round_0 summary for {aid} at {src}"
        rs = json.load(open(src))
        all_rounds_summary[aid].append(rs)
        json.dump(rs, open(round0_dir / f"{aid}.json", "w"), indent=2)
    print(f"[v11ctrl][round 0] reused {len(args.amb_ids)} baselines from experimental run "
          f"(identical: no clarification, no probes involved either way)", flush=True)

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
            print(f"[v11ctrl][round {r}] RESUMED from existing per-amb_id summaries, skipping", flush=True)
            continue

        # ---- 1. elicit exactly ONE question per amb_id ----
        all_questions = {}
        for i, aid in enumerate(args.amb_ids):
            cp = state[aid]["current_prompt"]
            uc = CONTROL_ELICIT_QUESTION_INSTRUCTION.format(original=cp)
            raw_q = generate_sane(
                tokenizer, model, uc, max_new_tokens=100,
                is_sane=is_sane_question,
                log_prefix=f"[v11ctrl][round {r}][elicit][{aid}] ",
            )
            q = extract_question(raw_q)
            all_questions[aid] = [q]  # list of 1, to match the {aid: {"answers": [...]}} oracle schema
            print(f"[v11ctrl][round {r}] elicited {i+1}/{len(args.amb_ids)} ({aid}: {q[:80]!r})", flush=True)

        q_path = round_dir / "questions.json"
        a_path = round_dir / "answers.json"
        json.dump(all_questions, open(q_path, "w"), indent=2)
        print(f"[v11ctrl][round {r}] elicitation done for all {len(args.amb_ids)} amb_ids "
              f"({time.time()-t0:.0f}s)", flush=True)

        # ---- 2. block for oracle answers ----
        print(f"[v11ctrl][round {r}] WAITING_FOR_ORACLE questions_file={q_path} answers_file={a_path}", flush=True)
        wait_for_file(a_path)
        all_answers = json.load(open(a_path))
        for aid in args.amb_ids:
            assert aid in all_answers, f"missing answers for {aid}"
            assert len(all_answers[aid]["answers"]) == 1, f"{aid}: expected exactly 1 answer"
        print(f"[v11ctrl][round {r}] got oracle answers for all {len(args.amb_ids)} amb_ids", flush=True)

        # ---- 3. apply the (only) clarification directly -- no scoring, no choice --
        #      then rewrite inline + codegen + test, same as the experimental script ----
        t1 = time.time()
        for i, aid in enumerate(args.amb_ids):
            item = held_out[aid]
            cp = state[aid]["current_prompt"]
            ip = state[aid]["inline_prompt"]
            q = all_questions[aid][0]
            a = all_answers[aid]["answers"][0]

            next_prompt = cp + f"\n\nClarification: {a}"
            detail = f"Add this clarification, integrated naturally into the relevant sentence: {a}"
            rewrite_user = STYLE_INSTRUCTION.format(original=ip, detail=detail)
            next_inline = generate_sane(
                tokenizer, model, rewrite_user, max_new_tokens=500,
                is_sane=lambda raw: is_sane_rewrite(raw, ip),
                log_prefix=f"[v11ctrl][round {r}][rewrite][{aid}] ",
            )

            code_user = CODE_INSTRUCTION.format(problem=next_inline)
            raw_code = generate_sane(
                tokenizer, model, code_user, max_new_tokens=600,
                is_sane=lambda raw: is_sane_code(raw, item["entry_point"]),
                log_prefix=f"[v11ctrl][round {r}][codegen][{aid}] ",
            )
            code = extract_code_block(raw_code)
            test_res = run_all(code, item["entry_point"], item["test_case"])

            round_dir_aid = round_dir / aid
            round_dir_aid.mkdir(exist_ok=True)
            (round_dir_aid / "generated_code.py").write_text(code)

            round_summary = {
                "round": r,
                "baseline_prompt": cp,
                "question": q, "answer": a,
                "next_prompt": next_prompt,
                "code_prompt": next_inline,
                "test_n_pass": test_res["n_pass"],
                "test_n_total": test_res["n_total"],
            }
            json.dump(round_summary, open(round_dir_aid / "summary.json", "w"), indent=2)
            all_rounds_summary[aid].append(round_summary)

            print(f"[v11ctrl][round {r}] {i+1}/{len(args.amb_ids)} {aid}: "
                  f"pass={test_res['n_pass']}/{test_res['n_total']}", flush=True)

            state[aid]["current_prompt"] = next_prompt
            state[aid]["inline_prompt"] = next_inline

        print(f"[v11ctrl][round {r}] all clarify/codegen/test done ({time.time()-t1:.0f}s)", flush=True)

    json.dump(all_rounds_summary, open(out_dir / "all_amb_ids_summary.json", "w"), indent=2)
    print("[v11ctrl] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
