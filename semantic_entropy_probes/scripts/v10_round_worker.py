"""v10: does the probe's uncertainty drop, when the candidate clarification
comes from a REAL question-answering loop instead of Qwen's own (often
wrong) self-assertions, predict actual downstream code-generation success?

For ONE held-out prompt, runs `rounds` rounds of:
  1. Qwen3-8B elicits up to `max_candidates` distinct clarifying QUESTIONS
     about the current (possibly already partially-clarified) prompt.
  2. [external oracle -- see below] answers each question, grounded in the
     true ClarifyCodeBench full spec + a verified-correct reference solution.
     This script does NOT answer its own questions -- it writes them to
     `questions.json` and blocks until `answers.json` appears, written by
     an isolated agent that sees ONLY the true prompt + reference code (not
     this pipeline's internal state), so its answers can't be contaminated
     by which candidate the probe already favors.
  3. Each (question, answer) pair is appended as a trailing clarification
     (`current_prompt + "\n\nClarification: {answer}"` -- the same
     reconstruction ClarifyCodeBench itself uses for its own K=0 class; see
     build_dataset.py) to form a candidate prompt. One forward pass per
     candidate (+ one for the unmodified baseline) is scored with rank1
     (LR, layer 20) / rank2 (MLP, layer 22) -- the layer choice already
     shown to generalize across hand-authored/Qwen-generated/self-elicited
     text (layer 36 does not, see results_layer36_v6v7 vs results_v8/9_layer36).
  4. The candidate with the largest mean P(ambiguous) drop is appended (as a
     trailing "Clarification:" line) to the prompt for real, becoming next
     round's baseline for elicitation/scoring. But the CODE-GENERATION input
     is a separate, inline-rewritten prompt: the same STYLE_INSTRUCTION
     inline-edit technique from v8/v9 is used to fold the chosen clarification
     directly into the existing prose, in the same terse AtCoder style as the
     original -- not a bolted-on trailing note. The coding agent -- openai/gpt-oss-120b,
     called via the Groq API -- then generates a full stdin/stdout Python
     solution for that inline-clarified prompt, and it's executed against the
     problem's real LiveCodeBench public+private test cases (scripts/v10_run_tests.py).
     Elicitation, the inline style-rewrite, and probe scoring stay on the local
     Qwen3-8B (probe scoring needs white-box hidden states, which an API can't give).

Usage: python v10_round_worker.py <amb_id> [--rounds 3] [--max_candidates 8]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v10_run_tests import run_all  # noqa: E402
from scripts.coding_agent import CODE_INSTRUCTION, extract_code_block, generate_code_agent  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-8B"  # local: elicitation, inline style-rewrite, probe scoring
# coding agent (codegen only): Groq gpt-oss-120b, falling back to OpenRouter Qwen3-Coder-480B-A35B
# when Groq is exhausted -- see coding_agent.py
HELD_OUT_PATH = ROOT / "data" / "v10_code_quality" / "held_out_3.json"
RANK1_PATH = ROOT / "models" / "rank1_lr_layer20_pca100_ccb_heval.joblib"
RANK2_PATH = ROOT / "models" / "rank2_mlp_layer22_pca100_ccb_heval.joblib"
RANK1_LAYER = json.load(open(ROOT / "models" / "rank1_lr_layer20_pca100_ccb_heval.json"))["layer"]
RANK2_LAYER = json.load(open(ROOT / "models" / "rank2_mlp_layer22_pca100_ccb_heval.json"))["layer"]

ELICIT_QUESTIONS_INSTRUCTION = (
    "You are reviewing a competitive-programming problem statement for specification ambiguity -- "
    "places where the statement under-specifies behavior, leaving multiple reasonable interpretations "
    "(e.g. edge cases, tie-breaking rules, formatting details, boundary conditions, data ranges, "
    "input constraints left implicit, terminology, etc.).\n\n"
    "Here is the problem statement:\n<problem>\n{original}\n</problem>\n\n"
    "List up to {max_candidates} distinct QUESTIONS you would need answered to remove ambiguity from this "
    "problem statement before implementing it. For EACH one, output a single line in this exact format:\n"
    "N. <one-sentence question>\n\n"
    "Only output the numbered list, nothing else -- no preamble, no explanation. Each item MUST be phrased "
    "as an actual question (ending in '?'), not a statement."
)

STYLE_INSTRUCTION = (
    "You are editing a competitive-programming problem statement (AtCoder/LiveCodeBench style: "
    "terse, formal, precise, no filler commentary, uses the same mathematical notation style as the original, "
    "and preserves the original's line-break structure -- one clause/sentence per line where the original does that).\n\n"
    "Here is the ORIGINAL problem statement:\n<original>\n{original}\n</original>\n\n"
    "Rewrite this problem statement to incorporate the following change, INLINE into the existing prose "
    "(edit an existing sentence, or insert one short sentence in the same place and style where a real "
    "problem-setter would have put it originally). Do NOT append a new bullet point or trailing note at the "
    "end. Do NOT change anything else about the text -- keep the rest word-for-word identical, including the "
    "Input/Output/Constraints sections and all line breaks. Output ONLY the full rewritten problem statement, "
    "with no preamble, no explanation, and no markdown formatting around it.\n\n"
    "Change to make: {detail}"
)


def build_prompt_text(tokenizer, user_content):
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def generate(tokenizer, model, user_content, max_new_tokens, temperature=0.7, top_p=0.8):
    """Always samples -- greedy decoding (do_sample=False) is a known source of
    degenerate/repetition-loop output that can leave a run stuck producing the same
    malformed text on every retry, since it's deterministic for a fixed prompt. Qwen3's
    own recommended non-thinking-mode sampling settings (temperature=0.7, top_p=0.8)."""
    text = build_prompt_text(tokenizer, user_content)
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p, pad_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def generate_sane(tokenizer, model, user_content, max_new_tokens, is_sane, max_retries=3, log_prefix=""):
    """generate(), retrying when is_sane(raw_output) rejects it -- Qwen3-8B occasionally
    produces malformed output (truncated, empty, or otherwise unusable) for these local
    generation steps. Every attempt samples (see generate()), so a retry on the same
    prompt has an actual chance at a different, sane result rather than reproducing the
    identical output. Falls back to the last attempt (with a warning) if every retry fails."""
    last = ""
    for attempt in range(max_retries):
        raw = generate(tokenizer, model, user_content, max_new_tokens)
        last = raw
        if is_sane(raw):
            if attempt > 0:
                print(f"{log_prefix}sane output on retry {attempt}", flush=True)
            return raw
        print(f"{log_prefix}attempt {attempt}: insane output ({len(raw)} chars), retrying", flush=True)
    print(f"{log_prefix}WARNING: no sane output after {max_retries} attempts, using last attempt", flush=True)
    return last


def parse_numbered(raw_text, max_items):
    items = []
    for line in raw_text.splitlines():
        line = line.strip()
        m = re.match(r'^\d+[\.\)]\s*(.+)$', line)
        if m:
            item = m.group(1).strip().strip('"')
            if item:
                items.append(item)
    return items[:max_items]


def is_sane_rewrite(raw_text: str, original: str) -> bool:
    """Reject obviously-broken STYLE_INSTRUCTION output: empty, or drastically shorter
    than the original (the usual signature of truncation or a degenerate generation) --
    a legitimate inline rewrite should be roughly the same length as the original, maybe
    a bit longer from the added clarification, never much shorter."""
    text = raw_text.strip()
    return bool(text) and len(text) >= 0.5 * len(original)


@torch.no_grad()
def hidden_states_at_layers(tokenizer, model, prompt_text, layers):
    text = build_prompt_text(tokenizer, prompt_text)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    return {l: hs[l][0, -1, :].float().cpu().numpy() for l in layers}


def score(rank1, rank2, vecs):
    x1 = vecs[RANK1_LAYER].reshape(1, -1)
    x2 = vecs[RANK2_LAYER].reshape(1, -1)
    p1 = float(rank1.predict_proba(x1)[0, list(rank1.classes_).index(1)])
    p2 = float(rank2.predict_proba(x2)[0, list(rank2.classes_).index(1)])
    return p1, p2


def wait_for_file(path: Path, poll_s: int = 5):
    while not path.exists():
        time.sleep(poll_s)
    # small extra sleep in case the writer is still flushing
    time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("amb_id")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max_candidates", type=int, default=8)
    ap.add_argument("--held_out_path", type=str, default=str(HELD_OUT_PATH))
    args = ap.parse_args()

    held_out = json.load(open(args.held_out_path))
    item = held_out[args.amb_id]
    test_cases = item["test_cases"]

    out_dir = ROOT / "data" / "v10_code_quality" / args.amb_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v10] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    rank1 = joblib.load(RANK1_PATH)
    rank2 = joblib.load(RANK2_PATH)
    print(f"[v10] model + probes loaded (rank1_layer={RANK1_LAYER}, rank2_layer={RANK2_LAYER})", flush=True)

    current_prompt = item["ambiguous_prompt"]  # trailing-append accumulation, for elicitation/scoring
    inline_prompt = item["ambiguous_prompt"]  # inline-rewritten accumulation, for code generation
    all_rounds_summary = []

    for r in range(1, args.rounds + 1):
        round_dir = out_dir / f"round_{r}"
        round_dir.mkdir(exist_ok=True)
        t0 = time.time()

        # ---- 1. elicit up to N candidate questions ----
        user_content = ELICIT_QUESTIONS_INSTRUCTION.format(
            original=current_prompt, max_candidates=args.max_candidates
        )
        log_prefix = f"[v10][{args.amb_id}][round {r}][elicit] "
        raw_q = generate_sane(
            tokenizer, model, user_content, max_new_tokens=500,
            is_sane=lambda raw: len(parse_numbered(raw, args.max_candidates)) >= 1,
            log_prefix=log_prefix,
        )
        questions = parse_numbered(raw_q, args.max_candidates)
        print(f"[v10][{args.amb_id}][round {r}] elicited {len(questions)} questions ({time.time()-t0:.0f}s)", flush=True)
        if not questions:
            print(f"[v10][{args.amb_id}][round {r}] WARNING: 0 questions parsed even after retries, "
                  f"raw={raw_q[:300]!r}", flush=True)

        q_path = round_dir / "questions.json"
        a_path = round_dir / "answers.json"
        json.dump({"amb_id": args.amb_id, "round": r, "questions": questions}, open(q_path, "w"), indent=2)

        cand_results = []
        if not questions:
            # elicitation produced nothing usable even after generate_sane's retries --
            # no point blocking on an oracle round-trip for zero questions. Treat this
            # round as a no-op: carry the current prompt forward unchanged (mirrors
            # v10_batch_worker.py's NO CANDIDATES handling; without this guard
            # `best = cand_results[0]` below would crash with an IndexError).
            print(f"[v10][{args.amb_id}][round {r}] NO QUESTIONS, skipping oracle/clarify/codegen for this round",
                  flush=True)
            best = None
        else:
            # ---- 2. block for external oracle answers ----
            print(f"[v10][{args.amb_id}][round {r}] WAITING_FOR_ORACLE questions_file={q_path} answers_file={a_path}", flush=True)
            wait_for_file(a_path)
            answers_obj = json.load(open(a_path))
            answers = answers_obj["answers"]
            assert len(answers) == len(questions), f"answers/questions length mismatch: {len(answers)} vs {len(questions)}"
            print(f"[v10][{args.amb_id}][round {r}] got {len(answers)} oracle answers", flush=True)

            # ---- 3. score baseline + each candidate ----
            t1 = time.time()
            layers = [RANK1_LAYER, RANK2_LAYER]
            base_vecs = hidden_states_at_layers(tokenizer, model, current_prompt, layers)
            base_p1, base_p2 = score(rank1, rank2, base_vecs)
            base_mean = (base_p1 + base_p2) / 2

            for i, (q, a) in enumerate(zip(questions, answers)):
                cand_prompt = current_prompt + f"\n\nClarification: {a}"
                vecs = hidden_states_at_layers(tokenizer, model, cand_prompt, layers)
                p1, p2 = score(rank1, rank2, vecs)
                mean_p = (p1 + p2) / 2
                delta = mean_p - base_mean
                cand_results.append({
                    "idx": i, "question": q, "answer": a, "prompt": cand_prompt,
                    "p1": p1, "p2": p2, "mean_p": mean_p, "delta": delta,
                })
            print(f"[v10][{args.amb_id}][round {r}] scored {len(cand_results)} candidates ({time.time()-t1:.0f}s)", flush=True)

            cand_results.sort(key=lambda c: c["delta"])
            top3 = cand_results[:3]
            print(f"[v10][{args.amb_id}][round {r}] TOP-3 uncertainty drops:", flush=True)
            for c in top3:
                print(f"    delta={c['delta']:+.4f}  Q: {c['question'][:90]}  A: {c['answer'][:90]}", flush=True)

            best = cand_results[0]

        if best is None:
            next_prompt = current_prompt
            next_inline_prompt = inline_prompt
            inline_unchanged = True
            top3 = []
            base_p1 = base_p2 = base_mean = None
        else:
            next_prompt = best["prompt"]

            # ---- 4. inline-rewrite the chosen clarification into the prose, then codegen ----
            t2 = time.time()
            detail = f"Add this clarification, integrated naturally into the relevant sentence: {best['answer']}"
            rewrite_user = STYLE_INSTRUCTION.format(original=inline_prompt, detail=detail)
            next_inline_prompt = generate_sane(
                tokenizer, model, rewrite_user, max_new_tokens=600,
                is_sane=lambda raw: is_sane_rewrite(raw, inline_prompt),
                log_prefix=f"[v10][{args.amb_id}][round {r}][rewrite] ",
            )
            inline_unchanged = (next_inline_prompt.strip() == inline_prompt.strip())
            print(f"[v10][{args.amb_id}][round {r}] inline rewrite done (unchanged={inline_unchanged}, "
                  f"{time.time()-t2:.0f}s)", flush=True)

        t3 = time.time()
        raw_code, coding_agent_provider = generate_code_agent(next_inline_prompt)
        code = extract_code_block(raw_code)
        (round_dir / "generated_code.py").write_text(code)
        test_res = run_all(code, test_cases)
        print(f"[v10][{args.amb_id}][round {r}] codegen+test ({coding_agent_provider}): "
              f"{test_res['n_pass']}/{test_res['n_total']} passed ({time.time()-t3:.0f}s)", flush=True)

        round_summary = {
            "round": r,
            "baseline_prompt": current_prompt,
            "baseline_p1": base_p1, "baseline_p2": base_p2, "baseline_mean_p": base_mean,
            "candidates": cand_results,
            "top3": [{"idx": c["idx"], "question": c["question"], "answer": c["answer"], "delta": c["delta"]} for c in top3],
            "chosen": ({"idx": best["idx"], "question": best["question"], "answer": best["answer"], "delta": best["delta"]}
                       if best else None),
            "next_prompt": next_prompt,
            "inline_prompt_before": inline_prompt,
            "inline_prompt_after": next_inline_prompt,
            "inline_rewrite_unchanged": inline_unchanged,
            "code_prompt": next_inline_prompt,
            "generated_code": code,
            "coding_agent_provider": coding_agent_provider,
            "test_n_pass": test_res["n_pass"],
            "test_n_total": test_res["n_total"],
            "test_results": test_res["results"],
        }
        json.dump(round_summary, open(round_dir / "summary.json", "w"), indent=2)
        all_rounds_summary.append(round_summary)

        current_prompt = next_prompt
        inline_prompt = next_inline_prompt

    json.dump(
        {"amb_id": args.amb_id, "rounds": all_rounds_summary},
        open(out_dir / "all_rounds_summary.json", "w"), indent=2,
    )
    print(f"[v10][{args.amb_id}] ALL ROUNDS DONE", flush=True)


if __name__ == "__main__":
    main()
