"""Live multi-round elicit -> oracle-answer -> rewrite -> score -> codegen -> test
loop on heval32, using the v16 SupCon+MLP ranking probe
(models/rank_probe_v16_supcon_raw_mlp_pooled.joblib) instead of v12's plain
LR probe -- the first live/behavioral test of whether SupCon's better LODO
generalization (FINDINGS.md's `## v15/v16` section: supcon_raw_mlp LODO mean
0.425 vs. pca_lr's 0.378) translates into better clarifying-question choices
in practice.

Differs from v12_heval32_live_loop.py in two ways:

1. Scoring uses pairwise win-counting (score_i = sum_j P(i beats j) over the
   MLPClassifier's pairwise-diff predictions), not a single dot product --
   the SupCon+MLP head can't reduce to one linear direction `w` the way the
   old LR probe could (see v15_train_ranking_probe.py's eval_ranking, "mlp"
   branch). The candidate activations are also run through the frozen SupCon
   projector (pipe.transform, GPU) before the MLP head, not just
   StandardScaler.

2. Elicitation AND oracle-answer+rewrite are both done via AWS Bedrock
   (DeepSeek V3.2), one call per problem per round for each stage, instead
   of local-Qwen elicitation + a separate out-of-process haiku oracle-answer
   step. This mirrors exactly how 416 of v15/v16's own 962 training rows
   were generated (scripts/v15_bedrock_oracle_rewrite.py) -- oracle-answer
   and inline-rewrite combined into one Bedrock call, grounded ONLY in the
   problem's privileged missing_detail/solution (never exposed to the
   elicitation call). Unlike v15's rewrite script this also does the
   elicitation step itself via Bedrock (v15's script assumed candidates
   already existed), and keeps the same HumanEval-docstring-specific
   elicitation wording v12_heval32_live_loop.py uses (NOT the competitive-
   programming-worded one) -- domain-matched elicitation phrasing was
   confirmed to matter for heval32 in that script's docstring. The
   inline-rewrite instruction, however, deliberately keeps v12's original
   "AtCoder/LiveCodeBench style" framing verbatim even though it's genre-
   mismatched for HumanEval docstrings, because the SupCon probe was
   trained on activations extracted from exactly that rewrite instruction
   (STYLE_INSTRUCTION) -- changing it would create a train/score format gap
   of the same kind v12's own docstring warns against.

Because Bedrock supplies oracle answers directly (grounded in privileged
per-problem ground truth, never seen by the elicitation call), this script
runs all rounds end-to-end in one invocation -- no WAITING_FOR_ORACLE
handoff.

Usage:
  python v16_heval32_live_loop_supcon.py --rounds 3
  python v16_heval32_live_loop_supcon.py --rounds 3 --amb_ids heval_1 heval_91
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path

import boto3
import joblib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.heval_run_tests import run_all  # noqa: E402
from scripts.coding_agent import extract_code_block  # noqa: E402
from scripts.v12_extract_activations import (  # noqa: E402
    MODEL_NAME,
    STYLE_INSTRUCTION,
    generate_sane_batch,
    hidden_states_at_layers_batch,
)
from v16_train_supcon_ranking import SupConPipe  # noqa: E402,F401 (needed for joblib unpickling)

HELD_OUT = ROOT / "data" / "v10_code_quality" / "held_out_orchid32.json"
ROUND0_REUSE_DIR = ROOT / "data" / "v10_code_quality" / "heval32" / "round_0"
OUT_DIR = ROOT / "data" / "v16_ranking" / "heval32_live_loop_supcon"
STATE_PATH = OUT_DIR / "state.json"
MODEL_PATH = ROOT / "models" / "rank_probe_v16_supcon_raw_mlp_pooled.joblib"
PROBE_LAYER = 20  # v16's supcon_raw_mlp selected layer (FINDINGS.md's `## v15/v16` table)

BEDROCK_MODEL_ID = "deepseek.v3.2"
BEDROCK_REGION = "ap-northeast-1"
BEDROCK_MAX_RETRIES = 8

# v12_heval32_live_loop.py's HumanEval-docstring-specific elicitation wording,
# unchanged -- confirmed domain-matched phrasing matters for this problem style.
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

# v12_extract_activations.py's ORACLE_RULE / STYLE_INSTRUCTION content, unchanged in
# substance (v15_bedrock_oracle_rewrite.py's SYSTEM_PROMPT pattern, reused verbatim).
ORACLE_RULE = (
    "You are answering a question about a Python function's specification. You have privileged "
    "knowledge of the TRUE intended specification, given below, which the person asking the question "
    "cannot see. Answer using ONLY the information in the true specification. If the question is not "
    "addressed by it, answer with the most standard/reasonable Python convention rather than inventing "
    "a new constraint. Answer in ONE concise sentence."
)

REWRITE_RULE = (
    "You are editing a competitive-programming problem statement (AtCoder/LiveCodeBench style: "
    "terse, formal, precise, no filler commentary, uses the same mathematical notation style as the original, "
    "and preserves the original's line-break structure -- one clause/sentence per line where the original does that). "
    "Rewrite the ORIGINAL problem statement to incorporate the given clarification, INLINE into the existing prose "
    "(edit an existing sentence, or insert one short sentence in the same place and style where a real "
    "problem-setter would have put it originally). Do NOT append a new bullet point or trailing note at the "
    "end. Do NOT change anything else about the text -- keep the rest word-for-word identical, including all "
    "code, signatures, imports, and line breaks."
)

ANSWER_REWRITE_SYSTEM_PROMPT = f"""For each candidate clarifying question about a Python function specification, do TWO things:

1. ANSWER: {ORACLE_RULE}

2. REWRITE: {REWRITE_RULE} The clarification to incorporate is your own answer from step 1, phrased as "Add this clarification, integrated naturally into the relevant sentence: <answer>".

Respond with ONLY a JSON object, no markdown fences:
{{"candidates": [{{"question": "<verbatim>", "answer": "<one sentence>", "inline_prompt": "<the FULL rewritten function specification, complete, not truncated>"}}]}}"""

CODE_INSTRUCTION = (
    "Complete the following Python function. Include any necessary imports (the signature line "
    "itself may omit them). Output ONLY the complete function -- the given signature plus your "
    "implementation -- optionally inside a single ```python ... ``` block. No explanation, no "
    "usage examples, no extra text.\n\n"
    "<problem>\n{problem}\n</problem>"
)


def load_aws_env() -> None:
    env_path = ROOT.parent / ".env"
    env = {}
    for line in open(env_path):
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = env["AWS"]


def bedrock_client():
    from botocore.config import Config

    # No timeout is set by default -- a hung/slow request (observed: one invoke_model
    # call stuck >45min with no server response) then blocks the whole run forever,
    # since it never reaches the exception-based retry logic below.
    config = Config(connect_timeout=30, read_timeout=180, retries={"max_attempts": 0})
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION, config=config)


def call_deepseek(rt, system_prompt: str | None, user_prompt: str, max_tokens: int = 3000) -> str:
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
        {"role": "user", "content": user_prompt}
    ]
    body = json.dumps({"messages": messages, "max_tokens": max_tokens, "temperature": 0.3})
    delay = 5
    last_err = None
    for attempt in range(BEDROCK_MAX_RETRIES):
        try:
            resp = rt.invoke_model(modelId=BEDROCK_MODEL_ID, body=body)
            payload = json.loads(resp["body"].read())
            return payload["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            if "Throttling" in msg or "TooManyRequests" in msg:
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            if attempt < BEDROCK_MAX_RETRIES - 1:
                time.sleep(min(delay, 20))
                delay = min(delay * 2, 120)
                continue
            raise
    raise last_err


def parse_numbered(raw: str, max_items: int) -> list[str]:
    lines = raw.strip().splitlines()
    out = []
    for line in lines:
        m = re.match(r"^\s*\d+[.)]\s*(.+)$", line.strip())
        if m:
            out.append(m.group(1).strip())
    return out[:max_items]


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    start = text.find("{")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON in Bedrock response")


def elicit_questions(rt, problem: str, max_candidates: int, max_sanity_retries: int = 3) -> list[str]:
    prompt = ELICIT_QUESTIONS_INSTRUCTION.format(original=problem, max_candidates=max_candidates)
    for _ in range(max_sanity_retries):
        raw = call_deepseek(rt, None, prompt, max_tokens=600)
        qs = parse_numbered(raw, max_candidates)
        if qs:
            return qs
    return []


def answer_and_rewrite(rt, current_prompt: str, missing_detail: str, solution: str,
                        questions: list[str]) -> list[dict]:
    cand_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    user_prompt = (
        f"--- ORIGINAL FUNCTION SPECIFICATION (to rewrite) ---\n{current_prompt}\n\n"
        f"--- TRUE SPECIFICATION DETAILS (privileged, for answering only) ---\n{missing_detail}\n\n"
        f"--- REFERENCE SOLUTION (privileged, for answering only) ---\n{solution}\n\n"
        f"--- CANDIDATE QUESTIONS ---\n{cand_text}"
    )
    for attempt in range(3):
        try:
            raw = call_deepseek(rt, ANSWER_REWRITE_SYSTEM_PROMPT, user_prompt, max_tokens=6000)
            parsed = extract_json(raw)
            results = parsed["candidates"]
            # Trust positional order when counts match -- the model sometimes echoes
            # back a slightly paraphrased question (not verbatim), so exact-text
            # matching is unreliable; callers use each result's own `question` field,
            # not the original list, so positional pairing is all that's needed.
            if len(results) != len(questions):
                raise ValueError(f"candidate count mismatch: sent {len(questions)}, got {len(results)}")
            return results
        except Exception:  # noqa: BLE001
            if attempt == 2:
                raise
            time.sleep(5)


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


def ensure_round0(amb_ids: list[str], held_out: dict, state: dict) -> None:
    round0_dir = OUT_DIR / "round_0"
    round0_dir.mkdir(parents=True, exist_ok=True)
    for aid in amb_ids:
        if aid not in state:
            state[aid] = {"prompt": held_out[aid]["ambiguous_prompt"]}
        summary_path = round0_dir / aid / "summary.json"
        if summary_path.exists():
            continue
        (round0_dir / aid).mkdir(exist_ok=True)
        reuse_src = ROUND0_REUSE_DIR / aid / "summary.json"
        assert reuse_src.exists(), f"no round-0 baseline to reuse for {aid} at {reuse_src}"
        rs = json.loads(reuse_src.read_text())
        summary_path.write_text(json.dumps(rs, indent=2))
        print(f"[v16-live][round 0] {aid}: reused baseline from {reuse_src}", flush=True)
    save_state(state)


def score_candidates(tokenizer, model, pipe, mlp_model, prompts: list[str]) -> list[float]:
    """SupCon+MLP pairwise win-counting score for each candidate prompt in `prompts`."""
    vecs = hidden_states_at_layers_batch(tokenizer, model, prompts, [PROBE_LAYER])[:, 0, :]
    z = pipe.transform(vecs)
    n = len(prompts)
    diffs, pair_idx = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diffs.append(z[i] - z[j])
            pair_idx.append(i)
    proba = mlp_model["clf"].predict_proba(np.array(diffs))[:, 1]
    scores = np.zeros(n)
    for p, i in zip(proba, pair_idx):
        scores[i] += p
    return scores.tolist()


def run_round(round_n: int, amb_ids: list[str], held_out: dict, state: dict,
              rt, tokenizer, model, pipe, mlp_model) -> None:
    round_dir = OUT_DIR / f"round_{round_n}"
    round_dir.mkdir(parents=True, exist_ok=True)

    for i, aid in enumerate(amb_ids):
        aid_dir = round_dir / aid
        summary_path = aid_dir / "summary.json"
        if summary_path.exists():
            print(f"[v16-live][round {round_n}][{aid}] already done, skipping", flush=True)
            state[aid]["prompt"] = json.loads(summary_path.read_text())["next_prompt"]
            continue
        aid_dir.mkdir(exist_ok=True)
        item = held_out[aid]
        prompt = state[aid]["prompt"]
        log_prefix = f"[v16-live][round {round_n}][{aid}] "

        try:
            _run_one(round_n, aid, item, prompt, log_prefix, i, len(amb_ids),
                     rt, tokenizer, model, pipe, mlp_model, aid_dir, summary_path, state)
        except Exception as e:  # noqa: BLE001
            # A single problem's Bedrock/codegen failure shouldn't kill the whole
            # multi-hour run -- log it and move on; state[aid]["prompt"] is left
            # unchanged so this problem is simply retried (not skipped) on the next
            # invocation of this script (summary_path won't exist).
            print(f"{log_prefix}FAILED: {type(e).__name__}: {e}", flush=True)
            if aid_dir.exists() and not any(aid_dir.iterdir()):
                aid_dir.rmdir()


def _run_one(round_n, aid, item, prompt, log_prefix, i, n_total,
             rt, tokenizer, model, pipe, mlp_model, aid_dir, summary_path, state):
    qs = elicit_questions(rt, prompt, max_candidates=12)
    print(f"{log_prefix}{len(qs)} questions elicited ({i+1}/{n_total})", flush=True)

    if not qs:
        print(f"{log_prefix}NO CANDIDATES, skipping clarify/codegen", flush=True)
        next_prompt = prompt
        cand_results = []
        best = None
    else:
        ar = answer_and_rewrite(rt, prompt, item["missing_detail"], item["solution"], qs)
        inline_prompts = [r["inline_prompt"] for r in ar]
        scores = score_candidates(tokenizer, model, pipe, mlp_model, inline_prompts)

        cand_results = [
            {"idx": idx, "question": r["question"], "answer": r["answer"], "prompt": r["inline_prompt"],
             "score": s}
            for idx, (r, s) in enumerate(zip(ar, scores))
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

    (aid_dir / "generated_code.py").write_text(code)
    round_summary = {
        "round": round_n,
        "prompt_before": prompt,
        "candidates": cand_results,
        "chosen": ({"idx": best["idx"], "question": best["question"], "answer": best["answer"],
                    "score": best["score"]} if best else None),
        "next_prompt": next_prompt,
        "test_n_pass": test_res["n_pass"],
        "test_n_total": test_res["n_total"],
    }
    summary_path.write_text(json.dumps(round_summary, indent=2))

    score_str = f"chosen_score={best['score']:+.3f}" if best else "no candidates"
    print(f"{log_prefix}pass={test_res['n_pass']}/{test_res['n_total']} {score_str}", flush=True)

    state[aid]["prompt"] = next_prompt
    save_state(state)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--amb_ids", nargs="+", default=None)
    args = ap.parse_args()

    load_aws_env()
    held_out = json.loads(HELD_OUT.read_text())
    amb_ids = args.amb_ids if args.amb_ids else sorted(held_out.keys(), key=lambda k: int(k.split("_")[1]))

    state = load_state()
    ensure_round0(amb_ids, held_out, state)

    print(f"[v16-live] loading SupCon probe from {MODEL_PATH} (layer={PROBE_LAYER})", flush=True)
    bundle = joblib.load(MODEL_PATH)
    pipe, mlp_model = bundle["pipe"], bundle["model"]

    print(f"[v16-live] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print("[v16-live] model loaded", flush=True)

    rt = bedrock_client()

    for round_n in range(1, args.rounds + 1):
        run_round(round_n, amb_ids, held_out, state, rt, tokenizer, model, pipe, mlp_model)

    print("[v16-live] DONE", flush=True)


if __name__ == "__main__":
    main()
