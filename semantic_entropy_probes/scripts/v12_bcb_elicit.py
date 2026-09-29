"""Stage B equivalent for a small orchid_bcb sample: elicit candidate
clarifying questions via the Groq-primary API agent (same
elicit_questions_via_api approach as v12_heval32_live_loop.py, reusing its
HumanEval/BigCodeBench-style ELICIT_QUESTIONS_INSTRUCTION since BigCodeBench
prompts are the same "function signature + docstring" format), for the same
N orchid_bcb source problems used in the earlier open-ended qualitative
probe (v12_control_qwen_bcb_probe.py).

Output: data/v12_ranking/bcb_candidates.jsonl, one row per
{task_id}_{lexical|vagueness} with its elicited candidate list -- ready for
manual/Claude tier-judging (Stage C equivalent) before running the same
zero-shot self-rank control used elsewhere.

Usage: python v12_bcb_elicit.py [--n 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.coding_agent import generate_agent  # noqa: E402
from scripts.v12_elicit_candidates import parse_numbered  # noqa: E402

RAW_PATH = ROOT / "data" / "raw" / "orchid_bcb.jsonl"
OUT_PATH = ROOT / "data" / "v12_ranking" / "bcb_candidates.jsonl"

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
    "Only output the numbered list, nothing else -- no preamble, no explanation."
)


def elicit_questions_via_api(problem: str, max_candidates: int, max_sanity_retries: int = 3):
    uc = ELICIT_QUESTIONS_INSTRUCTION.format(original=problem, max_candidates=max_candidates)
    last_qs, last_provider = [], None
    for _ in range(max_sanity_retries):
        raw, provider = generate_agent(uc, max_tokens=500)
        qs = parse_numbered(raw, max_candidates)
        last_qs, last_provider = qs, provider
        if qs:
            return qs, provider
    return last_qs, last_provider


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max_candidates", type=int, default=8)
    args = ap.parse_args()

    rows = []
    with RAW_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = rows[: args.n]

    out_rows = []
    for r in rows:
        for kind in ("Lexical", "Vagueness"):
            prompt = r[f"{kind}_prompt"]
            explanation = r[f"{kind}_ambiguity_explanation"]
            qs, provider = elicit_questions_via_api(prompt, args.max_candidates)
            print(f"[bcb-elicit] {r['task_id']} [{kind}] via {provider}: {len(qs)} questions", flush=True)
            for i, q in enumerate(qs):
                print(f"    {i}. {q}", flush=True)
            out_rows.append({
                "id": f"{r['task_id'].replace('/', '_')}_{kind.lower()}",
                "task_id": r["task_id"],
                "kind": kind,
                "prompt": prompt,
                "true_explanation": explanation,
                "questions": qs,
                "provider": provider,
            })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[bcb-elicit] saved {len(out_rows)} rows to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
