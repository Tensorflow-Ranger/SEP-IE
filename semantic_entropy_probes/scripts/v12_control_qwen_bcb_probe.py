"""Quick qualitative probe: orchid_bcb (BigCodeBench-derived ambiguities) was
NEVER included in v12 at all -- Stage A only pulled clarifycodebench +
orchid_heval (see FINDINGS.md's Stage E results note), and orchid_bcb has a
long history (pre-v12, the old i.i.d. classifier probes) of sitting at
~chance AUROC (~0.52) in-distribution, the one source the linear-probe
approach never learned well.

This is NOT a full tiered-candidate hit-rate eval like the heval32/ccb
controls (that would need Stage B elicitation + Stage C oracle-judging,
which never ran for this source). Instead: does Qwen, asked directly and
open-endedly (no candidate list, no probe) to state the single most
important clarifying question for a handful of orchid_bcb Lexical/Vagueness
ambiguous prompts, actually zero in on the SAME ambiguity that was
deliberately injected (known from data/raw/orchid_bcb.jsonl's
{Lexical,Vagueness}_ambiguity_explanation fields)? A human/Claude judge
reads the raw output against the explanation afterward -- this script only
does generation.

Usage: python v12_control_qwen_bcb_probe.py [--n 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_NAME = "Qwen/Qwen3-8B"
RAW_PATH = ROOT / "data" / "raw" / "orchid_bcb.jsonl"
OUT_PATH = ROOT / "data" / "v12_ranking" / "qwen_bcb_qualitative_results.json"

INSTRUCTION = """You are given a Python function specification. State the SINGLE most \
important clarifying question you would ask before implementing it -- the one \
piece of missing or ambiguous information that would most change your \
implementation depending on the answer.

Function specification:
{prompt}

Respond with just the one question, no preamble.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="number of source BigCodeBench problems (each yields 2 prompts: lexical + vagueness)")
    args = ap.parse_args()

    rows = []
    with RAW_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = rows[: args.n]
    print(f"[bcb-qual] {len(rows)} source problems -> {len(rows) * 2} prompts (lexical + vagueness each)", flush=True)

    print(f"[bcb-qual] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: "13GiB", "cpu": "40GiB"},
    )
    model.eval()
    print("[bcb-qual] model loaded", flush=True)

    results = []
    for r in rows:
        for kind in ("Lexical", "Vagueness"):
            prompt = r[f"{kind}_prompt"]
            explanation = r[f"{kind}_ambiguity_explanation"]

            messages = [{"role": "user", "content": INSTRUCTION.format(prompt=prompt)}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=150, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            row = {
                "task_id": r["task_id"],
                "kind": kind,
                "ambiguous_prompt": prompt,
                "true_explanation": explanation,
                "qwen_question": gen_text,
            }
            results.append(row)
            print(f"[bcb-qual] {r['task_id']} [{kind}]", flush=True)
            print(f"    true ambiguity: {explanation[:200]}", flush=True)
            print(f"    qwen asked:     {gen_text[:200]}", flush=True)

    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[bcb-qual] saved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
