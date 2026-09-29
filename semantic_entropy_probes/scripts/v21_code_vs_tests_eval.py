"""v21: the first real functional-correctness (code generation vs. executable
test cases) comparison of round-1 probe (`probe_1`) vs. SupCon round-2 probe
(v20j's best variant, `supcon_mlp_fixed_w10`) -- every prior v20g/h/i/j
comparison only measured RANKING hit-rate against dataset tier labels, never
whether the resulting clarified prompt actually leads to code that passes
real tests.

Per the user's explicit direction: reuse the SAME 244-group held-out test
split every v20c/g/h/i/j number was computed on (group-disjoint 80/20,
SEED=0, over the 1221 v19-pool groups that have round-2 branches) -- this
IS "the full test set used [for evaluating] round 1". Every group gets a
GENUINE two-clarification pipeline, not just the misses:
  - 83 chainable misses: round-1 wrong pick -> chain to that wrong
    candidate's pre-built round-2 branch -> SupCon probe picks the
    correction there. No live LLM calls needed (reuses existing
    deepseek_judged.jsonl / v12_ranking/judged.jsonl / answers_round2.jsonl).
  - 160 hits + 1 unrecoverable miss: no branch exists for a CORRECT round-1
    candidate (branches are only ever built from a wrong one, by design), so
    these get a genuinely FRESH round-2 elicitation+oracle-answer (Bedrock,
    grounded in the group's own reference_text/extra_context, same
    convention as the existing round-2 branches) on top of the correct
    round-1 clarification, then the SAME trained SupCon probe scores the
    new candidates. This measures a different question than the misses:
    not "can round-2 recover a wrong pick" but "does forcing an unforced
    second clarification on an already-correct answer help, hurt, or do
    nothing" to code correctness.
Either way, the final step is the same: build the actual clarified prompt
text (v19's own `CONCAT_TEMPLATE`) and generate code from it with Qwen3-8B
(the same model every ranking probe's activations were extracted from), run
against REAL test cases:
  - orchid_heval (47 groups): data/raw/orchid_heval.jsonl's {input,output}
    cases, via heval_run_tests.py (already existed).
  - orchid_bcb / orchid_bcb_expand (166 groups): data/raw/orchid_bcb*.jsonl's
    `test` field (a BigCodeBench-style unittest.TestCase), via the NEW
    v21_bcb_run_tests.py.
  - clarifycodebench (31 groups): ClarifyCodeBench problems are themselves
    AtCoder problems from LiveCodeBench (question_id e.g. "abc393_e") --
    ClarifyCodeBench.jsonl's own hidden tests aren't redistributed, but the
    ORIGINAL LiveCodeBench test cases are fetchable by that question_id
    (v21_fetch_ccb_test_cases.py, already run -> data/v20_branches_full/
    ccb_test_cases.json), and run via v10_run_tests.py's stdin/stdout runner.
  All 244 test groups are therefore scoreable (0 excluded).

Usage: python v21_code_vs_tests_eval.py
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import v19_train_ranking_probe as v19  # noqa: E402
import v20j_gpu_round2_ablation as v20j  # noqa: E402  (reuse its probe-training/scoring code as-is)
from v19_extract_activations import (  # noqa: E402
    JUDGED_PATH, V12_JUDGED_PATH, CONCAT_TEMPLATE, MODEL_NAME, LAYERS,
    load_answers_by_question, read_jsonl, hidden_states_at_layers_batch,
)
from v12_extract_activations import generate_sane_batch  # noqa: E402
from coding_agent import extract_code_block  # noqa: E402
from v21_bcb_run_tests import run_all as run_bcb_tests  # noqa: E402
from heval_run_tests import run_all as run_heval_tests  # noqa: E402
from v10_run_tests import run_all as run_stdio_tests  # noqa: E402

LAYER = 22
SEED = 0
TEST_FRAC = 0.2
MAX_NEW_TOKENS = 768

BRANCHES_PATH = ROOT / "data" / "v20_branches_full" / "branches_round2.json"
ACT_DIR = ROOT / "data" / "v20_branches_full" / "round2_activations"
ANSWERS_R2_PATH = ROOT / "data" / "v20_branches_full" / "answers_round2.jsonl"
CCB_TESTS_PATH = ROOT / "data" / "v20_branches_full" / "ccb_test_cases.json"
CHUNK_DIR = ROOT / "data" / "v13_rejudge" / "chunks"
RAW_HEVAL = ROOT / "data" / "raw" / "orchid_heval.jsonl"
RAW_BCB = ROOT / "data" / "raw" / "orchid_bcb.jsonl"
RAW_BCB_EXPAND = ROOT / "data" / "raw" / "orchid_bcb_expand.jsonl"
CCB_JSONL = Path("/home/ubuntu/SEP-IE/ClarifyCodeBench/data/ClarifyCodeBench.jsonl")

OUT_DIR = ROOT / "data" / "v21_code_vs_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUT_DIR / "results.json"
FRESH_R2_DIR = OUT_DIR / "fresh_round2_hits"
FRESH_R2_DIR.mkdir(parents=True, exist_ok=True)

BEDROCK_MODEL_ID = "deepseek.v3.2"
BEDROCK_REGION = "ap-northeast-1"
N_WORKERS = 8
N_ELICIT = 8

# Grounded exactly like v19_bedrock_oracle_answer.py's ORACLE_RULE/SYSTEM_PROMPT, minus the
# separate elicit step -- since these groups' round-1 pick already resolved the ONE deliberately
# injected ambiguity (reference_text), there's no tier-2 target left to judge candidates against;
# this only needs fresh candidate questions + grounded one-sentence answers for the SupCon probe
# to rank, not a repeat of the tier-labeling machinery.
FRESH_R2_SYSTEM_PROMPT = f"""You are reviewing a coding problem specification that already has one \
clarification appended at the end. Invent up to {N_ELICIT} diverse, plausible follow-up clarifying \
questions a careful reviewer would still want answered before implementing this exact prompt -- not \
generic boilerplate (type validation / negative numbers / empty input) unless it genuinely applies \
here. Then answer each one yourself. You have privileged knowledge of the TRUE reference given below, \
which the person asking cannot see. Answer using ONLY the information in the true reference; if a \
question isn't addressed by it, answer with the most standard/reasonable convention rather than \
inventing a new constraint. Answer in ONE concise sentence each.

Respond with ONLY a JSON object, no markdown fences:
{{"candidates": [{{"question": "<question>", "answer": "<one concise sentence>"}}, ...]}}"""


def load_aws_env() -> None:
    env = {}
    for line in open(ROOT.parent / ".env"):
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = env["AWS"]


def extract_json_obj(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    start = text.find("{")
    depth, in_str, esc = 0, False, False
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
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON")


def call_deepseek_elicit_answer(rt, round1_prompt: str, reference_text: str, extra_context: str) -> list[dict]:
    user_prompt = (
        f"--- CURRENT SPECIFICATION (already includes one clarification) ---\n{round1_prompt}\n\n"
        f"--- TRUE REFERENCE (privileged) ---\n{reference_text}\n\n{extra_context}"
    )
    body = json.dumps({
        "messages": [
            {"role": "system", "content": FRESH_R2_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 3000, "temperature": 0.3,
    })
    delay = 5
    for attempt in range(8):
        try:
            resp = rt.invoke_model(modelId=BEDROCK_MODEL_ID, body=body)
            payload = json.loads(resp["body"].read())
            raw = payload["choices"][0]["message"]["content"]
            return extract_json_obj(raw)["candidates"]
        except Exception as e:  # noqa: BLE001
            if attempt == 7:
                raise
            msg = str(e)
            delay = min(delay * 2, 120)
            time.sleep(delay if ("Throttling" in msg or "TooManyRequests" in msg) else min(delay, 20))


def fetch_fresh_round2_candidates(groups: list[str], round1_prompt_by_gid: dict[str, str],
                                   chunk_groups: dict[str, dict]) -> dict[str, list[dict]]:
    """Elicit+answer a fresh round-2 candidate pool for groups whose round-1 pick was already
    correct (or unrecoverably missed) -- no pre-built branch exists for these (branches are only
    ever built from a WRONG round-1 candidate). Resumable: one cache file per group id."""
    todo = [g for g in groups if not (FRESH_R2_DIR / f"{safe_name(g)}.json").exists()]
    print(f"[v21-fresh-r2] {len(groups)-len(todo)} cached, {len(todo)} to elicit via Bedrock", flush=True)

    def worker(gid):
        try:
            rt = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
            g = chunk_groups[gid]
            cands = call_deepseek_elicit_answer(rt, round1_prompt_by_gid[gid], g["reference_text"], g["extra_context"])
            (FRESH_R2_DIR / f"{safe_name(gid)}.json").write_text(json.dumps({"id": gid, "candidates": cands}))
            return gid, None
        except Exception as e:  # noqa: BLE001
            return gid, str(e)

    if todo:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(worker, gid): gid for gid in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                gid, err = fut.result()
                if err:
                    print(f"[v21-fresh-r2] ({i}/{len(todo)}) ERROR {gid}: {err[:200]}", flush=True)
                else:
                    print(f"[v21-fresh-r2] ({i}/{len(todo)}) {gid}: done", flush=True)

    out = {}
    for g in groups:
        p = FRESH_R2_DIR / f"{safe_name(g)}.json"
        if p.exists():
            out[g] = json.loads(p.read_text())["candidates"]
    return out


def safe_name(rid: str) -> str:
    return rid.replace("/", "__")


def with_retry(fn, tries=4, base_delay=3.0, log_prefix=""):
    """Transient NVML/CUDACachingAllocator errors have been observed mid-generation on this
    box (same flakiness v20j_gpu_round2_ablation.py's docstring documents for GPU encode calls) --
    retry with a cache-clear + backoff instead of losing the whole run to one bad allocation."""
    for attempt in range(tries):
        try:
            return fn()
        except RuntimeError as e:
            if attempt == tries - 1:
                raise
            print(f"{log_prefix}RuntimeError on attempt {attempt+1}/{tries} ({e}); "
                  f"clearing cache and retrying...", flush=True)
            torch.cuda.empty_cache()
            time.sleep(base_delay * (attempt + 1))

CODE_INSTRUCTION_HEVAL = (
    "Complete the following Python function. Include any necessary imports (the signature line "
    "itself may omit them). Output ONLY the complete function -- the given signature plus your "
    "implementation -- optionally inside a single ```python ... ``` block. No explanation, no "
    "usage examples, no extra text.\n\n<problem>\n{problem}\n</problem>"
)
CODE_INSTRUCTION_BCB = (
    "Complete the following Python function. Here is the existing code (imports and function "
    "signature):\n```python\n{code_prompt}\n```\n\nHere is the function's full specification "
    "(it may include a clarification not shown in the code above):\n<spec>\n{spec}\n</spec>\n\n"
    "Output ONLY the complete function definition -- starting from the imports and signature "
    "given above, with your implementation added -- inside a single ```python ... ``` block. "
    "No explanation, no usage examples, no extra text."
)
CODE_INSTRUCTION_CCB = (
    "Write a complete, self-contained Python program that solves the following competitive-"
    "programming problem. Read all input from standard input and print the required output to "
    "standard output, exactly in the format specified. Output ONLY the program inside a single "
    "```python ... ``` block. No explanation, no usage examples, no extra text.\n\n"
    "<problem>\n{problem}\n</problem>"
)


def load_chunk_groups() -> dict[str, dict]:
    groups = {}
    for f in sorted(CHUNK_DIR.glob("chunk_*.json")):
        for g in json.loads(f.read_text()):
            groups[g["id"]] = g
    return groups


def fresh_round2_final_prompts(needs_fresh_gids, prompt1_by_gid, chunk_groups, tokenizer, model,
                                supcon_pipe, head, li) -> dict[str, str | None]:
    """For hit/unrecoverable-miss groups (no pre-built wrong-candidate branch exists to chain
    from): elicit+answer a genuinely fresh round-2 candidate pool ON TOP of the round-1-correct
    prompt (Bedrock), then score with the SAME trained SupCon probe to pick a second
    clarification. Returns None for a group where elicitation failed (caller falls back to
    round-1-only for that group).

    One GPU forward pass PER GROUP (~8 candidates, small), not one giant batch -- the NVML/
    CUDACachingAllocator crash observed on this box hits large batched forward passes, and
    each group's winning prompt is cached to disk immediately so a mid-run crash only costs
    re-computing groups after the last checkpoint, not the whole set."""
    cand_map = fetch_fresh_round2_candidates(needs_fresh_gids, prompt1_by_gid, chunk_groups)
    out: dict[str, str | None] = {}
    for i, gid in enumerate(needs_fresh_gids, 1):
        picked_path = FRESH_R2_DIR / f"{safe_name(gid)}.picked.json"
        if picked_path.exists():
            cached = json.loads(picked_path.read_text())
            out[gid] = cached["prompt2"]
            continue
        cands = cand_map.get(gid) or []
        cands = [c for c in cands if c.get("question") and c.get("answer")]
        if not cands:
            out[gid] = None
            picked_path.write_text(json.dumps({"id": gid, "prompt2": None}))
            continue
        texts = [CONCAT_TEMPLATE.format(prompt=prompt1_by_gid[gid], answer=c["answer"]) for c in cands]
        if len(texts) < 2:
            # score_win_counting needs >=2 candidates to form pairwise diffs; with exactly one,
            # there's nothing to rank against -- just take it.
            out[gid] = texts[0]
        else:
            vecs = with_retry(lambda t=texts: hidden_states_at_layers_batch(tokenizer, model, t, LAYERS),
                               log_prefix=f"[v21-fresh-r2] [{i}/{len(needs_fresh_gids)}] ")
            vecs_l = vecs[:, li, :]
            top = int(np.argmax(v20j.score_win_counting(supcon_pipe, head, vecs_l)))
            out[gid] = texts[top]
        picked_path.write_text(json.dumps({"id": gid, "prompt2": out[gid]}))
        if i % 20 == 0:
            print(f"[v21-fresh-r2] scored {i}/{len(needs_fresh_gids)} groups' fresh candidates", flush=True)
    return out


def load_raw_lookups():
    heval_by_n = {int(r["name"].split("/")[1]): r for r in read_jsonl(RAW_HEVAL)}
    bcb_by_tid = {r["task_id"]: r for r in read_jsonl(RAW_BCB)}
    bcb_expand_by_tid = {r["task_id"]: r for r in read_jsonl(RAW_BCB_EXPAND)}
    ccb_by_tid = {r["task_id"]: r for r in read_jsonl(CCB_JSONL)}
    ccb_tests = json.loads(CCB_TESTS_PATH.read_text())
    return heval_by_n, bcb_by_tid, bcb_expand_by_tid, ccb_by_tid, ccb_tests


def problem_info(gid: str, source: str, lookups) -> dict:
    heval_by_n, bcb_by_tid, bcb_expand_by_tid, ccb_by_tid, ccb_tests = lookups
    if source == "orchid_heval":
        n = int(gid.split("_")[1])
        r = heval_by_n[n]
        return {"kind": "heval", "entry_point": r["entry_point"], "test_case": r["test_case"]}
    if source in ("orchid_bcb", "orchid_bcb_expand"):
        prefix = "bcb_" if source == "orchid_bcb" else "bcbexpand_"
        rest = gid[len(prefix):]
        tid = rest.rsplit("_", 1)[0]
        r = (bcb_by_tid if source == "orchid_bcb" else bcb_expand_by_tid)[tid]
        return {"kind": "bcb", "code_prompt": r["code_prompt"], "test": r["test"], "entry_point": r["entry_point"]}
    if source == "clarifycodebench":
        tid = gid.replace("ccb_amb_", "")
        r = ccb_by_tid[tid]
        return {"kind": "ccb", "test_cases": ccb_tests[r["question_id"]]}
    raise ValueError(source)


def round1_qa(gid: str, judged_rows: dict, v12_judged: dict) -> tuple[list[str], list[str]]:
    """Same candidate order/filter as v19_extract_activations.py's extraction loop --
    aligns 1:1 with per_group_vecs[gid]/per_group_tiers[gid]'s row order."""
    answers_by_q = load_answers_by_question(gid, v12_judged)
    qs, answers = [], []
    for c in judged_rows[gid]["judged"]["candidates"]:
        q = c["question"].strip()
        ans = answers_by_q.get(q)
        if ans is None:
            continue
        qs.append(q)
        answers.append(ans)
    return qs, answers


def round1_final_prompt(gid, top1_idx, prompt_by_gid, judged_rows, v12_judged):
    qs, answers = round1_qa(gid, judged_rows, v12_judged)
    return CONCAT_TEMPLATE.format(prompt=prompt_by_gid[gid], answer=answers[top1_idx])


def round2_final_prompt(branch_idx, top1_idx, branches, answers_r2):
    branch = branches[branch_idx]
    meta = json.loads((ACT_DIR / f"{branch_idx}.json").read_text())
    q = meta["questions"][top1_idx]
    ans_by_q = {a["question"].strip(): a["answer"] for a in answers_r2[branch_idx]["answers"]
                if "question" in a and "answer" in a}
    return CONCAT_TEMPLATE.format(prompt=branch["branch_prompt"], answer=ans_by_q[q])


def code_instruction(pinfo: dict, final_prompt: str) -> str:
    if pinfo["kind"] == "heval":
        return CODE_INSTRUCTION_HEVAL.format(problem=final_prompt)
    if pinfo["kind"] == "bcb":
        return CODE_INSTRUCTION_BCB.format(code_prompt=pinfo["code_prompt"], spec=final_prompt)
    return CODE_INSTRUCTION_CCB.format(problem=final_prompt)


def make_is_sane(pinfo: dict):
    def check(raw: str) -> bool:
        code = extract_code_block(raw)
        try:
            ast.parse(code)
        except SyntaxError:
            return False
        if pinfo["kind"] in ("heval", "bcb"):
            return f"def {pinfo['entry_point']}" in code
        return True
    return check


CODEGEN_CACHE_DIR = OUT_DIR / "codegen"
CODEGEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CODEGEN_BATCH = 12


def generate_cached(tokenizer, model, ids: list[str], round_tag: str, inputs: list[str],
                     sane_fns: list) -> list[str]:
    """Chunked + per-group-cached wrapper around generate_sane_batch -- a crash mid-run (the
    NVML/CUDACachingAllocator flakiness seen on this box) only costs the current small chunk,
    not the whole 244-group batch, and a rerun skips everything already generated."""
    cache_dir = CODEGEN_CACHE_DIR / round_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = [None] * len(ids)
    todo = []
    for i, gid in enumerate(ids):
        p = cache_dir / f"{safe_name(gid)}.txt"
        if p.exists():
            out[i] = p.read_text()
        else:
            todo.append(i)
    print(f"[v21-codegen-{round_tag}] {len(ids)-len(todo)} cached, {len(todo)} to generate", flush=True)
    for c0 in range(0, len(todo), CODEGEN_BATCH):
        idxs = todo[c0:c0 + CODEGEN_BATCH]
        batch_inputs = [inputs[i] for i in idxs]
        batch_sane = [sane_fns[i] for i in idxs]
        results = with_retry(lambda bi=batch_inputs, bs=batch_sane: generate_sane_batch(
            tokenizer, model, bi, MAX_NEW_TOKENS, bs, log_prefix=f"[v21-codegen-{round_tag}] "),
            log_prefix=f"[v21-codegen-{round_tag}] ")
        for i, text in zip(idxs, results):
            out[i] = text
            (cache_dir / f"{safe_name(ids[i])}.txt").write_text(text)
        print(f"[v21-codegen-{round_tag}] generated {min(c0+CODEGEN_BATCH, len(todo))}/{len(todo)}", flush=True)
    return out


def call_deepseek_code(rt, user_prompt: str, max_tokens: int = 2000) -> str:
    """DeepSeek-as-codegen counterpart to Qwen's generate_sane_batch -- same
    CODE_INSTRUCTION_* prompt text, sent to Bedrock instead of run locally."""
    body = json.dumps({
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": max_tokens, "temperature": 0.3,
    })
    delay = 5
    for attempt in range(8):
        try:
            resp = rt.invoke_model(modelId=BEDROCK_MODEL_ID, body=body)
            payload = json.loads(resp["body"].read())
            return payload["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            if attempt == 7:
                raise
            msg = str(e)
            delay = min(delay * 2, 120)
            time.sleep(delay if ("Throttling" in msg or "TooManyRequests" in msg) else min(delay, 20))


def generate_cached_deepseek(ids: list[str], round_tag: str, inputs: list[str], sane_fns: list,
                              cache_root: Path | None = None, max_retries: int = 3) -> list[str]:
    """DeepSeek-as-codegen counterpart to generate_cached -- same per-group disk cache
    convention (so a crash/interrupt only costs whatever's in flight), parallelized over
    Bedrock calls instead of batched on GPU, with the same is_sane retry-on-garbage-output
    contract generate_sane_batch provides for Qwen."""
    cache_dir = (cache_root or CODEGEN_CACHE_DIR) / round_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = [None] * len(ids)
    todo = []
    for i, gid in enumerate(ids):
        p = cache_dir / f"{safe_name(gid)}.txt"
        if p.exists():
            out[i] = p.read_text()
        else:
            todo.append(i)
    print(f"[v21-codegen-deepseek-{round_tag}] {len(ids)-len(todo)} cached, {len(todo)} to generate", flush=True)

    def worker(i):
        rt = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        text = ""
        for attempt in range(max_retries):
            text = call_deepseek_code(rt, inputs[i])
            if sane_fns[i](text):
                break
        (cache_dir / f"{safe_name(ids[i])}.txt").write_text(text)
        return i, text

    if todo:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(worker, i): i for i in todo}
            for n, fut in enumerate(as_completed(futs), 1):
                i, text = fut.result()
                out[i] = text
                if n % 20 == 0:
                    print(f"[v21-codegen-deepseek-{round_tag}] generated {n}/{len(todo)}", flush=True)
    return out


def run_tests_for(pinfo: dict, raw_code: str) -> dict:
    code = extract_code_block(raw_code)
    if pinfo["kind"] == "heval":
        res = run_heval_generic(code, pinfo["entry_point"], pinfo["test_case"])
    elif pinfo["kind"] == "bcb":
        res = run_bcb_tests(code, pinfo["test"])
    else:
        res = run_stdio_tests(code, pinfo["test_cases"])
    n_total = res["n_total"]
    return {"n_pass": res["n_pass"], "n_total": n_total, "all_pass": n_total > 0 and res["n_pass"] == n_total}


def run_heval_generic(code: str, entry_point: str, test_cases: list[dict]) -> dict:
    """heval_run_tests.run_all only handles relation=='==' (asserts on anything else).
    244 test groups include exactly 1 (orchidheval_4_semantic) with float-tolerance
    relation exprs like 'abs(candidate(...) - X) < 1e-6' -- the ORIGINAL HumanEval
    eval-harness convention (`candidate` = the function under test). Handle both."""
    if all(tc.get("relation") == "==" for tc in test_cases):
        return run_heval_tests(code, entry_point, test_cases)
    import subprocess
    import tempfile
    driver = (
        f"{code}\n\ncandidate = {entry_point}\n"
        f"_results = []\n"
        f"for _expr in {json.dumps([tc['relation'] for tc in test_cases])}:\n"
        f"    try:\n"
        f"        _results.append(bool(eval(_expr)))\n"
        f"    except Exception:\n"
        f"        _results.append(False)\n"
        f"import json as _json\nprint(_json.dumps(_results))\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(driver)
        path = f.name
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=15)
        if proc.returncode != 0:
            return {"n_pass": 0, "n_total": len(test_cases)}
        passed = json.loads(proc.stdout.strip().splitlines()[-1])
        return {"n_pass": sum(passed), "n_total": len(test_cases)}
    except subprocess.TimeoutExpired:
        return {"n_pass": 0, "n_total": len(test_cases)}
    finally:
        Path(path).unlink(missing_ok=True)


def main() -> None:
    import argparse
    global RESULTS_PATH, CODEGEN_CACHE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--codegen", choices=["qwen", "deepseek"], default="qwen",
                     help="which model writes the code -- selection/ranking (probe_1 + SupCon "
                          "round-2, which needs Qwen for activation extraction regardless) is "
                          "unaffected by this flag, only the final codegen step is")
    args = ap.parse_args()
    if args.codegen == "deepseek":
        RESULTS_PATH = OUT_DIR / "results_deepseek_codegen.json"
        CODEGEN_CACHE_DIR = OUT_DIR / "codegen_deepseek"
        CODEGEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[v21] codegen backend = {args.codegen}", flush=True)

    print("[v21] loading round-1 pool + round-2 branches ...", flush=True)
    ids, dataset_source, source_problem_id, layers, per_group_vecs, per_group_tiers = v19.load_groups()
    li = layers.tolist().index(LAYER)
    src_of = dict(zip(ids.tolist(), dataset_source.tolist()))

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
    train_groups_r2 = [g for g in groups_with_branches if g not in set(test_groups)]

    # --- Stage 1: probe_1 (identical to v20g/v20j) ---
    train_pools_r1 = [(per_group_vecs[g][li], per_group_tiers[g]) for g in train_groups_r2
                       if g in per_group_vecs and len(set(per_group_tiers[g].tolist())) >= 2]
    pipe1, w1 = v20j.fit_pairwise_lr(train_pools_r1)

    per_group = {}
    for g in test_groups:
        if g not in per_group_vecs or len(set(per_group_tiers[g].tolist())) < 2:
            continue
        vecs, tiers = per_group_vecs[g][li], per_group_tiers[g]
        scores = pipe1.transform(vecs) @ w1
        top1 = int(np.argmax(scores))
        per_group[g] = {"top1_idx": top1, "hit1": int(tiers[top1] == tiers.max())}
    n1 = len(per_group)
    hit1_rate = sum(v["hit1"] for v in per_group.values()) / n1
    print(f"[v21] probe_1 scored {n1} test groups, hit1 rate = {hit1_rate:.4f} "
          f"(parity check vs. v20g/j's 0.6557)", flush=True)

    # --- Stage 2: SupCon round-2 probe, best v20j variant (supcon_mlp_fixed_w10) ---
    train_pools_r2 = []
    for g in train_groups_r2:
        for branch_idx, _q in branches_by_group[g]:
            d = np.load(ACT_DIR / f"{branch_idx}.npz")
            vecs = d["candidates"][:, list(d["layers"]).index(LAYER), :]
            train_pools_r2.append((vecs, d["tiers"], branches[branch_idx]["dataset_source"]))
    print(f"[v21] fitting SupCon round-2 probe on {len(train_pools_r2)} branches ...", flush=True)
    supcon_pipe = v20j.fit_supcon_pipe(train_pools_r2, cross_source_only=False)
    zs_train = v20j.batch_encode(supcon_pipe, train_pools_r2)
    type_counts = v20j.count_pair_types(train_pools_r2)
    weight_of = v20j.weight_fn_for("fixed_w10", type_counts)
    X, y, w = v20j.build_pairwise_diffs(zs_train, train_pools_r2, weight_of)
    head = v20j.TorchPairwiseMLPHead(in_dim=zs_train[0].shape[1]).fit(X, y, w)

    judged_rows = {r["id"]: r for r in read_jsonl(JUDGED_PATH)}
    v12_judged = {r["id"]: r for r in read_jsonl(V12_JUDGED_PATH)}
    answers_r2 = {d["branch_idx"]: d for d in read_jsonl(ANSWERS_R2_PATH) if "error" not in d}
    chunk_groups = load_chunk_groups()
    prompt_by_gid = {g: chunk_groups[g]["prompt"] for g in chunk_groups}
    lookups = load_raw_lookups()

    # --- Chain misses to their round-2 branch, pick the SupCon probe's top-1 there ---
    # (hits and the 1 unrecoverable miss get round2_branch_idx=None -- no branch was ever built
    # from a CORRECT round-1 candidate, so these get a fresh elicitation below instead.)
    for g, info in per_group.items():
        info["round2_branch_idx"] = None
        if info["hit1"]:
            continue
        qs, _ = round1_qa(g, judged_rows, v12_judged)
        if info["top1_idx"] >= len(qs):
            continue
        wrong_q = qs[info["top1_idx"]]
        match = next((idx for idx, bq in branches_by_group[g] if bq == wrong_q), None)
        info["round2_branch_idx"] = match

    prompt1_by_gid = {
        g: round1_final_prompt(g, info["top1_idx"], prompt_by_gid, judged_rows, v12_judged)
        for g, info in per_group.items()
    }
    n_chained = sum(1 for v in per_group.values() if v["round2_branch_idx"] is not None)
    needs_fresh_gids = [g for g, v in per_group.items() if v["round2_branch_idx"] is None]
    print(f"[v21] {n_chained} groups chain to an existing wrong-candidate round-2 branch; "
          f"{len(needs_fresh_gids)} groups (hits + unrecoverable misses) need fresh round-2 "
          f"elicitation on top of their correct round-1 pick", flush=True)

    # --- Load Qwen3-8B once: used for fresh-candidate activation extraction below AND codegen ---
    print(f"[v21] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = with_retry(lambda: AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda"), log_prefix="[v21-load] ")
    model.eval()
    print("[v21] model loaded", flush=True)

    load_aws_env()
    fresh_prompt2 = fresh_round2_final_prompts(
        needs_fresh_gids, prompt1_by_gid, chunk_groups, tokenizer, model, supcon_pipe, head, li,
    )

    tasks = []
    for g, info in per_group.items():
        source = src_of[g]
        pinfo = problem_info(g, source, lookups)
        prompt1 = prompt1_by_gid[g]
        bidx = info["round2_branch_idx"]
        if bidx is not None:
            d = np.load(ACT_DIR / f"{bidx}.npz")
            vecs = d["candidates"][:, list(d["layers"]).index(LAYER), :]
            top2 = int(np.argmax(v20j.score_win_counting(supcon_pipe, head, vecs)))
            prompt2 = round2_final_prompt(bidx, top2, branches, answers_r2)
            hit2 = int(d["tiers"][top2] == d["tiers"].max())
        else:
            prompt2 = fresh_prompt2.get(g) or prompt1
            hit2 = None  # no labeled ambiguity left to judge a "best" second clarification against
        tasks.append({
            "id": g, "source": source, "pinfo": pinfo, "prompt1": prompt1, "prompt2": prompt2,
            "hit1": info["hit1"], "hit2": hit2, "same_prompt": prompt1 == prompt2,
        })
    n_fallback = sum(1 for g in needs_fresh_gids if fresh_prompt2.get(g) is None)
    print(f"[v21] {len(tasks)} tasks built; {n_fallback} fresh-elicitation groups fell back to "
          f"round1-only (elicitation failed)", flush=True)

    # --- Codegen: every task gets its OWN round-1 and round-2 generation (round-2 is now a
    # genuinely different prompt for every group except the rare elicitation-failure fallback).
    # Chunked + cached to disk (generate_cached) so GPU flakiness only costs a small batch. ---
    ids_list = [t["id"] for t in tasks]
    round1_inputs = [code_instruction(t["pinfo"], t["prompt1"]) for t in tasks]
    sane1 = [make_is_sane(t["pinfo"]) for t in tasks]
    print("[v21] generating round-1 code ...", flush=True)
    if args.codegen == "deepseek":
        raw1 = generate_cached_deepseek(ids_list, "r1", round1_inputs, sane1)
    else:
        raw1 = generate_cached(tokenizer, model, ids_list, "r1", round1_inputs, sane1)

    diff_idxs = [i for i, t in enumerate(tasks) if not t["same_prompt"]]
    diff_ids = [ids_list[i] for i in diff_idxs]
    round2_inputs = [code_instruction(tasks[i]["pinfo"], tasks[i]["prompt2"]) for i in diff_idxs]
    sane2 = [make_is_sane(tasks[i]["pinfo"]) for i in diff_idxs]
    print(f"[v21] generating round-2 code for {len(diff_idxs)} groups ...", flush=True)
    if diff_idxs:
        raw2_partial = (generate_cached_deepseek(diff_ids, "r2", round2_inputs, sane2) if args.codegen == "deepseek"
                        else generate_cached(tokenizer, model, diff_ids, "r2", round2_inputs, sane2))
    else:
        raw2_partial = []
    raw2 = list(raw1)
    for k, i in enumerate(diff_idxs):
        raw2[i] = raw2_partial[k]

    del model
    torch.cuda.empty_cache()

    # --- Run against real tests ---
    results = []
    for i, t in enumerate(tasks):
        r1 = run_tests_for(t["pinfo"], raw1[i])
        r2 = r1 if t["same_prompt"] else run_tests_for(t["pinfo"], raw2[i])
        results.append({
            "id": t["id"], "source": t["source"], "hit1": t["hit1"], "hit2": t["hit2"],
            "same_prompt": t["same_prompt"], "chained_miss": t["hit2"] is not None,
            "round1_n_pass": r1["n_pass"], "round1_n_total": r1["n_total"], "round1_pass": r1["all_pass"],
            "round2_n_pass": r2["n_pass"], "round2_n_total": r2["n_total"], "round2_pass": r2["all_pass"],
        })
        print(f"[v21] ({i+1}/{len(tasks)}) {t['id']} [{t['source']}] "
              f"round1={r1['n_pass']}/{r1['n_total']} round2={r2['n_pass']}/{r2['n_total']}", flush=True)

    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    def summarize(rows, key):
        n = len(rows)
        if n == 0:
            return 0.0, 0
        return sum(r[key] for r in rows) / n, n

    print("\n[v21] === SUMMARY: functional pass@1 rate (round-1 probe only vs. +SupCon round-2) ===", flush=True)
    r1_rate, n = summarize(results, "round1_pass")
    r2_rate, _ = summarize(results, "round2_pass")
    print(f"  OVERALL (n={n}): round1-only={r1_rate:.4f}  round1+supcon_round2={r2_rate:.4f}", flush=True)

    chained = [r for r in results if r["chained_miss"]]
    fresh = [r for r in results if not r["chained_miss"]]
    r1c, nc = summarize(chained, "round1_pass")
    r2c, _ = summarize(chained, "round2_pass")
    print(f"  chained misses  (n={nc:3d}): round1-only={r1c:.4f}  round1+supcon_round2={r2c:.4f}  "
          f"<- 'can round-2 recover a wrong pick'", flush=True)
    r1f, nf = summarize(fresh, "round1_pass")
    r2f, _ = summarize(fresh, "round2_pass")
    print(f"  hit/unrecov.    (n={nf:3d}): round1-only={r1f:.4f}  round1+supcon_round2={r2f:.4f}  "
          f"<- 'does an unforced 2nd clarification help/hurt an already-correct answer'", flush=True)

    by_source = defaultdict(list)
    for r in results:
        by_source[r["source"]].append(r)
    for src, rows in sorted(by_source.items()):
        r1s, n = summarize(rows, "round1_pass")
        r2s, _ = summarize(rows, "round2_pass")
        print(f"  {src:20s} (n={n:3d}): round1-only={r1s:.4f}  round1+supcon_round2={r2s:.4f}", flush=True)
    print(f"[v21] saved per-group results -> {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
