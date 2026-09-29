"""The (Groq-primary, OpenRouter-fallback) API agent: Groq-hosted
openai/gpt-oss-120b, falling back to OpenRouter's Qwen3-Coder-480B-A35B when
Groq is exhausted. Originally built purely as "the coding agent" (codegen
only, see generate_code_agent/CODE_INSTRUCTION below), but the same
provider-dispatch machinery (generate_agent) is reused for other tasks too --
e.g. v11_heval_batch_worker.py's question-elicitation step, testing whether a
more capable model proposes better clarifying questions than local Qwen3-8B.

Deliberately torch/transformers-free (only `requests`) so scripts that need
just an API call -- e.g. rerunning codegen+test against already-computed
clarified prompts from a prior round -- don't require a GPU/ML environment.
Imported by v10_round_worker.py (which pulls in torch/transformers for the
local scoring/style-rewrite model) and re-exported from there for backward
compatibility with scripts that do `from scripts.v10_round_worker import ...`.

History: originally OpenRouter's Qwen3-Coder-480B-A35B alone (module was named
openrouter_codegen.py). Switched to Groq/gpt-oss-120b after OpenRouter hit its
50-prompts/day account limit -- Groq doesn't host Qwen3-Coder-480B-A35B, so
that was a genuinely different (non-Qwen, non-coder-specialized) model. Then
Groq's own openai/gpt-oss-120b turned out to have a 200,000-tokens/day cap
that a single afternoon of iterating on this module blew through (observed:
"Used 198582/200000 ... try again in 13m21s" on a rolling window -- NOT the
per-minute x-ratelimit-* headers, which looked fine and were a red herring).
So: Groq is primary, OpenRouter is the fallback when Groq's retries are
exhausted. generate_agent()/generate_code_agent() return (content,
provider_used) so callers can record which model actually produced each
result -- worth keeping given results are no longer guaranteed to all come
from the same model.
"""

from __future__ import annotations

import os
import re
import time

import requests


def _parse_reset_duration(s: str) -> float:
    """Parse Groq's `x-ratelimit-reset-*` header format ("27m21.6s", "36.48s", "800ms")
    into seconds."""
    m = re.match(r'^(?:(\d+)m)?(\d+(?:\.\d+)?)(s|ms)?$', s.strip())
    if not m:
        return 10.0
    minutes, value, unit = m.groups()
    seconds = float(value) / 1000.0 if unit == "ms" else float(value)
    return (int(minutes) * 60 if minutes else 0) + seconds


CODE_INSTRUCTION = (
    "Write a complete Python 3 program that solves the following competitive-programming problem. "
    "The program must read all input from standard input and write the answer to standard output, exactly "
    "matching the I/O format described.\n\n"
    "<problem>\n{problem}\n</problem>\n\n"
    "Output ONLY the Python source code (optionally inside a single ```python ... ``` block). No explanation."
)

# provider configs, tried in order until one succeeds
PROVIDERS = {
    "groq": {
        "api_key_env": "GROQ",
        "model": "openai/gpt-oss-120b",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        # gpt-oss-120b is a reasoning model: without this, it can burn the entire
        # max_tokens budget on chain-of-thought and return empty `content` with
        # finish_reason "length" (observed: 1598/1600 tokens spent reasoning).
        "extra_payload": {"reasoning_effort": "low"},
    },
    "openrouter": {
        "api_key_env": "OpenRouter",
        "model": "qwen/qwen3-coder",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "extra_payload": {},
    },
}
CODING_AGENT_PROVIDER_ORDER = ["groq", "openrouter"]


def extract_code_block(raw_text: str) -> str:
    """Pull the actual solution out of `raw_text`. Despite CODE_INSTRUCTION asking for
    code only, gpt-oss-120b sometimes ignores that on harder problems and writes a full
    explanation with its own short, untagged ``` snippets (e.g. a one-line "A[x] xor A[y]
    = Z" fragment quoting part of the problem) before the real ```python ... ``` block --
    a naive first-match regex grabs that short snippet instead of the real solution. Prefer
    the LAST ```python-tagged block (the solution normally comes after any explanation);
    fall back to the LONGEST untagged ``` block if no python-tagged one exists."""
    python_blocks = re.findall(r"```python\s*\n(.*?)```", raw_text, re.DOTALL)
    if python_blocks:
        return python_blocks[-1].strip() + "\n"
    any_blocks = re.findall(r"```\s*\n(.*?)```", raw_text, re.DOTALL)
    if any_blocks:
        return max(any_blocks, key=len).strip() + "\n"
    return raw_text.strip() + "\n"


def _call_provider(provider: dict, user_content: str, max_tokens: int, max_retries: int) -> str:
    api_key = os.environ.get(provider["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"${provider['api_key_env']} is not set in this process's environment. It's exported "
            f"in ~/.bashrc but gated behind an interactive-shell check, so non-interactive "
            f"launchers (nohup, cron, a plain `bash script.sh`, ...) won't see it -- "
            f"`export {provider['api_key_env']}=...` in the shell that starts this script."
        )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    tokens_this_attempt = max_tokens
    last_err = None
    for attempt in range(max_retries):
        payload = {
            "model": provider["model"],
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": tokens_this_attempt,
            "temperature": 0.0,
            **provider["extra_payload"],
        }
        try:
            resp = requests.post(provider["url"], headers=headers, json=payload, timeout=180)
            if resp.status_code == 429:
                # x-ratelimit-reset-tokens can misreport ~0 even while still 429ing (observed
                # a run stuck retrying every ~1ms-reported-wait, exhausting all retries in
                # under a second) -- that header reflects the per-minute window, which is a
                # red herring next to a daily-cap 429 (see module docstring); floor the wait
                # so retries can't busy-loop faster than any window plausibly moves.
                reset_s = _parse_reset_duration(resp.headers.get("x-ratelimit-reset-tokens", "15s"))
                wait_s = max(min(reset_s + 1.0, 90.0), 8.0)
                last_err = RuntimeError(
                    f"429 rate limited: {resp.text[:300]}"
                )
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            if content:
                return content
            last_err = RuntimeError(
                f"empty content, finish_reason={data['choices'][0].get('finish_reason')!r}, "
                f"usage={data.get('usage')}"
            )
            tokens_this_attempt *= 2
            time.sleep(5 * (attempt + 1))
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{provider['model']} call failed after {max_retries} attempts: {last_err}")


def generate_agent(user_content: str, max_tokens: int = 3200, max_retries: int = 5) -> tuple[str, str]:
    """Send `user_content` as a single user message to the API agent. Tries each provider
    in CODING_AGENT_PROVIDER_ORDER until one succeeds. Returns (raw_content, provider_name)
    -- record provider_name alongside results since it's no longer guaranteed to be the
    same model for every call. Generic: task-specific callers (codegen, question
    elicitation, ...) build their own prompt and pass it in."""
    errors = []
    for name in CODING_AGENT_PROVIDER_ORDER:
        try:
            return _call_provider(PROVIDERS[name], user_content, max_tokens, max_retries), name
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError("all API agent providers failed -- " + " | ".join(errors))


def generate_code_agent(problem: str, max_tokens: int = 3200, max_retries: int = 5) -> tuple[str, str]:
    """Generate a full Python solution for `problem` via the coding agent. Returns
    (raw_content, provider_name) -- pass raw_content through extract_code_block()."""
    return generate_agent(CODE_INSTRUCTION.format(problem=problem), max_tokens, max_retries)
