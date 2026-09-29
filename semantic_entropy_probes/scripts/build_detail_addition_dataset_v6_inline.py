"""v6: same 16 double-held-out K=1 prompts as
scripts/score_rank1_rank2_full_testset.py, but instead of appending a
bullet (v2-v5), the redundant/missing detail is rewritten INLINE into the
prompt's own prose, in the same style as the surrounding sentence -- much
closer to what ClarifyCodeBench's real K=0 (never-edited original) text
looks like, and to what the deletion-edit process actually does/undoes.

Two things this controls for, on top of v5's header removal:
  1. Training-distribution shape: rank1/rank2's "clear" class is *never*
     edited-prose-plus-appended-sentence -- it's single-pass author prose.
     An inline rewrite is a much closer analogue to that shape than any
     appended bullet, headed or not.
  2. TBG extraction point: src/extract_activations.py takes the hidden
     state at the prompt's LAST token. Every v2-v5 variant's last token was
     from a sentence *I* freshly authored (the appended bullet), which
     could dominate the representation regardless of content. Here the
     prompt's ending is untouched (identical trailing text to baseline) --
     the edit sits mid-prompt, so the final-token context is the same as
     baseline's own natural ending in all three variants.

Base prompt text and exact clarification content are unchanged from
v2-v4 (pulled from those files, not re-authored) -- only *where* and *how*
the redundant/missing detail is inserted changes.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "detail_addition_probe_test_v6_inline.jsonl"

SOURCE_FILES = [
    ROOT / "data" / "detail_addition_probe_test_v2.jsonl",
    ROOT / "data" / "detail_addition_probe_test_v3_extra.jsonl",
    ROOT / "data" / "detail_addition_probe_test_v4_extra.jsonl",
]

# Each entry: base_id -> (redundant_old, redundant_new, missing_old, missing_new)
# All four substrings are applied via str.replace(old, new, 1) on the
# *baseline* prompt (which already has no "Constraints" special-casing --
# we edit the real ClarifyCodeBench text directly).
EDITS = {
    "amb_task_4": (
        "There are N bags, labeled bag 1, bag 2, \\ldots, bag N.\n",
        "There are N bags, labeled bag 1, bag 2, \\ldots, bag N, where N is a positive integer.\n",
        "Choose two bags A and B, and move stones from bag A into bag B.",
        "Choose two bags A and B, and move all of the stones from bag A into bag B, leaving bag A empty.",
    ),
    "amb_task_26": (
        "and an integer K.\n",
        "and an integer K, where N and K are both positive integers.\n",
        "Among all these values, find the K-th value.",
        "Among all these values, find the K-th largest value.",
    ),
    "amb_task_49": (
        "You are given a sequence of integers A = (A_1, A_2, A_3).\n",
        "You are given a sequence of three integers A = (A_1, A_2, A_3).\n",
        "Let B = (B_1, B_2, B_3) be any permutation related of A.",
        "Let B = (B_1, B_2, B_3) be any permutation of A, i.e., a rearrangement containing exactly the same three elements.",
    ),
    "amb_task_87": (
        "Takahashi ate oysters 1 and 2, and Aoki ate oysters 1 and 3.",
        "Takahashi ate two of the oysters, oysters 1 and 2, and Aoki ate two of the oysters, oysters 1 and 3.",
        "One or some types cause stomach trouble if eaten.",
        "Exactly one of the four types causes stomach trouble if eaten.",
    ),
    "amb_task_118": (
        "You are given a string S consisting of uppercase English letters.\n",
        "You are given a string S consisting of uppercase English letters, read from standard input.\n",
        "As long as the string contains WA as a substring, repeat the following operation:",
        "As long as the string contains the two adjacent characters W immediately followed by A, repeat the following operation:",
    ),
    "amb_task_198": (
        "There are N black balls and M white balls.\n",
        "There are N black balls and M white balls, each ball belonging to exactly one of the two colors.\n",
        "Choose zero or more balls so that the number of black balls chosen is the number of white balls chosen.",
        "Choose zero or more balls so that the number of black balls chosen is at least the number of white balls chosen.",
    ),
    "amb_task_212": (
        "We call a length-N sequence of non-negative integers A = (A_1, A_2, \\ldots, A_N) a good sequence",
        "We call a length-N sequence of non-negative integers A = (A_1, A_2, \\ldots, A_N), where N is a positive integer, a good sequence",
        "For every integer i with 1 \\le i \\le M, the XOR of A_{X_i} and A_{Y_i} is Z_i.",
        "For every integer i with 1 \\le i \\le M, where 1 \\le X_i, Y_i \\le N, the XOR of A_{X_i} and A_{Y_i} is Z_i.",
    ),
    "amb_task_279": (
        "Takahashi is standing at cell (R,C).\n",
        "Takahashi is standing at a single, fixed cell (R,C).\n",
        "There is an infinitely large two-dimensional grid, with a campfire at coordinate (0,0).",
        "There is an infinitely large two-dimensional grid, with a campfire at coordinate (0,0); at time t=0, smoke exists only at cell (0,0) and nowhere else.",
    ),
    "amb_task_321": (
        "You are given two positive integers N and M.\n",
        "You are given two positive integers N and M, given on a single line.\n",
        "Let X = \\displaystyle\\sum_{i = 0}^{M} N^i. If X, print the value of X. If X, print inf.",
        "Let X = \\displaystyle\\sum_{i = 0}^{M} N^i. If X > 10^9, print inf. Otherwise, print the value of X.",
    ),
    "amb_task_417": (
        "Each cell is either a road or a wall.\n",
        "Each cell is either a road or a wall, with no other possible cell states.\n",
        "When he performs a front kick, for each of the cells at most 2 steps away in that direction from the cell he is currently in, if that cell is a wall, it becomes a road.",
        "When he performs a front kick, for each of the cells at most 2 steps away in that direction from the cell he is currently in, even if such a cell would lie outside the grid, that cell becomes a road if it is a wall and lies inside the grid; cells outside the grid are unaffected.",
    ),
    "amb_task_142": (
        "There is a city divided into H \\times W blocks, and there is exactly one building in each block.\n",
        "There is a city divided into H \\times W blocks, where H and W are positive integers, and there is exactly one building in each block.\n",
        "H W\nF_{1,1} F_{1,2} \\ldots F_{1,W}\nF_{2,1} F_{2,2} \\ldots F_{2,W}\n\\vdots\nF_{H,1} F_{H,2} \\ldots F_{H,W}\n",
        "H W\nF_{1,1} F_{1,2} \\ldots F_{1,W}\nF_{2,1} F_{2,2} \\ldots F_{2,W}\n\\vdots\nF_{H,1} F_{H,2} \\ldots F_{H,W}\n(the i-th of these H lines gives the floor counts for the i-th block-row from the north, so a larger row index is farther south)\n",
    ),
    "amb_task_282": (
        "There are N people, labeled 1 to N. Person i has an integer A_i.\n",
        "There are N people, each with a distinct label from 1 to N. Person i has an integer A_i.\n",
        "If no person satisfies the condition, report that fact instead.",
        "If no person satisfies the condition, print -1 instead.",
    ),
    "amb_task_360": (
        "Consider a simple undirected graph G with (N+1) vertices numbered 0, 1, 2, \\ldots, N,",
        "Consider a simple undirected graph G, where N is a positive integer, with (N+1) vertices numbered 0, 1, 2, \\ldots, N,",
        "You are given a positive integer N and a length-N string s_0s_1\\ldots s_{N-1}.",
        "You are given a positive integer N and a length-N binary string s_0s_1\\ldots s_{N-1} consisting only of the characters 0 and 1.",
    ),
    "amb_task_28": (
        "For a binary string B = B_1 B_2 \\dots B_{3^n} of length 3^n (n \\geq 1), we define an operation",
        "For a binary string B = B_1 B_2 \\dots B_{3^n} of length 3^n, where n is a positive integer, we define an operation",
        "Partition the elements of B into groups of 3 and take the value from each group.",
        "Partition the elements of B into groups of 3 and take the majority value from each group.",
    ),
    "amb_task_125": (
        "You are given N strings S_1, S_2, \\ldots, S_N, each consisting of lowercase English letters.",
        "You are given N strings S_1, S_2, \\ldots, S_N, provided one per line, each consisting of lowercase English letters.",
        "Sort these strings, and then concatenate them in that order to form a single string.",
        "Sort these strings in ascending order of their length, and then concatenate them in that order to form a single string.",
    ),
    "amb_task_346": (
        "There are N squares aligned in a row from left to right.",
        "There are N squares, where N is a positive integer, aligned in a row from left to right.",
        "You are given two length-N strings A = A_1A_2 \\ldots A_N and B = B_1B_2 \\ldots B_N.",
        "You are given two length-N binary strings A = A_1A_2 \\ldots A_N and B = B_1B_2 \\ldots B_N, each consisting only of the characters 0 and 1.",
    ),
}


def load_baselines() -> dict:
    rows = {}
    for f in SOURCE_FILES:
        for line in f.open():
            r = json.loads(line)
            if r["variant"] == "baseline":
                rows[r["base_id"]] = r
    return rows


def main() -> None:
    baselines = load_baselines()
    rows = []
    for base_id, (red_old, red_new, mis_old, mis_new) in EDITS.items():
        b = baselines[base_id]
        base_prompt = b["prompt"]
        assert red_old in base_prompt, f"{base_id}: redundant anchor not found"
        assert mis_old in base_prompt, f"{base_id}: missing anchor not found"
        redundant_prompt = base_prompt.replace(red_old, red_new, 1)
        missing_prompt = base_prompt.replace(mis_old, mis_new, 1)
        assert redundant_prompt != base_prompt
        assert missing_prompt != base_prompt

        common = dict(base_id=base_id, source_problem_id=b["source_problem_id"], ambiguity_types=b["ambiguity_types"])
        rows.append(dict(common, id=f"{base_id}__baseline", variant="baseline", label=1, prompt=base_prompt))
        rows.append(dict(common, id=f"{base_id}__redundant", variant="redundant", label=1, prompt=redundant_prompt))
        rows.append(dict(common, id=f"{base_id}__missing", variant="missing", label=0, prompt=missing_prompt))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[build_v6] wrote {len(rows)} rows ({len(EDITS)} base prompts) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
