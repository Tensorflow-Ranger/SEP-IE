"""v7: expand v6's inline-edit test from 16 to the full 40 K=1 clarifycodebench
rows genuinely held out from rank1/rank2's saved pipelines (train+val-refit
aware -- see scripts/build_detail_addition_dataset_v3_extra.py's docstring).
16 were already done in scripts/build_detail_addition_dataset_v6_inline.py;
this file adds the remaining 24 (multiple rows per source problem where the
source has more than one K=1 row in the test fold -- all still genuinely
unseen by the saved pipeline).

Same methodology as v6: redundant/missing detail is rewritten INLINE into
the prompt's own prose (no appended bullet, no "Constraints" word
introduced, prompt's own ending untouched), each pair of edits anchored via
str.replace on the exact baseline text pulled from data/ambiguity_dataset.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "detail_addition_probe_test_v7_expand.jsonl"
DATASET_PATH = ROOT / "data" / "ambiguity_dataset.jsonl"

BASE_IDS = [
    "amb_task_5", "amb_task_6", "amb_task_29", "amb_task_30", "amb_task_50", "amb_task_51",
    "amb_task_88", "amb_task_89", "amb_task_119", "amb_task_120", "amb_task_126", "amb_task_127",
    "amb_task_143", "amb_task_144", "amb_task_199", "amb_task_200", "amb_task_213", "amb_task_214",
    "amb_task_280", "amb_task_347", "amb_task_348", "amb_task_361", "amb_task_362", "amb_task_418",
]

# base_id -> (redundant_old, redundant_new, missing_old, missing_new)
EDITS = {
    "amb_task_5": (
        "There are N bags, labeled bag 1, bag 2, \\ldots, bag N.\n",
        "There are N bags, labeled bag 1, bag 2, \\ldots, bag N, where N is a positive integer.\n",
        "Takahashi can perform the following operation any number of times:",
        "Takahashi can perform the following operation any number of times, including zero times:",
    ),
    "amb_task_6": (
        "There are N bags, labeled bag 1, bag 2, \\ldots, bag N.\n",
        "There are N bags, labeled bag 1, bag 2, \\ldots, bag N, where N is a positive integer.\n",
        "- B_1 \\oplus B_2 \\oplus \\cdots \\oplus B_N\n",
        "- B_1 \\oplus B_2 \\oplus \\cdots \\oplus B_N, where B_i is the final number of stones in bag i.\n",
    ),
    "amb_task_29": (
        "For a binary string B = B_1 B_2 \\dots B_{3^n} of length 3^n (n \\geq 1), we define an operation",
        "For a binary string B = B_1 B_2 \\dots B_{3^n} of length 3^n, where n is a positive integer, we define an operation",
        "Let A' = A'_1 be the length-1 string obtained by applying the above operation.",
        "Let A' = A'_1 be the length-1 string obtained by applying the above operation N times.",
    ),
    "amb_task_30": (
        "For a binary string B = B_1 B_2 \\dots B_{3^n} of length 3^n (n \\geq 1), we define an operation",
        "For a binary string B = B_1 B_2 \\dots B_{3^n} of length 3^n, where n is a positive integer, we define an operation",
        "Determine the minimum number of elements of A that must be changed in order to change the value of A'_1.",
        "Determine the minimum number of elements of A that must be changed, from 0 to 1 or from 1 to 0, whichever minimizes the count, in order to change the value of A'_1.",
    ),
    "amb_task_50": (
        "You are given a sequence of integers A = (A_1, A_2, A_3).\n",
        "You are given a sequence of three integers A = (A_1, A_2, A_3).\n",
        "Determine whether it is possible that B_1 \\times B_2 somehow equals B_3.",
        "Determine whether it is possible that B_1 \\times B_2 equals B_3 exactly, i.e., B_1 \\times B_2 = B_3 as integers.",
    ),
    "amb_task_51": (
        "You are given a sequence of integers A = (A_1, A_2, A_3).\n",
        "You are given a sequence of three integers A = (A_1, A_2, A_3).\n",
        "Print the result.",
        "Print Yes if such a permutation B exists, and No otherwise.",
    ),
    "amb_task_88": (
        "There are four types of oysters, labeled 1, 2, 3, and 4.",
        "There are exactly four types of oysters, labeled 1, 2, 3, and 4.",
        "Takahashi ate oysters, and Aoki ate oysters.",
        "Takahashi ate oysters 1 and 2, and Aoki ate oysters 1 and 3.",
    ),
    "amb_task_89": (
        "There are four types of oysters, labeled 1, 2, 3, and 4.",
        "There are exactly four types of oysters, labeled 1, 2, 3, and 4.",
        "Print the label of the oyster. \n",
        "Print only the numeric label (1-4) of the oyster type that causes stomach trouble. \n",
    ),
    "amb_task_119": (
        "You are given a string S consisting of uppercase English letters.\n",
        "You are given a string S consisting of uppercase English letters, read from standard input.\n",
        "- Among all occurrences of WA in the string, replace one with AC.\n",
        "- Among all occurrences of WA in the string, replace the leftmost one with AC.\n",
    ),
    "amb_task_120": (
        "You are given a string S consisting of uppercase English letters.\n",
        "You are given a string S consisting of uppercase English letters, read from standard input.\n",
        "Print the result.",
        "Print the final string S obtained after performing all replacements.",
    ),
    "amb_task_126": (
        "You are given N strings S_1, S_2, \\ldots, S_N, each consisting of lowercase English letters. \n",
        "You are given N strings S_1, S_2, \\ldots, S_N, provided one per line, each consisting of lowercase English letters. \n",
        "You are given N strings S_1, S_2, \\ldots, S_N, each consisting of lowercase English letters. \nSort these strings in ascending order of length,",
        "You are given N strings S_1, S_2, \\ldots, S_N, each consisting of lowercase English letters and each of a distinct length, so there is never a tie in length. \nSort these strings in ascending order of length,",
    ),
    "amb_task_127": (
        "You are given N strings S_1, S_2, \\ldots, S_N, each consisting of lowercase English letters. The lengths of these strings are all distinct.\n",
        "You are given N strings S_1, S_2, \\ldots, S_N, provided one per line, each consisting of lowercase English letters. The lengths of these strings are all distinct.\n",
        "Print the result.",
        "Print the concatenated string formed after sorting the strings by length.",
    ),
    "amb_task_143": (
        "There is a city divided into H \\times W blocks in the north-south-east-west directions, and there is exactly one building in each block.\n",
        "There is a city divided into H \\times W blocks, where H and W are positive integers, in the north-south-east-west directions, and there is exactly one building in each block.\n",
        "- Choose a building in a cardinally adjacent block, and move to the X-th floor of that building using a (sky) walkway.\n",
        "- Choose a building with at least X floors in a cardinally adjacent block, and move to the X-th floor of that building using a (sky) walkway.\n",
    ),
    "amb_task_144": (
        "There is a city divided into H \\times W blocks in the north-south-east-west directions, and there is exactly one building in each block.\n",
        "There is a city divided into H \\times W blocks, where H and W are positive integers, in the north-south-east-west directions, and there is exactly one building in each block.\n",
        "counts as 5 uses of stairs.)\nInput",
        "counts as 5 uses of stairs.)\nNote that the number of times he uses walkways does not need to be minimized.\nInput",
    ),
    "amb_task_199": (
        "There are N black balls and M white balls.\n",
        "There are N black balls and M white balls, each ball belonging to exactly one of the two colors.\n",
        "Among all such choices, find the sum of the values of the chosen balls.",
        "Among all such choices, find the maximum possible sum of the values of the chosen balls.",
    ),
    "amb_task_200": (
        "There are N black balls and M white balls.\n",
        "There are N black balls and M white balls, each ball belonging to exactly one of the two colors.\n",
        "Choose balls so that the number of black balls chosen is at least the number of white balls chosen.",
        "Choose zero or more balls so that the number of black balls chosen is at least the number of white balls chosen.",
    ),
    "amb_task_213": (
        "You are given integers N, M and three integer sequences",
        "You are given integers N, M, where N and M are positive integers, and three integer sequences",
        "- For every integer i with 1 \\le i \\le M, the XOR of A_{X_i} and A_{Y_i} is Z_i.\n",
        "- For every integer i with 1 \\le i \\le M, the bitwise exclusive OR (XOR) of A_{X_i} and A_{Y_i} is Z_i.\n",
    ),
    "amb_task_214": (
        "You are given integers N, M and three integer sequences",
        "You are given integers N, M, where N and M are positive integers, and three integer sequences",
        "If a good sequence exists, print one good sequence that minimizes the sum of its elements, separated by spaces.\nConstraints",
        "If a good sequence exists, print one good sequence that minimizes the sum of its elements, separated by spaces; if multiple such sequences exist, printing any one of them is accepted.\nConstraints",
    ),
    "amb_task_280": (
        "There is an infinitely large two-dimensional grid, with a campfire at coordinate (0,0).\n",
        "There is an infinitely large two-dimensional grid, with a single campfire located at coordinate (0,0).\n",
        "Print an N-character string.\n",
        "Print an N-character string consisting of 0 and 1, where the t-th character is 1 if smoke exists at cell (R,C) at time t+0.5, and 0 otherwise.\n",
    ),
    "amb_task_347": (
        "You are given two length-N strings A = A_1A_2 \\ldots A_N and B = B_1B_2 \\ldots B_N, each consisting of 0 and 1.\n",
        "You are given two length-N strings A = A_1A_2 \\ldots A_N and B = B_1B_2 \\ldots B_N, where N is a positive integer, each consisting of 0 and 1.\n",
        "- Move all pieces simultaneously one square closer to square i. \n",
        "- Move all pieces simultaneously one square closer to square i: for each piece at square j, it moves to square j-1 if i < j, to square j+1 if i > j, and stays at square j if i = j. \n",
    ),
    "amb_task_348": (
        "You are given two length-N strings A = A_1A_2 \\ldots A_N and B = B_1B_2 \\ldots B_N, each consisting of 0 and 1.\n",
        "You are given two length-N strings A = A_1A_2 \\ldots A_N and B = B_1B_2 \\ldots B_N, where N is a positive integer, each consisting of 0 and 1.\n",
        "find the number of operations needed to do so:",
        "find the minimum number of operations needed to do so:",
    ),
    "amb_task_361": (
        "You are given a positive integer N and a length-N string s_0s_1\\ldots s_{N-1} consisting only of 0 and 1.\n",
        "You are given a positive integer N and a length-N string s_0s_1\\ldots s_{N-1}, read from standard input, consisting only of 0 and 1.\n",
        "there is an undirected edge between vertices i and N if and only if s_i.\n",
        "there is an undirected edge between vertices i and N if and only if s_i = 1.\n",
    ),
    "amb_task_362": (
        "You are given a positive integer N and a length-N string s_0s_1\\ldots s_{N-1} consisting only of 0 and 1.\n",
        "You are given a positive integer N and a length-N string s_0s_1\\ldots s_{N-1}, read from standard input, consisting only of 0 and 1.\n",
        "Print the number, modulo , of distinct sequences (d_0, d_1, \\ldots, d_N) that can be obtained.",
        "Print the number, modulo 998244353, of distinct sequences (d_0, d_1, \\ldots, d_N) that can be obtained.",
    ),
    "amb_task_418": (
        "Each cell is either a road or a wall.\n",
        "Each cell is either a road or a wall, with no other possible cell states.\n",
        "Find the number of front kicks he needs in order to reach the fish shop.",
        "Find the minimum number of front kicks he needs in order to reach the fish shop; no path reconstruction is required, only the count.",
    ),
}


def load_baseline_rows() -> dict:
    rows = {}
    for line in DATASET_PATH.open():
        r = json.loads(line)
        if r["id"] in BASE_IDS:
            rows[r["id"]] = r
    return rows


def main() -> None:
    baselines = load_baseline_rows()
    assert set(baselines) == set(BASE_IDS), set(BASE_IDS) - set(baselines)

    rows = []
    for base_id in BASE_IDS:
        b = baselines[base_id]
        base_prompt = b["prompt"]
        red_old, red_new, mis_old, mis_new = EDITS[base_id]
        assert red_old in base_prompt, f"{base_id}: redundant anchor not found: {red_old!r}"
        assert mis_old in base_prompt, f"{base_id}: missing anchor not found: {mis_old!r}"
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
    print(f"[build_v7] wrote {len(rows)} rows ({len(BASE_IDS)} base prompts) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
