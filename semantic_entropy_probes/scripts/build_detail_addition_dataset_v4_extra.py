"""3 more K=1 prompts, completing coverage of every distinct source problem
available in rank1/rank2's genuine test split (train+val-refit-aware, see
scripts/build_detail_addition_dataset_v3_extra.py's docstring). Combined
with the 10 already scored in results_detail_test_v2/binary_comparison.csv
and the 2 in scripts/build_detail_addition_dataset_v3_extra.py, this uses
all 16 distinct source problems that have a K=1 clarifycodebench row in
rank1/rank2's held-out test split -- the maximum possible n for this check.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "detail_addition_probe_test_v4_extra.jsonl"

ITEMS = [
    dict(
        base_id="amb_task_28", source_problem_id="abc391_e",
        ambiguity_types=["Terminology"],
        prompt=(
            "For a binary string B = B_1 B_2 \\dots B_{3^n} of length 3^n (n \\geq 1), we define an operation to "
            "obtain a binary string C = C_1 C_2 \\dots C_{3^{n-1}} of length 3^{n-1} as follows:\n"
            "- Partition the elements of B into groups of 3 and take the value from each group. That is, for "
            "i=1,2,\\dots,3^{n-1}, let C_i be the value that appears most frequently among B_{3i-2}, B_{3i-1}, "
            "and B_{3i}.\n"
            "You are given a binary string A = A_1 A_2 \\dots A_{3^N} of length 3^N. Let A' = A'_1 be the length-1 "
            "string obtained by applying the above operation N times to A.\n"
            "Determine the minimum number of elements of A that must be changed (from 0 to 1 or from 1 to 0) in "
            "order to change the value of A'_1.\n"
            "Input\nThe input is given from Standard Input in the following format:\nN\nA_1 A_2 \\dots A_{3^N}\n"
            "Output\nPrint the answer.\n"
            "Constraints\n- N is an integer with 1 \\leq N \\leq 13.\n- A is a string of length 3^N consisting of 0 and 1."
        ),
        redundant_bullet="N is given as a single integer on its own line, followed by the string A on the next line, as shown in the input format above.",
        missing_bullet="The value taken from each group of 3 is the majority value among the three bits.",
    ),
    dict(
        base_id="amb_task_125", source_problem_id="abc394_b",
        ambiguity_types=["Comparison Rules"],
        prompt=(
            "You are given N strings S_1, S_2, \\ldots, S_N, each consisting of lowercase English letters. "
            "The lengths of these strings are all distinct.\n"
            "Sort these strings, and then concatenate them in that order to form a single string.\n"
            "Input\nThe input is given from Standard Input in the following format:\nN\nS_1\nS_2\n\\vdots\nS_N\n"
            "Output\nPrint the answer.\n"
            "Constraints\n- 2 \\leq N \\leq 50\n- N is an integer.\n"
            "- Each S_i is a string consisting of lowercase English letters with length between 1 and 50, inclusive.\n"
            "- If i \\neq j, the length of S_i is different from the length of S_j."
        ),
        redundant_bullet="N is given on the first line, followed by the N strings S_1,\\ldots,S_N, as shown in the input format above.",
        missing_bullet="The strings should be sorted in ascending order of their lengths.",
    ),
    dict(
        base_id="amb_task_346", source_problem_id="arc193_d",
        ambiguity_types=["Terminology"],
        prompt=(
            "You are given two length-N strings A = A_1A_2 \\ldots A_N and B = B_1B_2 \\ldots B_N.\n"
            "There are N squares aligned in a row from left to right. For i = 1, 2, \\ldots, N, the i-th square "
            "from the left is called square i. Initially, square i contains a piece if A_i = 1, and no piece if A_i = 0.\n"
            "You may repeat the following operation any number of times (possibly zero):\n"
            "- Choose an integer i between 1 and N, inclusive.\n"
            "- Move all pieces simultaneously one square closer to square i. That is, for each piece, let square j "
            "be its current position and square j' be its new position, and the following holds:\n"
            "- if i < j, then j' = j-1;\n- if i > j, then j' = j+1;\n- if i = j, then j' = j.\n"
            "Determine whether it is possible to reach a configuration satisfying the following condition, and if "
            "it is possible, find the minimum number of operations needed to do so:\n"
            "For every i = 1, 2, \\ldots, N, there is at least one piece in square i if and only if B_i = 1.\n"
            "You are given T independent test cases. Print the answer for each of them.\n"
            "Input\nThe input is given from Standard Input in the following format:\n"
            "T\n\\mathrm{case}_1\n\\mathrm{case}_2\n\\vdots\n\\mathrm{case}_T\n"
            "Here, \\mathrm{case}_i (i=1,2,\\ldots,T) denotes the i-th test case.\n"
            "Each test case is given in the following format:\nN\nA\nB\n"
            "Output\nPrint T lines.\n"
            "For each i = 1, 2, \\ldots, T, on the i-th line, print -1 if it is impossible to reach a configuration "
            "satisfying the condition for the i-th test case. Otherwise, print the minimum number of operations needed.\n"
            "Constraints\n- 1 \\leq T \\leq 2 \\times 10^5\n- 1 \\leq N \\leq 10^6\n- T and N are integers.\n"
            "- The sum of N over all test cases is at most 10^6."
        ),
        redundant_bullet="T is given on the first line, followed by T test cases each giving N, A, and B, as shown in the input format above.",
        missing_bullet="Both A and B consist only of the characters 0 and 1.",
    ),
]


def make_variant(prompt: str, bullet: str) -> str:
    if "Constraints" in prompt:
        return prompt + "\n- " + bullet
    return prompt + "\nConstraints\n- " + bullet


def main() -> None:
    rows = []
    for item in ITEMS:
        base = item["prompt"]
        common = dict(base_id=item["base_id"], source_problem_id=item["source_problem_id"],
                      ambiguity_types=item["ambiguity_types"])
        rows.append(dict(common, id=f"{item['base_id']}__baseline", variant="baseline", label=1, prompt=base))
        rows.append(dict(common, id=f"{item['base_id']}__redundant", variant="redundant", label=1,
                          prompt=make_variant(base, item["redundant_bullet"])))
        rows.append(dict(common, id=f"{item['base_id']}__missing", variant="missing", label=0,
                          prompt=make_variant(base, item["missing_bullet"])))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[build_v4_extra] wrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
