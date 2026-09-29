"""Build a small hand-authored probe test: for 10 K=1 ClarifyCodeBench
prompts, construct two edited variants each:

  - "redundant": one bullet added to Constraints that restates information
    already stated or directly implied elsewhere in the prompt. Does not
    resolve the annotated ambiguity -- a probe tracking real ambiguity
    should NOT move much (if at all) from baseline.
  - "missing": one bullet added to Constraints that is ClarifyCodeBench's
    own annotated clarification for that exact ambiguity point (reworded to
    fit as a natural constraint bullet rather than appended under a literal
    "Clarifications:" header -- avoiding the near-trivial header-token
    confound documented in FINDINGS.md's "Confound 1"). Should genuinely
    resolve the ambiguity -- a real ambiguity probe should move toward
    "less ambiguous."

Both edits are a single appended bullet, similar length, so a length-only
confound can't trivially explain a difference between the two conditions.

Base prompts and their ground-truth clarification text come directly from
data/ambiguity_dataset.jsonl (amb_task_* / full_task_* pairs sharing a
numeric suffix); the "missing" bullet text is lightly reworded from the
dataset's own "Clarifications:" bullet (see per-item `_source_clarification`
comment) to read as a natural spec constraint instead of a Q&A answer.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "detail_addition_probe_test.jsonl"

# Each item: (amb_task id, source_problem_id, ambiguity_types, base prompt,
# redundant bullet, missing-detail bullet [reworded clarification])
ITEMS = [
    dict(
        base_id="amb_task_1",
        source_problem_id="abc390_b",
        ambiguity_types=["Terminology"],
        prompt=(
            "You are given a length-N sequence A=(A_1,A_2,\\ldots,A_N).\n"
            "Determine whether A is a geometric progression.\n"
            "Input\n"
            "The input is given from Standard Input in the following format:\n"
            "N\nA_1 A_2 \\ldots A_N\n"
            "Output\n"
            "If A is a geometric progression, print Yes; otherwise, print No.\n"
            "Constraints\n"
            "- 2 \\leq N \\leq 100\n"
            "- 1 \\leq A_i \\leq 10^9"
        ),
        redundant_bullet="N is provided as a single integer, as shown in the input format above.",
        missing_bullet="All elements of the sequence are positive integers.",
    ),
    dict(
        base_id="amb_task_27",
        source_problem_id="abc391_a",
        ambiguity_types=["Terminology"],
        prompt=(
            "You are given a string D representing one of the eight directions. \n"
            "Print the string representing the direction opposite to the direction denoted by D.\n"
            "Input\n"
            "The input is given from Standard Input in the following format:\n"
            "D\n"
            "Output\n"
            "Print the answer."
        ),
        redundant_bullet="D is given on a single line of standard input, as shown in the input format above.",
        missing_bullet="D is one of N, E, W, S, NE, NW, SE, or SW.",
    ),
    dict(
        base_id="amb_task_51",
        source_problem_id="abc392_a",
        ambiguity_types=["Output Format"],
        prompt=(
            "You are given a sequence of integers A = (A_1, A_2, A_3).\n"
            "Let B = (B_1, B_2, B_3) be any permutation of A.\n"
            "Determine whether it is possible that B_1 \\times B_2 = B_3.\n"
            "Input\n"
            "The input is given from Standard Input in the following format:\n"
            "A_1 A_2 A_3\n"
            "Output\n"
            "Print the result.\n"
            "Constraints\n"
            "- All input values are integers.\n"
            "- 1 \\le A_1, A_2, A_3 \\le 100"
        ),
        redundant_bullet="A_1, A_2, A_3 are given on a single line, as shown in the input format above.",
        missing_bullet="Print \u201cYes\u201d if such a permutation B exists, and \u201cNo\u201d otherwise.",
    ),
    dict(
        base_id="amb_task_56",
        source_problem_id="abc392_g",
        ambiguity_types=["Terminology"],
        prompt=(
            "For integers A, B, C, a fine triplet.\n"
            "You are given a set of N distinct positive integers S = \\{ S_1, S_2, \\dots, S_N \\}. "
            "Find the number of fine triplets (A, B, C) with A, B, C \\in S.\n"
            "Input\n"
            "The input is given from Standard Input in the following format:\n"
            "N\nS_1 S_2 \\dots S_N\n"
            "Output\n"
            "Print the number of fine triplets as an integer.\n"
            "Constraints\n"
            "- All input values are integers.\n"
            "- 1 \\le N \\le 10^6\n"
            "- 1 \\le S_i \\le 10^6\n"
            "- The elements of S are distinct."
        ),
        redundant_bullet="N and the elements of S are given via standard input as shown in the input format above.",
        missing_bullet="A triplet (A, B, C) is a fine triplet if A < B < C and B \u2212 A = C \u2212 B.",
    ),
    dict(
        base_id="amb_task_29",
        source_problem_id="abc391_e",
        ambiguity_types=["Behavior"],
        prompt=(
            "For a binary string B = B_1 B_2 \\dots B_{3^n} of length 3^n (n \\geq 1), "
            "we define an operation to obtain a binary string C = C_1 C_2 \\dots C_{3^{n-1}} "
            "of length 3^{n-1} as follows:\n"
            "- Partition the elements of B into groups of 3 and take the majority value from each group. "
            "That is, for i=1,2,\\dots,3^{n-1}, let C_i be the value that appears most frequently among "
            "B_{3i-2}, B_{3i-1}, and B_{3i}.\n"
            "You are given a binary string A = A_1 A_2 \\dots A_{3^N} of length 3^N. "
            "Let A' = A'_1 be the length-1 string obtained by applying the above operation.\n"
            "Determine the minimum number of elements of A that must be changed (from 0 to 1 or from 1 to 0) "
            "in order to change the value of A'_1.\n"
            "Input\n"
            "The input is given from Standard Input in the following format:\n"
            "N\nA_1 A_2 \\dots A_{3^N}\n"
            "Output\n"
            "Print the answer.\n"
            "Constraints\n"
            "- N is an integer with 1 \\leq N \\leq 13.\n"
            "- A is a string of length 3^N consisting of 0 and 1."
        ),
        redundant_bullet="N is given as a single integer on its own line, as shown in the input format above.",
        missing_bullet="The reduction operation above is applied exactly N times, until a single character remains.",
    ),
    dict(
        base_id="amb_task_35",
        source_problem_id="abc391_g",
        ambiguity_types=["Units"],
        prompt=(
            "You are given a lowercase English string S of length N and an integer M. "
            "For each k=0,1,\\ldots,N, solve the following problem:\n"
            "- There are 26^M lowercase English strings of length M. Among these, find the number of "
            "strings whose longest common subsequence with S has length exactly k.\n"
            "Input\n"
            "The input is given from Standard Input in the following format:\n"
            "N M\nS\n"
            "Output\n"
            "Let \\mathrm{ans}_i be the answer for k=i. Print the answers in the following format:\n"
            "\\mathrm{ans}_0 \\mathrm{ans}_1 \\ldots \\mathrm{ans}_N\n"
            "Constraints\n"
            "- 1\\leq N\\leq 10\n"
            "- 1\\leq M\\leq 100\n"
            "- N and M are integers.\n"
            "- S is a lowercase English string of length N."
        ),
        redundant_bullet="N, M and S are given via standard input in the format shown above.",
        missing_bullet="All answers ans_i must be computed modulo 998244353.",
    ),
    dict(
        base_id="amb_task_63",
        source_problem_id="abc392_b",
        ambiguity_types=["Collection Semantics"],
        prompt=(
            "You are given a sequence of M integers A = (A_1, A_2, \\dots, A_M).\n"
            "Each element of A is an integer between 1 and N, inclusive.\n"
            "List all integers between 1 and N that do not appear in A in ascending order.\n"
            "Input\n"
            "The input is given from Standard Input in the following format:\n"
            "N M\nA_1 A_2 \\dots A_M\n"
            "Output\n"
            "Let (X_1, X_2, \\dots, X_C) be the sequence of all integers between 1 and N, inclusive, "
            "that do not appear in A, listed in ascending order.\n"
            "The output should be in the following format:\n"
            "C\nX_1 X_2 \\dots X_C\n"
            "Constraints\n"
            "- All input values are integers.\n"
            "- 1 \\le M \\le N \\le 1000\n"
            "- 1 \\le A_i \\le N"
        ),
        redundant_bullet="The value of M does not exceed N, as already stated above.",
        missing_bullet="All elements of A are distinct; no value appears more than once in A.",
    ),
    dict(
        base_id="amb_task_71",
        source_problem_id="abc392_d",
        ambiguity_types=["Behavior"],
        prompt=(
            "There are N dice.\n"
            "The i-th die has K_i faces, with the numbers A_{i,1}, A_{i,2}, \\ldots, A_{i,K_i} written on them.\n"
            "When you roll this die, each face appears with probability \\frac{1}{K_i}.\n"
            "You choose two dice from the N dice and roll them.\n"
            "Determine the probability that the two dice show the same number, when the dice are chosen optimally.\n"
            "Input\n"
            "The input is given from Standard Input in the following format:\n"
            "N\nK_1 A_{1,1} A_{1,2} \\dots A_{1,K_1}\n\\vdots\nK_N A_{N,1} A_{N,2} \\dots A_{N,K_N}\n"
            "Output\n"
            "Print the answer.\n"
            "Your answer is considered correct if the absolute or relative error from the true solution "
            "does not exceed 10^{-8}.\n"
            "Constraints\n"
            "- 2 \\leq N \\leq 100\n"
            "- 1 \\leq K_i\n"
            "- K_1 + K_2 + \\dots + K_N \\leq 10^5\n"
            "- 1 \\leq A_{i,j} \\leq 10^5\n"
            "- All input values are integers."
        ),
        redundant_bullet="Each die's faces are given on the same line as K_i, as shown in the input format above.",
        missing_bullet="Among all pairs of dice, output the maximum such probability.",
    ),
    dict(
        base_id="amb_task_78",
        source_problem_id="abc392_f",
        ambiguity_types=["Output Format"],
        prompt=(
            "There is an empty array A. For i = 1,2,\\ldots,N, perform the following operation in order:\n"
            "- Insert the number i into A so that it becomes the P_i-th element from the beginning.\n"
            "- More precisely, replace A with the concatenation of the first P_i-1 elements of A, then i, "
            "then the remaining elements of A starting from the P_i-th element, in this order.\n"
            "Output the final array A after all operations have been completed.\n"
            "Input\n"
            "The input is given from Standard Input in the following format:\n"
            "N\nP_1 P_2 \\ldots P_N\n"
            "Output\n"
            "Print A_1, A_2, \\ldots, A_N in this order, separated by spaces.\n"
            "Constraints\n"
            "- 1 \\leq N \\leq 5\\times 10^5\n"
            "- 1 \\leq P_i \\leq i\n"
            "- All input values are integers."
        ),
        redundant_bullet="N and the sequence P_1,\\ldots,P_N are read from standard input as shown above.",
        missing_bullet="Print the elements of A separated by single spaces, with no commas or brackets.",
    ),
    dict(
        base_id="amb_task_84",
        source_problem_id="abc392_c",
        ambiguity_types=["Terminology"],
        prompt=(
            "There are N people numbered from 1 to N.\n"
            "For each i = 1,2,\\ldots,N, find the number written on the bib of the person that the person "
            "wearing the bib with number i is staring at.\n"
            "Input\n"
            "The input is given from Standard Input in the following format:\n"
            "N\nP_1 P_2 \\dots P_N\nQ_1 Q_2 \\dots Q_N\n"
            "Output\n"
            "Let S_i be the number written on the bib of the person that the person wearing the bib with "
            "number i is staring at.\n"
            "Print S_1, S_2, \\ldots, S_N in this order, separated by a single space.\n"
            "Constraints\n"
            "- 2 \\leq N \\leq 3\\times 10^5\n"
            "- 1 \\leq P_i \\leq N\n"
            "- The values of P_i are distinct.\n"
            "- 1 \\leq Q_i \\leq N\n"
            "- The values of Q_i are distinct.\n"
            "- All input values are integers."
        ),
        redundant_bullet="P and Q are each given as N space-separated integers, as shown in the input format above.",
        missing_bullet="\u201cPerson i\u201d refers to the person whose index is i, not the person wearing bib number i.",
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
        rows.append(dict(
            id=f"{item['base_id']}__baseline",
            base_id=item["base_id"],
            source_problem_id=item["source_problem_id"],
            variant="baseline",
            ambiguity_types=item["ambiguity_types"],
            label=1,  # ClarifyCodeBench's own annotated K for this row
            prompt=base,
        ))
        rows.append(dict(
            id=f"{item['base_id']}__redundant",
            base_id=item["base_id"],
            source_problem_id=item["source_problem_id"],
            variant="redundant",
            ambiguity_types=item["ambiguity_types"],
            label=1,  # hypothesis: unchanged, no real ambiguity resolved
            prompt=make_variant(base, item["redundant_bullet"]),
        ))
        rows.append(dict(
            id=f"{item['base_id']}__missing",
            base_id=item["base_id"],
            source_problem_id=item["source_problem_id"],
            variant="missing",
            ambiguity_types=item["ambiguity_types"],
            label=0,  # hypothesis: resolved, K drops to 0
            prompt=make_variant(base, item["missing_bullet"]),
        ))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[build_detail_addition_dataset] wrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
