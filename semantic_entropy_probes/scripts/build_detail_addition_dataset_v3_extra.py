"""Two more K=1 prompts (amb_task_282, amb_task_360) that are genuinely
unseen by BOTH the multiclass probe's test split AND the rank1/rank2
probes' test split (the saved rank1/rank2 pipelines are refit on
train+val -- see scripts/save_best_models.py -- so only true test-split
rows are unseen by them). Together with amb_task_142 and amb_task_212
(already extracted/scored elsewhere -- see scripts/score_detail_addition_v2.py
and the ad hoc rank1/rank2-only scoring of amb_task_142), this gives 4
distinct source problems held out for every probe in this project.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "detail_addition_probe_test_v3_extra.jsonl"

ITEMS = [
    dict(
        base_id="amb_task_282", source_problem_id="abc398_c",
        ambiguity_types=["Output Format"],
        prompt=(
            "There are N people, labeled 1 to N. Person i has an integer A_i.\n"
            "Among the people who satisfy the condition \"None of the other N-1 people has the same integer as "
            "themselves,\" find the one with the greatest integer, and print that person's label.\n"
            "If no person satisfies the condition, report that fact instead.\n"
            "Input\nThe input is given from Standard Input in the following format:\nN\nA_1 A_2 \\ldots A_N\n"
            "Output\nAmong those who satisfy it, print the label of the person whose integer is the largest.\n"
            "Constraints\n- 1 \\leq N \\leq 3\\times 10^5\n- 1 \\leq A_i \\leq 10^9\n- All input values are integers."
        ),
        redundant_bullet="N is given on the first line, followed by A_1,\\ldots,A_N on the second line, as shown in the input format above.",
        missing_bullet="If no person satisfies the condition, print -1.",
    ),
    dict(
        base_id="amb_task_360", source_problem_id="arc193_b",
        ambiguity_types=["Terminology"],
        prompt=(
            "You are given a positive integer N and a length-N string s_0s_1\\ldots s_{N-1}.\n"
            "Consider a simple undirected graph G with (N+1) vertices numbered 0, 1, 2, \\ldots, N, and the "
            "following edges:\n"
            "- For each i = 0, 1, \\ldots, N-1, there is an undirected edge between vertices i and (i+1)\\bmod N.\n"
            "- For each i = 0, 1, \\ldots, N-1, there is an undirected edge between vertices i and N if and only if "
            "s_i = 1.\n"
            "- There are no other edges.\n"
            "Furthermore, create a directed graph G' by assigning a direction to each edge of G. That is, for each "
            "undirected edge \\lbrace u, v \\rbrace in G, replace it with either a directed edge (u, v) from u to v "
            "or a directed edge (v, u) from v to u.\n"
            "For each i = 0, 1, \\ldots, N, let d_i be the in-degree of vertex i in G'. Print the number, modulo "
            "998244353, of distinct sequences (d_0, d_1, \\ldots, d_N) that can be obtained.\n"
            "Input\nThe input is given from Standard Input in the following format:\nN\ns_0s_1\\ldots s_{N-1}\n"
            "Output\nPrint the answer.\n"
            "Constraints\n- 3 \\leq N \\leq 10^6\n- N is an integer."
        ),
        redundant_bullet="N is given on the first line, followed by the string s_0s_1\\ldots s_{N-1} on the second line, as shown in the input format above.",
        missing_bullet="The string s_0 s_1 \\ldots s_{N-1} consists only of the characters 0 and 1.",
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
    print(f"[build_v3_extra] wrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
