import json

DATA = {
"ccb_amb_task_101": [
    (1, "Only the GCD value is required for each i; no additional subset information is needed."),
    (0, "Yes, when K=1 the only choice is A_i itself, so the answer is A_i."),
    (1, "Positions are treated as distinct even if values repeat, so equal values at different indices are separate elements."),
    (0, "The order of the chosen elements does not matter since GCD is order-independent."),
    (0, "Yes, when N=K the only possible choice is the entire array, so the answer for each i is the GCD of the whole array."),
    (0, "Not specified beyond what's given; assume standard integer output, no special size restriction is implied by the constraints."),
    (0, "Each answer should be printed on its own line, as already stated in the Output section."),
    (0, "Not specified beyond what's given; assume standard whitespace-separated input format."),
],
"ccb_amb_task_102": [
    (2, "For each i, the task is to find the maximum possible GCD among all selections of K elements that include A_i."),
    (0, "Yes, when K=1 the only choice is A_i itself, so the answer is A_i."),
    (0, "The order of the chosen K elements does not matter since GCD is order-independent; they need not be contiguous."),
    (1, "Positions are treated as distinct even if values repeat, so equal values at different indices count as separate elements."),
    (0, "Yes, when N=K the only choice is the whole array, so the answer is its GCD."),
    (0, "Not specified beyond what's given; assume standard algorithmic time/memory limits for this constraint size."),
    (0, "Each answer should be printed on its own line, as already stated in the Output section."),
    (0, "This case cannot occur since all A_i >= 1, so the GCD of a non-empty selection is always positive."),
],
"ccb_amb_task_103": [
    (2, "Each of the N answers should be printed on a separate line, i.e. the j-th line corresponds to the answer for i=j."),
    (1, "Only the maximum GCD value is required, not which subset achieves it."),
    (0, "Yes, when K=1 the only choice is A_i itself, so the answer is A_i."),
    (1, "Positions are treated as distinct even if values repeat, so equal values at different indices count as separate elements."),
    (0, "The order of the chosen K elements does not matter since GCD is order-independent."),
    (0, "Not specified beyond what's given; the GCD is computed only over the selected K values."),
    (0, "Not specified beyond what's given; assume standard integer output, no modulo is implied."),
    (0, "Not specified beyond what's given; assume standard algorithmic time/memory limits for this constraint size."),
],
"ccb_amb_task_104": [
    (2, "For each i, the task is to find the maximum possible GCD among all selections of K elements that include A_i."),
    (0, "Yes, when K=1 the only choice is A_i itself, so the answer is A_i."),
    (1, "Positions are treated as distinct even if values repeat, so equal values at different indices count as separate elements."),
    (0, "The order of the chosen K elements does not matter since GCD is order-independent."),
    (0, "Yes, when N=K the only choice is the whole array, so the answer is its GCD."),
    (0, "Not specified beyond what's given; assume standard algorithmic time/memory limits for this constraint size."),
    (0, "Each answer should be printed on its own line, as already stated in the Output section."),
    (0, "Not specified beyond what's given; assume standard integer output with no modulo requirement."),
],
"ccb_amb_task_105": [
    (2, "Each of the N answers should be printed on a separate line, i.e. the j-th line corresponds to the answer for i=j."),
    (1, "Only the maximum GCD value is required, not which subset achieves it."),
    (0, "Yes, when K=1 the only choice is A_i itself, so the answer is A_i."),
    (1, "Positions are treated as distinct even if values repeat, so equal values at different indices count as separate elements."),
    (0, "The order of the chosen K elements does not matter since GCD is order-independent."),
    (0, "Yes, when N=K the only choice is the whole array, so the answer is its GCD."),
    (0, "Not specified beyond what's given; assume standard integer output with no modulo requirement."),
    (0, "Duplicate values do not fundamentally change the GCD computation or subset counting rules."),
],
"ccb_amb_task_106": [
    (2, "For each i, output the maximum possible GCD among all K-element subsets containing A_i."),
    (1, "Only the GCD value is required, not which subset achieves it."),
    (0, "Yes, when K=1 the only choice is A_i itself, so the answer is A_i."),
    (1, "Positions are treated as distinct even if values repeat, so equal values at different indices count as separate elements."),
    (2, "Each of the N answers should be printed on its own line, in increasing order of i."),
    (0, "Yes, answers should be printed in increasing order of i, matching the j-th line to i=j."),
    (0, "Not specified beyond what's given; assume standard algorithmic time/memory limits for this constraint size."),
    (0, "Not specified beyond what's given; assume standard integer types are sufficient."),
],
"ccb_amb_task_107": [
    (2, "For each i, output the maximum possible GCD among all K-element subsets containing A_i."),
    (1, "Only the GCD value is required, not which subset achieves it."),
    (1, "Positions are treated as distinct even if values repeat, so equal values at different indices count as separate elements."),
    (2, "Each of the N answers should be printed on its own line, not combined on a single line."),
    (0, "Yes, when K=1 the only choice is A_i itself, so the answer is A_i."),
    (0, "Not specified beyond what's given; assume standard algorithmic time/memory limits for this constraint size."),
    (0, "Yes, when N=K the only choice is the whole array, so the answer for every i is its GCD."),
    (0, "Not specified beyond what's given; assume standard integer types are sufficient."),
],
"ccb_amb_task_108": [
    (1, "The string is naturally 1-indexed as described, but implementation indexing is an implementation detail not fixed by the statement."),
    (0, "Not specified beyond what's given; each operation is one adjacent swap, and repeated swaps at the same or overlapping positions are allowed."),
    (0, "Each operation swaps only immediately adjacent characters; reaching non-adjacent effects requires multiple operations, as is standard."),
    (0, "Yes, the minimum number of operations is the total count of individual adjacent swaps performed."),
    (0, "Not specified beyond what's given; assume standard integer output, no modulo needed."),
    (0, "Not specified beyond what's given; assume standard algorithmic time/memory limits for this constraint size."),
    (0, "Yes, if the string already has all 1s contiguous, the expected output is 0."),
    (0, "Not specified beyond what's given; assume standard input format with no extra whitespace."),
],
"ccb_amb_task_109": [
    (0, "Yes, if the string already has all 1s contiguous, the expected output is 0."),
    (0, "Not specified beyond what's given; assume the answer fits in a standard 64-bit integer type."),
    (0, "Not specified beyond what's given; swapping two equal characters is a valid operation but never useful, so it wouldn't appear in an optimal solution."),
    (0, "Not specified beyond what's given; assume a plain integer followed by a newline is sufficient."),
    (1, "This case cannot occur since it is guaranteed that S contains at least one 1."),
    (1, "If the string consists entirely of 1s, they are already contiguous, so the answer is 0."),
    (0, "Not specified beyond what's given; assume standard algorithmic time/memory limits for this constraint size."),
    (0, "Not specified beyond what's given; assume a single test case per input file."),
],
"ccb_amb_task_110": [
    (0, "The number of adjacent swaps performed is counted individually; operations cannot be combined into a single step."),
    (0, "Not specified beyond what's given; assume 1-based indexing as described, with no leading spaces or extra characters."),
    (0, "Not specified beyond what's given; assume the answer fits in a standard 64-bit integer type."),
    (1, "You should print a single integer -- the minimum number of operations -- with no additional formatting."),
    (0, "Yes, if the string already has all 1s contiguous, the expected answer is 0, and this is a valid input."),
    (0, "Not specified beyond what's given; the string can in principle consist entirely of 1s."),
    (0, "Not specified beyond what's given; only the minimal number of swaps matters, not which final segment is chosen."),
    (0, "Not specified beyond what's given; assume an efficient O(N log N) or better solution is expected given the constraint size."),
],
"ccb_amb_task_111": [
    (0, "Yes, if the string already has all 1s contiguous, the expected output is 0."),
    (2, "The valid range of i is 1 <= i <= N-1; you can only swap adjacent characters within the string and cannot exceed its boundary."),
    (0, "Not specified beyond what's given; assume a plain integer on a single line with standard formatting."),
    (0, "Not specified beyond what's given; assume the answer fits in a standard 64-bit integer type."),
    (2, "It is guaranteed that S contains at least one 1, so this input cannot occur; if it had zero or one 1, the answer would trivially be 0."),
    (0, "Not specified beyond what's given; only the minimal number of swaps matters, not which final interval is chosen."),
    (0, "Not specified beyond what's given; assume no extra whitespace or blank lines in the input."),
    (0, "Yes, the input is guaranteed to contain exactly N characters in S."),
],
"ccb_amb_task_112": [
    (1, "The problem describes i using 1-based indexing; implementation may use any internal indexing as long as behavior matches."),
    (2, "Swapping is only allowed between adjacent positions i and i+1 with 1 <= i <= N-1, staying within the string's bounds."),
    (0, "The output must be the exact minimum number of swaps; no rounding or approximation is allowed."),
    (0, "Not specified beyond what's given; only the minimal count of swaps matters, not which final configuration is reached."),
    (0, "Not specified beyond what's given; assume the answer fits in a standard 64-bit integer type."),
    (0, "Not specified beyond what's given; assume no extra whitespace or blank lines in the input."),
    (0, "Yes, if the string already has all 1s contiguous, the expected answer is 0."),
    (0, "Not specified beyond what's given; assume a single test case per input file."),
],
"ccb_amb_task_113": [
    (0, "Yes, if the string already has all 1s contiguous, zero operations are needed."),
    (1, "You should print a single integer -- the minimum number of operations -- with no additional formatting required."),
    (0, "Not specified beyond what's given; assume the answer fits in a standard 64-bit integer type."),
    (2, "This case cannot occur since it is guaranteed that S contains at least one 1."),
    (1, "If the string consists entirely of 1s, they are already contiguous, so the answer is 0, and this is valid input."),
    (0, "Each adjacent swap is counted individually as one operation, regardless of whether multiple could be performed in sequence."),
    (0, "Not specified beyond what's given; assume standard algorithmic time/memory limits for this constraint size."),
    (0, "Yes, the input is guaranteed to contain exactly N characters on the second line."),
],
}

with open("data/v12_ranking/judged_chunk_1.jsonl", "a", encoding="utf-8") as out:
    cand = {json.loads(l)["id"]: json.loads(l) for l in open("data/v12_ranking/candidates.jsonl")}
    for rid, tiers_answers in DATA.items():
        questions = cand[rid]["questions"]
        assert len(questions) == len(tiers_answers), f"{rid}: {len(questions)} vs {len(tiers_answers)}"
        candidates = [
            {"idx": i, "question": q, "answer": a, "tier": t}
            for i, (q, (t, a)) in enumerate(zip(questions, tiers_answers))
        ]
        out.write(json.dumps({"id": rid, "candidates": candidates}, ensure_ascii=False) + "\n")
print("wrote", len(DATA), "rows")
