import json

# Each entry: id -> list of (tier, answer) aligned with candidates.jsonl's questions order
DATA = {
"ccb_amb_task_94": [
    (2, "The subsequence is not necessarily contiguous and preserves the original order of indices; it's formed by deleting any number of elements while keeping order."),
    (1, "Not specified beyond what's given; assume the standard convention that an empty subsequence is valid with length 0."),
    (1, "Not specified beyond what's given; assume elements must be at most X_i (i.e. <= X_i, inclusive)."),
    (1, "Only the maximum length is required, not a particular subsequence, so no tie-breaking rule is needed."),
    (0, "Yes, each query is answered independently against the fixed original array."),
    (0, "Queries are processed in the given order, each based on the prefix up to its own R_i, independent of prior queries."),
    (0, "Yes, X_i can equal the minimum element; this is exactly why the guarantee ensures a non-empty valid subsequence exists."),
    (0, "Not specified beyond what's given; assume a single test case per input file."),
],
"ccb_amb_task_95": [
    (0, "The problem already states the subsequence need not be contiguous but must preserve original index order, matching standard subsequence order."),
    (1, "Only the length is required; no particular subsequence needs to be identified."),
    (0, "Duplicate values are allowed in the array, but a strictly increasing subsequence cannot include two equal values."),
    (1, "You should output the length of the longest valid subsequence, i.e. the maximum length achievable."),
    (0, "Yes, X_i can equal that minimum; the guarantee just ensures a non-empty valid subsequence exists."),
    (0, "Yes, each query is answered independently based on the original array."),
    (0, "This case cannot occur given the stated guarantee, so it doesn't need special handling."),
    (0, "Each answer should be printed on its own line, as already stated in the Output section."),
],
"ccb_amb_task_96": [
    (0, "The problem already specifies the subsequence need not be contiguous and preserves original index order."),
    (0, "Duplicate values may exist in the array, but a strictly increasing subsequence cannot contain two equal values together."),
    (1, "Each element of the subsequence individually must satisfy the bound, not just the maximum."),
    (1, "Such elements are simply excluded from consideration when forming the subsequence."),
    (0, "The guarantee ensures at least one valid element exists, so the minimum answer is effectively at least 1 given a non-empty prefix condition."),
    (0, "Yes, each query considers only the original array, independent of other queries."),
    (2, "Each query's answer should be printed on a separate line, in the same order as the queries are given."),
    (0, "Not specified beyond what's given; assume typical AtCoder-style time/memory limits."),
],
"ccb_amb_task_97": [
    (2, "The subsequence is not necessarily contiguous; it preserves original index order (standard subsequence definition), not an arbitrary reordering."),
    (1, "Only the length is required, not a specific subsequence, so tie-breaking is unnecessary."),
    (0, "Duplicate values may appear in the array, but a strictly increasing subsequence excludes equal values."),
    (1, "Each element of the subsequence must individually satisfy the bound, not just the maximum element."),
    (0, "Yes, X_i can equal that minimum value; it just guarantees feasibility."),
    (0, "Yes, each query is answered independently against the original array."),
    (0, "Each query's answer should be printed on a separate line, as already stated in the Output section."),
    (0, "Not specified beyond what's given; assume a single test case per input file."),
],
"ccb_amb_task_98": [
    (2, "The subsequence is not necessarily contiguous; elements are chosen in order but need not be consecutive."),
    (1, "Only the maximum length is required, not a particular subsequence."),
    (0, "The minimum possible answer is 1, following from the stated guarantee that a valid element always exists."),
    (0, "Yes, each query is independent, evaluated against the original array."),
    (0, "Yes, R_i is inclusive, i.e. A_{R_i} is part of the considered prefix."),
    (0, "The guarantee is stated as an inequality (>=), so X_i can equal that minimum."),
    (0, "Not specified beyond what's given; assume a single test case per input file."),
    (2, "Each query's answer should be printed on a separate line, in the same order as the queries are given."),
],
"ccb_amb_task_99": [
    (0, "The problem already states the subsequence need not be contiguous but must preserve original index order."),
    (1, "Only the length is required; no specific subsequence needs to be reported."),
    (0, "Duplicate values may exist in the array, but a strictly increasing subsequence cannot include two equal values."),
    (1, "Each element of the subsequence must individually satisfy the bound, not only the maximum element."),
    (0, "The guarantee ensures a valid non-empty subsequence exists whenever the prefix has any qualifying element."),
    (0, "This case cannot occur given the stated guarantee."),
    (0, "Yes, each query is answered independently based on the original, unmodified array."),
    (2, "Each answer should be a plain integer on its own line, in the order the queries are given, with no extra spaces required beyond that."),
],
"ccb_amb_task_100": [
    (2, "The subsequence follows standard order (indices increasing), not an arbitrary reordering, though not necessarily contiguous."),
    (1, "Only the length is required; no tie-breaking or additional info is needed for multiple achievers."),
    (0, "Duplicate values may exist in the array, but a strictly increasing subsequence excludes equal values from being adjacent choices."),
    (1, "Each element of the subsequence must individually satisfy the bound, not just the last element."),
    (0, "Yes, X_i can equal that minimum value, per the stated guarantee, which still permits a valid non-empty subsequence."),
    (0, "Yes, each query is independent, evaluated against the original array."),
    (2, "Each answer should be printed as a plain integer per line with no additional formatting."),
    (0, "This case cannot occur since R_i >= 1 per the constraints."),
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
