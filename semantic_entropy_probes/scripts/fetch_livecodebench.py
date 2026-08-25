"""Fetch the LiveCodeBench v6 question bank and cache only the fields we need.

NOTE: this LiveCodeBench-paired dataset is kept as a *secondary* reference
only. Comparing `question_id`-matched pairs directly against ClarifyCodeBench's
`modified_content` turned out to have a severe formatting confound (see
`build_dataset.py`): LiveCodeBench's `question_content` almost always includes
"Sample Input/Output" sections, while ClarifyCodeBench's underspecified
rewrites essentially never do (1/419). A probe could trivially learn "does
this prompt have a Sample Input header" instead of anything about ambiguity.
The primary 0-class is instead reconstructed from ClarifyCodeBench's own
annotations (`modified_content` + the annotated QA answers) -- see
`build_dataset.py` for the full rationale.

We stream-parse the raw LiveCodeBench jsonl shards one at a time and keep only
{question_id, question_title, question_content, platform, difficulty}, then
delete each downloaded shard immediately -- the raw files are ~4.5GB total
(dominated by base64 test-case blobs we don't need), while the trimmed output
is a few MB.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "livecodebench/code_generation_lite"
REPO_TYPE = "dataset"
# release_v6 = test.jsonl .. test6.jsonl (matches the LiveCodeBench version
# ClarifyCodeBench was built on, per its data card).
SHARDS = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]

KEEP_FIELDS = ["question_id", "question_title", "question_content", "platform", "difficulty"]

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "livecodebench_v6_questions.jsonl"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seen: dict[str, dict] = {}

    for shard in SHARDS:
        print(f"[fetch] downloading {shard} ...")
        local_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=shard)

        n = 0
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qid = row["question_id"]
                seen[qid] = {k: row.get(k) for k in KEEP_FIELDS}
                n += 1
        print(f"[fetch] {shard}: {n} rows, {len(seen)} unique question_ids so far")

        # Free disk immediately -- shards are large (test-case blobs), we only
        # needed the text fields, which are now in `seen`. `local_path` is a
        # symlink into the HF cache's content-addressed blob store, so
        # unlinking it only removes the symlink, not the multi-hundred-MB
        # blob it points to -- resolve first, then delete the real file.
        real_path = Path(local_path).resolve()
        Path(local_path).unlink(missing_ok=True)
        real_path.unlink(missing_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in seen.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[fetch] wrote {len(seen)} unique questions -> {OUT_PATH}")


if __name__ == "__main__":
    main()
