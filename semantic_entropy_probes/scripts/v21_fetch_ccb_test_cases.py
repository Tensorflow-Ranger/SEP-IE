"""Fetch real executable test cases for the 24 unique LiveCodeBench question_ids
that back this eval's 31 clarifycodebench test-group problems (mapped via
ClarifyCodeBench.jsonl's task_id -> question_id, e.g. task_107 -> abc393_e).

fetch_livecodebench.py deliberately discarded public_test_cases/private_test_cases
(the multi-GB base64+zlib+json blobs) since the original probe-training task never
needed them. This eval does -- it runs generated code against real stdin/stdout
tests -- so this script re-streams the same 6 shards, decodes ONLY the needed
ids' test cases (public_test_cases: direct JSON string; private_test_cases:
base64 -> zlib.decompress -> JSON string, confirmed by hand against one row),
and writes {question_id: [{"input", "output"}, ...]} (public+private merged) to
disk. Same "download shard, keep only what's needed, delete the shard" pattern.

Usage: python v21_fetch_ccb_test_cases.py
"""
from __future__ import annotations

import base64
import json
import pickle
import zlib
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "livecodebench/code_generation_lite"
REPO_TYPE = "dataset"
SHARDS = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]

ROOT = Path(__file__).resolve().parents[1]
NEEDED_IDS_PATH = Path("/tmp/ccb_test_qids.json")
OUT_PATH = ROOT / "data" / "v20_branches_full" / "ccb_test_cases.json"


def decode_test_cases(raw: str) -> list[dict]:
    try:
        cases = json.loads(raw)
    except json.JSONDecodeError:
        cases = json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw))))
    return [{"input": c["input"], "output": c["output"]} for c in cases]


def main() -> None:
    needed = set(json.loads(NEEDED_IDS_PATH.read_text()))
    print(f"[v21-fetch] need test cases for {len(needed)} question_ids: {sorted(needed)}")
    found: dict[str, list[dict]] = {}

    for shard in SHARDS:
        if needed <= found.keys():
            print("[v21-fetch] all needed ids found, stopping early")
            break
        print(f"[v21-fetch] downloading {shard} ...", flush=True)
        local_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=shard)
        n_matched = 0
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qid = row["question_id"]
                if qid not in needed or qid in found:
                    continue
                cases = decode_test_cases(row["public_test_cases"])
                cases += decode_test_cases(row["private_test_cases"])
                found[qid] = cases
                n_matched += 1
        print(f"[v21-fetch] {shard}: matched {n_matched} ids, {len(found)}/{len(needed)} total", flush=True)
        real_path = Path(local_path).resolve()
        Path(local_path).unlink(missing_ok=True)
        real_path.unlink(missing_ok=True)

    missing = needed - found.keys()
    if missing:
        print(f"[v21-fetch] WARNING: never found test cases for {sorted(missing)}")

    OUT_PATH.write_text(json.dumps(found, indent=2))
    print(f"[v21-fetch] saved {len(found)} problems' test cases -> {OUT_PATH}")
    for qid, cases in found.items():
        print(f"  {qid}: {len(cases)} test cases")


if __name__ == "__main__":
    main()
