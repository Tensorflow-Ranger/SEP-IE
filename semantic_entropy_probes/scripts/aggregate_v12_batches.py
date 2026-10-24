"""Cross-batch aggregation over all v12_ranking live-loop round_*/*/summary.json files.

Produces, per batch directory:
  - heval32-style batches (have test_n_pass/test_n_total): per-task round-0 baseline,
    per-task trajectory across rounds, and an honest "genuinely-fixed-by-clarification"
    count that excludes tasks already fully passing at round 0.
  - ccb-style batches (no tests, selection-log only): descriptive stats only
    (n tasks, n rounds, candidate pool size, score gap between chosen and runner-up).

Writes a single markdown report to data/v12_ranking/CROSS_BATCH_REPORT.md.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve()
REPO = ROOT.parents[1]
RANKING_DIR = REPO / "data" / "v12_ranking"

HEVAL_BATCHES = [
    "heval32_live_loop",
    "heval32_live_loop_batch2",
    "heval32_live_loop_v1_8q_wrongprompt",
]
CCB_BATCHES = [
    "ccb_live_loop",
    "ccb_live_loop_batch2",
    "ccb_live_loop_haiku_elicit",
]


def load_heval_batch(name: str) -> dict:
    base = RANKING_DIR / name
    rounds = sorted(int(p.name.split("_")[1]) for p in base.glob("round_*") if p.is_dir())
    by_task: dict[str, dict[int, tuple[int, int]]] = {}
    for r in rounds:
        for f in sorted(base.glob(f"round_{r}/*/summary.json")):
            tid = f.parent.name
            j = json.loads(f.read_text())
            if "test_n_pass" not in j:
                continue
            by_task.setdefault(tid, {})[r] = (j["test_n_pass"], j["test_n_total"])

    n_tasks = len(by_task)
    r0_solved = 0
    r0_attempted = 0
    fixed_among_continued = 0
    still_broken_among_continued = 0
    unsolved_abandoned = 0
    regressed = 0
    n_unsolved_r0 = 0
    for tid, traj in by_task.items():
        if 0 not in traj:
            continue
        r0_attempted += 1
        p0, t0 = traj[0]
        full0 = t0 > 0 and p0 == t0
        if full0:
            r0_solved += 1
        else:
            n_unsolved_r0 += 1
            later_rounds = [r for r in traj if r > 0]
            if later_rounds:
                last_r = max(later_rounds)
                pl, tl = traj[last_r]
                if tl > 0 and pl == tl:
                    fixed_among_continued += 1
                else:
                    still_broken_among_continued += 1
            else:
                unsolved_abandoned += 1
        # regression check: any later round strictly worse than round 0 on a task that
        # started fully solved
        if full0:
            for r, (p, t) in traj.items():
                if r == 0:
                    continue
                if t > 0 and p < t0:
                    regressed += 1
                    break

    n_continued = sum(1 for traj in by_task.values() if len(traj) > 1)
    n_unsolved_continued = fixed_among_continued + still_broken_among_continued

    return {
        "n_tasks_round0": r0_attempted,
        "n_tasks_total_seen": n_tasks,
        "rounds": rounds,
        "r0_solved_full": r0_solved,
        "r0_unsolved": n_unsolved_r0,
        "n_continued_past_r0": n_continued,
        "unsolved_and_continued": n_unsolved_continued,
        "unsolved_and_abandoned": unsolved_abandoned,
        "fixed_among_continued": fixed_among_continued,
        "still_broken_among_continued": still_broken_among_continued,
        "regressed_from_solved": regressed,
    }


def load_ccb_batch(name: str) -> dict:
    base = RANKING_DIR / name
    rounds = sorted(int(p.name.split("_")[1]) for p in base.glob("round_*") if p.is_dir())
    n_files = 0
    n_candidates_total = 0
    gaps = []
    tasks = set()
    for r in rounds:
        for f in sorted(base.glob(f"round_{r}/*/summary.json")):
            n_files += 1
            tasks.add(f.parent.name)
            j = json.loads(f.read_text())
            cands = j.get("candidates", [])
            n_candidates_total += len(cands)
            if len(cands) >= 2:
                scores = sorted((c["score"] for c in cands), reverse=True)
                gaps.append(scores[0] - scores[1])
    return {
        "rounds": rounds,
        "n_task_round_files": n_files,
        "n_unique_tasks": len(tasks),
        "avg_candidates_per_round": (n_candidates_total / n_files) if n_files else 0.0,
        "avg_top2_score_gap": (sum(gaps) / len(gaps)) if gaps else None,
        "n_gap_samples": len(gaps),
    }


def overlap_note() -> list[str]:
    main_ids = set()
    for f in RANKING_DIR.glob("heval32_live_loop/round_1/*/summary.json"):
        main_ids.add(f.parent.name)
    wrong_ids = set()
    for f in RANKING_DIR.glob("heval32_live_loop_v1_8q_wrongprompt/round_0/*/summary.json"):
        wrong_ids.add(f.parent.name)
    shared = main_ids & wrong_ids
    return [
        "## Important caveat: `heval32_live_loop` rounds 1-3 are not a continuation of the "
        "full 32-problem round-0 cohort",
        "",
        f"`heval32_live_loop`'s round-0 covers all 32 problems (18 unsolved), but rounds 1-3 "
        f"only continue a fixed {len(main_ids)}-problem subset that is IDENTICAL to the "
        f"`heval32_live_loop_v1_8q_wrongprompt` comparison set ({len(shared)}/{len(main_ids)} "
        f"ids match). 6 of those 10 problems were already fully passing at round 0, so the "
        f"'converges to ~9-10/10 by round 2-3' figure from before mixes in problems the loop "
        f"never needed to fix. The real per-config test is only the problems that started "
        f"unsolved: main config fixed 3/4 (heval_91, heval_147, heval_150; heval_130 stayed "
        f"broken) with zero regressions; the wrongprompt config also fixed 4/5 on its slightly "
        f"different unsolved set but introduced one regression (heval_104: 4/4 -> 2/4 by round "
        f"3) that the main config did not.",
        "",
    ]


def main():
    lines = ["# v12 cross-batch aggregate report", ""]
    lines.append("Generated by scripts/aggregate_v12_batches.py over all "
                  "`data/v12_ranking/*_live_loop*/round_*/*/summary.json` files.\n")

    lines.extend(overlap_note())

    lines.append("## heval32-style batches (executable tests available)\n")
    for name in HEVAL_BATCHES:
        base = RANKING_DIR / name
        if not base.exists():
            continue
        stats = load_heval_batch(name)
        lines.append(f"### `{name}`")
        lines.append(f"- rounds present: {stats['rounds']}")
        lines.append(f"- tasks with a round-0 baseline: {stats['n_tasks_round0']} "
                      f"(full-32 run only for the base batch; a fixed subset for the others)")
        lines.append(f"- round-0 fully passing already: {stats['r0_solved_full']}/{stats['n_tasks_round0']}")
        lines.append(f"- round-0 unsolved: {stats['r0_unsolved']} "
                      f"(of which {stats['unsolved_and_continued']} were actually continued into later "
                      f"rounds, {stats['unsolved_and_abandoned']} were never retried)")
        caveat = (f" (not /{stats['r0_unsolved']} of all unsolved -- {stats['unsolved_and_abandoned']} "
                  f"unsolved tasks were never retried)") if stats['unsolved_and_abandoned'] else ""
        lines.append(f"- **genuinely fixed by the clarify loop** (unsolved at r0, continued, full pass by "
                      f"last round seen): {stats['fixed_among_continued']}/{stats['unsolved_and_continued']} "
                      f"of attempted{caveat}")
        lines.append(f"- still broken after being retried: {stats['still_broken_among_continued']}/{stats['unsolved_and_continued']}")
        lines.append(f"- **regressions** (fully passing at r0, worse in a later round): {stats['regressed_from_solved']}")
        lines.append("")

    lines.append("## ccb-style batches (selection-log only, no executable tests)\n")
    for name in CCB_BATCHES:
        base = RANKING_DIR / name
        if not base.exists():
            continue
        stats = load_ccb_batch(name)
        lines.append(f"### `{name}`")
        lines.append(f"- rounds present: {stats['rounds']}")
        lines.append(f"- unique tasks: {stats['n_unique_tasks']}, task-round files: {stats['n_task_round_files']}")
        lines.append(f"- avg candidates considered per round: {stats['avg_candidates_per_round']:.1f}")
        gap = stats['avg_top2_score_gap']
        lines.append(f"- avg score gap between top-1 and top-2 candidate: "
                      f"{gap:.3f}" if gap is not None else "- avg score gap: n/a")
        lines.append("")

    out_path = RANKING_DIR / "CROSS_BATCH_REPORT.md"
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
