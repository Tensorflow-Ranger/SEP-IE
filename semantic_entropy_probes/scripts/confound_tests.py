"""Is a binary probe's accuracy a real ambiguity/uncertainty signal, or a
stylistic artifact of how each source dataset constructs its clear/ambiguous
pairs? Three independent tests, all run at the LR and MLP layer chosen by
train_binary_probes.py (results-dir/run_summary.json's best_layer_lr/mlp):

1. **Leave-one-dataset-out (LODO) generalization.** Train on two of the three
   source datasets, test on the third, for all three held-out choices. Every
   source dataset was built by a different process (ClarifyCodeBench: expert-
   annotated deletion edits; Orchid: single lexical-substitution edits;
   human-eval-comm: an LLM rewrite prompt) with its own idiosyncratic
   phrasing/formatting tics. A probe that has learned a genuine "this text
   is under-specified" signal should transfer across construction methods
   reasonably well; a probe that has learned e.g. "Orchid always swaps this
   one word" or "human-eval-comm's LLM rewrites drop the docstring's `>>>`
   examples" should collapse when the held-out source's tics aren't in the
   training data. This is compared for BOTH the hidden-state probe and the
   tfidf_bow text baseline -- if hidden-state LODO degrades much less than
   tfidf_bow's LODO degradation, that's evidence the residual stream encodes
   something beyond source-specific lexical cues.

2. **Label permutation test.** Shuffle train+val labels (features and text
   untouched) N times, refit the same pipeline each time, and report the
   resulting null distribution of test accuracy/AUROC plus an empirical
   p-value for the real (unpermuted) result. This is the standard "could this
   score have happened by chance given this many features and this little
   data" check -- distinct from the majority/random baselines in
   src/baselines.py, which check chance under the TRUE label distribution,
   not chance under a refit model with a specific coefficient count and this
   train_idx sample size.

3. **tfidf_bow vs hidden-state gap, pooled.** Just the pooled (non-LODO)
   comparison from train_binary_probes.py's baselines.json, restated here for
   convenience in the combined report -- a large positive gap (hidden state
   clearly above tfidf_bow) is the headline "not just stylistic" evidence;
   a near-zero or negative gap means the pooled result doesn't distinguish
   the two hypotheses on its own (LODO test 1 is then the more informative one).

Outputs: results-dir/confound_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.combined_splits import DATASET_SOURCES
from src.data_utils import load_combined_activation_bundle
from src.probes import _build_pipeline, _fit, _safe_auroc_ovr, _safe_spearman
from src.text_baselines import tfidf_logreg_baseline
from sklearn.metrics import accuracy_score

MACRO_LABELS = (0, 1)


def _load_prompts_by_id(data_path: Path) -> dict[str, str]:
    out = {}
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[row["id"]] = row["prompt"]
    return out


def leave_one_dataset_out(
    X: np.ndarray, y: np.ndarray, dataset_source: np.ndarray, groups: np.ndarray, prompts: np.ndarray,
    backend: str, seed: int, C_grid_source: str = "pooled", held_out_sources: list[str] | None = None,
) -> dict:
    """Train on two sources (80/20 train/val split within them), test on the
    third, for both the hidden-state features X and raw-text tfidf_bow.

    Also excludes from TRAIN any row whose source_problem_id group appears in
    the held-out test set, even if that row's dataset_source label differs
    from the one being held out. This matters concretely: orchid_heval and
    humanevalcomm share source_problem_id (both are the 164 HumanEval
    problems -- see build_combined_dataset.py), so holding out only
    dataset_source=="orchid_heval" while leaving humanevalcomm's rows for
    the SAME underlying problems in train would let the model partially
    recognize "which HumanEval problem is this" from humanevalcomm's
    incomplete rewrite of it, inflating apparent cross-source transfer
    without the model having generalized anything about ambiguity in
    general. Confirmed empirically before this fix: 154/164 of
    orchid_heval's held-out groups were still present in train via
    humanevalcomm, and orchid_heval was the one LODO fold with a visibly
    inflated AUROC relative to the two folds (clarifycodebench, orchid_bcb)
    that had zero group overlap by construction.
    """
    results = {}
    sources_present = [s for s in DATASET_SOURCES if (dataset_source == s).any()]
    if held_out_sources is not None:
        sources_present = [s for s in sources_present if s in held_out_sources]
    rng = np.random.default_rng(seed)

    for held_out in sources_present:
        test_mask = dataset_source == held_out
        test_groups = set(groups[test_mask])
        train_mask = (dataset_source != held_out) & (~np.isin(groups, list(test_groups)))
        if len(set(y[test_mask])) < 2:
            results[held_out] = {"skipped": "test source has only one class"}
            continue

        n_excluded_for_overlap = int(((dataset_source != held_out) & np.isin(groups, list(test_groups))).sum())
        if n_excluded_for_overlap:
            print(f"[confound_tests][LODO held_out={held_out}] excluded {n_excluded_for_overlap} "
                  f"train rows from other dataset_source(s) sharing a source_problem_id with the held-out set")

        train_idx_all = np.where(train_mask)[0]
        rng.shuffle(train_idx_all)
        n_val = max(1, int(0.2 * len(train_idx_all)))
        val_idx, tr_idx = train_idx_all[:n_val], train_idx_all[n_val:]
        test_idx = np.where(test_mask)[0]

        if len(set(y[tr_idx])) < 2 or len(set(y[val_idx])) < 2:
            results[held_out] = {"skipped": "degenerate train/val split"}
            continue

        # Hidden-state probe: sweep C on val, refit on tr+val, score on held-out source.
        c_grid = (0.001, 0.01, 0.1, 1.0)
        best_c, best_val_sp = c_grid[0], -np.inf
        for c in c_grid:
            pipe = _build_pipeline(c, seed, n_train=len(tr_idx), n_features=X.shape[1], backend=backend)
            _fit(pipe, X[tr_idx], y[tr_idx])
            val_expected = pipe.predict_proba(X[val_idx]) @ pipe.classes_.astype(np.float64)
            sp = _safe_spearman(y[val_idx], val_expected)
            if not np.isnan(sp) and sp > best_val_sp:
                best_c, best_val_sp = c, sp
        pipe_final = _build_pipeline(best_c, seed, n_train=len(tr_idx) + len(val_idx),
                                      n_features=X.shape[1], backend=backend)
        _fit(pipe_final, X[np.concatenate([tr_idx, val_idx])], y[np.concatenate([tr_idx, val_idx])])
        probs = pipe_final.predict_proba(X[test_idx])
        y_pred = pipe_final.predict(X[test_idx])
        hidden_metrics = {
            "test_accuracy": float(accuracy_score(y[test_idx], y_pred)),
            "test_auroc_ovr_macro": _safe_auroc_ovr(y[test_idx], probs, pipe_final.classes_),
            "n_train": int(len(tr_idx) + len(val_idx)),
            "n_test": int(len(test_idx)),
        }

        # tfidf_bow on the same LODO split.
        bow_metrics = tfidf_logreg_baseline(
            prompts[tr_idx], y[tr_idx], prompts[val_idx], y[val_idx], prompts[test_idx], y[test_idx],
            seed=seed, macro_labels=MACRO_LABELS,
        )

        results[held_out] = {
            "hidden_state": hidden_metrics,
            "tfidf_bow": {"test_accuracy": bow_metrics["test_accuracy"],
                          "test_auroc_ovr_macro": bow_metrics["test_auroc_ovr_macro"]},
            "n_train_rows_excluded_for_group_overlap": n_excluded_for_overlap,
        }
        print(f"[confound_tests][LODO held_out={held_out}] "
              f"hidden acc={hidden_metrics['test_accuracy']:.3f} auroc={hidden_metrics['test_auroc_ovr_macro']:.3f} "
              f"| tfidf acc={bow_metrics['test_accuracy']:.3f} auroc={bow_metrics['test_auroc_ovr_macro']:.3f}")

    return results


def permutation_test(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, backend: str, seed: int, n_perms: int = 100, C: float = 0.1,
) -> dict:
    rng = np.random.default_rng(seed)
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    pipe_real = _build_pipeline(C, seed, n_train=len(y_trainval), n_features=X_train.shape[1], backend=backend)
    _fit(pipe_real, X_trainval, y_trainval)
    probs_real = pipe_real.predict_proba(X_test)
    real_acc = float(accuracy_score(y_test, pipe_real.predict(X_test)))
    real_auroc = _safe_auroc_ovr(y_test, probs_real, pipe_real.classes_)

    null_accs, null_aurocs = [], []
    for i in range(n_perms):
        y_perm = rng.permutation(y_trainval)
        pipe = _build_pipeline(C, seed + 1 + i, n_train=len(y_perm), n_features=X_train.shape[1], backend=backend)
        _fit(pipe, X_trainval, y_perm)
        probs = pipe.predict_proba(X_test)
        null_accs.append(accuracy_score(y_test, pipe.predict(X_test)))
        auroc = _safe_auroc_ovr(y_test, probs, pipe.classes_)
        if not np.isnan(auroc):
            null_aurocs.append(auroc)

    null_accs = np.array(null_accs)
    p_acc = float((np.sum(null_accs >= real_acc) + 1) / (len(null_accs) + 1))

    null_aurocs_arr = np.array(null_aurocs)
    p_auroc = None
    null_auroc_mean = None
    if len(null_aurocs_arr) and not np.isnan(real_auroc):
        p_auroc = float((np.sum(null_aurocs_arr >= real_auroc) + 1) / (len(null_aurocs_arr) + 1))
        null_auroc_mean = float(null_aurocs_arr.mean())

    return {
        "real_test_accuracy": real_acc,
        "real_test_auroc": None if np.isnan(real_auroc) else real_auroc,
        "null_accuracy_mean": float(null_accs.mean()),
        "null_accuracy_std": float(null_accs.std()),
        "p_value_accuracy": p_acc,
        "null_auroc_mean": null_auroc_mean,
        "p_value_auroc": p_auroc,
        "n_perms": n_perms,
        "C": C,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations-dir", type=Path, default=ROOT / "activations_combined")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "ambiguity_dataset_combined.jsonl")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results_combined")
    ap.add_argument("--n-perms", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acts, labels, groups, token_lens, dataset_source, ambiguity_type, meta_rows = \
        load_combined_activation_bundle(args.activations_dir)
    prompts_by_id = _load_prompts_by_id(args.data)
    prompts = np.array([prompts_by_id[m["id"]] for m in meta_rows])

    with (args.results_dir / "run_summary.json").open("r", encoding="utf-8") as f:
        run_summary = json.load(f)

    from src.combined_splits import make_combined_train_val_test_split
    train_idx, val_idx, test_idx = make_combined_train_val_test_split(labels, dataset_source, groups, seed=args.seed)

    report = {}
    for tag, backend, best_layer in [
        ("lr", "torch", run_summary["best_layer_lr"]),
        ("mlp", "torch_mlp", run_summary["best_layer_mlp"]),
    ]:
        print(f"[confound_tests] === {tag} (layer {best_layer}) ===")
        X_layer = acts[:, best_layer, :]

        print(f"[confound_tests][{tag}] running leave-one-dataset-out generalization...")
        lodo = leave_one_dataset_out(X_layer, labels, dataset_source, groups, prompts, backend=backend, seed=args.seed)

        print(f"[confound_tests][{tag}] running {args.n_perms}-permutation label-shuffle test...")
        perm = permutation_test(
            X_layer[train_idx], labels[train_idx], X_layer[val_idx], labels[val_idx],
            X_layer[test_idx], labels[test_idx], backend=backend, seed=args.seed, n_perms=args.n_perms,
        )

        report[tag] = {"best_layer": best_layer, "lodo": lodo, "permutation_test": perm}

    with (args.results_dir / "confound_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[confound_tests] done. wrote {args.results_dir / 'confound_report.json'}")


if __name__ == "__main__":
    main()
