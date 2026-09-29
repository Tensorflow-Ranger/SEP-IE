# Semantic Entropy Probes — Hidden-State Ambiguity Detection

**Question:** does Qwen3-8B's residual stream already encode *how much*
unresolved specification ambiguity a coding prompt has, in a single forward
pass — without the usual approach of sampling multiple generations and
measuring disagreement between them?

This is a first-pass feasibility check: a frozen-activation multinomial probe
per transformer layer, evaluated against trivial baselines. It does **not**
implement entropy estimation, contrastive learning, or clarification-question
generation — those are follow-on questions that only make sense once this one
is answered.

**Result: yes, moderately** (Spearman ρ≈0.31 at the best layer, vs. ρ≈0.01
random / ρ≈0.04 length-only baseline) — once a confound in the dataset's
0-class is identified and excluded. See **[FINDINGS.md](FINDINGS.md)** for
the full result and the confound diagnosis; the summary is also in the
Results section below.

## Target: ambiguity COUNT, not a binary flag

The target is K ∈ {0, 1, 2, 3} — the number of unresolved ambiguity points in
the prompt — not a collapsed "any ambiguity present" binary label. K=1 and
K=3 are genuinely different amounts of missing specification, and the
benchmark already gives us the count for free; collapsing it to binary would
throw that away. `label` is K throughout the canonical schema and every
downstream script.

## Data

[ClarifyCodeBench](https://github.com/fangz-cs/ClarifyCodeBench) (arXiv
[2607.00711](https://arxiv.org/abs/2607.00711)) provides 419 deliberately
underspecified coding problems — deletion-only edits of 80 unique
[LiveCodeBench](https://livecodebench.github.io/) v6 problems, each with 1–3
annotated ambiguity points (K=1: 199 tasks, K=2: 169, K=3: 51), each point
carrying a key clarification question and its ground-truth answer.

**K=0 construction, and a confound we found and avoided.** The first version
of this pipeline paired each `modified_content` (K≥1) against the *original*
LiveCodeBench `question_content` for the same `question_id` (K=0). That has a
fatal formatting confound: LiveCodeBench statements almost always carry
"Sample Input / Sample Output" sections, and ClarifyCodeBench's rewrites
essentially never do (1/419 contain the string "Sample Input", vs.
effectively all of the matched LiveCodeBench originals). A probe — or a
bag-of-words classifier — could hit near-perfect accuracy by detecting that
header alone, and it would look like every layer "encodes ambiguity." It was
also lopsided across K: 419 rewrites share only 80 source problems, so
pairing by `question_id` gives 80 K=0 examples against 199/169/51 for K=1/2/3.

Instead, the K=0 class uses the data card's own definition of "resolved":
*"each edited task is underspecified initially but becomes fully specified
once the annotated clarification answers are given."* So K=0 is
`modified_content` with its own annotated QA-pair answers appended as a
`Clarifications:` block. This gives **one K=0 example per ambiguous row (419,
not 80) — 838 rows total**, keeps every class in the same
provenance/formatting family (all derive from `modified_content`; none have
Sample I/O sections — verified: 1/419 for every class), and removes the
worst of the class imbalance. It introduces a smaller, expected confound of
its own — K=0 prompts are always at least as long as their source row (answers
are appended, never removed; mean 1056 vs. 890 characters for K≥1) and carry
a `Clarifications:` header that K≥1 rows never have — which is exactly what
the token-length baseline below is for. The LiveCodeBench-`question_id`-paired
version (with the Sample I/O confound, and only 80 K=0 examples) is still
written to `data/ambiguity_dataset_lcbfull.jsonl` for reference; nothing
downstream reads it by default.

Final K distribution: **{0: 419, 1: 199, 2: 169, 3: 51}** (838 rows, 80
unique source problems).

**Grouped splits are load-bearing here, not a nicety.** A K≥1 row and its
K=0 reconstruction differ by exactly the deleted information; if the same
`source_problem_id` appears on both sides of a split, a probe can partly solve
the task by recognizing the *problem* rather than judging its ambiguity.
Every split in this repo (`src/splits.py`) is grouped by `source_problem_id`
via `StratifiedGroupKFold` (which also handles the multiclass stratification).

## Pipeline

```
ClarifyCodeBench.jsonl ──> build_dataset.py ──> data/ambiguity_dataset.jsonl
                                                  (canonical schema, see below;
                                                   also writes the secondary
                                                   LiveCodeBench-paired file if
                                                   fetch_livecodebench.py has
                                                   already been run)
                                          │
                                          v
                          extract_activations.py  (Qwen3-8B, non-thinking, one fwd pass/prompt)
                                          │
                                          v
                     activations/activations.npy + metadata.jsonl
                                          │
                                          v
                               train_probes.py  (per-layer multinomial probes)
                                          │
                                          v
             results/layer_metrics.csv, layer_cv_spearman.csv, baselines.json
                                          │
                                          v
                                  plot_results.py  ->  results/layer_spearman.png
```

Canonical dataset schema (`data/ambiguity_dataset.jsonl`), one row per prompt:

```json
{
  "id": "amb_task_1",
  "source_problem_id": "abc390_b",
  "prompt": "You are given a length-N sequence ...",
  "label": 1,
  "ambiguity_types": ["Terminology"]
}
```

`label` = K, the ambiguity count (0–3). `label=0` rows are `modified_content`
with its annotated clarification answers appended; `label>=1` rows are
`modified_content` as-is. Everything downstream (extraction, splitting,
probing) is written against this schema — it never needs to know how a row's
label originated.

## Model & extraction details

- **Model:** `Qwen/Qwen3-8B`, loaded in bf16.
- **Non-thinking mode:** `apply_chat_template(..., enable_thinking=False)`.
  For Qwen3 this does *not* omit the thinking block — it emits an *empty* one
  (`<think>\n\n</think>\n\n`) right before generation would start. Verified
  directly against the tokenizer's chat template rather than trusted by name.
- **Extraction point:** the residual stream at the **last token of the fully
  templated prompt** (i.e. the last token the model conditions on before
  generating), for every layer. `output_hidden_states=True` returns
  `n_layers + 1` tensors (index 0 = embedding output, 1..36 = post-block
  residual streams) — all are kept; layer selection happens at the probing
  stage, not extraction.
- **Batch size 1.** Right-padding would put PAD tokens at the sequence end for
  shorter prompts, so a naive `[:, -1, :]` on a padded batch silently grabs
  the wrong token unless gathered via `attention_mask.sum(-1) - 1`. At ~800
  prompts, one forward pass per prompt is cheap enough on a 24GB GPU that the
  extra complexity isn't worth it.

## Probing approach

Each layer gets a standard multinomial `LogisticRegression` (not a dedicated
ordinal/proportional-odds model), `class_weight="balanced"`, with `C` swept
on the validation split. Metrics reported per layer:

- **test_spearman** — Spearman correlation between true K and the model's
  probability-weighted expected value `E[K] = sum_k P(k) * k`. This is the
  primary metric: it directly asks "does this layer's probe track the
  *degree* of ambiguity," which is the question this repo exists to answer,
  and it's also the criterion used for model selection (`C`) on the
  validation split.
- **test_accuracy**, **test_macro_f1** (macro-averaged over all 4 classes,
  so K=0's larger count doesn't dominate), **test_auroc_ovr_macro**
  (one-vs-rest, macro-averaged) — standard multiclass classification metrics.
- **test_mae** — mean absolute error between true K and the argmax predicted
  class, i.e. "off by how many ambiguity points, on average."

`layer_cv_spearman.csv` adds 5-fold grouped-CV Spearman (mean ± std)
per layer at the layer's selected `C`, since a single held-out test split
(~16 of 80 source-problem groups) is noisy across 37 layers on its own. Note
this CV runs over *all* rows including the held-out test split, so it isn't a
fully independent estimate from `test_spearman` — read it as a robustness
check on split-to-split variance, not as confirmation from an untouched set.

**`--min-k`, and why the headline result requires it.** Every K=0 row carries
an appended `Clarifications:` header that no K≥1 row has, which is trivially
separable and swamps the full-range (K=0–3) Spearman with a lexical-detection
signal rather than a degree signal — see **[FINDINGS.md](FINDINGS.md)** for
the full diagnosis (36/37 layers failed to converge on the full range; a
flat-with-depth Spearman curve; a length baseline that tracks the header).
`train_probes.py --min-k 1` restricts to K∈{1,2,3} — all `modified_content`
verbatim, zero header confound — and is the run that actually answers "does
the residual stream track ambiguity degree." Run both and compare; don't
report the full-range number alone.

## Reproduce

```bash
cd semantic_entropy_probes
python scripts/fetch_livecodebench.py      # -> data/livecodebench_v6_questions.jsonl
python scripts/build_dataset.py            # -> data/ambiguity_dataset.jsonl, ambiguity_dataset_lcbfull.jsonl, ambiguity_dataset_lcbclean.jsonl

# K=0 via appended Clarifications: header (most confounded)
python src/extract_activations.py --data data/ambiguity_dataset.jsonl --out-dir activations
python scripts/train_probes.py --activations-dir activations --min-k 0 --results-dir results
python scripts/train_probes.py --activations-dir activations --min-k 1 --results-dir results_k1   # load-bearing

# K=0 via LiveCodeBench text, Sample I/O stripped (less confounded, not clean)
python src/extract_activations.py --data data/ambiguity_dataset_lcbclean.jsonl --out-dir activations_lcbclean
python scripts/train_probes.py --activations-dir activations_lcbclean --min-k 0 --results-dir results_lcbclean
python scripts/train_probes.py --activations-dir activations_lcbclean --min-k 1 --results-dir results_lcbclean_k1  # determinism check

python scripts/plot_results.py --results-dir results
python scripts/plot_results.py --results-dir results_lcbclean
python scripts/plot_results.py --results-dir results_k1
```

Everything is seeded (`--seed`, default 0) for reproducibility. Activations
are cached to disk after extraction — `train_probes.py` never re-runs the
model, so re-running with different `--min-k`/`--seed` values is cheap.

## Baselines

- **Majority class** (always predict the most frequent train K) and
  **random** (draw from the train K distribution) — the required sanity
  floor. Majority's Spearman is undefined (a constant predictor has no
  correlation) and reported as NaN, not 0 — don't read a NaN row as "beats
  chance."
- **Token-length-only probe** — on the full range with the `Clarifications:`
  header K=0 construction, K=0 rows are always longer, so length alone
  tracks the confound (ρ≈0.21). On the LiveCodeBench-paired, Sample-I/O-
  stripped K=0 construction, length is close to chance (ρ≈0.02). On the
  K≥1-only subset (no K=0 at all), length is close to chance (ρ≈0.04). See
  FINDINGS.md.

## Results (summary — see FINDINGS.md for the full four-run analysis)

Four runs were tried, differing only in how the K=0 class was constructed —
this matters a lot, because **every K=0 construction tried inflates the
correlation to some degree**:

| Run | K=0 construction | Best layer | Test Spearman | Unconverged layers | Length baseline |
|---|---|---|---|---|---|
| `results/` | `modified_content` + appended `Clarifications:` answers | 20 | 0.855 | 36/37 | 0.21 |
| `results_lcbclean/` | LiveCodeBench original text, Sample I/O stripped & reformatted | 20 | 0.526 | 10/37 | 0.02 |
| **`results_k1/`** | **no K=0 — K∈{1,2,3} only** | **32** | **0.312** | **0/37** | 0.04 |

The two K≥1-only runs (`results_k1/` and `results_lcbclean_k1/`, run on two
different activation caches as a determinism check) reproduce each other to
8 decimal places. **This is the load-bearing result**: hidden states carry a
moderate, genuine, depth-increasing signal about ambiguity *degree* — rising
from ρ≈0.08 at layer 1 to ρ≈0.31 by layer 32 — clearly above random (0.007)
and length (0.043) baselines, in the one comparison where the K=0 confound
is structurally impossible (every row went through the same editing
process; only how much was deleted varies).

Both full-range runs are useful diagnostically (the confound visibly shrinks
as the K=0 construction improves: 0.855 → 0.526) but neither should be read
as the answer to "does the residual stream encode ambiguity degree" — see
FINDINGS.md's "Confound 1" and "Confound 2" sections for what each one's
non-convergence and flat/early-elevated Spearman curves actually indicate.

## Known limitations (by design, for this first pass)

- **No K=0 construction tried is fully clean.** The `Clarifications:` header
  version is confirmed confounded (36/37 layers non-convergent, flat Spearman
  curve). The LiveCodeBench-paired, Sample-I/O-stripped version is much
  cleaner (length baseline ≈ chance, all 80 pairs verified as genuine
  deletion-only edits) but still shows 10/37 non-convergent layers and a
  curve that's already elevated at layer 1 rather than building with depth —
  the working hypothesis in FINDINGS.md is a residual "was this text edited
  by ClarifyCodeBench annotators" stylistic signature, independent of
  ambiguity content. Only excluding K=0 entirely (`results_k1/`) removes
  this confound at the root.
- On the load-bearing K∈{1,2,3} subset, K=3 has only 51 examples out of 419,
  spread across the 80 source-problem groups; per-layer test-split metrics
  on this class especially are noisy, which is why macro-F1 forces all
  present classes into the denominator (`labels=[1,2,3]` for this run)
  rather than only scoring classes seen in a given split.
- The grouped-CV Spearman in `layer_cv_spearman.csv` is not independent of
  the single test-split `test_spearman` in `layer_metrics.csv` (the CV folds
  are drawn from all rows, including the held-out test split) — treat the
  two together as one robustness picture, not as two confirmations. On the
  K≥1 run they agree in direction and rough magnitude (0.31 test vs.
  0.21±0.05 CV at the best layer), which is reassuring but not fully
  independent evidence.

## Combined binary dataset: does this generalize past ClarifyCodeBench?

A second dataset combines four independently-built sources — ClarifyCodeBench
(deletion edits), [Orchid](https://huggingface.co/datasets/SII-YDD/Orchid)
(lexical/vagueness word-substitution edits, HumanEval + BigCodeBench +
BigCodeBench-Expand configs), and [human-eval-comm](https://github.com/jie-jw-wu/human-eval-comm)
(LLM-rewritten incompleteness edits) — into one **binary** clear-vs-ambiguous
target (K≥1 collapsed to one class), specifically to test whether the K-task's
finding holds up across construction methods, and to stress-test for
stylistic confounds the way Confounds 1–2 above already did once.

It found (and fixed) three more confounds this way — the same
`Clarifications:`-header bug reappearing via a stale file reference, a
last-token trailing/leading-whitespace artifact in Orchid-HEval, and a
group-leakage bug in the first cross-source generalization check — before
landing on a result in the same spirit as the K-task's: real,
statistically-significant in-distribution signal (AUROC≈0.75–0.77,
p≈0.01), but only weak-to-modest genuine cross-source transfer
(AUROC≈0.52–0.64 with plain logistic regression) once every known confound
is removed. A supervised-contrastive (SupCon) probe trained on raw,
un-PCA'd activations pushes cross-source transfer to ≈0.60 ± 0.01 (5-seed),
at a cost to pooled accuracy — a genuine, non-cherry-picked improvement over
every other probe/feature variant tried. Adding a 4th source
(Orchid-BCB-Expand, 976 more BigCodeBench problems, 2,745 rows) as a 4th
LODO fold left this picture unchanged: it transfers about as poorly as its
sibling Orchid-BCB (AUROC≈0.52–0.54), and the SupCon win holds up at the
same winning config on the larger 4-fold mean (0.597 vs. 0.560/0.556
baseline). Full writeup, including the concrete evidence for why one source
(Orchid-HEval) transfers better than the others and the full SupCon
ablation: **[FINDINGS.md's "Combined binary
dataset" section](FINDINGS.md#combined-binary-dataset-clarifycodebench--orchid--human-eval-comm)**.

Reproduce:

```bash
cd semantic_entropy_probes
python scripts/fetch_combined_sources.py
python scripts/build_combined_dataset.py
python src/extract_activations.py --data data/ambiguity_dataset_combined.jsonl --out-dir activations_combined
python scripts/train_binary_probes.py
python scripts/confound_tests.py
python scripts/improve_cross_transfer.py
python scripts/train_supcon_probe.py
python scripts/add_bcb_expand_fold.py  # fills in the orchid_bcb_expand LODO column only
```

**Two-source subset (clarifycodebench + orchid_heval only)**: dropping
orchid_bcb, orchid_bcb_expand, and humanevalcomm — the sources whose small
edit-fraction made them weak transfer partners — raises both the pooled
ceiling (AUROC 0.778/0.780) and the LODO floor (mean AUROC 0.616/0.599,
vs. 0.560/0.556 on all 5 sources). This also surfaced and fixed a real bug:
`cross_source_only` SupCon is structurally undefined for LODO with only 2
sources (each fold's train set has just 1 source, so the loss has zero valid
cross-source positive pairs and silently trains on no gradient signal) — a
first run mistakenly reported it as the best-ever cross-source result before
this was caught; with the bug fixed, that config is now skipped for LODO,
and plain LR/MLP (0.616/0.599) is the best cross-source number for this
subset, with SupCon roughly flat (0.607). Built by filtering
`activations_combined/` rather than re-extracting — see **[FINDINGS.md's
"Two-source subset" section](FINDINGS.md#two-source-subset-clarifycodebench--orchid_heval-only)**.

```bash
python scripts/build_ccb_heval_subset.py
python scripts/train_binary_probes.py --activations-dir activations_ccb_heval --data data/ambiguity_dataset_ccb_heval.jsonl --results-dir results_ccb_heval
python scripts/confound_tests.py --activations-dir activations_ccb_heval --data data/ambiguity_dataset_ccb_heval.jsonl --results-dir results_ccb_heval
python scripts/improve_cross_transfer.py --activations-dir activations_ccb_heval --results-dir results_ccb_heval
python scripts/train_supcon_probe.py --activations-dir activations_ccb_heval --results-dir results_ccb_heval
```
