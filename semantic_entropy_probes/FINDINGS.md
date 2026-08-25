# Semantic Entropy Probes — Findings

## Question

Does Qwen3-8B's residual stream encode the **degree of specification
ambiguity** in a coding prompt, from a single forward pass — without
sampling multiple generations and measuring disagreement between them?

**Target**: ambiguity count K ∈ {0,1,2,3}, from ClarifyCodeBench's own
annotations of missing specification details per task.

## Headline result

**Yes, moderately** — Spearman ρ ≈ 0.31 (test) / 0.21 ± 0.05 (5-fold grouped
CV) at the best layer, on the one dataset variant with **zero** construction
confound: the K∈{1,2,3} subset, where every prompt is `modified_content`
verbatim from ClarifyCodeBench, with no K=0 class at all. Three other
constructions of the K=0 class were tried, and all three inflate the
correlation by a different amount — the pattern across all four runs is
itself informative.

## Four runs, in order of decreasing K=0 confound

| Run | K=0 construction | Best layer | Test ρ | Unconverged layers | Length-only baseline ρ |
|---|---|---|---|---|---|
| `results/` | `modified_content` + appended `Clarifications:` answers | 20 | **0.855** | 36/37 | 0.21 |
| `results_lcbclean/` | Original LiveCodeBench text, Sample I/O stripped, reformatted to match | 20 | **0.526** | 10/37 | 0.02 |
| *(no run — see below)* | Original LiveCodeBench text, unmodified | — | — | — | — |
| `results_k1/` / `results_lcbclean_k1/` | **No K=0 class** — K∈{1,2,3} only | 32 | **0.312** | **0/37** | 0.04 |

(The unmodified-LiveCodeBench-text run was tried conceptually first but never
executed as a full pipeline pass — its Sample I/O confound was caught by
inspection before spending GPU time on it; see "Confound 1" below.)

The last two rows (`results_k1` and `results_lcbclean_k1`) are two
independent runs on different activation caches that **reproduce each other
to 8 decimal places** (ρ = 0.31230907994874046 both times) — expected, since
both restrict to the same 419 K≥1 rows, same model, same deterministic
pipeline, and confirms the extraction/probing code has no hidden
nondeterminism.

## Confound 1: the `Clarifications:` header (results/)

The first K=0 construction appended ClarifyCodeBench's own annotated
clarification answers to the ambiguous prompt, under a literal
`Clarifications:` header — no K≥1 row has this header. Signatures that this
inflated the score rather than measuring real degree-sensitivity:

- **36 of 37 layers failed to converge** during logistic regression fitting
  — coefficients diverging toward infinity is the textbook symptom of
  near-perfectly-separable classes, exactly what a literal marker token
  produces.
- **Length-only baseline reached ρ≈0.21** (K=0 rows are always longer, since
  answers are appended).
- **The Spearman-vs-layer curve was flat (~0.82–0.85) starting at layer 1** —
  no depth trend, i.e. available immediately rather than built up through
  the network.

## Confound 2: LiveCodeBench-paired, cleaned (results_lcbclean/)

At the user's request, this was replaced with the actual paired LiveCodeBench
`question_content` for each source problem — but with the Sample Input/Output
sections stripped and blank-line formatting collapsed to match
ClarifyCodeBench's style (`clean_full_text()` in `build_dataset.py`,
validated against all 80 relevant AtCoder problems: 0/80 contain "Sample
Input" after cleaning, 0/80 have leftover double-newlines, and mean character
length is now within 3% of the K≥1 rows: 919 vs. 890).

This is a much cleaner comparison — length-only baseline drops to ρ≈0.02
(chance) — but a residual signal remains:

- **10 of 37 layers still failed to converge** (down from 36, but not zero).
- **The curve starts already elevated at layer 1 (ρ≈0.31)**, comparable to
  where the confound-free K≥1-only curve *ends up* at its peak (layer 32,
  ρ≈0.31) — i.e. some of this run's signal is available before the network
  has done much processing, unlike the K≥1-only curve which visibly builds
  from ρ≈0.08 at layer 1 up to its peak.
- All 80 K=0/K≥1 pairs were verified to be genuine deletion-only edits
  (>80% of the K≥1 version's lines appear verbatim in its K=0 counterpart, as
  ClarifyCodeBench's methodology claims), and length variance is similar
  between classes (stdev 396 vs. 372) — so this isn't a length or
  content-volume artifact.

**Working hypothesis**: the residual signal is an *editing-provenance* tell,
not an ambiguity-content tell. K=0 rows are AtCoder contest-setter prose,
untouched; K≥1 rows passed through ClarifyCodeBench's manual deletion
editing (removing a sentence and smoothing the surrounding text). Even
"deletion-only" editing can leave a detectable stylistic signature —
independent of whether the resulting content is actually ambiguous — and a
probe grouped-split by source problem would still generalize on this, since
the signature is a property of the *editing process*, not of any specific
problem's content.

## The load-bearing result: K∈{1,2,3} only, no K=0 (results_k1/)

Excluding K=0 entirely removes both confounds at the root — every row in
this comparison went through the *same* ClarifyCodeBench editing process, so
"was this edited" cannot vary and cannot be a confound. The only thing that
varies is *how much* was deleted (K), which is exactly the quantity of
interest.

| | Spearman ρ |
|---|---|
| Random baseline | 0.007 |
| Token-length-only baseline | 0.043 |
| **Best-layer probe (layer 32)** | **0.312** (test), 0.214 ± 0.053 (5-fold grouped CV) |

**0/37 layers unconverged** — clean fits throughout. The layer-wise curve
rises steadily from ~0.08 at layer 1 through a plateau around 0.18–0.25 in
the middle layers to ~0.27–0.31 in the last third of the network (layers
27–36) — the shape of a property being progressively constructed through
depth, not read off the input.

## Interpretation

1. **Hidden states do encode something about ambiguity degree**, beyond
   chance and beyond the two length/lexical confounds tested, in the one
   comparison where those confounds are structurally impossible (K≥1-only).
   The effect is real but moderate: ρ≈0.31, not near-perfect separability.
2. **Every K=0 construction tried inflates the correlation**, by an amount
   that shrinks as the construction gets cleaner (0.855 → 0.526 → excluded
   entirely) but doesn't reach zero until K=0 is dropped. This is evidence
   that "resolved vs. unresolved" and "editing artifact vs. original prose"
   are hard to fully separate with this benchmark's K=0 options — not
   evidence that no real degree signal exists (the K≥1-only result rules
   that out).
3. **Don't report the full-range numbers as the headline finding.** Both
   `results/` and `results_lcbclean/` are useful diagnostically (they show
   the confound shrinking as construction improves) but neither is a clean
   measurement of "does the residual stream encode ambiguity degree."

## An OATML-style binary probe, on the confound-free K∈{1,2,3} subset

[OATML's Semantic Entropy Probes](https://github.com/OATML/semantic-entropy-probes)
(Kossen et al., arXiv:2406.15927) trains a probe on single-forward-pass
hidden states to predict *semantic entropy* -- a property of the model's
output computed by sampling many generations and clustering them by meaning.
**This project does not compute or verify semantic entropy anywhere** (no
sampling, no generation, no clustering); the target throughout is K,
ClarifyCodeBench's annotated ambiguity-point count -- a property of the
*input*, found by a human diffing the prompt against its fully-specified
original. K and semantic entropy are plausibly related but this project does
not test that relationship; conflating the two would overclaim what's
actually measured here.

What *is* reused from OATML, faithfully (functions ported close to verbatim
into `src/oatml_sep.py` from their `train-latent-probe.ipynb`, not
reimplemented from the paper description) is their **method**: rather than
treating an ordinal target as unrelated classes, find the single binary
split that minimizes within-group variance (`best_split`), then train a
plain per-layer `LogisticRegression()` and evaluate with bootstrapped-CI
AUROC. Applied to K restricted to {1,2,3} (K=0 excluded -- same reasoning as
above), `best_split` picks threshold≈1.0: **K=1 ("low", 199→ 40 in the test
split) vs. K∈{2,3} ("high", 220→41)** -- the most balanced, lowest-variance
cut, chosen automatically rather than hand-picked.

Result (`results_oatml/`, TBG token position -- identical to what this
project already extracts, so no re-extraction was needed):

| | AUROC |
|---|---|
| Chance | 0.500 |
| **Best layer (4)** | **0.695** [90% CI: 0.594, 0.790] |
| 5-fold grouped CV at layer 4 | 0.599 ± 0.042 |

Only 2/37 layers failed to converge (`max_iter=1000`, up from sklearn's
default 100, which does not converge at this project's D=4096 scale -- the
one deliberate deviation from OATML's exact `LogisticRegression()` call).

**Shape is notably different from the ordinal (Spearman) result above.** The
multinomial/Spearman analysis peaked late (layer 32) and rose smoothly with
depth (0.08 → 0.31), the signature of a representation being built up
through the network. This binary probe peaks early (**layer 4**, AUROC
0.695) then *declines* to a plateau around 0.60–0.65 for the rest of the
network (see `results_oatml/layer_auroc.png`). An early peak that decays is
consistent with the K=1-vs-K≥2 boundary being partly readable from
lower-level features (e.g. sentence/clause count, surface complexity cues
that correlate with "how many things are unclear") rather than requiring the
same depth of processing the fine-grained ordinal signal does. This is a
plausible interpretation, not a confirmed one -- worth flagging as a
difference in what the two probes may be picking up on, not as evidence
either result is wrong.

## OATML-style probe, 3-class (raw K∈{1,2,3}, not binarized)

Not part of OATML's notebook (see `src/oatml_sep.py` module docstring: their
framework always collapses to a binary decision). This trains the same
`LogisticRegression()` architecture directly on the three raw K values, and
also adds a Spearman correlation metric (`spearman_vs_expected` -- true K vs.
the model's probability-weighted `E[K]`) since OATML's own evaluation
(accuracy/AUROC/log-loss) has no ordinal-aware metric and K is ordinal.

**GPU backend added.** `sklearn.LogisticRegression` is CPU-only (no
scikit-learn GPU backend exists); rather than pull in a separate
CUDA-version-matched RAPIDS/cuML install, `src/torch_probe.py` implements the
same model (one linear layer + softmax = literally logistic regression) fit
with Adam on GPU instead of lbfgs on CPU. This is the same convex objective,
different optimizer -- validated against sklearn on 3 sample layers before
being trusted for the full run (see `src/torch_probe.py`'s docstring): an
initial patience setting stopped Adam far too early (51-501 iterations) and
landed at chance-level results on some layers; after fixing the schedule
(`min_iter=500`, `patience=1000`, `max_iter=20000`), the two backends agree
closely (e.g. layer 20: sklearn ρ=0.211/AUC=0.611 vs. torch ρ=0.197/AUC=0.625)
while torch is **~15-80x faster** at this project's D=4096 scale (0.2-1.2s
vs. 15-90s per layer).

Result (`results_oatml_multiclass/`, backend=torch, K∈{1,2,3}, no K=0):

| | Spearman ρ |
|---|---|
| **Best layer (17)** | **0.365** (test), 0.282 (5-fold grouped CV) |

**0/37 layers unconverged.** The layer curve (`results_oatml_multiclass/
layer_spearman.png`) is noisier at the individual-test-split level than the
PCA-reduced result above (this fits directly in the full 4096-dim space, no
PCA -- matching OATML's own approach of never reducing dimensionality --
which is a higher-variance estimator on an 83-example test set), but the
grouped-CV band tells a consistent story: real signal, roughly stable around
ρ≈0.15-0.28 from the middle layers onward, in the same moderate-effect-size
range as every other confound-free measurement in this project (0.21-0.37
across the Spearman-based analyses; 0.60-0.70 AUROC across the binary
analyses) rather than a qualitatively different number.

**A CPU (sklearn) run of the same 3-class experiment was launched in
parallel as a full-scale backend cross-check** (not just the 3-layer spot
check above) -- see `results_oatml_multiclass_sklearn/` if present; a
multi-hour CPU run at this scale means it may still be in progress. Compare
its `run_summary.json` against `results_oatml_multiclass/run_summary.json`
before treating the two as confirming each other.

The trained best-layer probe is saved to `results_oatml_multiclass/
probe.pkl` (mirroring OATML's own "save trained probes for inference" cell)
-- loadable via `src/oatml_sep.load_probe()`.

## TBG vs SLT: does OATML's better-performing position (SLT) also win here?

OATML's paper reports SLT (second-to-last token, taken from the model's own
generated response) consistently outperforming TBG by a modest margin on
their QA tasks. We tested this directly on the same K∈{1,2,3} data, using a
short-form-constrained generation (see below) and extracting **both**
positions from the *same* generation for a true apples-to-apples comparison
-- and got the opposite result.

**Short-form generation setup.** A first attempt with no output constraint
had Qwen3-8B write full explanations + code + worked examples per prompt --
confirmed empirically: all 8/8 prompts in a smoke test hit a 400-token
budget without producing EOS. Per OATML's own Section 5, their best results
come from a short-form setting (few-shot prompted to answer "as briefly as
possible" -- AUROC 0.7-1.0 vs. long-form's noisier 0.4-0.8, because
long-form's late-layer hidden states are "preoccupied with predicting the
next token... unrelated to the semantic uncertainty of the overall answer").
This project's code-generation analogue is a system prompt directing
"code only, no explanation" (`SHORT_FORM_SYSTEM_PROMPT` in
`src/extract_slt_activations.py`) -- after which all 8 smoke-test prompts
completed naturally in 82-157 tokens, and only 75/419 (18%, proportional
across K, not label-correlated) hit the token cap on the full run.

**Result, two independent probe methodologies, both K∈{1,2,3}, no K=0:**

| Method | TBG (short-form prompt) | SLT | 
|---|---|---|
| Own pipeline (PCA+multinomial, Spearman) | **0.398** @ layer 7 | 0.178 @ layer 12 (11/37 unconverged) |
| OATML-style (raw 4096-dim, multiclass) | **0.287** @ layer 5, AUROC 0.601 | 0.153 @ layer 35, AUROC 0.574 |

TBG wins by a wide margin in both -- the opposite of OATML's own finding.
The shape is also qualitatively different from everything else in this
project: TBG-short-form **peaks sharply early (layers 5-7)** then declines to
a low plateau (~0.05-0.15) for the rest of the network
(`results_shortform_tbg/layer_spearman.png`), unlike the bare-prompt TBG
result's smooth late rise (peak layer 32) or the OATML-binary result's
similar early-peak-then-plateau shape. SLT
(`results_shortform_slt/layer_spearman.png`) is low and flat throughout,
rarely clearing 0.15, indistinguishable in shape from noise.

**Why might this differ from OATML's finding?** Plausible, unconfirmed
hypotheses:
- *Target mismatch.* OATML's SLT position captures "how uncertain was I
  about the answer I just gave" for their own generated text. Here the
  target is K, an INPUT property (specification incompleteness) rather than
  an OUTPUT property (generation disagreement) -- see the "OATML-style binary
  probe" section above for why these aren't the same thing. TBG reads
  directly off the (still-ambiguous) input; SLT reads off code the model
  wrote *after* silently resolving the ambiguity one way, which may have
  scrubbed away the very information a K-probe needs.
- *Task mismatch.* OATML's tasks are short factual QA (Paris, a date, a
  name); ours is code generation, where the generated tokens are dominated
  by syntax (`def`, `for`, variable names) rather than content bearing on
  how ambiguous the original spec was.
- *Truncation.* 18% of SLT generations hit the token cap without producing
  a genuine EOS, so their "second-to-last token" is an artifact of the cutoff
  rather than OATML's intended "token before EOS."

This is a genuine, well-corroborated negative result for SLT in this
setting -- not a bug -- and a natural place for future work to dig further
(e.g. does a cleaner longer-budget SLT close the gap, or is TBG's advantage
here fundamentally about the target being an input- rather than
output-property).

## What this suggests for the next phase (not implemented here, per scope)

- The K≥1 signal (ρ≈0.3, peaking layers 27–36) is a plausible target for the
  eventual entropy-estimation work: if a linear probe partially recovers
  degree from a single forward pass, sampling-based entropy measurements are
  worth comparing against it as a check, not a replacement.
- A genuinely independent K=0 source — fully-specified coding prompts never
  touched by ClarifyCodeBench's editing pipeline (e.g. authored fresh, or
  drawn from a different benchmark entirely) — would be needed to test
  "resolved vs. unresolved" without the editing-provenance confound
  identified here. Out of scope for this pass.

## Reproducibility

Four results directories, all built from the same two activation caches (no
re-extraction needed to reproduce any of them):

- `activations/` (838 rows: 419 K≥1 + 419 K=0-via-Clarifications-header)
  → `results/` (`--min-k 0`, confounded) and `results_k1/` (`--min-k 1`,
  load-bearing).
- `activations_lcbclean/` (499 rows: 419 K≥1 + 80 K=0-via-cleaned-LiveCodeBench)
  → `results_lcbclean/` (`--min-k 0`, less confounded but not clean) and
  `results_lcbclean_k1/` (`--min-k 1`, reproduces `results_k1/` exactly).

Each results directory contains `layer_metrics.csv`, `layer_cv_spearman.csv`,
`baselines.json`, `run_summary.json`, and `layer_spearman.png`.

`results_oatml/` (`python scripts/train_oatml_style_probe.py --mode binary`)
reuses the same `activations/` cache, restricted to K≥1 in-script -- no
separate extraction. Contains `layer_metrics.csv` (with bootstrap CI
columns), `run_summary.json`, and `layer_auroc.png`.

`results_oatml_multiclass/` (`--mode multiclass --backend torch`) is the
3-class experiment on the confound-free K∈{1,2,3} subset. A parallel
`--backend sklearn` run of the same experiment was used only to spot-check
3 layers against torch (see src/torch_probe.py's docstring: layer 20 agreed
closely, ρ=0.211 sklearn vs. 0.197 torch) -- a full 37-layer sklearn run was
started but abandoned partway through once the spot-check confirmed
agreement, since it would have taken hours on raw 4096-dim features with no
PCA (see "Why torch is not always faster" below for when the raw-4096-dim
case differs from the PCA-reduced case). Each `results_oatml*` directory
also writes `probe.pkl` (the best layer's fitted model + metadata, loadable
via `src/oatml_sep.load_probe()`) and `layer_spearman.png` / `layer_auroc.png`.

`results_shortform_tbg/` and `results_shortform_slt/` (own PCA+multinomial
pipeline) and `results_oatml_shortform_tbg/` / `results_oatml_shortform_slt/`
(OATML-style, raw 4096-dim) are the TBG-vs-SLT comparison above, built from
`activations_shortform-tbg/` and `activations_shortform-slt/`
(`src/extract_slt_activations.py` -- both positions extracted from the same
greedy generation, short-form system prompt).

### Why torch is not always faster

`scripts/train_probes.py --backend torch` exists (src/torch_probe.py slots
into the same sklearn Pipeline as the PCA step) but is NOT what was used for
the shortform TBG/SLT runs above -- empirically, on this pipeline's
PCA-reduced 100-dim features, torch was *slower* than sklearn (17s vs. 13s
for one layer's full fit+CV sweep) because per-step Adam/GPU overhead
dominates once the problem is already small enough for CPU lbfgs to be fast.
GPU only wins decisively for the OATML-style port's raw, un-reduced 4096-dim
fits (15-80x faster there). Separately, the very first shortform run on
`sklearn` (default) backend appeared to hang at layer 0 for minutes; the
actual cause was BLAS thread oversubscription (a single process pinning all
32 cores for many small sequential fits, confirmed via `load average: 37.94`
on a 32-core machine) -- setting `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1` fixed it (1.28s for the same fit that had been hanging),
and `scripts/train_probes.py` / `scripts/train_oatml_style_probe.py` should
generally be run with these set when using the sklearn backend.

## Apples-to-apples: our pipeline vs. the OATML port, same binary task

`results_oatml/`'s binary-mode run (K=1 "low" vs. K∈{2,3} "high", via
OATML's own `best_split`/`binarize_entropy`, raw 4096-dim TBG features,
fixed C=1.0) reached **AUROC=0.6945 at layer 4**. Running our own
PCA(100)+LogisticRegression pipeline (`src/probes.py`'s `fit_and_eval`, C
swept by validation Spearman) on the *identical* binary target and the
*identical* bare-prompt TBG activation cache reaches **AUROC=0.7128 at
layer 18** — our pipeline wins by +0.018 on a true head-to-head. The most
likely explanation is methodological, not a bug in either: PCA whitening
plus a tuned C handles the D≫N (4096-dim, ~350-train-row) regime better than
raw features at a fixed default C. (This comparison also caught and fixed a
real bug: `src/probes.py`'s `_safe_auroc_ovr` unconditionally called
`roc_auc_score(..., multi_class="ovr", ...)`, which sklearn rejects for a
genuinely-binary `y_true` with a 2-column probability matrix — silently
swallowed into NaN by the surrounding `except ValueError`. Fixed by
branching to plain binary `roc_auc_score(y_true, probs[:, 1])` when
`len(classes) == 2`. This path is only exercised by genuinely-binary
targets; every previously-reported multiclass K result used 3-4 classes and
was unaffected.)

## Ambiguity TYPE probing (src/type_probes.py) — a different question from K

Everything above asks "how much ambiguity" (K). This is a separate
question: "what *kind*" — `ambiguity_types` in the dataset metadata is a
list of category tags drawn from 11 possibilities (Output Format,
Terminology, Behavior, Edge Cases, Indices & Ranges, Collection Semantics,
Comparison Rules, Ordering & Atomicity, Units, Numerical Precision, String &
Localization). A K=1 row has exactly one tag, verified: 11/838 rows
dataset-wide have `label != len(ambiguity_types)`, none of them K=1, so the
"K=1 rows have exactly one type" invariant holds cleanly for Stage 1 below.
K=2/K=3 rows are annotated to have *that many* types **in most cases but not
all** — the same 11-row mismatch includes some K=2/K=3 rows (5 of the 152
kept K=2 rows and 4 of the 43 kept K=3 rows have fewer in-vocab tags than
their K label; `multilabel_subset` doesn't assert `len(types) == k`, it just
uses whatever tags are actually present). This is a pre-existing annotation
artifact in ClarifyCodeBench itself, not a bug in this code — it means a
small minority of Stage-2 rows have a positive-label count below their K,
not that the multi-hot targets are wrong for those rows.

Staged by K per request, reusing the bare-prompt TBG `activations/` cache
(same one used for the K-count results above — no new extraction needed):

- **Stage 1 (K=1)**: ordinary nominal multiclass — which single type. No
  ordinal structure ("Output Format" isn't "more" than "Terminology"), so
  this drops Spearman/expected-value scoring (unlike `src/probes.py`) in
  favor of accuracy / macro-F1 / macro-AUROC (per-class OVR, averaged only
  over classes present in that test split — see module docstring for why).
- **Stage 2 (K=2, K=3)**: multi-label, binary relevance — one independent
  present/absent logistic head per type, sharing one PCA projection per
  layer/split. Run **separately** per K (not pooled), per explicit user
  choice, using the *same fixed type vocabulary* computed from K=1 support
  so results are comparable across K.

**Rare-type handling** (explicit user decision, applied consistently across
all three stages): 5 of 11 types have too few K=1 examples to survive
grouped splitting (Ordering & Atomicity=5, Units=3, Numerical Precision=2,
String & Localization=1, out of 199 K=1 rows). Rows carrying these types are
**dropped**, not collapsed into "Other" — for K=2/K=3 this means dropping a
row if *any* of its types falls outside vocab (keeping it would silently
mislabel it: a K=2 row with one in-vocab and one dropped type would look
like a clean single-positive-label row when it actually contains a second,
unmodeled ambiguity). This yields a fixed 7-type vocabulary: Output Format,
Terminology, Behavior, Edge Cases, Indices & Ranges, Collection Semantics,
Comparison Rules (≥8 K=1 examples each, 188/199 K=1 rows retained).

**Splitting**: grouped by `source_problem_id` (same leakage rationale as
`src/splits.py`) via plain `GroupShuffleSplit`/`GroupKFold` — unstratified,
since within a fixed-K subset there's no scalar label left to stratify by
(Stage 2's target is a multi-hot vector; sklearn has no built-in multi-label
group stratification).

**Baselines, and why macro-AUROC is the headline metric, not accuracy/F1**:
per user instruction, given how imbalanced and small these subsets are
(Output Format is ~30-40% of every K), majority-class and prevalence-matched
random baselines are computed alongside every probe run
(`majority_baseline_multiclass`/`random_baseline_multiclass` for Stage 1,
`majority_and_random_baseline_multilabel` for Stage 2) specifically so a
probe's F1/accuracy can't be mistaken for skill when a trivial baseline
already gets there by exploiting class frequency. Macro-AUROC is reported as
the primary number because it's threshold-free and prevalence-invariant — a
constant/majority predictor has no ranking information and scores
undefined/chance (NaN or 0.5) by construction, unlike accuracy or F1.

### Results (within-K, held-out-problem)

| Stage | n (kept/total) | Unique source problems | Best layer | Single-split macro-AUROC | Grouped-CV macro-AUROC | Length-only baseline | Majority baseline | Random baseline |
|---|---|---|---|---|---|---|---|---|
| K=1 (multiclass, `results_types_k1/`) | 188/199 | 78 | 16 | 0.602 (6/7 classes present in test) | **0.571 ± 0.052** | macro-AUROC=0.411 | acc=0.35, macro-F1=0.074 | acc=0.20, macro-F1=0.132, AUROC=0.5 |
| K=2 (multi-label, `results_types_k2/`) | 152/169 | 66 | 35 | 0.654 (7/7 types) | **0.597 ± 0.075** | macro-AUROC=0.510 | macro-F1=0.095, exact-match=0.029 | macro-F1=0.260, AUROC=0.5 |
| K=3 (multi-label, `results_types_k3/`) | 43/51 | 43 | 10 | 0.875 (6/7 types — Comparison Rules had 0 test positives) | **0.719 ± 0.043** | macro-AUROC=0.564 | macro-F1=0.269, exact-match=0.111 | macro-F1=0.393, AUROC=0.5 |

All three CV-macro-AUROC numbers now use the same metric (`cv_multiclass_macro_auroc`
for K=1, `cv_macro_auroc_multilabel` for K=2/K=3 — both grouped, both
per-class/per-type AUC averaged only over classes actually present in that
fold's test set) so they're comparable at face value. Caveat on that
comparability: `GroupKFold`'s folds are **deterministic given `n_splits`**
(no `shuffle`/`random_state` — the `seed` argument to `cv_multiclass_macro_auroc`/
`cv_macro_auroc_multilabel` is accepted but has no effect on which rows land
in which fold). The tight K=3 CV std (±0.043 despite ~8-row test folds) is
therefore variance across *one fixed partition's* folds, not evidence of
robustness to partition choice — don't read it as more reassuring than a
single train/val/test split would be. Length-only baselines are close to
chance everywhere (0.41-0.56), ruling out prompt-length as a confound for
any of the three stages, same as the K-count probes above.

### Cross-K transfer check — the trend this data does NOT support

The obvious reading of the table above — signal gets stronger from K=1
(0.57) to K=2 (0.60) to K=3 (0.72) — **should not be reported as a trend**,
though the reason turned out to be more subtle than first thought (see the
positive-control result below, which overturned the first interpretation
of this check).

**Why K=3 is suspect in principle**: K=1 rows average ~2.4 per source
problem and K=2 ~2.3 (multiple different rewrites of the same base problem,
so the type label varies *within* a group — verified directly: of K=1's 67
multi-row groups, 66 have varying labels across their rows; of K=2's 43
multi-row groups, **all 43** have varying type-pair labels, none identical).
**K=3's 43 kept rows come from 43 distinct problems — exactly one K=3 row
per problem.** Grouped CV at K=3 is therefore vacuous for forcing
deletion-specific signal: a model could in principle score well by learning
"what kind of problem is this" (topic/domain correlates with which
ambiguity types annotators tend to introduce for it) without ever having to
distinguish one deleted sentence from another. K=1 and K=2, by contrast,
have verified, strong within-group label contrast, so this specific concern
does not apply to their own within-stage numbers.

**The attempted discriminating test, and why it came back inconclusive**
(`scripts/cross_k_type_transfer.py`): train the same per-type binary heads
on K=1 rows restricted to source problems that share no group with the K=3
eval set, then evaluate on all 43 K=3 rows.

| Transfer | Train rows (groups) | Eval rows | Best-layer macro-AUROC | Range across layers |
|---|---|---|---|---|
| K=1 → K=3 (`results_types_cross_k1_to_k3/`) | 43 (35 problems) | 43 (all of kept K=3) | 0.566 (layer 18) | 0.44 – 0.57 |
| K=1 → K=2 (`results_types_cross_k1_to_k2/`) | 10 (8 problems) | 152 (all of kept K=2) | 0.500 (layer 0) | 0.44 – 0.53 |

Both land near chance. The K=1→K2 row was already flagged as underpowered
(only 8 group-disjoint problems). The K=1→K3 row's 43 training rows looked
large enough to be informative — comparable to K=3's own 25-row within-stage
train split — but **a positive control shows it isn't**
(`scripts/cross_k_positive_control.py`): evaluating the *exact same*
K=1-trained classifiers (same 43/16 train/val split, same seed) on the 129
K=1 rows held out of that training pool — same task, same feature
distribution, still group-disjoint from training, the only thing that
changed is which problems are being scored — reaches only **0.544 macro-AUROC
at its best layer**, barely above chance and well below the 0.571±0.052 a
properly-sized within-K1 split achieves. **The 43-row training set can't
reliably learn even the in-distribution K=1 task**, so its near-chance
performance on K=3 is uninformative — it would look identical whether or
not a transferable type signature exists. Both cross-K checks are therefore
**inconclusive given this dataset's size**, not evidence against transfer.

This also means the K=3-vs-confound question stays genuinely open, with (at
least) three live explanations for why 0.719±0.043 doesn't survive the
transfer attempt: (1) the problem-topic confound described above, (2) a
real signature that's specific to the *K=3 context* (three co-occurring
ambiguities altering the representation in a way a single-ambiguity-trained
model wouldn't recognize), or (3) plain distribution shift — the
StandardScaler+PCA basis is fit on K=1 activations (one sentence deleted)
and applied unchanged to K=3 activations (three sentences deleted, more
different from the source text), which could fail to span the right region
of activation space regardless of whether type information is present.
This check cannot distinguish between the three, and settling it would need
more group-disjoint K=1 data than this dataset provides (all 43 K=3
problems already overlap the 78-problem K=1 pool).

**Bottom line**: report the three within-K numbers (0.57 / 0.60 / 0.72) as
three separate, real-but-modest results, each beating its own length/majority/
random baselines — not as a monotonic trend. K=1 and K=2's own numbers are
well-supported by verified within-group label contrast; K=3's number is
reproducible on held-out K=3 problems but its source (deletion-specific
signal vs. problem-topic confound vs. K=3-specific context effect) is
unresolved, and the cross-K transfer checks that could have helped were both
underpowered on this dataset.

Per-type breakdown (`results_types_k2/layer_per_type.json`,
`results_types_k3/layer_per_type.json`) shows Output Format and Terminology
(the two most prevalent types) carry the most reliable within-K signal
across layers; Comparison Rules and Collection Semantics (rarest, most
prevalence-imbalanced in test splits) are the noisiest per-type AUROCs,
sometimes below 0.5 at individual layers — expected given how few positive
test examples they get at these sample sizes.

Code: `src/type_probes.py` (module docstring has full methodology),
`scripts/train_type_probe_k1.py`, `scripts/train_type_probe_multilabel.py --k {2,3}`,
`scripts/cross_k_type_transfer.py --eval-k {2,3}`, `scripts/cross_k_positive_control.py`.
