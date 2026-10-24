# SEP-IE: Reading Ambiguity Off a Model's Hidden States

Can a small language model tell, in **one forward pass**, that a coding request
is ambiguous, and which clarifying question is worth asking about it?

This repo is the research code, data, and write-up behind that question. If you
are new to it, read **[The problem](#the-problem)** first. If you want to run or
navigate the code, skip to **[The repo](#the-repo)**.

---

# The problem

## 1. Coding prompts are often underspecified

Coding assistants are now agents. They get a natural-language task, then plan,
write, run, and revise code with little human input. A common failure is that
the request does not pin down the behavior the user wants. For example:

> *"Sort the records by date."*
> Ascending or descending? The model may well represent that uncertainty
> internally, yet it will still emit one concrete implementation.

A one-shot generation silently commits to one reading. If it is the wrong one,
the code looks fine and fails the hidden tests. The standard fix is to let the
agent **ask a clarifying question before writing code**. That turns one
generation step into a short loop:

```
specification -> hidden-state uncertainty -> targeted clarification -> updated specification -> reduced uncertainty
```

In more operational terms, the loop is:

```
detect ambiguity -> propose candidate questions -> pick one -> get an answer -> write code
```

This raises two sub-problems:

| Sub-problem | Question |
|---|---|
| **Detection** | Is this prompt ambiguous enough to interrupt the user for? |
| **Ranking** | Given several candidate questions, which one resolves the ambiguity that matters most? |

## 2. The usual approach is expensive

The standard way to estimate uncertainty is **semantic entropy**: sample the
model many times and measure how much the answers or programs disagree. It
works, but it has two drawbacks:

1. **Cost.** Every sample is another autoregressive generation, and the cost
   scales with the number of samples drawn from the *coding* model, usually the
   largest and most expensive model in the pipeline. Every decision to ask a
   question costs several full code generations before any code exists.
2. **Conflation.** Disagreement between samples mixes up *specification
   ambiguity* with ordinary *coding errors*. Two programs can differ because
   the request was unclear or because the model simply got one wrong.

## 3. The idea tested here

**Semantic Entropy Probes (SEPs)** (Kossen et al., OATML) train a small linear
probe on a model's hidden states to predict semantic entropy directly. This
avoids sampling at inference time. This repo adapts that idea to a different
target:

- not the entropy of a model's *outputs*, but the ambiguity of an *input* coding prompt;
- not only detection, but also **ranking candidate clarifying questions**;
- using a small, decoupled model (**Qwen3-8B**) as the reader, whatever model
  later writes the code.

The cost argument is that a probe needs one forward pass of a small model. It
needs no autoregressive generation from the coding model.

The project started from the hypothesis that the hidden state already holds
information about *unresolved intent*, detectable before the model commits to an
implementation. It also started with two open questions, which shaped how the
work was done:

- **What should the target uncertainty be?** The repo tries several: the
  *count* of missing details (K), a binary clear/ambiguous label, the ambiguity
  *type*, and which candidate question best resolves the ambiguity.
- **How do you know the probe reads intent uncertainty and not a dataset or
  stylistic artifact?** This is the reason for the grouped splits, the
  length and bag-of-words baselines, the leave-one-source-out evaluation, and
  the many confound checks recorded in `FINDINGS.md`. Several early "strong"
  results turned out to be artifacts.

## 4. What the experiments found

The work moved through four stages. Each stage's headline result and caveat is
below. Most of the effort went into finding and removing confounds, so the
caveats matter as much as the numbers.

**Stage 1: does the hidden state encode ambiguity degree at all?**
Yes, moderately. On ClarifyCodeBench prompts with K ∈ {1, 2, 3} missing details,
a per-layer probe reaches Spearman ρ ≈ 0.31 (about 0.21 ± 0.05 in grouped CV)
at layer 32. Random is about 0.01 and a length-only baseline about 0.04. The
signal grows with depth. Every construction of a "fully specified" (K=0) class
that was tried inflated the result through a formatting confound (ρ up to
0.855), so the headline number deliberately excludes K=0.

**Stage 2: does it generalize beyond one benchmark?**
Only weakly. Across four independently built sources (ClarifyCodeBench,
Orchid-HumanEval, Orchid-BigCodeBench, Orchid-BCB-Expand), in-distribution AUROC
is about 0.75–0.77. Leave-one-source-out transfer is only about 0.52–0.64 once
confounds are removed. A supervised-contrastive probe helps modestly (about 0.60).
Transfer is best for sources whose edit is large relative to the prompt.

**Stage 3: can a probe rank clarifying questions?**
Close to a zero-shot baseline, but it does not beat it. On 1,229 groups under
leave-one-source-out splitting, the rank-1 hit rate (does the top pick match the
labeled ambiguity) is:

| | Probe (PCA + LR) | Zero-shot Qwen self-rank |
|---|---|---|
| LODO mean | 0.637 | **0.710** |
| Held-out ClarifyCodeBench | **0.701** | 0.663 |
| Held-out Orchid-BCB-Expand | 0.525 | **0.742** |

An earlier, smaller evaluation showed the probe winning. That result did not
survive the larger, stricter pool, and the write-up says so. The honest framing
is a **cost–quality tradeoff**, not an accuracy win. The cost measurement is
still provisional (see [Known gaps](#known-gaps)).

**Stage 4: does better ranking produce better code?**
Mostly no, but asking at all helps. Using real executable tests on 244 held-out
problems:

- Asking *any* clarifying question beats asking none: +4.0pp (round 1) and
  +6.3pp (rounds 1+2) in mean fractional pass rate, Wilcoxon p ≤ 0.002.
- *Which* question is asked matters much less than expected. Hitting the labeled
  ambiguity shows no significant effect on the all-or-nothing pass metric. On the
  fractional pass rate there is a +9.9pp gain (p = 0.035) on Orchid-sourced
  problems only.
- Most problems are underspecified in several independent ways at once, so
  resolving one or two caps the achievable gain.
- A separate cheap fix helps: restoring parameter types, requirements, and an
  example to BigCodeBench-derived prompts (without re-leaking the injected
  ambiguity) raised round-1 mean fractional pass from 0.796 to 0.863 on 106
  problems.

## 5. Where to read more

| You want... | Read |
|---|---|
| The condensed story as a paper draft | [`semantic_entropy_probes/paper/main.pdf`](semantic_entropy_probes/paper/main.pdf) (source in `paper/sections/`) |
| Every experiment, confound, bug, and reversal, in order | [`semantic_entropy_probes/FINDINGS.md`](semantic_entropy_probes/FINDINGS.md) (about 4,500 lines) |
| Stage 1 setup in detail (data, extraction, probing) | [`semantic_entropy_probes/README.md`](semantic_entropy_probes/README.md) |
| The underlying benchmark | [`ClarifyCodeBench/README.md`](ClarifyCodeBench/README.md) |

`FINDINGS.md` is organized by experiment version. Its headings run: the K-probe
and its confounds, ambiguity-type probes, the combined binary dataset, then
`v12` onward (ranking probes, rejudging, round-specific probes, code-vs-tests
evaluation, prompt augmentation). Where a later section corrects an earlier one,
the later section says so.

---

# The repo

## Layout

```
SEP-IE/
├── ClarifyCodeBench/            # Git submodule (upstream benchmark): 419 underspecified LiveCodeBench tasks
│   ├── data/ClarifyCodeBench.jsonl
│   └── clarifycodebench/        # Interactive eval protocol, TKQR/ORA metrics
└── semantic_entropy_probes/     # All the research code of this project
    ├── src/                     # Reusable library: extraction, splits, probes, SupCon, baselines
    ├── scripts/                 # One script per experiment step, named by version (v10 ... v29)
    ├── data/                    # Datasets, candidate pools, oracle answers, judged labels, codegen outputs
    ├── activations*/            # Cached Qwen3-8B hidden states (+ metadata.jsonl) per dataset variant
    ├── results*/                # Metrics, CSVs, plots per experiment
    ├── models/                  # Trained probes (.joblib + .json sidecars)
    ├── logs/                    # Run logs for the longer jobs
    ├── paper/                   # LaTeX draft (IEEEtran)
    ├── README.md                # Stage 1 write-up
    └── FINDINGS.md              # Full lab notebook
```

## Data sources

| Source | What it is | Edit type |
|---|---|---|
| [ClarifyCodeBench](https://arxiv.org/abs/2607.00711) | 419 human-edited LiveCodeBench v6 tasks with 1–3 annotated ambiguity points and ground-truth answers | Deletion |
| [Orchid](https://huggingface.co/datasets/SII-YDD/Orchid) (HumanEval, BigCodeBench, BCB-Expand) | Programmatic ambiguity injection | Lexical / vagueness substitution |
| [human-eval-comm](https://github.com/jie-jw-wu/human-eval-comm) | LLM-rewritten incomplete prompts | Incompleteness |

## How the scripts map to the stages

Scripts are numbered by experiment version and are meant to be read alongside
the matching `FINDINGS.md` section.

| Stage | Scripts | Notes |
|---|---|---|
| 1. K-probe | `fetch_livecodebench.py`, `build_dataset.py`, `src/extract_activations.py`, `train_probes.py`, `plot_results.py` | Reproduce commands are in `semantic_entropy_probes/README.md` |
| 2. Combined binary / cross-source | `fetch_combined_sources.py`, `build_combined_dataset.py`, `train_binary_probes.py`, `confound_tests.py`, `improve_cross_transfer.py`, `train_supcon_probe.py` | Confound tests are the point of this stage |
| 3. Ranking probes | `v12_*` (pairwise ranking, live loop), `v13`/`v14` (rejudging), `v15`–`v19` (retraining, LODO, SupCon, zero-shot control, cost) | Zero-shot control is `v19_control_qwen_selfrank.py` |
| 3b. Round-specific probes | `v20*` | `v20e` fixes a label-dropping bug; its numbers supersede `v20`–`v20d` |
| 4. Code vs. real tests | `v21`–`v27` | Fractional pass-rate analysis in `paper_recompute_fractional_pass.py` and `v23_significance_analysis.py` |
| 4b. Prompt augmentation | `v28`/`v29` | `v29` corrects `v28` |

`src/` holds the shared pieces: grouped splits (`splits.py`, `combined_splits.py`),
probes (`probes.py`, `torch_probe.py`, `torch_mlp_probe.py`), supervised
contrastive training (`supcon.py`), the OATML-style probe port (`oatml_sep.py`),
and text-only baselines (`text_baselines.py`).

## Setup

- **Clone with the benchmark:** `ClarifyCodeBench/` is a git submodule pinned to
  a specific upstream commit. Use `git clone --recurse-submodules <url>`, or run
  `git submodule update --init` after a plain clone. Without it, the Stage 3+
  scripts that read `ClarifyCodeBench/data/ClarifyCodeBench.jsonl` will not find it.
- **Python:** `semantic_entropy_probes/requirements.txt` lists `torch`,
  `transformers`, `accelerate`, `datasets`, `scikit-learn`, `numpy`, `matplotlib`.
  There is no checked-in virtualenv.
- **GPU:** activation extraction runs `Qwen/Qwen3-8B` in bf16. It fits a 24GB
  card (e.g. A10G). On a 16GB T4 it only fits with CPU offload, which removes
  the speed advantage the cost argument depends on.
- **Probing is cheap:** once activations are cached, `train_probes.py` never
  re-runs the model, so re-running with another seed or `--min-k` is fast.
- **API keys (Stage 3 onward):** elicitation, oracle answering, judging, and
  DeepSeek code generation call external LLM APIs (Groq, OpenRouter, AWS
  Bedrock, DeepSeek). Keys live in a gitignored `.env`; you must supply your own.

## Quick start: reproduce Stage 1

```bash
cd semantic_entropy_probes
python scripts/fetch_livecodebench.py
python scripts/build_dataset.py
python src/extract_activations.py --data data/ambiguity_dataset.jsonl --out-dir activations
python scripts/train_probes.py --activations-dir activations --min-k 1 --results-dir results_k1   # the load-bearing run
python scripts/plot_results.py --results-dir results_k1
```

The `--min-k 1` run is the one that answers the research question. Running
`--min-k 0` shows the K=0 confound, which is useful for understanding it but
should not be reported alone. Later stages are listed with their commands in the
"Reproduce" blocks of `FINDINGS.md`.

## Conventions worth knowing

- **Grouped splits everywhere.** Many rows share one source problem, so every
  split is grouped by `source_problem_id`. Otherwise a probe can score well by
  recognizing the problem instead of judging its ambiguity.
- **Last-token residual stream.** Features are the hidden state at the final
  token of the fully chat-templated prompt (Qwen3 non-thinking mode, which emits
  an *empty* think block), for every layer. Layer choice happens at probing time.
- **Confound-hunting is part of the method.** Stylistic cues (a `Clarifications:`
  header, missing Sample I/O sections, trailing whitespace, a leaked
  `</original>` template tag) can each fake an "ambiguity" signal. When a result
  looks too good, check `FINDINGS.md` for which confound was ruled out and how.
- **Superseded results stay in the notebook.** Several numbers were later
  corrected (see the `v20e`, `v26`, and `v29` sections). Treat the latest
  section on a topic as authoritative.

## Known gaps

Collected from the paper draft and `FINDINGS.md` caveats:

- The inference-cost comparison is **provisional**: it times a bare-prompt probe
  pass against zero-shot generation, not the batched N-candidate extraction the
  deployed ranker actually runs. A fresh benchmark is pending.
- The chained-miss arm of the prompt-augmentation experiment needs the original
  GPU environment and has not been re-run.
- Some Stage 3 steps (oracle answering, hand-checked judging) were done through
  Claude sub-agents rather than a single deterministic script, so they cannot be
  replayed end to end from the scripts alone.
- Small subsets (for example `orchid_bcb`, n=21 in the code-vs-tests analysis)
  give suggestive but not conclusive statistics.
- The paper draft still has placeholder author and affiliation fields.

## Size and storage

Cached hidden-state activations (`activations*/` and the per-experiment
`activations/`, `round2_activations/`, `heval32*_activations/` folders under
`data/`) are gitignored and not part of the repo. They are large (several GB) and
fully regenerable: `src/extract_activations.py` and the
`scripts/v*_extract_activations.py` scripts rebuild them from the datasets in
`data/`, and probe training reads them from there. Git LFS is not used. Results,
labels, candidate pools and logs are tracked.

## Citing and licensing

- ClarifyCodeBench is MIT-licensed (`ClarifyCodeBench/LICENSE`). Its underlying
  problems and hidden tests follow the LiveCodeBench license.
- This repo has no top-level license file. Add one before reusing the code.
- If you use the benchmark, cite Fang et al., *ClarifyCodeBench*,
  [arXiv:2607.00711](https://arxiv.org/abs/2607.00711). The probing approach
  builds on Kossen et al.'s semantic entropy probes.
