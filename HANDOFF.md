# Paper-writing handoff

This note summarizes the completed TabAgent experiment for collaborators
preparing the paper. The frozen result CSVs and committed traces are the record
of the experiment; analysis outputs are derived and can be regenerated.

## What was run

The study compares conventional tabular baselines, a plain LLM agent, the same
agent with a read-only experiment-memory pool, and a no-LLM candidate zoo.
There are eight binary OpenML-CC18 datasets and three seeds (0, 1, 2). Agent
candidates were selected using five-fold CV on each run's training split; the
held-out test split was reserved for the final score. The memory pool was built
from plain traces and retrieval excluded the target dataset.

### Result files

- `results/agent_results.csv` — one summary row per completed plain or memory
  agent run, including held-out score, compute, tokens, and failure counters.
- `results/baselines.csv` — per-dataset/seed results for default XGBoost and
  AutoGluon.
- `results/zoo_results.csv` — per-dataset/seed CV-selected zoo winner and its
  final held-out score.
- `results/traces/*.jsonl` — committed live plain/memory run headers and
  iteration-level attempts; this is the record of what the agents ran.
- `results/zoo_traces/*.jsonl` — per-candidate CV outcomes and code for the
  no-LLM zoo runs.
- `results/dev_artifacts/` — development artifacts, not part of the frozen
  experiment; do not use for analysis.
- `results/memory_pool.db` — a copy of the frozen `data/memory.db`, kept in the
  repository because `data/` is git-ignored. Its SHA-256 is
  `60E697B6881C66AC2CA8C1B044D591B148ABB495E4B67844BA5BC378F430EFCE`; it is
  the memory pool used by the memory runs. This copy is tracked by Git.

### Derived analysis files

- `results/analysis/summary.md` — all detailed report tables, methods,
  sensitivity analyses, and caveats.
- `results/analysis/curves.csv` — mean best-so-far CV AUC at iterations 0–6
  for plain and memory, with failed iterations carried forward.
- `results/analysis/curves.png` — plot of the full best-so-far CV curves.
- `results/analysis/curves_clean.csv` — successful/non-noop progress curves,
  including the early-clean subset.
- `results/analysis/curves_clean.png` — plot of full and early-clean
  successful/non-noop progress curves.
- `results/analysis/memory_plain_by_dataset.csv` — plain and memory test AUC
  means and paired per-seed differences for each dataset.
- `results/analysis/test_auc_by_dataset.png` — grouped test-AUC bars for the
  methods available in the frozen result files.
- `results/analysis/cost_vs_auc.csv` — plotted mean test AUC, dataset bootstrap
  interval, and median compute time for each method.
- `results/analysis/cost_vs_auc.png` — mean test AUC versus median compute
  time, with dataset-bootstrap error bars.

## Key findings

- **Mean rank across datasets (1 = best):** zoo 1.7500, AutoGluon 2.5000,
  plain 2.9375, memory 3.1875, and default XGBoost 4.6250. Source:
  `results/analysis/summary.md`, “Mean rank across datasets (1 = best)”.
- **Memory versus plain final test AUC:** mean paired difference (memory minus
  plain) -0.0009, dataset-level bootstrap 95% CI [-0.007678, 0.005133].
  Source: “Paired test-AUC comparisons”.
- **Memory versus zoo:** mean paired difference -0.0050, dataset-level
  bootstrap 95% CI [-0.007839, -0.002109]. The zoo's mean rank was 1.7500.
  Source: “Paired test-AUC comparisons” and “Mean rank across datasets”.
- **Error attempts:** plain had 44 errors in 180 attempts (0.2444); memory
  had 13 in 156 attempts (0.0833). These are per-attempt rates, not run rates.
  Source: “Agent failure rates”.
- **CV progress:** memory minus plain was +0.0156 at iteration 1 (95% dataset
  bootstrap CI [0.006003, 0.025644]) and +0.0021 at iteration 6 (CI
  [-0.000855, 0.004959]). Source: “Paired memory-minus-plain CV differences”.
- **Versus default XGBoost:** mean paired differences were +0.0197 for plain,
  +0.0188 for memory, and +0.0238 for zoo, from “Paired test-AUC comparisons”.
  For AutoGluon, the difference is +0.018536: computed from `results/baselines.csv`
  as the mean of the eight dataset-level test-AUC means for AutoGluon minus
  default XGBoost.
- **Zoo versus AutoGluon:** the mean paired difference (zoo minus AutoGluon)
  was +0.0053, with dataset-level bootstrap 95% CI [-0.000536, 0.012630].
  Source: “Paired test-AUC comparisons”.
- **Cost:** median compute seconds per run were 57.6284 for zoo, 15.9099 for
  AutoGluon, 251.1376 for plain, 210.9568 for memory, and 1.4136 for default
  XGBoost. Source: “Cost by method”.
- **Candidate quality:** among successful non-noop attempts with a recorded
  iteration-0 baseline, mean CV-AUC change was +0.0056 for plain and +0.0029
  for memory. Across all attempts, 53 of 180 plain attempts (0.2944) and 67
  of 156 memory attempts (0.4295) were valid, non-noop, and improved CV AUC by
  more than 0.002 over iteration 0. The mean deltas favor plain, while the
  all-attempt shares favor memory. Source: “Per-attempt candidate quality”.

## What the results do NOT show

- Memory did not improve final accuracy over plain; the paired estimate is
  near zero and its bootstrap interval includes zero.
- The LLM agents did not beat the fixed no-LLM zoo on final test AUC.
- The reported p-values are uncorrected and descriptive, not confirmatory.
- The zoo was added after seeing the agent results and is a post-hoc control.
- The experiment uses only eight small datasets and three seeds; the seed also
  changes the train/test split.
- The agent results represent one quantized 7B model.
- AutoGluon used a `medium_quality`, tree-only configuration and took about
  16 seconds median against a 300-second limit.
- Per-attempt analyses cannot determine whether memory proposes better ideas
  or only avoids invalid candidates.
- Objective 3 ablations (structured versus free-text memory, a memory-size
  curve, and shuffled memory) were **not run**.

## Disclosures that must appear in the paper

- `wdbc` seed 2 and `ilpd` seed 2 (memory) were truncated by system sleep and
  rerun with the same seeds after clearing the LLM cache. The original runs
  are kept in `backup_before_rerun/`.
- Plain `diabetes` seed 1 had a stalled LLM attempt lasting 873.1127 seconds.
  It was retained because plain traces form the memory pool; the sensitivity
  analysis excludes that dataset-seed pair.
- The no-LLM zoo was added after inspecting agent results and must be described
  as post-hoc.

## Do not cite without checking

Any reference list received from the project lead has not been verified.
Check every citation against the original publication and bibliographic record
before including it in the paper.

## Regenerating tables and figures

From the repository root, regenerate the detailed analysis tables and its
standard figures with:

```powershell
python scripts\summarize.py
```

Then regenerate the additional cost-versus-AUC figure and its plotted-data CSV
with:

```powershell
python scripts\make_figures.py
```

The summary command writes `summary.md`, `curves.csv`/`curves.png`,
`curves_clean.csv`/`curves_clean.png`, `memory_plain_by_dataset.csv`, and
`test_auc_by_dataset.png` under `results/analysis/`. The figure script writes
`cost_vs_auc.csv` and `cost_vs_auc.png` there. The curve figures show
training-only CV progress; the grouped bars show final test AUC by dataset;
the cost-versus-AUC scatter shows each method's median compute and mean
dataset-level test AUC with a 10,000-resample dataset bootstrap interval.
The cost-versus-AUC bootstrap uses fixed random seed `20261020`.
