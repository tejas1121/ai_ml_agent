"""Build the offline dashboard from the frozen experiment results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dashboard" / "index.html"
METHODS = [
    ("xgboost_default", "Default XGBoost"),
    ("autogluon", "AutoGluon"),
    ("plain", "LLM agent"),
    ("memory", "LLM agent + memory"),
    ("zoo", "No-LLM zoo"),
]
DATASET_LABELS = {
    "blood-transfusion-service-center": "Blood transfusion",
    "credit-approval": "Credit-approval",
    "credit-g": "Credit-g",
    "diabetes": "Diabetes",
    "ilpd": "ILPD",
    "kc1": "KC1",
    "phoneme": "Phoneme",
    "wdbc": "WDBC",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            return list(csv.DictReader(source))
    except OSError as exc:
        raise RuntimeError(f"Could not read required input {path.relative_to(ROOT)}: {exc}") from exc


def number(row: dict[str, str], column: str, source: Path) -> float:
    value = row.get(column, "")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Missing or invalid {column!r} in {source.relative_to(ROOT)}: {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite {column!r} in {source.relative_to(ROOT)}")
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
    print(f"OK: {message}")


def markdown_table(lines: list[str], heading: str) -> list[dict[str, str]]:
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    except StopIteration as exc:
        raise RuntimeError(f"Required section {heading!r} not found in summary.md") from exc

    table: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("#"):
            break
        if line.strip().startswith("|"):
            table.append(line)
        elif table:
            break
    if len(table) < 3:
        raise RuntimeError(f"Required Markdown table missing under {heading!r}")

    headers = [cell.strip() for cell in table[0].strip().strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in table[2:]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != len(headers):
            raise RuntimeError(f"Malformed table row under {heading!r}: {line}")
        rows.append(dict(zip(headers, cells)))
    return rows


def required_row(
    rows: list[dict[str, str]], selectors: dict[str, str], description: str
) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in selectors.items())]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {description} in summary.md; found {len(matches)}")
    return matches[0]


def parse_ci(value: str, description: str) -> list[float]:
    match = re.fullmatch(r"\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]", value)
    if not match:
        raise RuntimeError(f"Could not parse {description} confidence interval: {value!r}")
    try:
        return [float(match.group(1)), float(match.group(2))]
    except ValueError as exc:
        raise RuntimeError(f"Invalid {description} confidence interval: {value!r}") from exc


def trace_identity(path: Path) -> tuple[str, str, int]:
    match = re.fullmatch(r"(.+)_(plain|memory)_(\d+)\.jsonl", path.name)
    if not match:
        raise RuntimeError(f"Unexpected live trace filename: {path.name}")
    return match.group(1), match.group(2), int(match.group(3))


def load_trace(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        with path.open(encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read live trace {path.relative_to(ROOT)}: {exc}") from exc
    if not records or records[0].get("type") != "header":
        raise RuntimeError(f"Live trace has no header: {path.relative_to(ROOT)}")
    return records[0], records[1:]


def build_data() -> dict[str, Any]:
    agent_path = ROOT / "results" / "agent_results.csv"
    baseline_path = ROOT / "results" / "baselines.csv"
    zoo_path = ROOT / "results" / "zoo_results.csv"
    curves_path = ROOT / "results" / "analysis" / "curves.csv"
    summary_path = ROOT / "results" / "analysis" / "summary.md"
    agent_rows = read_csv(agent_path)
    baseline_rows = read_csv(baseline_path)
    zoo_rows = read_csv(zoo_path)
    model_names = sorted({row.get("model", "") for row in agent_rows if row.get("model", "")})
    if not model_names:
        raise RuntimeError("No model names found in results/agent_results.csv")
    seed_count = len({row.get("seed", "") for row in agent_rows})

    test_scores: dict[str, dict[str, list[float]]] = {
        dataset: {method: [] for method, _ in METHODS} for dataset in DATASET_LABELS
    }
    means_by_method: dict[str, list[float]] = {method: [] for method, _ in METHODS}
    dataset_scores: dict[str, dict[str, list[float]]] = {
        dataset: {method: [] for method, _ in METHODS} for dataset in DATASET_LABELS
    }

    def collect_score(rows: list[dict[str, str]], path: Path, allowed: set[str]) -> None:
        for row in rows:
            dataset, method = row.get("dataset", ""), row.get("method", "")
            if dataset not in DATASET_LABELS or method not in allowed:
                raise RuntimeError(f"Unexpected dataset/method in {path.relative_to(ROOT)}: {dataset}/{method}")
            dataset_scores[dataset][method].append(number(row, "test_auc", path))

    collect_score(agent_rows, agent_path, {"plain", "memory"})
    collect_score(baseline_rows, baseline_path, {"xgboost_default", "autogluon"})
    collect_score(zoo_rows, zoo_path, {"zoo"})

    datasets = []
    for dataset, label in DATASET_LABELS.items():
        row_means: dict[str, float] = {}
        for method, _ in METHODS:
            observations = dataset_scores[dataset][method]
            if not observations:
                raise RuntimeError(f"No test-AUC observations for {dataset}/{method}")
            row_means[method] = statistics.mean(observations)
            means_by_method[method].append(row_means[method])
        datasets.append({"key": dataset, "label": label, "values": row_means})
    means = {method: statistics.mean(values) for method, values in means_by_method.items()}

    rank_sums = {method: 0.0 for method, _ in METHODS}
    for dataset in datasets:
        ordered = sorted(dataset["values"].items(), key=lambda item: item[1], reverse=True)
        ranks = {}
        position = 1
        while position <= len(ordered):
            end = position
            while end < len(ordered) and ordered[end][1] == ordered[position - 1][1]:
                end += 1
            rank = (position + end) / 2
            for method, _ in ordered[position - 1 : end]:
                ranks[method] = rank
            position = end + 1
        for method, rank in ranks.items():
            rank_sums[method] += rank
    ranks = {method: rank_sums[method] / len(datasets) for method, _ in METHODS}

    # Use agent compute_seconds, baseline seconds, and zoo compute_seconds; these are compute times, not wall_seconds.
    compute_values: dict[str, list[float]] = {method: [] for method, _ in METHODS}
    for row in agent_rows:
        compute_values[row["method"]].append(number(row, "compute_seconds", agent_path))
    for row in baseline_rows:
        compute_values[row["method"]].append(number(row, "seconds", baseline_path))
    for row in zoo_rows:
        compute_values[row["method"]].append(number(row, "compute_seconds", zoo_path))
    compute_medians = {method: statistics.median(values) for method, values in compute_values.items()}

    summary_lines = summary_path.read_text(encoding="utf-8").splitlines()
    test_comparisons = markdown_table(summary_lines, "## Paired test-AUC comparisons")
    memory_plain = required_row(
        test_comparisons, {"agent": "memory", "comparator": "plain"}, "memory-minus-plain test-AUC comparison"
    )
    memory_zoo = required_row(
        test_comparisons, {"agent": "memory", "comparator": "zoo"}, "memory-minus-zoo test-AUC comparison"
    )
    cv_comparisons = markdown_table(summary_lines, "### Paired memory-minus-plain CV differences")
    cv_leads = {}
    for iteration in ("1", "6"):
        row = required_row(cv_comparisons, {"iteration": iteration}, f"iteration-{iteration} CV lead")
        cv_leads[iteration] = {
            "difference": float(row["mean_difference_memory_minus_plain"]),
            "ci": parse_ci(row["dataset_mean_bootstrap_95_ci"], f"iteration-{iteration} CV lead"),
        }

    curves = []
    for row in read_csv(curves_path):
        curves.append(
            {
                "iteration": int(number(row, "iteration", curves_path)),
                "plain": number(row, "plain_all_successful_non_noop", curves_path),
                "memory": number(row, "memory_all_successful_non_noop", curves_path),
            }
        )
    if not curves:
        raise RuntimeError("No learning-curve data found in results/analysis/curves.csv")

    trace_files = sorted((ROOT / "results" / "traces").glob("*.jsonl"))
    run_lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    attempts = {
        method: {"attempts": 0, "errors": 0, "noops": 0, "useful": 0}
        for method in ("plain", "memory")
    }
    for path in trace_files:
        if "mock" in path.parts or "mock" in path.name.lower():
            continue
        dataset, method, seed = trace_identity(path)
        header, events = load_trace(path)
        if header.get("mode") != "live":
            continue
        if header.get("method") != method:
            raise RuntimeError(f"Method in trace header does not match filename: {path.name}")
        if dataset not in DATASET_LABELS:
            raise RuntimeError(f"Unknown dataset in trace filename: {path.name}")

        baseline_cv = next(
            (event.get("cv_auc") for event in events if int(event.get("iteration", -1)) == 0),
            None,
        )
        if method == "memory":
            context = header.get("memory_context")
            if not isinstance(context, str):
                raise RuntimeError(f"Memory trace has no string memory_context: {path.name}")
            if dataset.casefold() in context.casefold():
                raise RuntimeError(f"Target dataset {dataset!r} appears in memory_context for {path.name}")
        else:
            context = ""

        run_events = []
        for event in events:
            try:
                iteration = int(event.get("iteration", -1))
                attempt = int(event.get("attempt", -1))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid iteration/attempt in {path.name}") from exc
            cv_auc = event.get("cv_auc")
            if cv_auc is not None and not isinstance(cv_auc, (int, float)):
                raise RuntimeError(f"Invalid cv_auc in {path.name}, iteration {iteration}")
            compact = {
                "iteration": iteration,
                "attempt": attempt,
                "hypothesis": str(event.get("hypothesis", "")),
                "status": str(event.get("status", "")),
                "cv_auc": cv_auc,
                "noop": bool(event.get("noop", False)),
                "error": str(event.get("error", ""))[:180],
                "best_so_far": event.get("best_so_far"),
                "code": str(event.get("code", "")),
            }
            run_events.append(compact)
            if iteration >= 1:
                attempts[method]["attempts"] += 1
                # Timeouts are tracked separately from code errors in the traces and summary.
                attempts[method]["errors"] += compact["status"] == "error"
                attempts[method]["noops"] += compact["noop"]
                attempts[method]["useful"] += (
                    compact["status"] == "ok"
                    and not compact["noop"]
                    and cv_auc is not None
                    and baseline_cv is not None
                    and cv_auc - baseline_cv > 0.002
                )
        run_lookup[(dataset, method, seed)] = {
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "memory_context": context,
            "events": run_events,
        }

    result_by_run = {}
    for row in agent_rows:
        key = (row["dataset"], row["method"], int(row["seed"]))
        result_by_run[key] = {
            "test_auc": number(row, "test_auc", agent_path),
            "compute_seconds": number(row, "compute_seconds", agent_path),
            "prompt_tokens": int(number(row, "prompt_tokens", agent_path)),
            "completion_tokens": int(number(row, "completion_tokens", agent_path)),
        }
    runs = []
    for key, trace in sorted(run_lookup.items()):
        if key not in result_by_run:
            raise RuntimeError(f"No agent_results.csv row matches live trace {key}")
        runs.append({**trace, **result_by_run[key]})
    if len(run_lookup) != len(result_by_run):
        missing = sorted(set(result_by_run) - set(run_lookup))
        raise RuntimeError(f"Agent result rows without matching live traces: {missing[:3]}")

    require(len(datasets) == 8, "8 datasets")
    require(len([run for run in runs if run["method"] == "plain"]) == 24, "24 plain runs")
    require(len([run for run in runs if run["method"] == "memory"]) == 24, "24 memory runs")
    for method, expected_text, expected_value in [
        ("zoo", "1.75", 1.75),
        ("autogluon", "2.50", 2.50),
        ("plain", "2.9375", 2.9375),
        ("memory", "3.1875", 3.1875),
        ("xgboost_default", "4.625", 4.625),
    ]:
        require(
            math.isclose(ranks[method], expected_value, abs_tol=1e-9),
            f"mean rank {method} = {expected_text}",
        )
    require(
        (attempts["plain"]["errors"], attempts["plain"]["attempts"]) == (44, 180),
        "plain errors = 44 of 180",
    )
    require(
        (attempts["memory"]["errors"], attempts["memory"]["attempts"]) == (13, 156),
        "memory errors = 13 of 156",
    )
    require(
        (attempts["plain"]["useful"], attempts["plain"]["attempts"]) == (53, 180),
        "useful attempts plain = 53 of 180",
    )
    require(
        (attempts["memory"]["useful"], attempts["memory"]["attempts"]) == (67, 156),
        "useful attempts memory = 67 of 156",
    )

    return {
        "methods": [{"key": key, "label": label} for key, label in METHODS],
        "dataset_count": len(datasets),
        "seed_count": seed_count,
        "model_names": model_names,
        "datasets": datasets,
        "means": means,
        "ranks": ranks,
        "compute_medians": compute_medians,
        "reliability": attempts,
        "curves": curves,
        "comparisons": {
            "memory_plain": {
                "difference": float(memory_plain["mean_difference_paired_dataset_seed"]),
                "ci": parse_ci(
                    memory_plain["dataset_mean_bootstrap_95_ci"], "memory-minus-plain test-AUC"
                ),
            },
            "memory_zoo": {
                "difference": float(memory_zoo["mean_difference_paired_dataset_seed"]),
                "ci": parse_ci(memory_zoo["dataset_mean_bootstrap_95_ci"], "memory-minus-zoo test-AUC"),
            },
        },
        "cv_leads": cv_leads,
        "runs": runs,
    }


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>TabAgent results</title>
<style>
:root{--bg:#f4f6f8;--panel:#fff;--ink:#101828;--mut:#566074;--line:#d9dfe7;--plain:#1d4ed8;--mem:#b45309;--zoo:#0f766e;--base:#7a8394;--good:#0f766e;--bad:#b42318;box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0e1319;--panel:#151c25;--ink:#e8edf3;--mut:#9aa6b8;--line:#2a3441;--plain:#7aa2ff;--mem:#f0a35a;--zoo:#4cc9b0;--base:#8b95a7;--good:#4cc9b0;--bad:#ff8a7a}}
:root[data-theme="dark"]{--bg:#0e1319;--panel:#151c25;--ink:#e8edf3;--mut:#9aa6b8;--line:#2a3441;--plain:#7aa2ff;--mem:#f0a35a;--zoo:#4cc9b0;--base:#8b95a7;--good:#4cc9b0;--bad:#ff8a7a}
html{scroll-padding-top:calc(env(safe-area-inset-top,0px) + 56px)}*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
nav{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:var(--bg);border-bottom:1px solid var(--line)}
nav div{max-width:1080px;margin:auto;padding:12px 20px;display:flex;gap:22px;align-items:center;overflow-x:auto;white-space:nowrap}
nav b{font-family:Georgia,"Iowan Old Style",serif;font-size:18px;margin-right:auto}
nav a{color:var(--mut);text-decoration:none;font-size:14px}nav a:hover,nav button:hover{color:var(--ink)}
nav button,.control button{background:none;border:1px solid var(--line);color:var(--mut);border-radius:6px;padding:4px 10px;cursor:pointer;font:inherit;font-size:13px}
a:focus-visible,button:focus-visible,select:focus-visible,summary:focus-visible{outline:2px solid var(--plain);outline-offset:2px}
main{max-width:1080px;margin:auto;padding:0 20px 64px}
h1,h2{font-family:Georgia,"Iowan Old Style",serif;font-weight:600;letter-spacing:-.01em;margin:0}
h1{font-size:clamp(30px,5vw,46px);line-height:1.12;max-width:20ch;margin-top:48px}
h2{font-size:26px;margin-top:64px}
.lead{color:var(--mut);max-width:62ch;margin:14px 0 0}.sub{color:var(--mut);max-width:70ch;margin:6px 0 20px}
.plate{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:20px;margin-top:20px;overflow-x:auto}
.rank{margin-top:28px}.rank svg{width:100%;min-width:560px;height:auto;display:block}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:0;margin-top:28px;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.kpis div{padding:16px 20px 16px 0}.kpis strong{display:block;font:600 24px/1.2 Georgia,serif}.kpis span{color:var(--mut);font-size:14px}
table{border-collapse:collapse;width:100%;min-width:640px;font-variant-numeric:tabular-nums;font-size:14.5px}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}
th{font-weight:600;color:var(--mut);font-size:13px}td.best{font-weight:700}
tr.foot td{font-weight:600;border-bottom:0}tr.foot:first-of-type td{border-top:2px solid var(--ink)}
svg text{fill:var(--mut);font:12px system-ui,sans-serif}svg .gr{stroke:var(--line)}svg text.lb{fill:var(--ink);font-weight:600}
.leg{display:flex;gap:18px;flex-wrap:wrap;font-size:14px;color:var(--mut);margin-bottom:8px}
.leg i{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.bars{display:grid;gap:18px}.row b{display:block;font-weight:600;font-size:15px}.row small{color:var(--mut)}
.bar{display:flex;align-items:center;gap:10px;margin-top:6px;font-size:14px;font-variant-numeric:tabular-nums}
.bar u{display:block;height:14px;border-radius:2px;text-decoration:none;max-width:60%}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:40px;margin-top:20px}
.cols h3{font:600 17px Georgia,serif;margin:0 0 8px}
ul{margin:0;padding-left:20px;color:var(--mut)}li{margin:6px 0}
.controls{display:flex;gap:14px;flex-wrap:wrap}.control{display:grid;gap:5px;color:var(--mut);font-size:13px}
.control select{min-width:160px;max-width:100%;background:var(--panel);border:1px solid var(--line);border-radius:5px;color:var(--ink);padding:8px;font:inherit}
.run-meta{display:flex;gap:12px 24px;flex-wrap:wrap;color:var(--mut);font-size:14px;margin:16px 0}
.run-meta strong{color:var(--ink)}.reporting{border-left:3px solid var(--mem);padding:10px 14px;background:color-mix(in srgb,var(--mem) 8%,transparent)}
.reporting strong{display:block}.memory-context{white-space:pre-wrap;overflow-wrap:anywhere;max-height:260px;overflow:auto;border:1px solid var(--line);padding:12px;border-radius:4px;color:var(--mut);font-size:13px}
.run-table-wrap{overflow-x:auto}.run-table{min-width:980px}.run-table td{vertical-align:top}.run-table td.hyp{min-width:180px;max-width:260px;text-align:left;white-space:normal}
.run-table td.err{min-width:170px;max-width:240px;text-align:left;white-space:normal;color:var(--bad)}
.run-table details{min-width:125px;text-align:left}.run-table summary{cursor:pointer;color:var(--plain);font-size:13px}
pre{font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre;max-height:360px;max-width:min(72vw,760px);overflow:auto;background:var(--bg);border:1px solid var(--line);padding:12px;color:var(--ink)}
footer{margin-top:64px;padding-top:16px;border-top:1px solid var(--line);color:var(--mut);font-size:13px}
@media(max-width:520px){nav div{gap:14px;padding-inline:14px}main{padding-inline:14px}.plate{padding:14px}.kpis{grid-template-columns:1fr 1fr}.kpis div{padding-right:10px}.kpis strong{font-size:20px}.cols{grid-template-columns:1fr}}
</style>
</head>
<body>
<nav><div><b>TabAgent</b><a href="#accuracy">Accuracy</a><a href="#progress">Progress</a><a href="#cost">Cost</a><a href="#reliability">Reliability</a><a href="#explorer">Run explorer</a><a href="#caveats">Caveats</a><button id="theme" type="button" aria-label="Toggle color theme">Theme</button></div></nav>
<main>
<h1>Frozen results from a fixed model sweep.</h1>
<p class="lead" id="lead"></p>
<div class="plate rank"><div class="leg" style="margin-bottom:0">Mean rank across datasets, 1 is best</div><div id="rank"></div></div>
<div class="kpis" id="kpis"></div>
<h2 id="accuracy">Test AUC by dataset</h2>
<p class="sub">Mean over seeds. Shading runs from the lowest to the highest method in each row; bold marks the best.</p>
<div class="plate" id="heat"></div>
<h2 id="progress">Progress within a run</h2>
<p class="sub">Best cross-validation AUC so far, averaged over dataset-seed runs. Curves use the all-successful-non-noop series.</p>
<div class="plate"><div class="leg"><span><i style="background:var(--plain)"></i>LLM agent</span><span><i style="background:var(--mem)"></i>LLM agent + memory</span></div><div id="curve"></div></div>
<h2 id="cost">Cost against accuracy</h2>
<p class="sub">Median compute per run (log scale) against mean test AUC. Up and to the left is better.</p>
<div class="plate"><div id="cost-chart"></div></div>
<h2 id="reliability">Reliability</h2>
<p class="sub" id="reliability-sub"></p>
<div class="plate"><div class="leg"><span><i style="background:var(--plain)"></i>LLM agent</span><span><i style="background:var(--mem)"></i>LLM agent + memory</span></div><div class="bars" id="bars"></div></div>
<h2 id="explorer">Run explorer</h2>
<p class="sub">Inspect each recorded attempt and its generated code. Test AUC is shown only as a final reporting metric.</p>
<div class="plate">
<div class="controls">
<label class="control">Dataset<select id="run-dataset"></select></label>
<label class="control">Method<select id="run-method"><option value="plain">Plain</option><option value="memory">Memory</option></select></label>
<label class="control">Seed<select id="run-seed"></select></label>
</div>
<div id="run-summary"></div>
</div>
<h2 id="caveats">What the results do not show</h2>
<div class="cols">
<div><h3>Read with care</h3><ul><li>Memory did not improve final accuracy; the interval includes zero.</li><li>The no-LLM zoo was added after seeing the agent results, so it is a post-hoc control.</li><li>P-values are uncorrected and descriptive.</li></ul></div>
<div><h3>Scope</h3><ul><li id="scope-count"></li><li id="scope-model"></li><li>AutoGluon ran medium quality, tree models only.</li><li>Structured versus free-text memory and shuffled memory were not run.</li><li id="memory-pool"></li></ul></div>
</div>
<footer>Sources: frozen result CSVs, live traces and results/analysis summary. Held-out test AUC is reporting only.</footer>
</main>
<script id="dashboard-data" type="application/json">__DASHBOARD_DATA__</script>
<script>
const DATA=JSON.parse(document.getElementById("dashboard-data").textContent);
const $=s=>document.querySelector(s), esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const COLORS={xgboost_default:"var(--base)",autogluon:"var(--base)",plain:"var(--plain)",memory:"var(--mem)",zoo:"var(--zoo)"};
const fmt=(v,n=4)=>Number(v).toFixed(n), pct=(n,d)=>d?100*n/d:0, ci=a=>`[${fmt(a[0],4)}, ${fmt(a[1],4)}]`;
const methodLabel=key=>DATA.methods.find(m=>m.key===key).label;
const average=values=>values.reduce((a,b)=>a+b,0)/values.length;
function buildRank(){
 const W=900,H=190,left=40,right=40,x=v=>left+(v-1)/4*(W-left-right);let out=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Mean rank by method">`;
 for(let tick=1;tick<=DATA.methods.length;tick++)out+=`<line class="gr" x1="${x(tick)}" x2="${x(tick)}" y1="30" y2="155"/><text x="${x(tick)}" y="177" text-anchor="middle">${tick}</text>`;
 const order=[...DATA.methods].sort((a,b)=>DATA.ranks[a.key]-DATA.ranks[b.key]);
 order.forEach((m,i)=>{const y=40+i*26,rank=DATA.ranks[m.key];out+=`<line class="gr" x1="${left}" x2="${W-right}" y1="${y}" y2="${y}"/><circle cx="${x(rank)}" cy="${y}" r="8" style="fill:${COLORS[m.key]}"/><text class="lb" x="${x(rank)+14}" y="${y+4}">${esc(m.label)} ${fmt(rank,2)}</text>`});
 $("#rank").innerHTML=out+"</svg>";
}
function buildKpis(){
 const plain=DATA.reliability.plain,memory=DATA.reliability.memory,mp=DATA.comparisons.memory_plain,mz=DATA.comparisons.memory_zoo;
 const k=[
  [`${fmt(mp.difference,4)}`,`Memory minus plain test AUC; 95% CI ${ci(mp.ci)}.`],
  [`${fmt(mz.difference,4)}`,`Memory minus zoo test AUC; 95% CI ${ci(mz.ci)}.`],
  [`${fmt(DATA.cv_leads["1"].difference,4)} → ${fmt(DATA.cv_leads["6"].difference,4)}`,`Memory CV lead at iteration 1 (95% CI ${ci(DATA.cv_leads["1"].ci)}) and iteration 6 (95% CI ${ci(DATA.cv_leads["6"].ci)}).`],
  [`${fmt(DATA.compute_medians.zoo,1)} s vs ${fmt(DATA.compute_medians.plain,1)} s`,`Median compute: no-LLM zoo versus plain agent.`]
 ];
 $("#kpis").innerHTML=k.map(([value,label])=>`<div><strong>${esc(value)}</strong><span>${esc(label)}</span></div>`).join("");
}
function buildScope(){
 const models=DATA.model_names.map(esc).join(", ");
 $("#lead").textContent=`${DATA.dataset_count} binary OpenML-CC18 datasets, ${DATA.seed_count} seeds, ${DATA.model_names.length} local model${DATA.model_names.length===1?"":"s"}. These are frozen results; held-out test AUC is for reporting only and was never used for selection.`;
 $("#scope-count").textContent=`${DATA.dataset_count} small datasets and ${DATA.seed_count} seeds; the seed also changes the split.`;
 $("#scope-model").textContent=`${DATA.model_names.length===1?"One":"Several"} quantized model${DATA.model_names.length===1?"":"s"}: ${models}.`;
 $("#memory-pool").textContent=`Memory pool = ${DATA.dataset_count-1} other datasets per run, so “similar dataset” retrieval is limited.`;
}
function buildHeat(){
 let out="<table><thead><tr><th>Dataset</th>"+DATA.methods.map(m=>`<th>${esc(m.label)}</th>`).join("")+"</tr></thead><tbody>";
 DATA.datasets.forEach(row=>{const vals=DATA.methods.map(m=>row.values[m.key]),min=Math.min(...vals),max=Math.max(...vals);
  out+=`<tr><td>${esc(row.label)}</td>`+DATA.methods.map(m=>{const value=row.values[m.key],shade=max===min?0:Math.round((value-min)/(max-min)*34);return `<td${value===max?' class="best"':""} style="background:color-mix(in srgb,var(--plain) ${shade}%,transparent)">${fmt(value)}</td>`}).join("")+"</tr>";
 });
 out+='<tr class="foot"><td>Mean test AUC</td>'+DATA.methods.map(m=>`<td>${fmt(DATA.means[m.key])}</td>`).join("")+'</tr><tr class="foot"><td>Median compute (s)</td>'+DATA.methods.map(m=>`<td>${fmt(DATA.compute_medians[m.key],1)}</td>`).join("")+"</tr></tbody></table>";
 $("#heat").innerHTML=out;
}
const W=720,H=300,p={l:58,r:35,t:18,b:42};
function svgStart(label){return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;min-width:520px;height:auto" role="img" aria-label="${esc(label)}">`}
function buildCurve(){
 const rows=DATA.curves,lo=Math.min(...rows.flatMap(r=>[r.plain,r.memory])),hi=Math.max(...rows.flatMap(r=>[r.plain,r.memory])),pad=Math.max((hi-lo)*.16,.0001),ymin=lo-pad,ymax=hi+pad;
 const xmin=Math.min(...rows.map(r=>r.iteration)),xmax=Math.max(...rows.map(r=>r.iteration)),x=v=>p.l+(v-xmin)/(xmax-xmin||1)*(W-p.l-p.r),y=v=>H-p.b-(v-ymin)/(ymax-ymin)*(H-p.t-p.b);
 let out=svgStart("Learning curves");
 for(let i=0;i<=4;i++){const val=ymin+(ymax-ymin)*i/4;out+=`<line class="gr" x1="${p.l}" x2="${W-p.r}" y1="${y(val)}" y2="${y(val)}"/><text x="${p.l-8}" y="${y(val)+4}" text-anchor="end">${fmt(val,3)}</text>`}
 rows.forEach(r=>out+=`<text x="${x(r.iteration)}" y="${H-p.b+18}" text-anchor="middle">${r.iteration}</text>`);
 out+=`<text x="${(W+p.l)/2}" y="${H-6}" text-anchor="middle">Iteration</text>`;
 [["plain","var(--plain)"],["memory","var(--mem)"]].forEach(([key,color])=>{const vals=rows.map(r=>`${x(r.iteration)},${y(r[key])}`).join(" ");out+=`<polyline fill="none" stroke-width="2.5" style="stroke:${color}" points="${vals}"/>`+rows.map(r=>`<circle cx="${x(r.iteration)}" cy="${y(r[key])}" r="4" style="fill:${color}"><title>${key} ${r.iteration}: ${r[key]}</title></circle>`).join("")});
 $("#curve").innerHTML=out+"</svg>";
}
function buildCost(){
 const methods=DATA.methods,values=methods.map(m=>DATA.compute_medians[m.key]),min=Math.min(...values),max=Math.max(...values),lo=Math.pow(10,Math.floor(Math.log10(min))),hi=Math.pow(10,Math.ceil(Math.log10(max))),xmin=Math.log10(lo),xmax=Math.log10(hi);
 const x=v=>p.l+(Math.log10(v)-xmin)/(xmax-xmin||1)*(W-p.l-p.r),aucValues=methods.map(m=>DATA.means[m.key]),amin=Math.min(...aucValues),amax=Math.max(...aucValues),ypad=Math.max((amax-amin)*.18,.001),ymin=amin-ypad,ymax=amax+ypad,y=v=>H-p.b-(v-ymin)/(ymax-ymin)*(H-p.t-p.b);
 let out=svgStart("Median compute compared with mean test AUC");
 for(let tick=lo;tick<=hi;tick*=10){out+=`<line class="gr" x1="${x(tick)}" x2="${x(tick)}" y1="${p.t}" y2="${H-p.b}"/><text x="${x(tick)}" y="${H-p.b+18}" text-anchor="middle">${tick}</text>`}
 for(let i=0;i<=4;i++){const val=ymin+(ymax-ymin)*i/4;out+=`<line class="gr" x1="${p.l}" x2="${W-p.r}" y1="${y(val)}" y2="${y(val)}"/><text x="${p.l-8}" y="${y(val)+4}" text-anchor="end">${fmt(val,3)}</text>`}
 out+=`<text x="${(W+p.l)/2}" y="${H-6}" text-anchor="middle">Median compute per run, seconds (log scale)</text>`;
 methods.forEach(m=>{const px=x(DATA.compute_medians[m.key]),py=y(DATA.means[m.key]);out+=`<circle cx="${px}" cy="${py}" r="8" style="fill:${COLORS[m.key]}"/><text class="lb" x="${px+12}" y="${py+4}">${esc(m.label)}</text>`});
 $("#cost-chart").innerHTML=out+"</svg>";
}
function buildReliability(){
 const p=DATA.reliability.plain,m=DATA.reliability.memory;
 $("#reliability-sub").textContent=`Shares of all LLM attempts after iteration 0: ${p.attempts} for the plain agent and ${m.attempts} for the memory agent.`;
 const rows=[
  ["Candidate errors","Attempts with status error",p.errors,m.errors],
  ["No-op candidates","Attempts that repeated an earlier candidate or the incumbent score",p.noops,m.noops],
  ["Useful attempts","Status ok, not a no-op, and CV AUC more than 0.002 above that run’s iteration-0 CV AUC",p.useful,m.useful]
 ];
 $("#bars").innerHTML=rows.map(([title,desc,plain,memory])=>`<div class="row"><b>${esc(title)}</b><small>${esc(desc)}</small><div class="bar"><u style="width:${pct(plain,p.attempts)*4}px;background:var(--plain)"></u>Plain ${fmt(pct(plain,p.attempts),1)}% (${plain}/${p.attempts})</div><div class="bar"><u style="width:${pct(memory,m.attempts)*4}px;background:var(--mem)"></u>Memory ${fmt(pct(memory,m.attempts),1)}% (${memory}/${m.attempts})</div></div>`).join("");
}
function buildExplorer(){
 const datasets=[...new Set(DATA.runs.map(r=>r.dataset))],datasetSelect=$("#run-dataset"),methodSelect=$("#run-method"),seedSelect=$("#run-seed");
 datasetSelect.innerHTML=datasets.map(key=>`<option value="${esc(key)}">${esc(DATA.datasets.find(d=>d.key===key).label)}</option>`).join("");
 function updateSeeds(){
  const ds=datasetSelect.value,method=methodSelect.value,current=seedSelect.value;
  const seeds=DATA.runs.filter(r=>r.dataset===ds&&r.method===method).map(r=>r.seed).sort((a,b)=>a-b);
  seedSelect.innerHTML=seeds.map(seed=>`<option value="${seed}">${seed}</option>`).join("");
  if(seeds.includes(Number(current)))seedSelect.value=current;
  render();
 }
 function render(){
  const run=DATA.runs.find(r=>r.dataset===datasetSelect.value&&r.method===methodSelect.value&&r.seed===Number(seedSelect.value));
  if(!run){$("#run-summary").innerHTML="<p>No recorded run for this selection.</p>";return}
  const events=[...run.events].sort((a,b)=>a.iteration-b.iteration||a.attempt-b.attempt);
  let out=`<div class="run-meta"><span>Dataset: <strong>${esc(DATA.datasets.find(d=>d.key===run.dataset).label)}</strong></span><span>Method: <strong>${esc(run.method)}</strong></span><span>Seed: <strong>${run.seed}</strong></span></div>`;
  if(run.method==="memory")out+=`<h3>What the agent was told about other datasets</h3><pre class="memory-context">${esc(run.memory_context)}</pre>`;
  out+=`<div class="run-meta"><span>Compute: <strong>${fmt(run.compute_seconds,1)} s</strong></span><span>Prompt tokens: <strong>${run.prompt_tokens.toLocaleString()}</strong></span><span>Completion tokens: <strong>${run.completion_tokens.toLocaleString()}</strong></span></div>`;
  out+=`<div class="reporting"><strong>Final test AUC: ${fmt(run.test_auc)}</strong>Reporting only; never used for selection.</div>`;
  out+='<div class="run-table-wrap"><table class="run-table"><thead><tr><th>Iteration</th><th>Attempt</th><th>Hypothesis</th><th>Status</th><th>CV AUC</th><th>No-op</th><th>Short error</th><th>Best-so-far</th><th>Generated code</th></tr></thead><tbody>';
  out+=events.map(e=>`<tr><td>${e.iteration}</td><td>${e.attempt}</td><td class="hyp">${esc(e.hypothesis)}</td><td>${esc(e.status)}</td><td>${e.cv_auc==null?"—":fmt(e.cv_auc)}</td><td>${e.noop?"Yes":"No"}</td><td class="err">${esc(e.error||"—")}</td><td>${e.best_so_far==null?"—":fmt(e.best_so_far)}</td><td><details><summary>View code</summary><pre>${esc(e.code||"(No code recorded)")}</pre></details></td></tr>`).join("");
  out+="</tbody></table></div>";
  $("#run-summary").innerHTML=out;
 }
 datasetSelect.addEventListener("change",updateSeeds);methodSelect.addEventListener("change",updateSeeds);seedSelect.addEventListener("change",render);
 updateSeeds();
}
buildScope();buildRank();buildKpis();buildHeat();buildCurve();buildCost();buildReliability();buildExplorer();
$("#theme").onclick=()=>{const root=document.documentElement,dark=root.dataset.theme?root.dataset.theme==="dark":matchMedia("(prefers-color-scheme:dark)").matches;root.dataset.theme=dark?"light":"dark"};
</script>
</body>
</html>
'''


def render(data: dict[str, Any]) -> str:
    embedded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", r"<\/")
    return PAGE.replace("__DASHBOARD_DATA__", embedded)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or verify the offline results dashboard.")
    parser.add_argument("--check", action="store_true", help="compare current generated data without writing")
    args = parser.parse_args()
    try:
        expected = render(build_data())
        if args.check:
            try:
                actual = OUTPUT.read_text(encoding="utf-8")
            except OSError as exc:
                raise RuntimeError(f"Cannot check dashboard; {OUTPUT.relative_to(ROOT)} is missing: {exc}") from exc
            marker = '<script id="dashboard-data" type="application/json">'
            if marker not in actual:
                raise RuntimeError("Existing dashboard has no embedded dashboard-data JSON block")
            actual_data = actual.split(marker, 1)[1].split("</script>", 1)[0]
            expected_data = json.loads(expected.split(marker, 1)[1].split("</script>", 1)[0])
            try:
                parsed_actual = json.loads(actual_data)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Existing dashboard data is invalid JSON: {exc}") from exc
            if parsed_actual != expected_data:
                raise RuntimeError("Dashboard embedded data differs from current source results")
            print("OK: dashboard embedded data matches current source results (no files written)")
            return 0
        OUTPUT.write_text(expected, encoding="utf-8", newline="\n")
        print(f"Wrote {OUTPUT.relative_to(ROOT)}")
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
