from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


FREQUENCY_ORDER = {"monthly": 0, "quarterly": 1, "yearly": 2}
REMOVED_SYSTEMS = {
    "forecast_regime_oracle",
    "pattern_forecast_regime_oracle",
    "late_qrels_oracle_upper_bound",
    "pattern_late_qrels_oracle_upper_bound",
    "historical_backtest_profile",
    "pattern_historical_backtest_profile",
    "forecast_profile_early",
    "pattern_forecast_profile_early",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frequency_sort_key(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    return FREQUENCY_ORDER.get(name, 99), name


def discover_run_dirs(root: Path) -> list[Path]:
    if (root / "research_results.json").exists():
        return [root]
    if not root.exists():
        return []
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / "research_results.json").exists()],
        key=frequency_sort_key,
    )


def experiment_table(run_dir: Path, report: dict[str, Any], name: str) -> pd.DataFrame:
    csv_path = run_dir / f"{name}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    rows = report.get("experiments", {}).get(name, {}).get("results", [])
    return pd.DataFrame(rows)


def run_frequency(run_dir: Path, report: dict[str, Any]) -> str:
    params = report.get("base_configuration", {}).get("parameters", {})
    return str(params.get("frequency") or run_dir.name)


def metric_value(row: pd.Series, metric: str) -> float:
    return float(pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0])


def fmt(value: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def collect_dataset_rows(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for run_dir in run_dirs:
        report = load_json(run_dir / "research_results.json")
        params = report.get("base_configuration", {}).get("parameters", {})
        summary = report.get("data_summary", {})
        splits = summary.get("split_counts", {})
        lengths = summary.get("series_length", {})
        rows.append(
            {
                "frequency": run_frequency(run_dir, report),
                "n_series": summary.get("n_series"),
                "index": splits.get("index"),
                "dev": splits.get("dev"),
                "test": splits.get("test"),
                "min_len": lengths.get("min"),
                "median_len": lengths.get("median"),
                "max_len": lengths.get("max"),
                "forecast_h": params.get("forecast_h"),
                "pattern_len": params.get("pattern_len"),
            }
        )
    return rows


def collect_main_rows(run_dirs: list[Path], metric: str) -> list[dict[str, Any]]:
    rows = []
    for run_dir in run_dirs:
        report = load_json(run_dir / "research_results.json")
        freq = run_frequency(run_dir, report)
        df = experiment_table(run_dir, report, "exp1_pattern_main")
        if df.empty or metric not in df.columns or "system" not in df.columns:
            continue
        df = df[~df["system"].isin(REMOVED_SYSTEMS)].copy()
        random_metric = None
        random_rows = df[df["system"] == "random"]
        if not random_rows.empty:
            random_metric = metric_value(random_rows.iloc[0], metric)
        content = df[df["system"] != "random"].copy()
        if content.empty:
            continue
        content = content.sort_values(metric, ascending=False)
        best = content.iloc[0]
        runner = content.iloc[1] if len(content) > 1 else None
        best_metric = metric_value(best, metric)
        rows.append(
            {
                "frequency": freq,
                "best_system": best["system"],
                metric: best_metric,
                "runner_up": runner["system"] if runner is not None else "-",
                f"runner_{metric}": metric_value(runner, metric) if runner is not None else None,
                "random": random_metric,
                "delta_vs_random": best_metric - random_metric if random_metric is not None else None,
            }
        )
    return rows


def collect_whole_pattern_rows(run_dirs: list[Path], metric: str) -> list[dict[str, Any]]:
    rows = []
    for run_dir in run_dirs:
        report = load_json(run_dir / "research_results.json")
        freq = run_frequency(run_dir, report)
        df = experiment_table(run_dir, report, "exp2_whole_vs_pattern")
        if df.empty or metric not in df.columns:
            continue
        pivot = df.pivot_table(index="comparison", columns="approach", values=metric, aggfunc="first")
        row = {"frequency": freq}
        deltas = []
        for comparison in ["cosine", "dtw", "tsfel"]:
            if comparison not in pivot.index:
                continue
            whole = pivot.loc[comparison].get("whole-series")
            pattern = pivot.loc[comparison].get("pattern")
            delta = pattern - whole if pd.notna(whole) and pd.notna(pattern) else None
            row[f"{comparison}_delta"] = delta
            if delta is not None:
                deltas.append(float(delta))
        row["avg_delta"] = sum(deltas) / len(deltas) if deltas else None
        rows.append(row)
    return rows


def collect_length_rows(run_dirs: list[Path], metric: str) -> list[dict[str, Any]]:
    rows = []
    for run_dir in run_dirs:
        report = load_json(run_dir / "research_results.json")
        freq = run_frequency(run_dir, report)
        df = experiment_table(run_dir, report, "exp4_pattern_length_sensitivity")
        if df.empty or metric not in df.columns:
            continue
        df = df[~df["system"].isin(REMOVED_SYSTEMS)].copy()
        if df.empty:
            continue
        best = df.sort_values(metric, ascending=False).iloc[0]
        by_len = (
            df.groupby("pattern_len")[metric]
            .mean()
            .sort_index()
            .map(lambda x: fmt(x))
            .to_dict()
        )
        rows.append(
            {
                "frequency": freq,
                "best_len": int(best["pattern_len"]),
                "best_system": best["system"],
                metric: metric_value(best, metric),
                "mean_by_len": ", ".join(f"{k}:{v}" for k, v in by_len.items()),
            }
        )
    return rows


def collect_qrels_density_rows(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for run_dir in run_dirs:
        report = load_json(run_dir / "research_results.json")
        freq = run_frequency(run_dir, report)
        path = run_dir / "ir_qrels_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or "task" not in df.columns:
            continue
        sub = df[(df["task"] == "task2") & (df.get("split", "test") == "test")]
        if sub.empty:
            continue
        row = sub.iloc[0].to_dict()
        rows.append(
            {
                "frequency": freq,
                "n_queries": row.get("n_queries"),
                "mean_qrels": row.get("mean_positive_qrels_per_query"),
                "median_qrels": row.get("median_positive_qrels_per_query"),
                "mean_relevance": row.get("mean_relevance"),
                "rel1": row.get("relevance_1_rows"),
                "rel2": row.get("relevance_2_rows"),
                "rel3": row.get("relevance_3_rows"),
                "rel4": row.get("relevance_4_rows"),
            }
        )
    return rows


def collect_ir_robustness_rows(
    run_dirs: list[Path], main_rows: list[dict[str, Any]], metric: str
) -> list[dict[str, Any]]:
    best_by_freq = {row["frequency"]: row["best_system"] for row in main_rows}
    rows = []
    for run_dir in run_dirs:
        report = load_json(run_dir / "research_results.json")
        freq = run_frequency(run_dir, report)
        best_system = best_by_freq.get(freq)
        if not best_system:
            continue
        uncertainty_path = run_dir / "ir_system_uncertainty.csv"
        pairwise_path = run_dir / "ir_pairwise_random_tests.csv"
        if not uncertainty_path.exists() or not pairwise_path.exists():
            continue
        uncertainty = pd.read_csv(uncertainty_path)
        pairwise = pd.read_csv(pairwise_path)
        u = uncertainty[
            (uncertainty["task"] == "task2")
            & (uncertainty["system"] == best_system)
            & (uncertainty["metric"] == metric)
        ]
        p = pairwise[
            (pairwise["task"] == "task2")
            & (pairwise["system"] == best_system)
            & (pairwise["metric"] == metric)
        ]
        if u.empty or p.empty:
            continue
        urow = u.iloc[0]
        prow = p.iloc[0]
        rows.append(
            {
                "frequency": freq,
                "best_system": best_system,
                "mean": urow.get("mean"),
                "ci95": f"[{fmt(urow.get('ci95_low'))}, {fmt(urow.get('ci95_high'))}]",
                "delta_vs_random": prow.get("mean_delta"),
                "delta_ci95": f"[{fmt(prow.get('ci95_delta_low'))}, {fmt(prow.get('ci95_delta_high'))}]",
                "p_randomization": prow.get("paired_randomization_p"),
                "wins": prow.get("wins"),
                "losses": prow.get("losses"),
            }
        )
    return rows


def print_table(title: str, rows: list[dict[str, Any]], columns: list[str], metric_cols: set[str] | None = None) -> None:
    print(f"\n{title}")
    if not rows:
        print("  no data")
        return
    metric_cols = metric_cols or set()
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    out = df[columns].copy()
    for col in metric_cols:
        if col in out.columns:
            out[col] = out[col].map(fmt)
    print(out.to_string(index=False))


def print_verdict(main_rows: list[dict[str, Any]], wp_rows: list[dict[str, Any]], length_rows: list[dict[str, Any]], metric: str) -> None:
    print("\nVerdict")
    if main_rows:
        best = max(main_rows, key=lambda r: float(r.get(metric) or float("-inf")))
        print(f"  best overall: {best['frequency']} / {best['best_system']} ({metric}={fmt(best[metric])})")
        improved = [r for r in main_rows if r.get("delta_vs_random") is not None and float(r["delta_vs_random"]) > 0]
        print(f"  content beats random: {len(improved)}/{len(main_rows)} frequencies")
    if wp_rows:
        comparisons = 0
        wins = 0
        for row in wp_rows:
            for key in ["cosine_delta", "dtw_delta", "tsfel_delta"]:
                if row.get(key) is not None and pd.notna(row.get(key)):
                    comparisons += 1
                    wins += int(float(row[key]) > 0)
        print(f"  pattern beats whole-series: {wins}/{comparisons} comparisons")
    if length_rows:
        counts = pd.Series([r["best_len"] for r in length_rows]).value_counts().sort_index()
        best_lengths = ", ".join(f"L{k}:{v}" for k, v in counts.items())
        print(f"  best pattern length counts: {best_lengths}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paper-ready retrieval results by frequency.")
    parser.add_argument("--root", default="results_ri_bins_paper", help="Root folder containing frequency subfolders.")
    parser.add_argument("--metric", default="ndcg@10", help="Metric to summarize, e.g. ndcg@10 or ndcg@15.")
    args = parser.parse_args()

    root = Path(args.root)
    run_dirs = discover_run_dirs(root)
    if not run_dirs:
        print(f"No completed runs found under {root}. Expected research_results.json in root or frequency subfolders.")
        return

    print(f"Results root: {root}")
    print("Runs:", ", ".join(str(p.relative_to(root)) if p != root else "." for p in run_dirs))
    print(f"Metric: {args.metric}")

    dataset_rows = collect_dataset_rows(run_dirs)
    main_rows = collect_main_rows(run_dirs, args.metric)
    qrels_density_rows = collect_qrels_density_rows(run_dirs)
    ir_robustness_rows = collect_ir_robustness_rows(
        run_dirs, main_rows, args.metric
    )
    wp_rows = collect_whole_pattern_rows(run_dirs, args.metric)
    length_rows = collect_length_rows(run_dirs, args.metric)

    print_table(
        "Dataset / Split",
        dataset_rows,
        ["frequency", "n_series", "index", "dev", "test", "forecast_h", "pattern_len", "median_len"],
    )
    print_table(
        "Main Pattern Retrieval",
        main_rows,
        ["frequency", "best_system", args.metric, "runner_up", f"runner_{args.metric}", "random", "delta_vs_random"],
        {args.metric, f"runner_{args.metric}", "random", "delta_vs_random"},
    )
    print_table(
        "Qrels Density (Pattern Test)",
        qrels_density_rows,
        ["frequency", "n_queries", "mean_qrels", "median_qrels", "mean_relevance", "rel1", "rel2", "rel3", "rel4"],
        {"mean_qrels", "median_qrels", "mean_relevance"},
    )
    print_table(
        "IR Robustness vs Random (Pattern Test)",
        ir_robustness_rows,
        ["frequency", "best_system", "mean", "ci95", "delta_vs_random", "delta_ci95", "p_randomization", "wins", "losses"],
        {"mean", "delta_vs_random", "p_randomization"},
    )
    print_table(
        "Pattern minus Whole-Series",
        wp_rows,
        ["frequency", "cosine_delta", "dtw_delta", "tsfel_delta", "avg_delta"],
        {"cosine_delta", "dtw_delta", "tsfel_delta", "avg_delta"},
    )
    print_table(
        "Pattern-Length Sensitivity",
        length_rows,
        ["frequency", "best_len", "best_system", args.metric, "mean_by_len"],
        {args.metric},
    )
    print_verdict(main_rows, wp_rows, length_rows, args.metric)


if __name__ == "__main__":
    main()
