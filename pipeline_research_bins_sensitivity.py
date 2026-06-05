from __future__ import annotations

import argparse
import itertools
import json
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import pipeline_research_bins as research

GENERAL_HORIZONS = [3, 6, 12]
QUARTERLY_HORIZONS = [1, 2, 4, 8]
YEARLY_HORIZONS = [1, 2, 4, 6]
REGIME_DEFINITIONS = ["full", "error_only", "error_model", "error_disagreement"]


def parse_int_grid(text: str | None, default: list[int]) -> list[int]:
    if not text:
        return list(default)
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values or list(default)


def parse_str_grid(text: str | None, default: list[str]) -> list[str]:
    if not text:
        return list(default)
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(part)
    return values or list(default)


def safe_name(value: Any) -> str:
    return str(value).replace(" ", "_").replace("/", "-").replace("\\", "-")


def run_id(params: dict[str, Any]) -> str:
    parts = [
        f"freq{params.get('frequency', 'monthly')}",
        f"bins{params['qrels_n_bins']}",
        f"h{params['forecast_h']}",
        f"seed{params['seed']}",
        f"pps{params['patterns_per_series']}",
        f"reg{params['regime_definition']}",
    ]
    return "__".join(safe_name(part) for part in parts)


def qrels_stats_for_run(
    run_dir: Path, metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(run_dir.glob("*_qrels_*.csv")):
        name = path.stem
        if not (name.endswith("_qrels_dev") or name.endswith("_qrels_test")):
            continue
        split = (
            "dev" if name.endswith("_dev") else "test" if name.endswith("_test") else ""
        )
        task = name.replace("_qrels_dev", "").replace("_qrels_test", "")
        df = pd.read_csv(path)
        if df.empty:
            stats = {
                "qrels_rows": 0,
                "n_queries_with_positive_qrels": 0,
                "mean_positive_qrels_per_query": 0.0,
                "median_positive_qrels_per_query": 0.0,
                "min_positive_qrels_per_query": 0,
                "max_positive_qrels_per_query": 0,
                "mean_relevance": np.nan,
            }
        else:
            per_query = df.groupby("query_id").size()
            stats = {
                "qrels_rows": int(len(df)),
                "n_queries_with_positive_qrels": int(per_query.size),
                "mean_positive_qrels_per_query": float(per_query.mean()),
                "median_positive_qrels_per_query": float(per_query.median()),
                "min_positive_qrels_per_query": int(per_query.min()),
                "max_positive_qrels_per_query": int(per_query.max()),
                "mean_relevance": float(
                    pd.to_numeric(df["relevance"], errors="coerce").mean()
                ),
            }
        rows.append({**metadata, "task": task, "split": split, **stats})
    return rows


def collect_ir_diagnostic_rows(
    run_dir: Path, metadata: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    files = {
        "qrels": run_dir / "ir_qrels_summary.csv",
        "uncertainty": run_dir / "ir_system_uncertainty.csv",
        "pairwise": run_dir / "ir_pairwise_random_tests.csv",
    }
    out: dict[str, list[dict[str, Any]]] = {
        "qrels": [],
        "uncertainty": [],
        "pairwise": [],
    }
    for key, path in files.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for row in df.to_dict(orient="records"):
            out[key].append({**metadata, **research._json_safe(row)})
    return out


def collect_table_rows(
    tables: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for experiment, df in tables.items():
        for row in df.to_dict(orient="records"):
            rows.append(
                {**metadata, "experiment": experiment, **research._json_safe(row)}
            )
    return rows


def collect_regime_rows(
    whole: dict[str, Any],
    pattern_main: dict[str, Any],
    length_results: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for row in whole.get("late_regime_counts", []):
        rows.append({**metadata, "scope": "whole_series", "pattern_len": np.nan, **row})
    for row in pattern_main.get("late_regime_counts", []):
        rows.append(
            {
                **metadata,
                "scope": "pattern_main",
                "pattern_len": pattern_main.get("length"),
                **row,
            }
        )
    for result in length_results:
        for row in result.get("late_regime_counts", []):
            rows.append(
                {
                    **metadata,
                    "scope": "pattern_length",
                    "pattern_len": result.get("length"),
                    **row,
                }
            )
    return rows


def run_one(
    cfg: research.Config,
    series_map: dict[str, np.ndarray],
    outdir: Path,
    write_plots: bool,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    research.cleanup_streamlined_outputs(outdir, cfg)
    research.set_seed(cfg.seed)
    splits = research.split_ids(
        list(series_map.keys()), cfg.seed, cfg.index_frac, cfg.dev_frac
    )
    (outdir / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    if cfg.save_series_meta:
        pd.DataFrame(
            {
                "unique_id": list(series_map.keys()),
                "length": [len(series_map[k]) for k in series_map],
            }
        ).to_csv(outdir / "series_meta.csv", index=False)

    whole = research.run_whole_experiment(series_map, splits, cfg, outdir)
    pattern_main = research.run_pattern_experiment_for_length(
        cfg.pattern_len, series_map, splits, cfg, outdir, "task2"
    )

    length_values = sorted(
        set(research.parse_int_list(cfg.pattern_lens, [12, 24, 36]) + [cfg.pattern_len])
    )
    length_results = []
    for length in length_values:
        if int(length) == int(cfg.pattern_len):
            length_results.append(pattern_main)
        else:
            length_results.append(
                research.run_pattern_experiment_for_length(
                    int(length), series_map, splits, cfg, outdir, f"task2_L{length}"
                )
            )

    tables = research.build_experiment_tables(whole, pattern_main, length_results)
    plot_paths = research.write_plots(tables, outdir) if write_plots else {}
    report_path = research.write_report(
        cfg,
        series_map,
        splits,
        whole,
        pattern_main,
        length_results,
        tables,
        plot_paths,
        outdir,
    )
    return {
        "whole": whole,
        "pattern_main": pattern_main,
        "length_results": length_results,
        "tables": tables,
        "report_path": str(report_path),
    }


def write_summary(metrics: pd.DataFrame, outdir: Path) -> None:
    if metrics.empty:
        pd.DataFrame().to_csv(outdir / "sensitivity_summary.csv", index=False)
        return
    metric_cols = [
        col
        for col in ["ndcg@10", "ap@10", "p@10", "ndcg@15", "ap@15", "p@15"]
        if col in metrics.columns
    ]
    group_cols = [
        col
        for col in [
            "experiment",
            "system",
            "comparison",
            "approach",
            "pattern_len",
            "frequency",
            "qrels_n_bins",
            "forecast_h",
            "patterns_per_series",
            "regime_definition",
        ]
        if col in metrics.columns
    ]
    summary = (
        metrics.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.columns = [
        (
            "_".join(str(part) for part in col if str(part))
            if isinstance(col, tuple)
            else str(col)
        )
        for col in summary.columns
    ]
    summary.to_csv(outdir / "sensitivity_summary.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run sensitivity analyses around pipeline_research_bins.py."
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset-name", default="dataset")
    ap.add_argument(
        "--quarterly",
        action="store_true",
        help="Use quarterly defaults: seasonal-period=4 and forecast horizons 1,2,4,8 unless overridden.",
    )
    ap.add_argument(
        "--yearly",
        action="store_true",
        help="Use yearly defaults: seasonal-period=1 and forecast horizons 1,2,3,6 unless overridden.",
    )
    ap.add_argument("--seasonal-period", type=int)
    ap.add_argument("--forecast-h-grid", help="Comma-separated forecast horizons.")
    ap.add_argument("--qrels-n-bins-grid", default="2,3,4,5")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--patterns-per-series-grid", default="1,2,3")
    ap.add_argument("--regime-definitions", default=",".join(REGIME_DEFINITIONS))
    ap.add_argument("--pattern-len", type=int)
    ap.add_argument("--pattern-lens", default="12,24,36")
    ap.add_argument("--raw-resample-len", type=int, default=64)
    ap.add_argument("--ml-lags", default="1,2,3,4,6,12")
    ap.add_argument("--ml-n-estimators", type=int, default=80)
    ap.add_argument("--tsfel-standardize-series", action="store_true")
    ap.add_argument("--save-rankings", action="store_true")
    ap.add_argument(
        "--save-dev-artifacts",
        action="store_true",
        help="Also write dev qrels and dev metrics CSVs inside each sensitivity run.",
    )
    ap.add_argument(
        "--save-series-meta",
        action="store_true",
        help="Also write per-series metadata CSV inside each sensitivity run.",
    )
    ap.add_argument("--write-plots", action="store_true")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs that already have research_results.json.",
    )
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.quarterly and args.yearly:
        raise ValueError("--quarterly and --yearly are mutually exclusive.")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    frequency = (
        "yearly" if args.yearly else "quarterly" if args.quarterly else "monthly"
    )
    if frequency == "yearly":
        horizons_default = YEARLY_HORIZONS
    elif frequency == "quarterly":
        horizons_default = QUARTERLY_HORIZONS
    else:
        horizons_default = GENERAL_HORIZONS
    horizons = parse_int_grid(args.forecast_h_grid, horizons_default)
    seasonal_period = (
        args.seasonal_period
        if args.seasonal_period is not None
        else research.FREQUENCY_DEFAULTS[frequency]["seasonal_period"]
    )
    pattern_lengths = parse_int_grid(args.pattern_lens, [12, 24, 36])
    pattern_len = (
        args.pattern_len if args.pattern_len is not None else pattern_lengths[0]
    )
    qrels_bins = parse_int_grid(args.qrels_n_bins_grid, [2, 3, 4, 5])
    seeds = parse_int_grid(args.seeds, [1, 2, 3, 4, 5])
    patterns_per_series_values = parse_int_grid(
        args.patterns_per_series_grid, [1, 2, 3]
    )
    regime_definitions = parse_str_grid(args.regime_definitions, REGIME_DEFINITIONS)
    invalid_regimes = sorted(set(regime_definitions) - set(REGIME_DEFINITIONS))
    if invalid_regimes:
        raise ValueError(f"Invalid regime definitions: {invalid_regimes}")

    grid = list(
        itertools.product(
            qrels_bins,
            horizons,
            seeds,
            patterns_per_series_values,
            regime_definitions,
        )
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": args.input,
        "output_dir": str(outdir),
        "frequency": frequency,
        "seasonal_period": seasonal_period,
        "forecast_h_grid": horizons,
        "qrels_n_bins_grid": qrels_bins,
        "seeds": seeds,
        "patterns_per_series_grid": patterns_per_series_values,
        "regime_definitions": regime_definitions,
        "pattern_len": pattern_len,
        "pattern_lens": pattern_lengths,
        "n_runs": len(grid),
    }
    (outdir / "sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    research.log(f"Sensitivity grid has {len(grid)} runs")
    if args.dry_run:
        for values in grid:
            params = {
                "qrels_n_bins": values[0],
                "forecast_h": values[1],
                "seed": values[2],
                "patterns_per_series": values[3],
                "regime_definition": values[4],
                "frequency": frequency,
            }
            print(run_id(params), flush=True)
        return

    research.log("Loading long-format dataset")
    df = research.ensure_long_df(pd.read_csv(args.input))
    series_map = research.build_series_map(df)

    run_rows = []
    metric_rows = []
    qrels_rows = []
    regime_rows = []
    ir_qrels_rows = []
    ir_uncertainty_rows = []
    ir_pairwise_rows = []

    for i, (n_bins, horizon, seed, patterns_per_series, regime_definition) in enumerate(
        grid, start=1
    ):
        params = {
            "qrels_n_bins": n_bins,
            "forecast_h": horizon,
            "seed": seed,
            "patterns_per_series": patterns_per_series,
            "regime_definition": regime_definition,
            "frequency": frequency,
        }
        rid = run_id(params)
        run_dir = outdir / rid
        metadata = {
            "run_id": rid,
            "run_dir": str(run_dir),
            "dataset_name": args.dataset_name,
            "frequency": frequency,
            "seasonal_period": seasonal_period,
            "pattern_len": pattern_len,
            "pattern_lens": ",".join(str(x) for x in pattern_lengths),
            **params,
        }
        research.log(f"[{i}/{len(grid)}] {rid}")
        if args.resume and (run_dir / "research_results.json").exists():
            run_rows.append({**metadata, "status": "skipped_existing"})
            qrels_rows.extend(qrels_stats_for_run(run_dir, metadata))
            ir_rows = collect_ir_diagnostic_rows(run_dir, metadata)
            ir_qrels_rows.extend(ir_rows["qrels"])
            ir_uncertainty_rows.extend(ir_rows["uncertainty"])
            ir_pairwise_rows.extend(ir_rows["pairwise"])
            continue

        cfg = research.Config(
            input=args.input,
            output_dir=str(run_dir),
            dataset_name=args.dataset_name,
            frequency=frequency,
            seasonal_period=seasonal_period,
            pattern_len=pattern_len,
            pattern_lens=",".join(str(x) for x in pattern_lengths),
            patterns_per_series=patterns_per_series,
            raw_resample_len=args.raw_resample_len,
            forecast_h=horizon,
            qrels_n_bins=n_bins,
            regime_definition=regime_definition,
            ml_lags=args.ml_lags,
            ml_n_estimators=args.ml_n_estimators,
            seed=seed,
            tsfel_standardize_series=args.tsfel_standardize_series,
            save_rankings=args.save_rankings,
            save_dev_artifacts=args.save_dev_artifacts,
            save_series_meta=args.save_series_meta,
        )

        try:
            result = run_one(cfg, series_map, run_dir, args.write_plots)
            run_rows.append(
                {
                    **metadata,
                    "status": "ok",
                    "config": json.dumps(
                        research._json_safe(asdict(cfg)), sort_keys=True
                    ),
                    "report_path": result["report_path"],
                }
            )
            metric_rows.extend(collect_table_rows(result["tables"], metadata))
            qrels_rows.extend(qrels_stats_for_run(run_dir, metadata))
            ir_rows = collect_ir_diagnostic_rows(run_dir, metadata)
            ir_qrels_rows.extend(ir_rows["qrels"])
            ir_uncertainty_rows.extend(ir_rows["uncertainty"])
            ir_pairwise_rows.extend(ir_rows["pairwise"])
            regime_rows.extend(
                collect_regime_rows(
                    result["whole"],
                    result["pattern_main"],
                    result["length_results"],
                    metadata,
                )
            )
        except Exception as exc:
            run_rows.append(
                {
                    **metadata,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            pd.DataFrame(run_rows).to_csv(outdir / "sensitivity_runs.csv", index=False)
            if args.fail_fast:
                raise

    runs_df = pd.DataFrame(run_rows)
    metrics_df = pd.DataFrame(metric_rows)
    qrels_df = pd.DataFrame(qrels_rows)
    regimes_df = pd.DataFrame(regime_rows)
    ir_qrels_df = pd.DataFrame(ir_qrels_rows)
    ir_uncertainty_df = pd.DataFrame(ir_uncertainty_rows)
    ir_pairwise_df = pd.DataFrame(ir_pairwise_rows)

    runs_df.to_csv(outdir / "sensitivity_runs.csv", index=False)
    metrics_df.to_csv(outdir / "sensitivity_metrics_long.csv", index=False)
    qrels_df.to_csv(outdir / "sensitivity_qrels.csv", index=False)
    regimes_df.to_csv(outdir / "sensitivity_regime_counts.csv", index=False)
    ir_qrels_df.to_csv(outdir / "sensitivity_ir_qrels_summary.csv", index=False)
    ir_uncertainty_df.to_csv(
        outdir / "sensitivity_ir_system_uncertainty.csv", index=False
    )
    ir_pairwise_df.to_csv(
        outdir / "sensitivity_ir_pairwise_random_tests.csv", index=False
    )
    write_summary(metrics_df, outdir)
    research.log(f"Done. Sensitivity artifacts written to {outdir}")


if __name__ == "__main__":
    main()
