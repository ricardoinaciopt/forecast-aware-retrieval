from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import pipeline_research_bins as research

WHOLE_TASK = "whole"
PATTERN_TASK = "pattern"
REMOVED_ORACLE_SYSTEMS = {
    "late_qrels_oracle_upper_bound",
    "pattern_late_qrels_oracle_upper_bound",
    "forecast_regime_oracle",
    "pattern_forecast_regime_oracle",
    "historical_backtest_profile",
    "pattern_historical_backtest_profile",
    "forecast_profile_early",
    "pattern_forecast_profile_early",
}
_ACTIVE_EXPLORERS: list[Any] = []


@dataclass
class RetrievalResult:
    task: str
    split: str
    query_id: str
    system: str
    ranking: pd.DataFrame
    process: list[str]
    pattern_len: int | None = None


def launch_explorer(
    data_csv: str | Path,
    results_dir: str | Path,
    top_k: int = 10,
) -> Any:
    """Create and display the notebook retrieval explorer."""
    explorer = RetrievalNotebookExplorer(data_csv=data_csv, results_dir=results_dir)
    _ACTIVE_EXPLORERS.append(explorer)
    return explorer.show(top_k=top_k)


def export_qualitative_example(
    data_csv: str | Path,
    results_dir: str | Path,
    output_path: str | Path,
    query_id: str | None = None,
    system: str | None = None,
    task: str = WHOLE_TASK,
    pattern_len: int | None = None,
    top_k: int = 3,
    split: str = "test",
) -> RetrievalResult:
    """Export a compact paper figure for one query and its top retrieved items."""
    explorer = RetrievalNotebookExplorer(data_csv=data_csv, results_dir=results_dir)
    length = pattern_len or explorer.pattern_len
    if system is None:
        system = (
            "raw_dtw+tsfel"
            if task == WHOLE_TASK
            else "pattern_raw_cosine+pattern_tsfel"
        )
    if query_id is None:
        queries = explorer.query_ids(task, split, length)
        if not queries:
            raise ValueError(f"No {split} queries are available for task={task!r}.")
        query_id = str(np.random.default_rng().choice(queries))
    result = explorer.rank(
        query_id=str(query_id),
        task=task,
        split=split,
        system=system,
        pattern_len=length,
        top_k=top_k,
    )
    explorer.export_paper_figure(result, output_path=output_path, top_k=top_k)
    return result


class RetrievalNotebookExplorer:
    """Interactive retrieval explorer for pipeline_research_bins.py artifacts.

    Typical notebook use:

    ```python
    from notebook_retrieval_explorer import launch_explorer
    launch_explorer("m3_monthly.csv", "results_ri_bins")
    ```

    The explorer computes raw, DTW, TSFEL, and content-fusion rankings on demand.
    If saved ranking CSVs exist, it can also display those. Qrels are used
    only as relevance labels in the result table.
    """

    def __init__(self, data_csv: str | Path, results_dir: str | Path):
        self.data_csv = Path(data_csv)
        self.results_dir = Path(results_dir)
        self.report = self._load_report()
        self.params = self.report.get("base_configuration", {}).get("parameters", {})
        self.forecast_h = int(self.params.get("forecast_h", 6))
        self.raw_resample_len = int(self.params.get("raw_resample_len", 64))
        self.pattern_len = int(self.params.get("pattern_len", 24))
        self.pattern_lens = research.parse_int_list(
            str(self.params.get("pattern_lens", "12,24,36")), [12, 24, 36]
        )
        if self.pattern_len not in self.pattern_lens:
            self.pattern_lens.append(self.pattern_len)
        self.pattern_lens = sorted(set(int(x) for x in self.pattern_lens))
        self.patterns_per_series = int(self.params.get("patterns_per_series", 1))
        self.tsfel_standardize_series = bool(
            self.params.get("tsfel_standardize_series", False)
        )

        df = research.ensure_long_df(pd.read_csv(self.data_csv))
        self.series_map = research.build_series_map(df)
        self.series_dates = self._build_date_map(df)
        self.splits = self._load_splits()
        self.visible_series_map = research.retrieval_visible_series_map(
            self.series_map, self.forecast_h
        )

        self._qrels_cache: dict[tuple[str, str, int | None], pd.DataFrame] = {}
        self._regime_cache: dict[tuple[str, int | None], pd.DataFrame] = {}
        self._tsfel_cache: dict[
            tuple[str, int | None], tuple[pd.DataFrame, StandardScaler]
        ] = {}
        self._pattern_cache: dict[
            int,
            tuple[
                dict[str, np.ndarray],
                dict[str, str],
                dict[str, tuple[int | None, int | None]],
            ],
        ] = {}
        self._saved_rankings = self._discover_saved_rankings()

    def show(self, top_k: int = 10) -> Any:
        try:
            import ipywidgets as widgets
        except Exception as exc:
            raise RuntimeError(
                "Install ipywidgets and IPython to use the notebook UI."
            ) from exc

        task = widgets.Dropdown(
            options=[("Pattern objects", PATTERN_TASK), ("Whole series", WHOLE_TASK)],
            value=PATTERN_TASK,
            description="Task",
            layout=widgets.Layout(width="310px"),
        )
        query_split = "test"
        pattern_len = widgets.Dropdown(
            options=self.pattern_lens,
            value=(
                self.pattern_len
                if self.pattern_len in self.pattern_lens
                else self.pattern_lens[0]
            ),
            description="Pattern L",
            layout=widgets.Layout(width="310px"),
        )
        system = widgets.Dropdown(
            description="System", layout=widgets.Layout(width="390px")
        )
        query = widgets.Combobox(
            description="Query",
            ensure_option=True,
            placeholder="Type or choose a query id",
            layout=widgets.Layout(width="390px"),
        )
        status = widgets.HTML(layout=widgets.Layout(width="auto"))
        k_slider = widgets.IntSlider(
            value=int(top_k),
            min=3,
            max=30,
            step=1,
            description="Top K",
            continuous_update=False,
            layout=widgets.Layout(width="390px"),
        )
        run_button = widgets.Button(
            description="Run Retrieval",
            button_style="primary",
            icon="refresh",
            layout=widgets.Layout(width="220px", height="38px"),
        )
        output = widgets.Output(layout=widgets.Layout(width="100%"))
        style = widgets.HTML(
            """
            <style>
            .retrieval-explorer .retrieval-secondary {
                color: #c65f00 !important;
            }
            .retrieval-explorer .jp-RenderedHTMLCommon code,
            .retrieval-explorer .jp-OutputArea-output code,
            .retrieval-explorer code {
                color: #c65f00 !important;
                background: rgba(198, 95, 0, 0.10) !important;
            }
            .retrieval-explorer .widget-label,
            .retrieval-explorer .widget-readout {
                color: #c65f00 !important;
            }
            </style>
            """
        )

        def refresh_options(*_: Any) -> None:
            length = int(pattern_len.value)
            previous_system = system.value
            previous_query = query.value
            systems = self.available_systems(task.value, length, query_split)
            system.options = systems
            if systems:
                system.value = (
                    previous_system if previous_system in systems else systems[0]
                )
            queries = self.query_ids(task.value, query_split, length)
            query.options = queries
            if queries:
                query.value = (
                    previous_query if previous_query in queries else queries[0]
                )
            else:
                query.value = ""
            index_count = len(self._index_object_ids(task.value, length))
            status.value = (
                f"<span class='retrieval-secondary' style='font-size: 12px;'>"
                f"{len(queries)} selectable <b>test</b> query objects; "
                f"retrieval ranks against <b>{index_count}</b> indexed candidates. "
                f"Press <b>Run</b> to compute."
                f"</span>"
            )

        def run(_: Any = None) -> None:
            refresh_options()
            output.clear_output(wait=True)
            run_button.disabled = True
            previous_description = run_button.description
            run_button.description = "Running..."
            with output:
                try:
                    if not query.value or not system.value:
                        print("No query/system is available for this selection.")
                        return
                    result = self.rank(
                        query_id=str(query.value),
                        task=str(task.value),
                        split=query_split,
                        system=str(system.value),
                        pattern_len=int(pattern_len.value),
                        top_k=int(k_slider.value),
                    )
                    self.display_result(result, top_k=int(k_slider.value))
                except Exception as exc:
                    print(f"{type(exc).__name__}: {exc}")
                finally:
                    run_button.description = previous_description
                    run_button.disabled = False

        for widget in [task, pattern_len]:
            widget.observe(refresh_options, names="value")
        run_button.on_click(run)
        refresh_options()

        action_bar = widgets.HBox(
            [
                run_button,
                status,
            ],
            layout=widgets.Layout(
                width="100%",
                align_items="center",
                border="1px solid #d0d7de",
                padding="10px",
                margin="0 0 10px 0",
            ),
        )
        selector_panel = widgets.VBox(
            [
                widgets.HTML("<b>Selectors</b>"),
                widgets.HBox([task, pattern_len]),
                widgets.HBox([system, query, k_slider]),
            ],
            layout=widgets.Layout(
                width="100%",
                border="1px solid #d0d7de",
                padding="10px",
                margin="0 0 12px 0",
            ),
        )
        results_panel = widgets.VBox(
            [
                widgets.HTML("<b>Results</b>"),
                output,
            ],
            layout=widgets.Layout(width="100%"),
        )
        ui = widgets.VBox(
            [
                style,
                widgets.HTML("<b>Retrieval Explorer</b>"),
                action_bar,
                selector_panel,
                results_panel,
            ],
            layout=widgets.Layout(width="100%"),
        )
        ui.add_class("retrieval-explorer")
        self._widget_state = {
            "task": task,
            "query_split": query_split,
            "pattern_len": pattern_len,
            "system": system,
            "query": query,
            "status": status,
            "top_k": k_slider,
            "run_button": run_button,
            "output": output,
            "style": style,
            "action_bar": action_bar,
            "selector_panel": selector_panel,
            "results_panel": results_panel,
            "ui": ui,
        }
        return ui

    def available_systems(
        self, task: str, pattern_len: int | None = None, split: str | None = None
    ) -> list[str]:
        if task == WHOLE_TASK:
            systems = ["raw_cosine", "raw_dtw"]
            if self._has_tsfel(task, None):
                systems.append("tsfel")
                if self._fusion_weights("raw_dtw+tsfel"):
                    systems.append("raw_dtw+tsfel")
            systems += self._saved_systems(task, None)
            return list(dict.fromkeys(systems))

        length = int(pattern_len or self.pattern_len)
        systems = ["pattern_raw_cosine", "pattern_raw_dtw"]
        if self._has_tsfel(task, length):
            systems.append("pattern_tsfel")
            if self._fusion_weights("pattern_raw_cosine+pattern_tsfel", length):
                systems.append("pattern_raw_cosine+pattern_tsfel")
        systems += self._saved_systems(task, length)
        return list(dict.fromkeys(systems))

    def query_ids(
        self, task: str, split: str, pattern_len: int | None = None
    ) -> list[str]:
        ids = list(self.splits.get(split, []))
        if task == WHOLE_TASK:
            return [uid for uid in ids if uid in self.visible_series_map]
        length = int(pattern_len or self.pattern_len)
        _, owner, _ = self.pattern_objects(length)
        split_set = set(ids)
        return sorted(
            pid
            for pid, uid in owner.items()
            if uid in split_set and pid in self.pattern_visible_series(length)
        )

    def rank(
        self,
        query_id: str,
        task: str = PATTERN_TASK,
        split: str = "test",
        system: str = "pattern_raw_cosine",
        pattern_len: int | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        length = int(pattern_len or self.pattern_len) if task == PATTERN_TASK else None
        process = self._process_description(system, task)

        saved = self._saved_ranking(task, length, system)
        if saved is not None:
            ranking = saved[saved["query_id"].astype(str) == str(query_id)].copy()
            if ranking.empty:
                raise ValueError(
                    f"No saved ranking for query {query_id!r} and system {system!r}."
                )
            ranking = ranking.sort_values("rank")
        elif task == WHOLE_TASK:
            ranking = self._rank_whole(query_id, system)
        else:
            ranking = self._rank_pattern(query_id, system, int(length))

        ranking = self._attach_metadata(ranking, query_id, task, split, length)
        if top_k:
            ranking = ranking.head(int(top_k)).reset_index(drop=True)
        return RetrievalResult(
            task=task,
            split=split,
            query_id=query_id,
            system=system,
            ranking=ranking,
            process=process,
            pattern_len=length,
        )

    def display_result(self, result: RetrievalResult, top_k: int = 10) -> None:
        from IPython.display import Markdown, display

        display(Markdown(self._markdown_summary(result)))
        for step in result.process:
            print(f"- {step}")
        display_cols = [
            "rank",
            "doc_id",
            "owner_id",
            "score",
            "relevance",
            "error_regime",
            "best_model",
            "disagreement_regime",
        ]

        display(Markdown("**Query**"))
        query_table = pd.DataFrame([self._query_display_row(result)])
        display(query_table[[c for c in display_cols if c in query_table.columns]])

        display(Markdown("**Retrieved index objects**"))
        table = result.ranking.head(top_k).copy()
        display(table[[c for c in display_cols if c in table.columns]])
        display(Markdown(self._topk_diagnostic_markdown(result, top_k=top_k)))
        self.plot_result(result, top_k=top_k)

    def _query_display_row(self, result: RetrievalResult) -> dict[str, Any]:
        row: dict[str, Any] = {
            "rank": "query",
            "doc_id": result.query_id,
            "owner_id": self._owner_id(
                result.query_id, result.task, result.pattern_len
            ),
            "score": np.nan,
            "relevance": np.nan,
        }
        regimes = self.regimes(result.task, result.pattern_len)
        if not regimes.empty:
            id_col = "object_id" if "object_id" in regimes.columns else "doc_id"
            match = regimes[regimes[id_col].astype(str) == str(result.query_id)]
            if not match.empty:
                first = match.iloc[0]
                for col in ["error_regime", "best_model", "disagreement_regime"]:
                    if col in match.columns:
                        row[col] = first[col]
        return row

    def _topk_diagnostic_markdown(self, result: RetrievalResult, top_k: int) -> str:
        top = result.ranking.head(top_k).copy()
        if top.empty or "relevance" not in top.columns:
            return (
                "**How to read this run**  \n"
                "`score` is the retrieval similarity used for ranking. `relevance` is the late-window "
                "forecasting-regime match used only for evaluation."
            )

        rel = pd.to_numeric(top["relevance"], errors="coerce").fillna(0)
        relevant = int((rel > 0).sum())
        strong = int((rel >= 3).sum())
        mean_rel = float(rel.mean()) if len(rel) else 0.0
        best_idx = int(rel.idxmax()) if len(rel) else -1
        best_rank = int(top.loc[best_idx, "rank"]) if best_idx in top.index else None
        best_rel = int(rel.max()) if len(rel) else 0
        query = self._query_display_row(result)
        query_regime = (
            f"{query.get('error_regime', '?')} error, "
            f"{query.get('best_model', '?')} best model, "
            f"{query.get('disagreement_regime', '?')} disagreement"
        )
        best_text = (
            f"best relevance is `{best_rel}` at rank `{best_rank}`"
            if best_rank is not None
            else "no relevance labels were found"
        )
        return (
            "**How to read this run**  \n"
            f"The query's late forecasting regime is `{query_regime}`. "
            "`score` is the retrieval similarity used to sort the table; it does not use qrels. "
            "`relevance` is the held-out late-window forecasting-regime match used after ranking.  \n"
            f"Top-{len(top)} diagnostic: `{relevant}` have relevance > 0, `{strong}` have relevance >= 3, "
            f"mean relevance is `{mean_rel:.2f}`, and {best_text}."
        )

    def plot_result(self, result: RetrievalResult, top_k: int = 10) -> None:
        import matplotlib.pyplot as plt

        top = result.ranking.head(top_k)
        if top.empty:
            print("No ranking rows to plot.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
        self._plot_query_context(axes[0], result)
        self._plot_normalized_overlay(axes[1], result, top)
        self._plot_score_table(axes[2], top)
        fig.tight_layout()

    def export_paper_figure(
        self,
        result: RetrievalResult,
        output_path: str | Path,
        top_k: int = 3,
        dpi: int = 300,
    ) -> Path:
        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        top = result.ranking.head(top_k).copy()
        objects = [("Query", result.query_id, None)] + [
            (f"Rank {int(row['rank'])}", str(row["doc_id"]), row)
            for _, row in top.iterrows()
        ]
        n_plot_cols = 2
        n_plot_rows = int(np.ceil(len(objects) / n_plot_cols))
        fig = plt.figure(figsize=(10.8, 6.2))
        grid = fig.add_gridspec(
            n_plot_rows + 1,
            n_plot_cols,
            height_ratios=[1.0] * n_plot_rows + [0.42],
            hspace=0.48,
            wspace=0.18,
        )

        query_row = self._query_display_row(result)
        table_rows = [
            [
                "Query",
                result.query_id,
                "-",
                "-",
                self._format_regime(query_row),
            ]
        ]

        for i, (label, object_id, row) in enumerate(objects):
            ax = fig.add_subplot(grid[i // n_plot_cols, i % n_plot_cols])
            if row is None:
                title = f"Query: {object_id}"
                color = "#1f77b4"
            else:
                title = (
                    f"{label}: {object_id} "
                    f"(score={float(row['score']):.2f}, rel={int(row.get('relevance', 0))})"
                )
                color = "#4c78a8"
                table_rows.append(
                    [
                        label,
                        object_id,
                        f"{float(row['score']):.2f}",
                        int(row.get("relevance", 0)),
                        self._format_regime(row),
                    ]
                )
            self._plot_paper_object(
                ax,
                object_id,
                result.task,
                result.pattern_len,
                title,
                color,
            )

        for i in range(len(objects), n_plot_rows * n_plot_cols):
            fig.add_subplot(grid[i // n_plot_cols, i % n_plot_cols]).axis("off")

        table_ax = fig.add_subplot(grid[n_plot_rows, :])
        self._draw_paper_summary_table(table_ax, table_rows)
        fig.suptitle(
            f"Qualitative retrieval example: {result.system}",
            y=0.985,
            fontsize=13,
            fontweight="bold",
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def _plot_paper_object(
        self,
        ax: Any,
        object_id: str,
        task: str,
        length: int | None,
        title: str,
        color: str,
    ) -> None:
        y = self._object_values(object_id, task, length, visible=False)
        visible = self._object_values(object_id, task, length, visible=True)
        y_plot = research.zscore(np.asarray(y, dtype=float))
        ax.plot(np.arange(len(y_plot)), y_plot, color=color, lw=1.7)
        boundary = max(0, len(visible) - 1)
        ax.axvline(boundary, color="#111111", ls="--", lw=0.9)
        if len(visible) < len(y_plot):
            ax.axvspan(
                len(visible),
                len(y_plot) - 1,
                color="#f58518",
                alpha=0.16,
            )
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xlabel("step", fontsize=8)
        ax.set_ylabel("z(y)", fontsize=8)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7.5)

    def _draw_metadata_box(
        self, ax: Any, title: str, values: dict[str, Any]
    ) -> None:
        ax.axis("off")
        rows = [[str(k), str(v)] for k, v in values.items()]
        table = ax.table(
            cellText=rows,
            colLabels=[title, ""],
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.25)
        for (r, _c), cell in table.get_celld().items():
            cell.set_edgecolor("#d0d0d0")
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#f1f1f1")

    def _draw_paper_summary_table(self, ax: Any, rows: list[list[Any]]) -> None:
        ax.axis("off")
        table = ax.table(
            cellText=rows,
            colLabels=["Item", "Object", "Score", "Rel.", "Regime"],
            cellLoc="left",
            colLoc="left",
            loc="center",
            colWidths=[0.11, 0.36, 0.09, 0.08, 0.36],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.0)
        table.scale(1.0, 1.15)
        for (r, _c), cell in table.get_celld().items():
            cell.set_edgecolor("#d6d6d6")
            cell.PAD = 0.035
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#efefef")

    def _format_regime(self, row: Any) -> str:
        getter = row.get if hasattr(row, "get") else lambda key, default=None: default
        return (
            f"{getter('error_regime', '?')} / "
            f"{getter('best_model', '?')} / "
            f"{getter('disagreement_regime', '?')}"
        )
        plt.show()

    def _plot_score_table(self, ax: Any, top: pd.DataFrame) -> None:
        ax.axis("off")
        rows = [
            [int(row["rank"]), str(row["doc_id"]), f"{float(row['score']):.2f}"]
            for _, row in top.iterrows()
        ]
        table = ax.table(
            cellText=rows,
            colLabels=["rank", "doc_id", "score"],
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.05, 1.55)
        ax.set_title("Top scores")

    def pattern_objects(
        self,
        length: int,
    ) -> tuple[
        dict[str, np.ndarray], dict[str, str], dict[str, tuple[int | None, int | None]]
    ]:
        length = int(length)
        if length not in self._pattern_cache:
            pattern_series, owner = research.build_pattern_objects(
                self.series_map, length, self.patterns_per_series
            )
            spans = {}
            for uid, y in self.series_map.items():
                starts = self._pattern_starts(len(y), length, self.patterns_per_series)
                for j, start in enumerate(starts):
                    pid = f"{uid}::pattern{j}::L{length}"
                    spans[pid] = (start, None if start is None else start + length)
            self._pattern_cache[length] = (pattern_series, owner, spans)
        return self._pattern_cache[length]

    def pattern_visible_series(self, length: int) -> dict[str, np.ndarray]:
        series, _, _ = self.pattern_objects(length)
        pattern_h = self.pattern_forecast_h(length)
        return research.retrieval_visible_series_map(series, pattern_h)

    def pattern_forecast_h(self, length: int) -> int:
        raw_runs = self.report.get("raw_task_outputs", {}).get(
            "pattern_length_runs", []
        )
        for run in raw_runs:
            if int(run.get("length", -1)) == int(length):
                return int(
                    run.get(
                        "pattern_forecast_h",
                        max(2, min(self.forecast_h, max(2, int(length) // 4))),
                    )
                )
        main = self.report.get("raw_task_outputs", {}).get("task2_pattern_main", {})
        if int(main.get("length", -1)) == int(length):
            return int(
                main.get(
                    "pattern_forecast_h",
                    max(2, min(self.forecast_h, max(2, int(length) // 4))),
                )
            )
        return max(2, min(self.forecast_h, max(2, int(length) // 4)))

    def _rank_whole(self, query_id: str, system: str) -> pd.DataFrame:
        if query_id not in self.visible_series_map:
            raise ValueError(
                f"Query {query_id!r} is unavailable after pre-holdout truncation."
            )
        index_series = {
            uid: self.visible_series_map[uid]
            for uid in self.splits.get("index", [])
            if uid in self.visible_series_map
        }
        q = self.visible_series_map[query_id]
        if system == "raw_cosine":
            scores = research.score_raw(
                q, index_series, self.raw_resample_len, "cosine"
            )
        elif system == "raw_dtw":
            scores = research.score_raw(q, index_series, self.raw_resample_len, "dtw")
        elif system == "tsfel":
            scores = self._score_tsfel(query_id, q, index_series, WHOLE_TASK, None)
        elif system == "raw_dtw+tsfel":
            weights = self._fusion_weights(system) or {"raw_dtw": 0.5, "tsfel": 0.5}
            dtw = research.score_raw(q, index_series, self.raw_resample_len, "dtw")
            tsfel = self._score_tsfel(query_id, q, index_series, WHOLE_TASK, None)
            scores = research.fuse_scores(
                [dtw, tsfel],
                [float(weights.get("raw_dtw", 0.5)), float(weights.get("tsfel", 0.5))],
            )
        else:
            raise ValueError(f"Unsupported whole-series system: {system}")
        return research.rank_from_scores(query_id, scores)

    def _rank_pattern(self, query_id: str, system: str, length: int) -> pd.DataFrame:
        visible = self.pattern_visible_series(length)
        _, owner, _ = self.pattern_objects(length)
        index_patterns = {
            pid: visible[pid]
            for pid, uid in owner.items()
            if uid in set(self.splits.get("index", [])) and pid in visible
        }
        if query_id not in visible:
            raise ValueError(
                f"Pattern query {query_id!r} is unavailable after pre-holdout truncation."
            )
        q = visible[query_id]
        if system == "pattern_raw_cosine":
            scores = research.score_raw(
                q, index_patterns, self.raw_resample_len, "cosine"
            )
        elif system == "pattern_raw_dtw":
            scores = research.score_raw(q, index_patterns, self.raw_resample_len, "dtw")
        elif system == "pattern_tsfel":
            scores = self._score_tsfel(
                query_id, q, index_patterns, PATTERN_TASK, length
            )
        elif system == "pattern_raw_cosine+pattern_tsfel":
            weights = self._fusion_weights(system, length) or {
                "pattern_raw_cosine": 0.5,
                "pattern_tsfel": 0.5,
            }
            raw = research.score_raw(q, index_patterns, self.raw_resample_len, "cosine")
            tsfel = self._score_tsfel(query_id, q, index_patterns, PATTERN_TASK, length)
            scores = research.fuse_scores(
                [raw, tsfel],
                [
                    float(weights.get("pattern_raw_cosine", 0.5)),
                    float(weights.get("pattern_tsfel", 0.5)),
                ],
            )
        else:
            raise ValueError(f"Unsupported pattern system: {system}")
        return research.rank_from_scores(query_id, scores)

    def _score_tsfel(
        self,
        query_id: str,
        query_values: np.ndarray,
        index_series: dict[str, np.ndarray],
        task: str,
        length: int | None,
    ) -> dict[str, float]:
        raw, scaler = self._index_tsfel(task, length)
        feature_cols = [c for c in raw.columns if c != "unique_id"]
        idx_scaled = scaler.transform(raw[feature_cols].to_numpy(dtype=float))
        idx_feats = {
            uid: idx_scaled[i] for i, uid in enumerate(raw["unique_id"].astype(str))
        }
        q_raw = research.extract_tsfel_features(
            {query_id: query_values}, self.tsfel_standardize_series
        )
        for col in feature_cols:
            if col not in q_raw.columns:
                q_raw[col] = 0.0
        q_raw = q_raw[["unique_id"] + feature_cols].fillna(0.0)
        q_vec = scaler.transform(q_raw[feature_cols].to_numpy(dtype=float))[0]
        idx_feats = {uid: vec for uid, vec in idx_feats.items() if uid in index_series}
        return research.score_tsfel(q_vec, idx_feats)

    def _index_tsfel(
        self, task: str, length: int | None
    ) -> tuple[pd.DataFrame, StandardScaler]:
        key = (task, length)
        if key in self._tsfel_cache:
            return self._tsfel_cache[key]
        path = self._tsfel_path(task, length)
        if path is None:
            raise ValueError(
                "No saved index TSFEL feature CSV was found for this task."
            )
        raw = pd.read_csv(path)
        feature_cols = [c for c in raw.columns if c != "unique_id"]
        scaler = StandardScaler()
        scaler.fit(raw[feature_cols].fillna(0.0).to_numpy(dtype=float))
        self._tsfel_cache[key] = (raw, scaler)
        return raw, scaler

    def _attach_metadata(
        self,
        ranking: pd.DataFrame,
        query_id: str,
        task: str,
        split: str,
        length: int | None,
    ) -> pd.DataFrame:
        ranking = ranking.copy().sort_values("rank").reset_index(drop=True)
        qrels = self.qrels(task, split, length)
        if not qrels.empty:
            rel_map = {
                (str(r.query_id), str(r.doc_id)): int(r.relevance)
                for r in qrels.itertuples(index=False)
            }
            ranking["relevance"] = [
                rel_map.get((str(query_id), str(doc)), 0) for doc in ranking["doc_id"]
            ]
        else:
            ranking["relevance"] = np.nan

        regimes = self.regimes(task, length)
        if not regimes.empty:
            regimes = regimes.rename(columns={"object_id": "doc_id"})
            keep = [
                "doc_id",
                "error_regime",
                "best_model",
                "disagreement_regime",
                "mean_mae",
                "disagreement",
                "best_gap",
            ]
            ranking = ranking.merge(
                regimes[[c for c in keep if c in regimes.columns]],
                on="doc_id",
                how="left",
            )

        stats_rows = []
        for doc_id in ranking["doc_id"].astype(str):
            arr = self._object_values(doc_id, task, length, visible=True)
            row = self._series_stats(arr)
            row["doc_id"] = doc_id
            row["owner_id"] = self._owner_id(doc_id, task, length)
            stats_rows.append(row)
        stats = pd.DataFrame(stats_rows)
        return ranking.merge(stats, on="doc_id", how="left")

    def _plot_query_context(self, ax: Any, result: RetrievalResult) -> None:
        if result.task == WHOLE_TASK:
            uid = result.query_id
            y = self.series_map[uid]
            x = np.arange(len(y))
            visible_len = len(self.visible_series_map.get(uid, y))
            ax.plot(x, y, color="#1f77b4", lw=1.8, label="query")
            ax.axvline(
                max(0, visible_len - 1),
                color="black",
                ls="--",
                lw=1.0,
                label="holdout boundary",
            )
            if visible_len < len(y):
                ax.axvspan(
                    visible_len,
                    len(y) - 1,
                    color="#d62728",
                    alpha=0.12,
                    label="late qrels window",
                )
            ax.set_title(f"Query series: {uid}")
            ax.set_xlabel("time")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
            return

        length = int(result.pattern_len or self.pattern_len)
        full_patterns, _, _ = self.pattern_objects(length)
        y = full_patterns[result.query_id]
        visible_len = len(self.pattern_visible_series(length).get(result.query_id, y))
        ax.plot(np.arange(len(y)), y, color="#1f77b4", lw=1.8, label="query pattern")
        ax.axvline(
            max(0, visible_len - 1),
            color="black",
            ls="--",
            lw=1.0,
            label="pattern holdout boundary",
        )
        if visible_len < len(y):
            ax.axvspan(
                visible_len,
                len(y) - 1,
                color="#d62728",
                alpha=0.12,
                label="late qrels window",
            )
        ax.set_title(f"Query pattern: {result.query_id}")
        ax.set_xlabel("pattern step")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    def _plot_normalized_overlay(
        self, ax: Any, result: RetrievalResult, top: pd.DataFrame
    ) -> None:
        q = self._object_values(
            result.query_id, result.task, result.pattern_len, visible=True
        )
        q_plot = research.zscore(research.resample_1d(q, self.raw_resample_len))
        ax.plot(q_plot, color="black", lw=2.3, label="query")
        for _, row in top.iterrows():
            doc_id = str(row["doc_id"])
            arr = self._object_values(
                doc_id, result.task, result.pattern_len, visible=True
            )
            y = research.zscore(research.resample_1d(arr, self.raw_resample_len))
            label = f"{int(row['rank'])}. {doc_id} ({float(row['score']):.4f})"
            ax.plot(y, lw=1.2, alpha=0.65, label=label)
        ax.set_title("Normalized retrieval objects")
        ax.set_xlabel("resampled step")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="best")

    def _markdown_summary(self, result: RetrievalResult) -> str:
        warning = "\n\nRetrieval uses only pre-holdout visible data."
        length = (
            f", pattern length `{result.pattern_len}`" if result.pattern_len else ""
        )
        return (
            f"**Query:** `{result.query_id}`  \n"
            f"**Task:** `{result.task}`{length}  \n"
            f"**System:** `{result.system}`  \n"
            f"**Split:** `{result.split}`"
            f"{warning}"
        )

    def _process_description(self, system: str, task: str) -> list[str]:
        if "dtw" in system and "+" not in system:
            return [
                "Take the query object's pre-holdout visible prefix.",
                "Z-score and resample query and indexed objects.",
                "Compute constrained DTW distance and rank by negative distance.",
            ]
        if "cosine" in system and "+" not in system:
            return [
                "Take the query object's pre-holdout visible prefix.",
                "Z-score and resample query and indexed objects.",
                "Compute cosine similarity and rank descending.",
            ]
        if "tsfel" in system and "+" not in system:
            return [
                "Take the query object's pre-holdout visible prefix.",
                "Extract TSFEL statistical and temporal features for the query.",
                "Scale query features with the indexed-feature scaler and rank by cosine similarity.",
            ]
        if "+" in system:
            return [
                "Compute each component retrieval score on pre-holdout visible data.",
                "Min-max normalize component scores within the candidate set.",
                "Fuse with dev-tuned weights from the benchmark report, then rank descending.",
            ]
        return ["Load or compute the selected retrieval ranking."]

    def _object_values(
        self, object_id: str, task: str, length: int | None, visible: bool
    ) -> np.ndarray:
        if task == WHOLE_TASK:
            source = self.visible_series_map if visible else self.series_map
            return source[str(object_id)]
        if length is None:
            raise ValueError("Pattern length is required for pattern objects.")
        source = (
            self.pattern_visible_series(int(length))
            if visible
            else self.pattern_objects(int(length))[0]
        )
        return source[str(object_id)]

    def _owner_id(self, object_id: str, task: str, length: int | None) -> str:
        if task == WHOLE_TASK:
            return str(object_id)
        _, owner, _ = self.pattern_objects(int(length or self.pattern_len))
        return owner.get(str(object_id), "")

    def _index_object_ids(self, task: str, length: int | None) -> list[str]:
        if task == WHOLE_TASK:
            return [
                uid
                for uid in self.splits.get("index", [])
                if uid in self.visible_series_map
            ]
        _, owner, _ = self.pattern_objects(int(length or self.pattern_len))
        visible = self.pattern_visible_series(int(length or self.pattern_len))
        index_set = set(self.splits.get("index", []))
        return sorted(
            pid for pid, uid in owner.items() if uid in index_set and pid in visible
        )

    def _series_stats(self, arr: np.ndarray) -> dict[str, float | int]:
        arr = np.asarray(arr, dtype=float)
        return {
            "length": int(len(arr)),
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
        }

    def qrels(self, task: str, split: str, length: int | None) -> pd.DataFrame:
        key = (task, split, length)
        if key not in self._qrels_cache:
            path = self._qrels_path(task, split, length)
            self._qrels_cache[key] = (
                pd.read_csv(path)
                if path and path.exists()
                else pd.DataFrame(columns=["query_id", "doc_id", "relevance"])
            )
        return self._qrels_cache[key]

    def regimes(self, task: str, length: int | None) -> pd.DataFrame:
        key = (task, length)
        if key not in self._regime_cache:
            path = self._regime_path(task, length)
            self._regime_cache[key] = (
                pd.read_csv(path) if path and path.exists() else pd.DataFrame()
            )
        return self._regime_cache[key]

    def _has_qrels(self, task: str, split: str, length: int | None) -> bool:
        path = self._qrels_path(task, split, length)
        return bool(path and path.exists())

    def _has_tsfel(self, task: str, length: int | None) -> bool:
        return research.tsfel is not None and self._tsfel_path(task, length) is not None

    def _task_prefix(self, task: str, length: int | None = None) -> str:
        if task == WHOLE_TASK:
            return "task1"
        length = int(length or self.pattern_len)
        return "task2" if length == self.pattern_len else f"task2_L{length}"

    def _qrels_path(self, task: str, split: str, length: int | None) -> Path | None:
        path = self.results_dir / f"{self._task_prefix(task, length)}_qrels_{split}.csv"
        return path if path.exists() else None

    def _regime_path(self, task: str, length: int | None) -> Path | None:
        path = (
            self.results_dir
            / f"{self._task_prefix(task, length)}_late_forecast_regimes.csv"
        )
        return path if path.exists() else None

    def _tsfel_path(self, task: str, length: int | None) -> Path | None:
        path = (
            self.results_dir / f"{self._task_prefix(task, length)}_tsfel_index_raw.csv"
        )
        return path if path.exists() else None

    def _saved_ranking_path(
        self, task: str, length: int | None, system: str
    ) -> Path | None:
        path = (
            self.results_dir
            / f"{self._task_prefix(task, length)}_rankings_{system}.csv"
        )
        return path if path.exists() else None

    def _saved_ranking(
        self, task: str, length: int | None, system: str
    ) -> pd.DataFrame | None:
        path = self._saved_ranking_path(task, length, system)
        return pd.read_csv(path) if path else None

    def _saved_systems(self, task: str, length: int | None) -> list[str]:
        prefix = self._task_prefix(task, length)
        systems = []
        for path in self.results_dir.glob(f"{prefix}_rankings_*.csv"):
            system = path.stem.replace(f"{prefix}_rankings_", "")
            if system not in REMOVED_ORACLE_SYSTEMS:
                systems.append(system)
        return sorted(systems)

    def _discover_saved_rankings(self) -> list[Path]:
        return sorted(self.results_dir.glob("*_rankings_*.csv"))

    def _fusion_weights(
        self, system: str, length: int | None = None
    ) -> dict[str, float] | None:
        raw = self.report.get("raw_task_outputs", {})
        if system == "raw_dtw+tsfel":
            fusion = raw.get("task1_whole_series", {}).get("fusion", {})
            return fusion.get("weights") if fusion.get("system") == system else None
        if system == "pattern_raw_cosine+pattern_tsfel":
            if length is None or int(length) == self.pattern_len:
                fusion = raw.get("task2_pattern_main", {}).get("fusion", {})
                return fusion.get("weights") if fusion.get("system") == system else None
            for run in raw.get("pattern_length_runs", []):
                if int(run.get("length", -1)) == int(length):
                    fusion = run.get("fusion", {})
                    return (
                        fusion.get("weights")
                        if fusion.get("system") == system
                        else None
                    )
        return None

    def _load_report(self) -> dict[str, Any]:
        path = self.results_dir / "research_results.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_splits(self) -> dict[str, list[str]]:
        path = self.results_dir / "splits.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        seed = int(self.params.get("seed", 42))
        return research.split_ids(list(self.series_map.keys()), seed, 0.6, 0.2)

    def _build_date_map(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        out = {}
        for uid, g in df.groupby("unique_id", sort=False):
            out[str(uid)] = g["ds"].to_numpy()
        return out

    @staticmethod
    def _pattern_starts(
        series_len: int, length: int, n_patterns: int
    ) -> list[int | None]:
        if series_len < length:
            return [None]
        if n_patterns <= 1:
            return [max(0, (series_len - length) // 2)]
        return [
            int(x)
            for x in np.linspace(0, series_len - length, n_patterns).round().astype(int)
        ]


def parse_pattern_id(pattern_id: str) -> tuple[str, int, int] | None:
    match = re.match(r"(.+)::pattern(\d+)::L(\d+)$", str(pattern_id))
    if not match:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))
