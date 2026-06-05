from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
_tmp_dir = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(_tmp_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_tmp_dir / "cache"))
os.environ.setdefault("NIXTLA_ID_AS_COL", "1")

try:
    import tsfel
except Exception:
    tsfel = None

try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS
except Exception:
    StatsForecast = None
    AutoARIMA = None
    AutoETS = None

try:
    from mlforecast import MLForecast
except Exception:
    MLForecast = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


@dataclass
class Config:
    input: str
    output_dir: str
    dataset_name: str = "dataset"
    frequency: str = "monthly"
    seed: int = 42
    seasonal_period: int = 12
    raw_resample_len: int = 64
    pattern_len: int = 24
    pattern_lens: str = "12,24,36"
    patterns_per_series: int = 1
    tsfel_standardize_series: bool = False
    index_frac: float = 0.6
    dev_frac: float = 0.2
    forecast_h: int = 6
    qrels_top_high: int = 5
    qrels_top_mid: int = 15
    qrels_n_bins: int = 3
    regime_definition: str = "full"
    ml_lags: str = "1,2,3,4,6,12"
    ml_n_estimators: int = 80
    skip_ml_models: bool = False
    save_rankings: bool = False
    save_dev_artifacts: bool = False
    save_series_meta: bool = False


FREQUENCY_DEFAULTS = {
    "monthly": {"seasonal_period": 12, "forecast_h": 12},
    "quarterly": {"seasonal_period": 4, "forecast_h": 4},
    "yearly": {"seasonal_period": 1, "forecast_h": 4},
}

FREQUENCY_ALIASES = {
    "m": "monthly",
    "month": "monthly",
    "monthly": "monthly",
    "q": "quarterly",
    "quarter": "quarterly",
    "quarterly": "quarterly",
    "y": "yearly",
    "year": "yearly",
    "annual": "yearly",
    "yearly": "yearly",
}

IR_N_BOOTSTRAPS = 2000
IR_N_PERMUTATIONS = 5000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_int_list(text: str, default: list[int]) -> list[int]:
    if not text:
        return default
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    return vals or default


def normalize_frequency(value: str | None) -> str:
    key = str(value or "monthly").strip().lower()
    if key not in FREQUENCY_ALIASES:
        raise ValueError(
            f"Unsupported frequency {value!r}; use monthly/M, quarterly/Q, or yearly/Y."
        )
    return FREQUENCY_ALIASES[key]


def parse_frequency_list(text: str | None, default: list[str]) -> list[str]:
    if not text:
        return list(default)
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(normalize_frequency(part))
    return values or list(default)


def ensure_long_df(df: pd.DataFrame) -> pd.DataFrame:
    req = {"unique_id", "ds", "y"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")
    out = df[["unique_id", "ds", "y"]].copy()
    out["unique_id"] = out["unique_id"].astype(str)
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["unique_id", "ds", "y"])
    return out.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)


def build_series_map(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for uid, g in df.groupby("unique_id", sort=False):
        y = g["y"].to_numpy(dtype=float)
        if len(y) >= 8:
            out[str(uid)] = y
    if not out:
        raise ValueError("No valid series with length >= 8 were found.")
    return out


def split_ids(
    ids: list[str], seed: int, index_frac: float, dev_frac: float
) -> dict[str, list[str]]:
    ids = list(ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_index = max(1, int(round(index_frac * n)))
    n_dev = max(1, int(round(dev_frac * n)))
    if n_index + n_dev >= n:
        n_index = max(1, n - 2)
        n_dev = 1
    return {
        "index": ids[:n_index],
        "dev": ids[n_index : n_index + n_dev],
        "test": ids[n_index + n_dev :],
    }


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.nan_to_num(
        (x - np.nanmean(x)) / (np.nanstd(x) + eps), nan=0.0, posinf=0.0, neginf=0.0
    )


def resample_1d(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == length:
        return x.copy()
    if len(x) == 1:
        return np.repeat(x[0], length)
    xp = np.linspace(0.0, 1.0, len(x))
    xnew = np.linspace(0.0, 1.0, length)
    return np.interp(xnew, xp, x)


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if den <= eps else float(np.dot(a, b) / den)


def dtw_distance(a: np.ndarray, b: np.ndarray, window: int | None = None) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    if window is None:
        window = max(n, m)
    window = max(window, abs(n - m))
    prev = np.full(m + 1, np.inf)
    curr = np.full(m + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr[:] = np.inf
        for j in range(max(1, i - window), min(m, i + window) + 1):
            curr[j] = abs(a[i - 1] - b[j - 1]) + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(prev[m])


def extract_tsfel_features(
    series_map: dict[str, np.ndarray], standardize_series: bool
) -> pd.DataFrame:
    if tsfel is None:
        raise RuntimeError(
            "tsfel is not installed. Install it or remove TSFEL systems from the experiment."
        )
    cfg = {}
    cfg.update(tsfel.get_features_by_domain("statistical"))
    cfg.update(tsfel.get_features_by_domain("temporal"))
    rows, ids = [], []
    total = len(series_map)
    for i, (uid, y) in enumerate(series_map.items(), start=1):
        if i == 1 or i % 100 == 0 or i == total:
            log(f"TSFEL {i}/{total}")
        x = zscore(y) if standardize_series else np.asarray(y, float)
        fb = tsfel.time_series_features_extractor(
            cfg, pd.Series(x, index=pd.RangeIndex(len(x))), fs=1, verbose=0
        )
        rows.append(fb)
        ids.append(uid)
    feats = pd.concat(rows, ignore_index=True)
    feats.columns = [str(c).replace("0_", "") for c in feats.columns]
    feats = feats.loc[:, ~feats.columns.duplicated()].replace([np.inf, -np.inf], np.nan)
    feats.insert(0, "unique_id", ids)
    keep = feats.columns[1:][feats.iloc[:, 1:].isna().mean() <= 0.2].tolist()
    return feats[["unique_id"] + keep].fillna(0.0)


def normalize_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop(columns=["unique_id"]).to_numpy(dtype=float))
    out = pd.DataFrame(X, columns=[c for c in df.columns if c != "unique_id"])
    out.insert(0, "unique_id", df["unique_id"].tolist())
    return out, scaler


def apply_feature_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    X = scaler.transform(df.drop(columns=["unique_id"]).to_numpy(dtype=float))
    out = pd.DataFrame(X, columns=[c for c in df.columns if c != "unique_id"])
    out.insert(0, "unique_id", df["unique_id"].tolist())
    return out


def build_candidate_tsfel(
    cand_series: dict[str, np.ndarray],
    standardize_series: bool,
    scaler: StandardScaler | None,
):
    feats = extract_tsfel_features(cand_series, standardize_series)
    if scaler is None:
        feats_n, scaler = normalize_feature_table(feats)
    else:
        feats_n = apply_feature_scaler(feats, scaler)
    return {
        r[0]: np.asarray(r[1:], dtype=float)
        for r in feats_n.itertuples(index=False, name=None)
    }, scaler


def extract_patterns(y: np.ndarray, length: int, n_patterns: int) -> list[np.ndarray]:
    if len(y) < length:
        return [resample_1d(y, length)]
    if n_patterns <= 1:
        start = max(0, (len(y) - length) // 2)
        return [y[start : start + length].copy()]
    starts = np.linspace(0, len(y) - length, n_patterns).round().astype(int)
    return [y[s : s + length].copy() for s in starts]


def _window_split(
    y: np.ndarray, h: int, origin: str
) -> tuple[np.ndarray, np.ndarray] | None:
    y = np.asarray(y, dtype=float)
    if len(y) < 8:
        return None
    hh = min(int(h), max(2, len(y) // 4))
    if origin == "late":
        if len(y) < hh + 6:
            return None
        train, test = y[:-hh], y[-hh:]
    elif origin == "early":
        if len(y) < 2 * hh + 6:
            return None
        train, test = y[: -(2 * hh)], y[-(2 * hh) : -hh]
    else:
        raise ValueError("origin must be 'early' or 'late'")
    if len(train) < max(6, hh + 1) or len(test) < 2:
        return None
    return train, test


def retrieval_visible_series_map(
    series_map: dict[str, np.ndarray], h: int
) -> dict[str, np.ndarray]:
    """Return only the pre-holdout prefix available to retrieval systems."""
    visible = {}
    for uid, y in series_map.items():
        split = _window_split(y, h, "late")
        if split is not None:
            visible[uid] = split[0]
    return visible


def _forecast_panel_names(skip_ml_models: bool) -> list[str]:
    names = ["AutoETS", "AutoARIMA"]
    if not skip_ml_models:
        names += ["LGBM", "XGB"]
    return names


def _check_forecast_dependencies(skip_ml_models: bool) -> None:
    if StatsForecast is None or AutoETS is None or AutoARIMA is None:
        raise RuntimeError("statsforecast with AutoETS and AutoARIMA is required.")
    if not skip_ml_models:
        missing = []
        if MLForecast is None:
            missing.append("mlforecast")
        if LGBMRegressor is None:
            missing.append("lightgbm")
        if XGBRegressor is None:
            missing.append("xgboost")
        if missing:
            raise RuntimeError(
                "Missing packages for the 2 statistical + 2 ML forecasting panel: "
                + ", ".join(missing)
                + ". Install them or pass --skip-ml-models."
            )


def _normalize_forecast_output(
    fcst: pd.DataFrame, required_cols: list[str], source: str
) -> pd.DataFrame:
    out = fcst.copy()
    if "unique_id" not in out.columns or "ds" not in out.columns:
        out = out.reset_index()
    rename_map = {}
    if "unique_id" not in out.columns:
        for candidate in ["index", "level_0"]:
            if candidate in out.columns:
                rename_map[candidate] = "unique_id"
                break
    if "ds" not in out.columns and "level_1" in out.columns:
        rename_map["level_1"] = "ds"
    if rename_map:
        out = out.rename(columns=rename_map)
    missing = [col for col in required_cols if col not in out.columns]
    if missing:
        raise RuntimeError(
            f"{source} forecast output is missing columns {missing}. "
            f"Available columns: {list(out.columns)}; index names: {list(fcst.index.names)}"
        )
    out = out[required_cols].copy()
    out["unique_id"] = out["unique_id"].astype(str)
    return out


def _naive_forecasts(train_df: pd.DataFrame, h: int, model_col: str) -> pd.DataFrame:
    rows = []
    for uid, g in train_df.groupby("unique_id", sort=False):
        g = g.sort_values("ds")
        last_ds = int(g["ds"].iloc[-1])
        last_y = float(g["y"].iloc[-1])
        for step in range(1, int(h) + 1):
            rows.append((str(uid), last_ds + step, last_y))
    return pd.DataFrame(rows, columns=["unique_id", "ds", model_col])


def _statsforecast_model_forecast(
    train_df: pd.DataFrame, h: int, model_factory, model_col: str
) -> pd.DataFrame:
    sf = StatsForecast(models=[model_factory()], freq=1, n_jobs=1)
    fcst = sf.forecast(df=train_df, h=int(h))
    return _normalize_forecast_output(
        fcst, ["unique_id", "ds", model_col], "StatsForecast"
    )


def _statsforecast_model_forecast_with_tiny_fallback(
    train_df: pd.DataFrame,
    h: int,
    model_factory,
    model_col: str,
) -> pd.DataFrame:
    try:
        return _statsforecast_model_forecast(train_df, h, model_factory, model_col)
    except NotImplementedError as exc:
        if "tiny datasets" not in str(exc).lower():
            raise
    sizes = train_df.groupby("unique_id").size()
    tiny_ids = {str(uid) for uid in sizes[sizes <= 6].index}
    uid_as_str = train_df["unique_id"].astype(str)
    tiny_df = train_df[uid_as_str.isin(tiny_ids)].copy()
    other_df = train_df[~uid_as_str.isin(tiny_ids)].copy()
    pieces = []
    if not other_df.empty:
        try:
            pieces.append(
                _statsforecast_model_forecast(other_df, h, model_factory, model_col)
            )
        except NotImplementedError as exc:
            if "tiny datasets" not in str(exc).lower():
                raise
            tiny_df = pd.concat([tiny_df, other_df], ignore_index=True)
    if not tiny_df.empty:
        log(
            f"{model_col} fell back to naive forecasts for "
            f"{tiny_df['unique_id'].nunique()} tiny training windows."
        )
        pieces.append(_naive_forecasts(tiny_df, h, model_col))
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["unique_id", "ds"], kind="mergesort"
    )


def _statistical_forecasts(
    train_df: pd.DataFrame, h: int, season_length: int
) -> pd.DataFrame:
    sp_len = (
        season_length
        if h >= 2 and train_df.groupby("unique_id").size().min() > 2 * season_length
        else 1
    )
    ets = _statsforecast_model_forecast_with_tiny_fallback(
        train_df,
        h,
        lambda: AutoETS(season_length=sp_len, model="ZZZ"),
        "AutoETS",
    )
    arima = _statsforecast_model_forecast_with_tiny_fallback(
        train_df,
        h,
        lambda: AutoARIMA(season_length=sp_len),
        "AutoARIMA",
    )
    return ets.merge(arima, on=["unique_id", "ds"], how="inner")


def _ml_forecasts(
    train_df: pd.DataFrame, h: int, lags: list[int], seed: int, n_estimators: int
) -> pd.DataFrame:
    min_train = int(train_df.groupby("unique_id").size().min())
    valid_lags = [int(l) for l in lags if 0 < int(l) < min_train]
    if not valid_lags:
        valid_lags = [1]
    models = {
        "LGBM": LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        ),
        "XGB": XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        ),
    }
    fcst = MLForecast(models=models, freq=1, lags=valid_lags)
    fcst.fit(train_df)
    pred = fcst.predict(h=int(h))
    return _normalize_forecast_output(
        pred, ["unique_id", "ds", "LGBM", "XGB"], "MLForecast"
    )


def build_forecast_profile_map(
    series_map: dict[str, np.ndarray],
    h: int,
    season_length: int,
    origin: str,
    ml_lags: list[int],
    seed: int,
    ml_n_estimators: int,
    skip_ml_models: bool,
) -> dict[str, np.ndarray]:
    _check_forecast_dependencies(skip_ml_models)
    rows = []
    actuals: dict[str, np.ndarray] = {}
    horizons: dict[str, int] = {}
    for uid, y in series_map.items():
        split = _window_split(y, h, origin)
        if split is None:
            continue
        train, test = split
        hh = int(len(test))
        actuals[uid] = test
        horizons[uid] = hh
        for t, val in enumerate(train, start=1):
            rows.append((uid, t, float(val), hh))
    if not rows:
        raise ValueError(
            f"No objects are long enough to build {origin} forecast profiles."
        )
    train_all = pd.DataFrame(rows, columns=["unique_id", "ds", "y", "h"])
    profiles: dict[str, np.ndarray] = {}
    expected_cols = _forecast_panel_names(skip_ml_models)
    for hh in sorted(train_all["h"].unique()):
        sub = train_all[train_all["h"] == hh][["unique_id", "ds", "y"]].copy()
        log(
            f"Forecast profiles ({origin}): {sub['unique_id'].nunique()} objects, h={int(hh)}, panel={'+'.join(expected_cols)}"
        )
        stat_fcst = _statistical_forecasts(sub, int(hh), season_length)
        if skip_ml_models:
            fcst = stat_fcst
        else:
            ml_fcst = _ml_forecasts(sub, int(hh), ml_lags, seed, ml_n_estimators)
            fcst = stat_fcst.merge(ml_fcst, on=["unique_id", "ds"], how="inner")
        missing_cols = [c for c in expected_cols if c not in fcst.columns]
        if missing_cols:
            raise RuntimeError(
                f"Forecast panel did not produce expected columns: {missing_cols}"
            )
        for uid, g in fcst.groupby("unique_id", sort=False):
            true = actuals[uid][: len(g)]
            feats = []
            model_mae = []
            for col in expected_cols:
                pred = g[col].to_numpy(dtype=float)
                err = pred - true
                mae = float(np.mean(np.abs(err)))
                rmse = float(np.sqrt(np.mean(err**2)))
                denom = np.abs(true) + np.abs(pred) + 1e-8
                smape = float(np.mean(200.0 * np.abs(err) / denom))
                bias = float(np.mean(err))
                feats.extend([mae, rmse, smape, bias])
                model_mae.append(mae)
            order = np.argsort(model_mae)
            rank_vec = np.empty(len(model_mae), dtype=float)
            for rank, idx in enumerate(order):
                rank_vec[idx] = rank
            feats.extend(rank_vec.tolist())
            mean_mae = float(np.mean(model_mae))
            std_mae = float(np.std(model_mae))
            sorted_mae = sorted(model_mae)
            best_gap = (
                float(sorted_mae[1] - sorted_mae[0]) if len(sorted_mae) > 1 else 0.0
            )
            feats.extend([mean_mae, std_mae, best_gap])
            profiles[uid] = np.asarray(feats, dtype=float)
    return profiles


def _profile_summary(profile: np.ndarray, panel_names: list[str]) -> dict[str, Any]:
    profile = np.asarray(profile, dtype=float)
    n_models = len(panel_names)
    if len(profile) < 5 * n_models + 3:
        raise ValueError(
            "Forecast profile has unexpected length for the supplied forecast panel."
        )
    model_mae = np.array([profile[4 * i] for i in range(n_models)], dtype=float)
    best_idx = int(np.argmin(model_mae))
    return {
        "model_mae": model_mae,
        "best_model": panel_names[best_idx],
        "best_model_index": best_idx,
        "mean_mae": float(profile[5 * n_models]),
        "disagreement": float(profile[5 * n_models + 1]),
        "best_gap": float(profile[5 * n_models + 2]),
    }


def _quantile_thresholds(values: list[float], n_bins: int) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.asarray([], dtype=float)
    if n_bins <= 1 or float(np.nanmax(vals) - np.nanmin(vals)) <= 1e-12:
        return np.asarray([], dtype=float)
    qs = [i / n_bins for i in range(1, n_bins)]
    thresholds = np.quantile(vals, qs)
    thresholds = np.asarray(sorted(set(float(x) for x in thresholds)), dtype=float)
    return thresholds


def _assign_bin(value: float, thresholds: np.ndarray, n_bins: int) -> int:
    if n_bins <= 1 or len(thresholds) == 0:
        return 0
    return int(
        min(n_bins - 1, max(0, np.searchsorted(thresholds, float(value), side="right")))
    )


def forecast_regime_table_from_profiles(
    profile_map: dict[str, np.ndarray],
    index_ids: list[str],
    panel_names: list[str],
    n_bins: int = 3,
) -> pd.DataFrame:
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2 for regime qrels.")
    summaries = {
        uid: _profile_summary(vec, panel_names) for uid, vec in profile_map.items()
    }
    ref_ids = [uid for uid in index_ids if uid in summaries]
    if not ref_ids:
        return pd.DataFrame(
            columns=[
                "object_id",
                "mean_mae",
                "disagreement",
                "best_gap",
                "best_model",
                "best_model_index",
                "error_bin",
                "disagreement_bin",
                "error_regime",
                "disagreement_regime",
            ]
        )
    error_thresholds = _quantile_thresholds(
        [summaries[uid]["mean_mae"] for uid in ref_ids], n_bins
    )
    disagreement_thresholds = _quantile_thresholds(
        [summaries[uid]["disagreement"] for uid in ref_ids], n_bins
    )
    bin_names = (
        ["low", "medium", "high"]
        if n_bins == 3
        else [f"bin_{i}" for i in range(n_bins)]
    )
    rows = []
    for uid, summary in summaries.items():
        error_bin = _assign_bin(summary["mean_mae"], error_thresholds, n_bins)
        disagreement_bin = _assign_bin(
            summary["disagreement"], disagreement_thresholds, n_bins
        )
        rows.append(
            {
                "object_id": uid,
                "mean_mae": summary["mean_mae"],
                "disagreement": summary["disagreement"],
                "best_gap": summary["best_gap"],
                "best_model": summary["best_model"],
                "best_model_index": summary["best_model_index"],
                "error_bin": error_bin,
                "disagreement_bin": disagreement_bin,
                "error_regime": (
                    bin_names[error_bin]
                    if error_bin < len(bin_names)
                    else str(error_bin)
                ),
                "disagreement_regime": (
                    bin_names[disagreement_bin]
                    if disagreement_bin < len(bin_names)
                    else str(disagreement_bin)
                ),
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["error_thresholds"] = error_thresholds.tolist()
    df.attrs["disagreement_thresholds"] = disagreement_thresholds.tolist()
    df.attrs["n_bins"] = n_bins
    return df


def regime_relevance(
    query: pd.Series, candidate: pd.Series, definition: str = "full"
) -> int:
    same_error = int(query["error_bin"]) == int(candidate["error_bin"])
    adjacent_error = abs(int(query["error_bin"]) - int(candidate["error_bin"])) == 1
    same_model = str(query["best_model"]) == str(candidate["best_model"])
    same_disagreement = int(query["disagreement_bin"]) == int(
        candidate["disagreement_bin"]
    )
    if definition == "error_only":
        if same_error:
            return 2
        if adjacent_error:
            return 1
        return 0
    if definition == "error_model":
        if same_error and same_model:
            return 3
        if same_error:
            return 2
        if adjacent_error and same_model:
            return 1
        return 0
    if definition == "error_disagreement":
        if same_error and same_disagreement:
            return 3
        if same_error:
            return 2
        if adjacent_error and same_disagreement:
            return 1
        return 0
    if definition != "full":
        raise ValueError(
            "regime_definition must be one of: full, error_only, error_model, error_disagreement"
        )
    if same_error and same_model and same_disagreement:
        return 4
    if same_error and (same_model or same_disagreement):
        return 3
    if same_error:
        return 2
    if adjacent_error and same_model and same_disagreement:
        return 1
    return 0


def qrels_from_forecast_regimes(
    query_ids: list[str],
    index_ids: list[str],
    regime_table: pd.DataFrame,
    definition: str = "full",
) -> pd.DataFrame:
    if regime_table.empty:
        return pd.DataFrame(columns=["query_id", "doc_id", "relevance"])
    lookup = {str(r.object_id): r for r in regime_table.itertuples(index=False)}
    rows = []
    for qid in query_ids:
        q = lookup.get(str(qid))
        if q is None:
            continue
        q_series = pd.Series(q._asdict())
        for did in index_ids:
            d = lookup.get(str(did))
            if d is None:
                continue
            d_series = pd.Series(d._asdict())
            rel = regime_relevance(q_series, d_series, definition)
            if rel > 0:
                rows.append((qid, did, int(rel)))
    return pd.DataFrame(rows, columns=["query_id", "doc_id", "relevance"])


def random_score_bank(
    query_ids: list[str], index_ids: list[str], system_name: str, seed: int
) -> dict[str, dict[str, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    out = {}
    for qid in query_ids:
        out[qid] = {system_name: {uid: float(rng.random()) for uid in index_ids}}
    return out


def score_raw(
    query_y: np.ndarray,
    cand_series: dict[str, np.ndarray],
    resample_len: int,
    mode: str,
) -> dict[str, float]:
    q = zscore(resample_1d(query_y, resample_len))
    out = {}
    for cid, arr in cand_series.items():
        x = zscore(resample_1d(arr, resample_len))
        out[cid] = (
            cosine_sim(q, x)
            if mode == "cosine"
            else -dtw_distance(q, x, window=max(4, resample_len // 8))
        )
    return out


def score_tsfel(
    query_feat: np.ndarray, cand_feats: dict[str, np.ndarray]
) -> dict[str, float]:
    return {cid: cosine_sim(query_feat, vec) for cid, vec in cand_feats.items()}


def minmax_scores(scores: dict[str, float]) -> dict[str, float]:
    vals = np.array(list(scores.values()), float)
    lo, hi = float(vals.min()), float(vals.max())
    return (
        {k: 0.0 for k in scores}
        if hi - lo < 1e-12
        else {k: float((v - lo) / (hi - lo)) for k, v in scores.items()}
    )


def fuse_scores(
    score_dicts: list[dict[str, float]], weights: list[float]
) -> dict[str, float]:
    normed = [minmax_scores(s) for s in score_dicts]
    keys = set(score_dicts[0])
    for s in score_dicts[1:]:
        keys &= set(s)
    return {k: float(sum(w * s[k] for w, s in zip(weights, normed))) for k in keys}


def rank_from_scores(query_id: str, scores: dict[str, float]) -> pd.DataFrame:
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return pd.DataFrame(
        {
            "query_id": [query_id] * len(items),
            "doc_id": [k for k, _ in items],
            "score": [float(v) for _, v in items],
            "rank": list(range(1, len(items) + 1)),
        }
    )


def dcg(rels: list[int]) -> float:
    return float(sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels)))


def evaluate_query_metrics(
    rankings: pd.DataFrame, qrels: pd.DataFrame, k_values: tuple[int, int] = (10, 15)
) -> pd.DataFrame:
    cols = [
        "query_id",
        "total_relevant",
        *[f"{m}@{k}" for k in k_values for m in ["p", "ap", "ndcg"]],
    ]
    if qrels.empty or rankings.empty:
        return pd.DataFrame(columns=cols)
    qmap = {
        (str(r.query_id), str(r.doc_id)): int(r.relevance)
        for r in qrels.itertuples(index=False)
    }
    positives_by_query = (
        qrels.groupby("query_id")["relevance"]
        .apply(lambda s: int((s > 0).sum()))
        .to_dict()
    )
    positives_by_query = {str(k): int(v) for k, v in positives_by_query.items()}
    queries = [str(qid) for qid, npos in positives_by_query.items() if npos > 0]
    rows = []
    for qid in queries:
        sub = rankings[rankings["query_id"].astype(str) == qid].sort_values("rank")
        rel_all = [qmap.get((qid, str(d)), 0) for d in sub["doc_id"].tolist()]
        ideal = sorted([r for (qq, _), r in qmap.items() if qq == qid], reverse=True)
        total_relevant = int(positives_by_query[qid])
        row = {"query_id": qid, "total_relevant": total_relevant}
        for k in k_values:
            rel, ideal_k = rel_all[:k], ideal[:k]
            row[f"p@{k}"] = float(sum(r > 0 for r in rel) / max(1, len(rel)))
            nrel = 0
            ap = 0.0
            for i, r in enumerate(rel, start=1):
                if r > 0:
                    nrel += 1
                    ap += nrel / i
            row[f"ap@{k}"] = float(ap / max(1, total_relevant))
            idcg = dcg(ideal_k)
            row[f"ndcg@{k}"] = float(dcg(rel) / idcg) if idcg > 0 else 0.0
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def evaluate_run(
    rankings: pd.DataFrame, qrels: pd.DataFrame, k_values: tuple[int, int] = (10, 15)
) -> dict[str, float]:
    df = evaluate_query_metrics(rankings, qrels, k_values)
    if df.empty:
        return {
            "n_queries": 0,
            **{f"{m}@{k}": 0.0 for m in ["p", "ap", "ndcg"] for k in k_values},
        }
    out = {"n_queries": int(len(df))}
    for k in k_values:
        for m in ["p", "ap", "ndcg"]:
            out[f"{m}@{k}"] = float(df[f"{m}@{k}"].mean())
    return out


def query_metrics_for_systems(
    rankings_by_system: dict[str, pd.DataFrame],
    qrels: pd.DataFrame,
    systems: list[str],
    k_values: tuple[int, int] = (10, 15),
) -> pd.DataFrame:
    frames = []
    for system in systems:
        df = evaluate_query_metrics(
            rankings_by_system.get(system, pd.DataFrame()), qrels, k_values
        )
        if df.empty:
            continue
        df.insert(0, "system", system)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def qrels_summary(
    qrels: pd.DataFrame, query_ids: list[str], task: str, split: str
) -> pd.DataFrame:
    query_ids = [str(qid) for qid in query_ids]
    if qrels.empty:
        base = {
            "task": task,
            "split": split,
            "n_queries": len(query_ids),
            "n_queries_with_positive_qrels": 0,
            "qrels_rows": 0,
            "mean_positive_qrels_per_query": 0.0,
            "median_positive_qrels_per_query": 0.0,
            "min_positive_qrels_per_query": 0,
            "max_positive_qrels_per_query": 0,
            "mean_relevance": np.nan,
        }
        for grade in range(1, 5):
            base[f"relevance_{grade}_rows"] = 0
        return pd.DataFrame([base])

    per_query = (
        qrels.assign(query_id=qrels["query_id"].astype(str))
        .groupby("query_id")
        .size()
        .reindex(query_ids, fill_value=0)
    )
    grade_counts = (
        pd.to_numeric(qrels["relevance"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .to_dict()
    )
    row = {
        "task": task,
        "split": split,
        "n_queries": len(query_ids),
        "n_queries_with_positive_qrels": int((per_query > 0).sum()),
        "qrels_rows": int(len(qrels)),
        "mean_positive_qrels_per_query": float(per_query.mean()),
        "median_positive_qrels_per_query": float(per_query.median()),
        "min_positive_qrels_per_query": int(per_query.min()) if len(per_query) else 0,
        "max_positive_qrels_per_query": int(per_query.max()) if len(per_query) else 0,
        "mean_relevance": float(
            pd.to_numeric(qrels["relevance"], errors="coerce").mean()
        ),
    }
    for grade in range(1, 5):
        row[f"relevance_{grade}_rows"] = int(grade_counts.get(grade, 0))
    return pd.DataFrame([row])


def bootstrap_mean_ci(
    values: pd.Series | np.ndarray,
    seed: int,
    n_boot: int = IR_N_BOOTSTRAPS,
    alpha: float = 0.05,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def paired_randomization_pvalue(
    deltas: pd.Series | np.ndarray,
    seed: int,
    n_perm: int = IR_N_PERMUTATIONS,
) -> float:
    arr = np.asarray(deltas, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    observed = abs(float(arr.mean()))
    if observed < 1e-15:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(arr)))
    permuted = np.abs((signs * arr).mean(axis=1))
    return float((np.sum(permuted >= observed) + 1) / (n_perm + 1))


def _query_metric_frame(task: str, result: dict[str, Any]) -> pd.DataFrame:
    df = result.get("dataframes", {}).get("test_query_metrics", pd.DataFrame()).copy()
    if df.empty:
        return pd.DataFrame()
    df.insert(0, "task", task)
    return df


def _qrels_summary_frame(task: str, result: dict[str, Any]) -> pd.DataFrame:
    df = result.get("dataframes", {}).get("qrels_test_summary", pd.DataFrame()).copy()
    if df.empty:
        return pd.DataFrame()
    if "task" not in df.columns:
        df.insert(0, "task", task)
    return df


def system_uncertainty_table(
    query_metrics: pd.DataFrame, seed: int, metric: str = "ndcg@10"
) -> pd.DataFrame:
    if query_metrics.empty or metric not in query_metrics.columns:
        return pd.DataFrame()
    rows = []
    for i, ((task, system), group) in enumerate(
        query_metrics.groupby(["task", "system"], dropna=False)
    ):
        vals = pd.to_numeric(group[metric], errors="coerce")
        lo, hi = bootstrap_mean_ci(vals, seed + 1000 + i)
        row = {
            "task": task,
            "system": system,
            "metric": metric,
            "n_queries": int(vals.notna().sum()),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0,
            "ci95_low": lo,
            "ci95_high": hi,
        }
        for extra in ["p@10", "ap@10", "ndcg@15", "p@15", "ap@15"]:
            if extra in group.columns:
                row[f"mean_{extra}"] = float(
                    pd.to_numeric(group[extra], errors="coerce").mean()
                )
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_random_tests(
    query_metrics: pd.DataFrame, seed: int, metric: str = "ndcg@10"
) -> pd.DataFrame:
    if query_metrics.empty or metric not in query_metrics.columns:
        return pd.DataFrame()
    rows = []
    for task, task_df in query_metrics.groupby("task", dropna=False):
        random_df = task_df[task_df["system"] == "random"][
            ["query_id", metric]
        ].rename(columns={metric: "random_metric"})
        if random_df.empty:
            continue
        for i, (system, sys_df) in enumerate(task_df.groupby("system", dropna=False)):
            if system == "random":
                continue
            paired = sys_df[["query_id", metric]].merge(
                random_df, on="query_id", how="inner"
            )
            if paired.empty:
                continue
            deltas = (
                pd.to_numeric(paired[metric], errors="coerce")
                - pd.to_numeric(paired["random_metric"], errors="coerce")
            )
            lo, hi = bootstrap_mean_ci(deltas, seed + 2000 + i)
            rows.append(
                {
                    "task": task,
                    "system": system,
                    "baseline": "random",
                    "metric": metric,
                    "n_queries": int(deltas.notna().sum()),
                    "mean_system": float(pd.to_numeric(paired[metric]).mean()),
                    "mean_baseline": float(
                        pd.to_numeric(paired["random_metric"]).mean()
                    ),
                    "mean_delta": float(deltas.mean()),
                    "ci95_delta_low": lo,
                    "ci95_delta_high": hi,
                    "paired_randomization_p": paired_randomization_pvalue(
                        deltas, seed + 3000 + i
                    ),
                    "wins": int((deltas > 0).sum()),
                    "ties": int((np.isclose(deltas, 0.0)).sum()),
                    "losses": int((deltas < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def write_query_delta_plot(
    query_metrics: pd.DataFrame, pattern_metrics: pd.DataFrame, outdir: Path
) -> str | None:
    if query_metrics.empty or pattern_metrics.empty:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(
            f"Skipping query-level diagnostic plot because matplotlib is unavailable: {type(exc).__name__}"
        )
        return None
    content_systems = [
        "pattern_raw_cosine",
        "pattern_raw_dtw",
        "pattern_tsfel",
        "pattern_raw_cosine+pattern_tsfel",
    ]
    metric_rows = pattern_metrics[pattern_metrics["system"].isin(content_systems)]
    if metric_rows.empty:
        return None
    best_system = (
        metric_rows.sort_values("ndcg@10", ascending=False).iloc[0]["system"]
    )
    task_df = query_metrics[query_metrics["task"] == "task2"].copy()
    best = task_df[task_df["system"] == best_system][["query_id", "ndcg@10"]].rename(
        columns={"ndcg@10": "best_ndcg@10"}
    )
    random_df = task_df[task_df["system"] == "random"][["query_id", "ndcg@10"]].rename(
        columns={"ndcg@10": "random_ndcg@10"}
    )
    paired = best.merge(random_df, on="query_id", how="inner")
    if paired.empty:
        return None
    paired["delta"] = paired["best_ndcg@10"] - paired["random_ndcg@10"]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.hist(paired["delta"], bins=24, color="#4c78a8", edgecolor="white")
    ax.axvline(0.0, color="#111111", linewidth=1.1)
    ax.axvline(float(paired["delta"].mean()), color="#f58518", linewidth=1.5)
    ax.set_title(f"Query-level nDCG@10 delta vs random: {best_system}")
    ax.set_xlabel("nDCG@10(system) - nDCG@10(random)")
    ax.set_ylabel("Number of queries")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = outdir / "fig_query_ndcg10_delta_vs_random.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def write_ir_diagnostics(
    cfg: Config,
    whole: dict[str, Any],
    pattern_main: dict[str, Any],
    length_results: list[dict[str, Any]],
    outdir: Path,
) -> dict[str, Any]:
    query_frames = [
        _query_metric_frame("task1", whole),
        _query_metric_frame("task2", pattern_main),
    ]
    query_metrics = pd.concat(
        [df for df in query_frames if not df.empty], ignore_index=True
    )
    qrels_frames = [
        _qrels_summary_frame("task1", whole),
        _qrels_summary_frame("task2", pattern_main),
    ]
    for result in length_results:
        task = "task2" if int(result.get("length", -1)) == cfg.pattern_len else f"task2_L{result.get('length')}"
        qrels_frames.append(_qrels_summary_frame(task, result))
    qrels_table = pd.concat(
        [df for df in qrels_frames if not df.empty], ignore_index=True
    )
    metrics_for_tests = [
        metric for metric in ["ndcg@10", "ndcg@15"] if metric in query_metrics.columns
    ]
    uncertainty = (
        pd.concat(
            [
                system_uncertainty_table(query_metrics, cfg.seed, metric)
                for metric in metrics_for_tests
            ],
            ignore_index=True,
        )
        if metrics_for_tests
        else pd.DataFrame()
    )
    pairwise = (
        pd.concat(
            [
                pairwise_random_tests(query_metrics, cfg.seed, metric)
                for metric in metrics_for_tests
            ],
            ignore_index=True,
        )
        if metrics_for_tests
        else pd.DataFrame()
    )

    artifacts: dict[str, Any] = {}
    if not query_metrics.empty:
        path = outdir / "ir_query_metrics_test.csv"
        query_metrics.to_csv(path, index=False)
        artifacts["query_metrics_test"] = str(path)
    if not qrels_table.empty:
        path = outdir / "ir_qrels_summary.csv"
        qrels_table.drop_duplicates(
            subset=[c for c in ["task", "split"] if c in qrels_table.columns]
        ).to_csv(path, index=False)
        artifacts["qrels_summary"] = str(path)
    if not uncertainty.empty:
        path = outdir / "ir_system_uncertainty.csv"
        uncertainty.to_csv(path, index=False)
        artifacts["system_uncertainty"] = str(path)
    if not pairwise.empty:
        path = outdir / "ir_pairwise_random_tests.csv"
        pairwise.to_csv(path, index=False)
        artifacts["pairwise_random_tests"] = str(path)
    plot_path = write_query_delta_plot(
        query_metrics,
        pattern_main.get("dataframes", {}).get("test_metrics", pd.DataFrame()),
        outdir,
    )
    if plot_path:
        artifacts["query_delta_plot"] = plot_path
    return artifacts


def tune_pairwise_fusion(
    dev_a: dict[str, dict[str, float]],
    dev_b: dict[str, dict[str, float]],
    qrels_dev: pd.DataFrame,
    name: str,
) -> float:
    best_alpha, best_val = 1.0, -1.0
    common = sorted(set(dev_a) & set(dev_b))
    for alpha in np.linspace(0, 1, 11):
        rk = pd.concat(
            [
                rank_from_scores(
                    q, fuse_scores([dev_a[q], dev_b[q]], [alpha, 1 - alpha])
                )
                for q in common
            ],
            ignore_index=True,
        )
        mt = evaluate_run(rk, qrels_dev)
        if mt["ndcg@10"] > best_val:
            best_alpha, best_val = float(alpha), mt["ndcg@10"]
    log(f"Tuned {name}: alpha={best_alpha:.2f}, dev ndcg@10={best_val:.4f}")
    return best_alpha


def summarize_systems(
    score_bank: dict[str, dict[str, dict[str, float]]],
    qrels: pd.DataFrame,
    systems: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    metrics_rows, rankings_out = [], {}
    for system in systems:
        queries = [q for q in score_bank if system in score_bank[q]]
        if not queries:
            metrics_rows.append(
                {"system": system, **evaluate_run(pd.DataFrame(), qrels)}
            )
            rankings_out[system] = pd.DataFrame()
            continue
        rk = pd.concat(
            [rank_from_scores(q, score_bank[q][system]) for q in queries],
            ignore_index=True,
        )
        metrics_rows.append({"system": system, **evaluate_run(rk, qrels)})
        rankings_out[system] = rk
    return pd.DataFrame(metrics_rows), rankings_out


def merge_score_banks(
    *banks: dict[str, dict[str, dict[str, float]]]
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for bank in banks:
        for qid, systems in bank.items():
            out.setdefault(qid, {}).update(systems)
    return out


def whole_task(
    query_ids: list[str],
    index_ids: list[str],
    series_map: dict[str, np.ndarray],
    index_tsfel: dict[str, np.ndarray],
    tsfel_scaler: StandardScaler,
    cfg: Config,
    split_name: str,
) -> dict[str, dict[str, dict[str, float]]]:
    score_bank = {}
    cand_series = {k: series_map[k] for k in index_ids if k in series_map}
    for i, qid in enumerate(query_ids, start=1):
        if qid not in series_map:
            continue
        log(f"Task 1 [{split_name}] query {i}/{len(query_ids)}: {qid}")
        q = series_map[qid]
        q_feat = build_candidate_tsfel(
            {qid: q}, cfg.tsfel_standardize_series, tsfel_scaler
        )[0][qid]
        score_bank[qid] = {
            "raw_cosine": score_raw(q, cand_series, cfg.raw_resample_len, "cosine"),
            "raw_dtw": score_raw(q, cand_series, cfg.raw_resample_len, "dtw"),
            "tsfel": score_tsfel(q_feat, index_tsfel),
        }
    return score_bank


def build_pattern_objects(
    series_map: dict[str, np.ndarray], length: int, n_patterns: int
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    pattern_series_all = {}
    pattern_owner = {}
    for uid, y in series_map.items():
        pats = extract_patterns(y, length, n_patterns)
        for j, arr in enumerate(pats):
            pid = f"{uid}::pattern{j}::L{length}"
            pattern_series_all[pid] = arr
            pattern_owner[pid] = uid
    return pattern_series_all, pattern_owner


def pattern_task(
    query_ids: list[str],
    index_pattern_series: dict[str, np.ndarray],
    index_pattern_tsfel: dict[str, np.ndarray],
    pattern_tsfel_scaler: StandardScaler,
    pattern_series_all: dict[str, np.ndarray],
    cfg: Config,
    split_name: str,
) -> dict[str, dict[str, dict[str, float]]]:
    score_bank = {}
    for i, qid in enumerate(query_ids, start=1):
        if qid not in pattern_series_all:
            continue
        qpat = pattern_series_all[qid]
        log(f"Task 2 [{split_name}] query {i}/{len(query_ids)}: {qid}")
        q_feat = build_candidate_tsfel(
            {qid: qpat}, cfg.tsfel_standardize_series, pattern_tsfel_scaler
        )[0][qid]
        score_bank[qid] = {
            "pattern_raw_cosine": score_raw(
                qpat, index_pattern_series, cfg.raw_resample_len, "cosine"
            ),
            "pattern_raw_dtw": score_raw(
                qpat, index_pattern_series, cfg.raw_resample_len, "dtw"
            ),
            "pattern_tsfel": score_tsfel(q_feat, index_pattern_tsfel),
        }
    return score_bank


def add_fusion_system(
    score_bank_dev: dict[str, dict[str, dict[str, float]]],
    score_bank_test: dict[str, dict[str, dict[str, float]]],
    qrels_dev: pd.DataFrame,
    left: str,
    right: str,
    fused_name: str,
) -> dict[str, Any]:
    dev_a = {
        q: score_bank_dev[q][left]
        for q in score_bank_dev
        if left in score_bank_dev[q] and right in score_bank_dev[q]
    }
    dev_b = {
        q: score_bank_dev[q][right]
        for q in score_bank_dev
        if left in score_bank_dev[q] and right in score_bank_dev[q]
    }
    alpha = tune_pairwise_fusion(dev_a, dev_b, qrels_dev, fused_name)
    for bank in [score_bank_dev, score_bank_test]:
        for q in list(bank):
            if left in bank[q] and right in bank[q]:
                bank[q][fused_name] = fuse_scores(
                    [bank[q][left], bank[q][right]], [alpha, 1 - alpha]
                )
    return {"system": fused_name, "weights": {left: alpha, right: 1 - alpha}}


def run_pattern_experiment_for_length(
    length: int,
    series_map: dict[str, np.ndarray],
    splits: dict[str, list[str]],
    cfg: Config,
    outdir: Path,
    save_prefix: str | None,
) -> dict[str, Any]:
    log(f"Preparing pattern experiment for length={length}")
    ml_lags = parse_int_list(cfg.ml_lags, [1, 2, 3, 4, 6, 12])
    pattern_h = max(2, min(cfg.forecast_h, max(2, length // 4)))
    pattern_series_all, pattern_owner = build_pattern_objects(
        series_map, length, cfg.patterns_per_series
    )
    index_pattern_ids = [
        pid for pid, owner in pattern_owner.items() if owner in set(splits["index"])
    ]
    dev_pattern_ids = [
        pid for pid, owner in pattern_owner.items() if owner in set(splits["dev"])
    ]
    test_pattern_ids = [
        pid for pid, owner in pattern_owner.items() if owner in set(splits["test"])
    ]

    log(f"Building late qrels profiles for patterns, length={length}")
    profiles_late = build_forecast_profile_map(
        pattern_series_all,
        pattern_h,
        1,
        "late",
        ml_lags,
        cfg.seed,
        cfg.ml_n_estimators,
        cfg.skip_ml_models,
    )
    panel_names = _forecast_panel_names(cfg.skip_ml_models)
    late_regimes = forecast_regime_table_from_profiles(
        profiles_late, index_pattern_ids, panel_names, cfg.qrels_n_bins
    )
    qrels_dev = qrels_from_forecast_regimes(
        dev_pattern_ids, index_pattern_ids, late_regimes, cfg.regime_definition
    )
    qrels_test = qrels_from_forecast_regimes(
        test_pattern_ids, index_pattern_ids, late_regimes, cfg.regime_definition
    )
    pattern_retrieval_series_all = retrieval_visible_series_map(
        pattern_series_all, pattern_h
    )

    log(f"Building index pattern TSFEL, length={length}")
    index_pattern_series = {
        pid: pattern_retrieval_series_all[pid]
        for pid in index_pattern_ids
        if pid in pattern_retrieval_series_all
    }
    index_pattern_tsfel_raw = extract_tsfel_features(
        index_pattern_series, cfg.tsfel_standardize_series
    )
    index_pattern_tsfel_norm, pattern_tsfel_scaler = normalize_feature_table(
        index_pattern_tsfel_raw
    )
    index_pattern_tsfel = {
        r[0]: np.asarray(r[1:], dtype=float)
        for r in index_pattern_tsfel_norm.itertuples(index=False, name=None)
    }

    log(f"Running content pattern retrieval, length={length}")
    content_dev = pattern_task(
        dev_pattern_ids,
        index_pattern_series,
        index_pattern_tsfel,
        pattern_tsfel_scaler,
        pattern_retrieval_series_all,
        cfg,
        f"dev L={length}",
    )
    content_test = pattern_task(
        test_pattern_ids,
        index_pattern_series,
        index_pattern_tsfel,
        pattern_tsfel_scaler,
        pattern_retrieval_series_all,
        cfg,
        f"test L={length}",
    )

    random_dev = random_score_bank(
        dev_pattern_ids, index_pattern_ids, "random", cfg.seed + length + 10
    )
    random_test = random_score_bank(
        test_pattern_ids, index_pattern_ids, "random", cfg.seed + length + 20
    )

    dev_bank = merge_score_banks(content_dev, random_dev)
    test_bank = merge_score_banks(content_test, random_test)
    fusion = add_fusion_system(
        dev_bank,
        test_bank,
        qrels_dev,
        "pattern_raw_cosine",
        "pattern_tsfel",
        "pattern_raw_cosine+pattern_tsfel",
    )

    systems = [
        "random",
        "pattern_raw_cosine",
        "pattern_raw_dtw",
        "pattern_tsfel",
        "pattern_raw_cosine+pattern_tsfel",
    ]
    dev_metrics, dev_rankings = summarize_systems(dev_bank, qrels_dev, systems)
    test_metrics, test_rankings = summarize_systems(test_bank, qrels_test, systems)
    test_query_metrics = query_metrics_for_systems(test_rankings, qrels_test, systems)
    qrels_test_summary = qrels_summary(
        qrels_test, test_pattern_ids, save_prefix or f"task2_L{length}", "test"
    )
    dev_metrics = dev_metrics.sort_values(
        ["ndcg@10", "ap@10", "p@10"], ascending=False
    ).reset_index(drop=True)
    test_metrics = test_metrics.sort_values(
        ["ndcg@10", "ap@10", "p@10"], ascending=False
    ).reset_index(drop=True)

    if save_prefix:
        qrels_test.to_csv(outdir / f"{save_prefix}_qrels_test.csv", index=False)
        late_regimes.to_csv(
            outdir / f"{save_prefix}_late_forecast_regimes.csv", index=False
        )
        test_metrics.to_csv(outdir / f"{save_prefix}_metrics_test.csv", index=False)
        test_query_metrics.to_csv(
            outdir / f"{save_prefix}_query_metrics_test.csv", index=False
        )
        qrels_test_summary.to_csv(
            outdir / f"{save_prefix}_qrels_summary_test.csv", index=False
        )
        index_pattern_tsfel_raw.to_csv(
            outdir / f"{save_prefix}_tsfel_index_raw.csv", index=False
        )
        if cfg.save_dev_artifacts:
            qrels_dev.to_csv(outdir / f"{save_prefix}_qrels_dev.csv", index=False)
            dev_metrics.to_csv(outdir / f"{save_prefix}_metrics_dev.csv", index=False)
        if cfg.save_rankings:
            for system, rk in test_rankings.items():
                rk.to_csv(outdir / f"{save_prefix}_rankings_{system}.csv", index=False)

    return {
        "length": length,
        "pattern_forecast_h": pattern_h,
        "n_index_patterns": len(index_pattern_ids),
        "n_dev_patterns": len(dev_pattern_ids),
        "n_test_patterns": len(test_pattern_ids),
        "qrels_rows": {"dev": int(len(qrels_dev)), "test": int(len(qrels_test))},
        "late_regime_counts": df_to_records(
            late_regimes.groupby(["error_regime", "best_model", "disagreement_regime"])
            .size()
            .reset_index(name="n")
        ),
        "fusion": fusion,
        "metrics": {
            "dev": df_to_records(dev_metrics),
            "test": df_to_records(test_metrics),
        },
        "dataframes": {
            "dev_metrics": dev_metrics,
            "test_metrics": test_metrics,
            "test_query_metrics": test_query_metrics,
            "qrels_test_summary": qrels_test_summary,
        },
    }


def run_whole_experiment(
    series_map: dict[str, np.ndarray],
    splits: dict[str, list[str]],
    cfg: Config,
    outdir: Path,
) -> dict[str, Any]:
    log("Preparing whole-series experiment")
    ml_lags = parse_int_list(cfg.ml_lags, [1, 2, 3, 4, 6, 12])

    log("Building late qrels profiles for whole series")
    profiles_late = build_forecast_profile_map(
        series_map,
        cfg.forecast_h,
        cfg.seasonal_period,
        "late",
        ml_lags,
        cfg.seed,
        cfg.ml_n_estimators,
        cfg.skip_ml_models,
    )
    panel_names = _forecast_panel_names(cfg.skip_ml_models)
    late_regimes = forecast_regime_table_from_profiles(
        profiles_late, splits["index"], panel_names, cfg.qrels_n_bins
    )
    qrels_dev = qrels_from_forecast_regimes(
        splits["dev"], splits["index"], late_regimes, cfg.regime_definition
    )
    qrels_test = qrels_from_forecast_regimes(
        splits["test"], splits["index"], late_regimes, cfg.regime_definition
    )
    retrieval_series = retrieval_visible_series_map(series_map, cfg.forecast_h)

    log("Building index TSFEL for whole series")
    index_series = {
        k: retrieval_series[k] for k in splits["index"] if k in retrieval_series
    }
    tsfel_index_raw = extract_tsfel_features(index_series, cfg.tsfel_standardize_series)
    tsfel_index_raw.to_csv(outdir / "task1_tsfel_index_raw.csv", index=False)
    tsfel_index_norm, tsfel_scaler = normalize_feature_table(tsfel_index_raw)
    index_tsfel = {
        r[0]: np.asarray(r[1:], dtype=float)
        for r in tsfel_index_norm.itertuples(index=False, name=None)
    }

    log("Running whole-series content retrieval")
    content_dev = whole_task(
        splits["dev"],
        splits["index"],
        retrieval_series,
        index_tsfel,
        tsfel_scaler,
        cfg,
        "dev",
    )
    content_test = whole_task(
        splits["test"],
        splits["index"],
        retrieval_series,
        index_tsfel,
        tsfel_scaler,
        cfg,
        "test",
    )

    random_dev = random_score_bank(
        splits["dev"], splits["index"], "random", cfg.seed + 1001
    )
    random_test = random_score_bank(
        splits["test"], splits["index"], "random", cfg.seed + 1002
    )

    dev_bank = merge_score_banks(content_dev, random_dev)
    test_bank = merge_score_banks(content_test, random_test)
    fusion = add_fusion_system(
        dev_bank, test_bank, qrels_dev, "raw_dtw", "tsfel", "raw_dtw+tsfel"
    )

    systems = ["random", "raw_cosine", "raw_dtw", "tsfel", "raw_dtw+tsfel"]
    dev_metrics, dev_rankings = summarize_systems(dev_bank, qrels_dev, systems)
    test_metrics, test_rankings = summarize_systems(test_bank, qrels_test, systems)
    test_query_metrics = query_metrics_for_systems(test_rankings, qrels_test, systems)
    qrels_test_summary = qrels_summary(qrels_test, splits["test"], "task1", "test")
    dev_metrics = dev_metrics.sort_values(
        ["ndcg@10", "ap@10", "p@10"], ascending=False
    ).reset_index(drop=True)
    test_metrics = test_metrics.sort_values(
        ["ndcg@10", "ap@10", "p@10"], ascending=False
    ).reset_index(drop=True)

    qrels_test.to_csv(outdir / "task1_qrels_test.csv", index=False)
    late_regimes.to_csv(outdir / "task1_late_forecast_regimes.csv", index=False)
    test_metrics.to_csv(outdir / "task1_metrics_test.csv", index=False)
    test_query_metrics.to_csv(outdir / "task1_query_metrics_test.csv", index=False)
    qrels_test_summary.to_csv(outdir / "task1_qrels_summary_test.csv", index=False)
    if cfg.save_dev_artifacts:
        qrels_dev.to_csv(outdir / "task1_qrels_dev.csv", index=False)
        dev_metrics.to_csv(outdir / "task1_metrics_dev.csv", index=False)
    if cfg.save_rankings:
        for system, rk in test_rankings.items():
            rk.to_csv(outdir / f"task1_rankings_{system}.csv", index=False)

    return {
        "qrels_rows": {"dev": int(len(qrels_dev)), "test": int(len(qrels_test))},
        "late_regime_counts": df_to_records(
            late_regimes.groupby(["error_regime", "best_model", "disagreement_regime"])
            .size()
            .reset_index(name="n")
        ),
        "fusion": fusion,
        "metrics": {
            "dev": df_to_records(dev_metrics),
            "test": df_to_records(test_metrics),
        },
        "dataframes": {
            "dev_metrics": dev_metrics,
            "test_metrics": test_metrics,
            "test_query_metrics": test_query_metrics,
            "qrels_test_summary": qrels_test_summary,
        },
    }


def df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_safe(row) for row in df.to_dict(orient="records")]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items() if k != "dataframes"}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.DataFrame,)):
        return df_to_records(value)
    if not isinstance(value, str):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def _metric_row(df: pd.DataFrame, system: str) -> dict[str, Any] | None:
    sub = df[df["system"] == system]
    if sub.empty:
        return None
    return _json_safe(sub.iloc[0].to_dict())


def _best_content_row(df: pd.DataFrame, task: str) -> dict[str, Any] | None:
    if task == "task2":
        systems = [
            "pattern_raw_cosine",
            "pattern_raw_dtw",
            "pattern_tsfel",
            "pattern_raw_cosine+pattern_tsfel",
        ]
    else:
        systems = ["raw_cosine", "raw_dtw", "tsfel", "raw_dtw+tsfel"]
    sub = df[df["system"].isin(systems)]
    if sub.empty:
        return None
    return _json_safe(sub.sort_values("ndcg@10", ascending=False).iloc[0].to_dict())


def build_experiment_tables(
    whole: dict[str, Any],
    pattern_main: dict[str, Any],
    length_results: list[dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    task1_test = whole["dataframes"]["test_metrics"]
    task2_test = pattern_main["dataframes"]["test_metrics"]

    exp1 = task2_test[
        task2_test["system"].isin(
            [
                "random",
                "pattern_raw_cosine",
                "pattern_raw_dtw",
                "pattern_tsfel",
                "pattern_raw_cosine+pattern_tsfel",
            ]
        )
    ].copy()
    exp1.insert(0, "experiment", "exp1_pattern_main")

    pairs = []
    for whole_sys, pattern_sys, label in [
        ("raw_cosine", "pattern_raw_cosine", "cosine"),
        ("raw_dtw", "pattern_raw_dtw", "dtw"),
        ("tsfel", "pattern_tsfel", "tsfel"),
    ]:
        wr = _metric_row(task1_test, whole_sys)
        pr = _metric_row(task2_test, pattern_sys)
        if wr:
            rows = {"comparison": label, "approach": "whole-series", **wr}
            pairs.append(rows)
        if pr:
            rows = {"comparison": label, "approach": "pattern", **pr}
            pairs.append(rows)
    exp2 = pd.DataFrame(pairs)

    best_content = _best_content_row(task2_test, "task2")
    random_row = _metric_row(task2_test, "random")
    exp3_rows = []
    for label, row in [
        ("random", random_row),
        ("best_content_baseline", best_content),
    ]:
        if row:
            exp3_rows.append({"approach": label, **row})
    exp3 = pd.DataFrame(exp3_rows)

    exp4_rows = []
    keep = ["pattern_raw_dtw", "pattern_raw_cosine", "pattern_tsfel"]
    for res in length_results:
        df = res["dataframes"]["test_metrics"]
        for row in df[df["system"].isin(keep)].to_dict(orient="records"):
            exp4_rows.append({"pattern_len": int(res["length"]), **_json_safe(row)})
    exp4 = pd.DataFrame(exp4_rows)

    return {
        "exp1_pattern_main": exp1,
        "exp2_whole_vs_pattern": exp2,
        "exp3_retrieval_ablation": exp3,
        "exp4_pattern_length_sensitivity": exp4,
    }


def _safe_ylim(
    values: pd.Series, min_top: float = 0.05, upper_cap: float | None = None
) -> tuple[float, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    top = max(min_top, float(vals.max()) * 1.15 if len(vals) else min_top)
    if upper_cap is not None:
        top = min(upper_cap, max(min_top, top))
    return 0.0, top


def write_plots(tables: dict[str, pd.DataFrame], outdir: Path) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as exc:
        log(f"Skipping plots because matplotlib is unavailable: {type(exc).__name__}")
        return {}

    plot_paths: dict[str, str] = {}
    pdf_path = outdir / "research_results_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        exp1 = tables["exp1_pattern_main"].copy()
        if not exp1.empty:
            exp1 = exp1.sort_values("ndcg@10", ascending=True)
            fig, ax = plt.subplots(figsize=(9.2, 4.8))
            ax.barh(exp1["system"], exp1["ndcg@10"])
            ax.set_title(
                "Experiment 1: Pattern retrieval under late forecast-regime relevance"
            )
            ax.set_xlabel("nDCG@10")
            ax.set_xlim(*_safe_ylim(exp1["ndcg@10"], min_top=0.1, upper_cap=1.05))
            ax.grid(axis="x", alpha=0.25)
            for i, val in enumerate(exp1["ndcg@10"]):
                ax.text(
                    float(val) + 0.005, i, f"{float(val):.3f}", va="center", fontsize=8
                )
            fig.tight_layout()
            path = outdir / "fig_exp1_pattern_main_ndcg10.png"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            pdf.savefig(fig)
            plt.close(fig)
            plot_paths["exp1_pattern_main"] = str(path)

        exp2 = tables["exp2_whole_vs_pattern"].copy()
        if not exp2.empty:
            pivot = exp2.pivot_table(
                index="comparison",
                columns="approach",
                values="ndcg@10",
                aggfunc="first",
            )
            fig, ax = plt.subplots(figsize=(7.8, 4.4))
            pivot.plot(kind="bar", ax=ax, width=0.72)
            ax.set_title("Experiment 2: Whole-series vs pattern retrieval")
            ax.set_ylabel("nDCG@10")
            ax.set_xlabel("")
            ax.set_ylim(*_safe_ylim(exp2["ndcg@10"], min_top=0.1, upper_cap=1.05))
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(axis="x", labelrotation=0)
            fig.tight_layout()
            path = outdir / "fig_exp2_whole_vs_pattern_ndcg10.png"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            pdf.savefig(fig)
            plt.close(fig)
            plot_paths["exp2_whole_vs_pattern"] = str(path)

        exp3 = tables["exp3_retrieval_ablation"].copy()
        if not exp3.empty:
            order = ["random", "best_content_baseline"]
            exp3["approach"] = pd.Categorical(
                exp3["approach"], categories=order, ordered=True
            )
            exp3 = exp3.sort_values("approach")
            fig, ax = plt.subplots(figsize=(8.4, 4.2))
            ax.bar(exp3["approach"].astype(str), exp3["ndcg@10"])
            ax.set_title("Experiment 3: No-leak retrieval ablation")
            ax.set_ylabel("nDCG@10")
            ax.set_xlabel("")
            ax.set_ylim(*_safe_ylim(exp3["ndcg@10"], min_top=0.1, upper_cap=1.05))
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(axis="x", labelrotation=20)
            for i, val in enumerate(exp3["ndcg@10"]):
                ax.text(
                    i,
                    float(val) + 0.01,
                    f"{float(val):.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            fig.tight_layout()
            path = outdir / "fig_exp3_retrieval_ablation_ndcg10.png"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            pdf.savefig(fig)
            plt.close(fig)
            plot_paths["exp3_retrieval_ablation"] = str(path)

        exp4 = tables["exp4_pattern_length_sensitivity"].copy()
        if not exp4.empty:
            pivot = exp4.pivot_table(
                index="pattern_len", columns="system", values="ndcg@10", aggfunc="first"
            ).sort_index()
            fig, ax = plt.subplots(figsize=(9.2, 4.8))
            pivot.plot(marker="o", ax=ax)
            ax.set_title("Experiment 4: Pattern-length sensitivity")
            ax.set_ylabel("nDCG@10")
            ax.set_xlabel("Pattern length")
            ax.set_ylim(*_safe_ylim(exp4["ndcg@10"], min_top=0.1, upper_cap=1.05))
            ax.grid(axis="both", alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            path = outdir / "fig_exp4_pattern_length_sensitivity.png"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            pdf.savefig(fig)
            plt.close(fig)
            plot_paths["exp4_pattern_length_sensitivity"] = str(path)

    plot_paths["all_experiments_pdf"] = str(pdf_path)
    log(f"Wrote plots PDF: {pdf_path}")
    return plot_paths


def write_report(
    cfg: Config,
    series_map: dict[str, np.ndarray],
    splits: dict[str, list[str]],
    whole: dict[str, Any],
    pattern_main: dict[str, Any],
    length_results: list[dict[str, Any]],
    tables: dict[str, pd.DataFrame],
    plot_paths: dict[str, str],
    outdir: Path,
) -> Path:
    for name, df in tables.items():
        df.to_csv(outdir / f"{name}.csv", index=False)
    ir_artifacts = write_ir_diagnostics(
        cfg, whole, pattern_main, length_results, outdir
    )
    if ir_artifacts.get("query_delta_plot"):
        plot_paths["query_delta_vs_random"] = ir_artifacts["query_delta_plot"]
    relevance_notes = {
        "full": {
            "relevance_4": "same error bin, same best model, and same disagreement bin",
            "relevance_3": "same error bin and either same best model or same disagreement bin",
            "relevance_2": "same error bin only",
            "relevance_1": "adjacent error bin, same best model, and same disagreement bin",
            "relevance_0": "otherwise",
        },
        "error_only": {
            "relevance_2": "same error bin",
            "relevance_1": "adjacent error bin",
            "relevance_0": "otherwise",
        },
        "error_model": {
            "relevance_3": "same error bin and same best model",
            "relevance_2": "same error bin only",
            "relevance_1": "adjacent error bin and same best model",
            "relevance_0": "otherwise",
        },
        "error_disagreement": {
            "relevance_3": "same error bin and same disagreement bin",
            "relevance_2": "same error bin only",
            "relevance_1": "adjacent error bin and same disagreement bin",
            "relevance_0": "otherwise",
        },
    }
    report = {
        "schema_version": 4,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"generator": "pipeline_research", "output_dir": str(outdir)},
        "base_configuration": {
            "description": (
                "Research-grade retrieval benchmark with four experiments: pattern-only retrieval under late forecasting-regime qrels, "
                "whole-series versus pattern retrieval, no-leak retrieval ablation, "
                f"and pattern-length sensitivity. Relevance judgments are built from {cfg.qrels_n_bins}-bin late-window forecasting regimes. "
                "Retrieval systems use only pre-holdout visible histories. IR diagnostics include query-level metrics, qrels density, "
                "bootstrap confidence intervals, and paired randomization tests against random ranking."
            ),
            "parameters": _json_safe(asdict(cfg)),
            "forecast_panel": _forecast_panel_names(cfg.skip_ml_models),
            "forecast_panel_note": (
                "Default panel uses two statistical models from StatsForecast (AutoETS, AutoARIMA) and two machine-learning models "
                "through MLForecast (LightGBM, XGBoost). Pass --skip-ml-models to use only AutoETS and AutoARIMA."
            ),
            "retrieval_policy": {
                "systems": "rank using only information available before each object's held-out late window",
                "content_baselines": "raw, DTW, TSFEL, and their fusions are computed on pre-holdout visible prefixes",
            },
            "qrels_definition": {
                "profile_window": "late",
                "type": "forecasting-regime graded qrels",
                "definition": cfg.regime_definition,
                "n_bins": cfg.qrels_n_bins,
                "regime_fields": [
                    "mean_error_bin",
                    "best_model",
                    "model_disagreement_bin",
                ],
                "binning": "quantile bins fitted on the indexed collection and applied to query objects",
                "retrieval_information": "retrieval systems use only the pre-holdout prefix; qrels use the held-out late window only as evaluation labels",
                **relevance_notes[cfg.regime_definition],
            },
        },
        "data_summary": {
            "n_series": int(len(series_map)),
            "split_counts": {k: int(len(v)) for k, v in splits.items()},
            "series_length": {
                "min": int(min(len(v) for v in series_map.values())),
                "median": float(np.median([len(v) for v in series_map.values()])),
                "max": int(max(len(v) for v in series_map.values())),
            },
        },
        "experiments": {
            "exp1_pattern_main": {
                "question": "Which pattern retrieval systems recover late-window forecasting-regime relevance?",
                "primary_metric": "ndcg@10",
                "results": df_to_records(tables["exp1_pattern_main"]),
                "plot": plot_paths.get("exp1_pattern_main"),
            },
            "exp2_whole_vs_pattern": {
                "question": "Are local patterns better retrieval objects than complete series under forecasting-regime relevance?",
                "primary_metric": "ndcg@10",
                "results": df_to_records(tables["exp2_whole_vs_pattern"]),
                "plot": plot_paths.get("exp2_whole_vs_pattern"),
            },
            "exp3_retrieval_ablation": {
                "question": "How much late forecast-regime relevance is recovered by random ranking and the best content retrieval baseline?",
                "primary_metric": "ndcg@10",
                "anti_circularity": "qrels use late forecast profiles as evaluation labels; all retrieval systems use only pre-holdout information.",
                "results": df_to_records(tables["exp3_retrieval_ablation"]),
                "plot": plot_paths.get("exp3_retrieval_ablation"),
            },
            "exp4_pattern_length_sensitivity": {
                "question": "How sensitive are the retrieval conclusions to the chosen subsequence length?",
                "primary_metric": "ndcg@10",
                "results": df_to_records(tables["exp4_pattern_length_sensitivity"]),
                "plot": plot_paths.get("exp4_pattern_length_sensitivity"),
            },
        },
        "raw_task_outputs": {
            "task1_whole_series": _json_safe(whole),
            "task2_pattern_main": _json_safe(pattern_main),
            "pattern_length_runs": _json_safe(length_results),
        },
        "ir_diagnostics": ir_artifacts,
        "plots": plot_paths,
    }
    path = outdir / "research_results.json"
    path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    log(f"Wrote research JSON: {path}")
    return path


def run_configured_pipeline(cfg: Config, write_figures: bool = True) -> Path:
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cleanup_streamlined_outputs(outdir, cfg)
    set_seed(cfg.seed)

    log(f"Loading long-format dataset: {cfg.input}")
    df = ensure_long_df(pd.read_csv(cfg.input))
    series_map = build_series_map(df)
    splits = split_ids(list(series_map.keys()), cfg.seed, cfg.index_frac, cfg.dev_frac)
    (outdir / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    if cfg.save_series_meta:
        pd.DataFrame(
            {
                "unique_id": list(series_map.keys()),
                "length": [len(series_map[k]) for k in series_map],
            }
        ).to_csv(outdir / "series_meta.csv", index=False)

    whole = run_whole_experiment(series_map, splits, cfg, outdir)
    pattern_main = run_pattern_experiment_for_length(
        cfg.pattern_len, series_map, splits, cfg, outdir, "task2"
    )

    length_values = sorted(
        set(parse_int_list(cfg.pattern_lens, [12, 24, 36]) + [cfg.pattern_len])
    )
    length_results = []
    for length in length_values:
        if int(length) == int(cfg.pattern_len):
            length_results.append(pattern_main)
        else:
            length_results.append(
                run_pattern_experiment_for_length(
                    int(length), series_map, splits, cfg, outdir, f"task2_L{length}"
                )
            )

    tables = build_experiment_tables(whole, pattern_main, length_results)
    plot_paths = write_plots(tables, outdir) if write_figures else {}
    report_path = write_report(
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

    log("=== Experiment 1: pattern retrieval held-out test ===")
    print(tables["exp1_pattern_main"].to_string(index=False), flush=True)
    log("=== Experiment 2: whole-series vs pattern retrieval ===")
    print(tables["exp2_whole_vs_pattern"].to_string(index=False), flush=True)
    log("=== Experiment 3: retrieval ablation ===")
    print(tables["exp3_retrieval_ablation"].to_string(index=False), flush=True)
    log("=== Experiment 4: pattern-length sensitivity ===")
    print(tables["exp4_pattern_length_sensitivity"].to_string(index=False), flush=True)
    log(f"Done. Artifacts written to {outdir}")
    return report_path


def cleanup_streamlined_outputs(outdir: Path, cfg: Config) -> None:
    stale_patterns = []
    if not cfg.save_dev_artifacts:
        stale_patterns.extend(["*_qrels_dev.csv", "*_metrics_dev.csv"])
    if not cfg.save_series_meta:
        stale_patterns.append("series_meta.csv")
    if not cfg.save_rankings:
        stale_patterns.append("*_rankings_*.csv")
    stale_patterns.extend(["fig_exp3_oracle_gap*.png", "*oracle_gap*.csv"])

    removed_system_tokens = [
        "forecast_regime_oracle",
        "late_qrels_oracle_upper_bound",
        "historical_backtest_profile",
        "forecast_profile_early",
    ]
    stale_patterns.extend(
        [f"*_rankings_*{token}*.csv" for token in removed_system_tokens]
    )

    for pattern in stale_patterns:
        for path in outdir.glob(pattern):
            if path.is_file():
                path.unlink()


def _infer_sibling_input(
    input_path: str | None, source_freq: str, target_freq: str
) -> str | None:
    if not input_path:
        return None
    path = Path(input_path)
    candidates = []
    lower_name = path.name.lower()
    if source_freq in lower_name:
        candidates.append(path.with_name(lower_name.replace(source_freq, target_freq)))
    freq_tokens = ["monthly", "quarterly", "yearly", "annual"]
    for token in freq_tokens:
        if token in lower_name:
            candidates.append(path.with_name(lower_name.replace(token, target_freq)))
    if target_freq == "yearly":
        for token in freq_tokens:
            if token in lower_name:
                candidates.append(path.with_name(lower_name.replace(token, "annual")))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _resolve_frequency_input(
    args: argparse.Namespace, frequency: str, multi_run: bool
) -> str | None:
    if frequency == "monthly":
        return args.monthly_input or args.input
    if frequency == "quarterly":
        explicit = args.quarterly_input
    elif frequency == "yearly":
        explicit = args.yearly_input
    else:
        explicit = None
    if explicit:
        return explicit
    inferred = _infer_sibling_input(args.input, "monthly", frequency)
    if inferred:
        return inferred
    return None if multi_run else args.input


def _build_config(
    args: argparse.Namespace, frequency: str, input_path: str, output_dir: str
) -> Config:
    defaults = FREQUENCY_DEFAULTS[frequency]
    return Config(
        input=input_path,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        frequency=frequency,
        seasonal_period=(
            args.seasonal_period
            if args.seasonal_period is not None
            else defaults["seasonal_period"]
        ),
        pattern_len=args.pattern_len,
        pattern_lens=args.pattern_lens,
        patterns_per_series=args.patterns_per_series,
        raw_resample_len=args.raw_resample_len,
        forecast_h=(
            args.forecast_h if args.forecast_h is not None else defaults["forecast_h"]
        ),
        qrels_top_high=args.qrels_top_high,
        qrels_top_mid=args.qrels_top_mid,
        qrels_n_bins=args.qrels_n_bins,
        regime_definition=args.regime_definition,
        ml_lags=args.ml_lags,
        ml_n_estimators=args.ml_n_estimators,
        seed=args.seed,
        tsfel_standardize_series=args.tsfel_standardize_series,
        skip_ml_models=args.skip_ml_models,
        save_rankings=args.save_rankings,
        save_dev_artifacts=args.save_dev_artifacts,
        save_series_meta=args.save_series_meta,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        help="Long-format CSV used for a single run, or the monthly CSV in --frequencies M,Q,Y mode.",
    )
    ap.add_argument(
        "--monthly-input", help="Long-format monthly CSV for multi-frequency runs."
    )
    ap.add_argument(
        "--quarterly-input", help="Long-format quarterly CSV for multi-frequency runs."
    )
    ap.add_argument(
        "--yearly-input", help="Long-format yearly CSV for multi-frequency runs."
    )
    ap.add_argument(
        "--frequency",
        default="monthly",
        help="Single-run frequency: monthly/M, quarterly/Q, or yearly/Y.",
    )
    ap.add_argument(
        "--quarterly",
        action="store_true",
        help="Shortcut for --frequency quarterly in a single main-pipeline run.",
    )
    ap.add_argument(
        "--yearly",
        action="store_true",
        help="Shortcut for --frequency yearly in a single main-pipeline run.",
    )
    ap.add_argument(
        "--frequencies",
        help="Comma-separated frequencies to run, e.g. M,Q,Y. Multi-run output is written to per-frequency subdirectories.",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset-name", default="dataset")
    ap.add_argument(
        "--seasonal-period",
        type=int,
        help="Overrides the frequency default: monthly=12, quarterly=4, yearly=1.",
    )
    ap.add_argument("--pattern-len", type=int, default=24)
    ap.add_argument("--pattern-lens", default="12,24,36")
    ap.add_argument("--patterns-per-series", type=int, default=1)
    ap.add_argument("--raw-resample-len", type=int, default=64)
    ap.add_argument(
        "--forecast-h",
        type=int,
        help="Overrides the frequency default: monthly=6, quarterly=4, yearly=4.",
    )
    ap.add_argument("--qrels-top-high", type=int, default=5)
    ap.add_argument(
        "--qrels-top-mid",
        type=int,
        default=15,
        help="Deprecated for regime qrels; kept for backward compatibility.",
    )
    ap.add_argument(
        "--qrels-n-bins",
        type=int,
        default=3,
        help="Number of quantile bins for error and disagreement regimes.",
    )
    ap.add_argument(
        "--regime-definition",
        choices=["full", "error_only", "error_model", "error_disagreement"],
        default="full",
        help="Forecast-regime relevance definition used for qrels.",
    )
    ap.add_argument("--ml-lags", default="1,2,3,4,6,12")
    ap.add_argument("--ml-n-estimators", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tsfel-standardize-series", action="store_true")
    ap.add_argument("--skip-ml-models", action="store_true")
    ap.add_argument("--save-rankings", action="store_true")
    ap.add_argument(
        "--save-dev-artifacts",
        action="store_true",
        help="Also write dev qrels and dev metrics CSVs. Dev is still used internally for fusion tuning.",
    )
    ap.add_argument(
        "--save-series-meta",
        action="store_true",
        help="Also write per-series metadata CSV.",
    )
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and print the run configuration without executing experiments.",
    )
    args = ap.parse_args()

    if args.quarterly and args.yearly:
        ap.error("--quarterly and --yearly are mutually exclusive.")
    if (args.quarterly or args.yearly) and args.frequencies:
        ap.error(
            "--quarterly/--yearly cannot be combined with --frequencies; use --frequencies Q, --frequencies Y, or --frequencies M,Q,Y."
        )
    if args.quarterly:
        args.frequency = "quarterly"
    if args.yearly:
        args.frequency = "yearly"

    multi_run = bool(args.frequencies)
    frequencies = parse_frequency_list(
        args.frequencies, [normalize_frequency(args.frequency)]
    )
    reports = []
    for frequency in frequencies:
        input_path = _resolve_frequency_input(args, frequency, multi_run)
        if not input_path:
            flag_by_frequency = {
                "monthly": "--monthly-input",
                "quarterly": "--quarterly-input",
                "yearly": "--yearly-input",
            }
            flag = flag_by_frequency.get(frequency, "--input")
            ap.error(f"{flag} is required when running frequency {frequency!r}.")
        if not Path(input_path).exists():
            ap.error(f"Input file does not exist for {frequency}: {input_path}")

        run_output_dir = (
            str(Path(args.output_dir) / frequency) if multi_run else args.output_dir
        )
        cfg = _build_config(args, frequency, input_path, run_output_dir)
        if multi_run:
            cfg.dataset_name = f"{args.dataset_name}_{frequency}"
        log(
            f"=== Running {cfg.dataset_name}: frequency={cfg.frequency}, "
            f"seasonal_period={cfg.seasonal_period}, forecast_h={cfg.forecast_h} ==="
        )
        if args.dry_run:
            print(json.dumps(_json_safe(asdict(cfg)), indent=2), flush=True)
            reports.append(Path(cfg.output_dir) / "research_results.json")
            continue
        reports.append(run_configured_pipeline(cfg, write_figures=not args.no_plots))

    if multi_run:
        log("Completed multi-frequency main runs:")
        for report in reports:
            print(report, flush=True)


if __name__ == "__main__":
    main()
