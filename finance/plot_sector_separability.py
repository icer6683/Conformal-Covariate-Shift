#!/usr/bin/env python3
"""
plot_sector_separability.py
===========================
Diagnostic: for each S&P 500 sector, how separable is the sector from the
rest of the market on a panel of 8 ticker-level features derived from daily
OHLCV data?

Motivation: Healthcare as currently defined does not separate cleanly on the
4 features used by the finance experiments. This script tests whether other
sectors would give the LR classifier a stronger signal, and whether adding
standard return-distribution features helps.

Inputs:
  - Ticker list + sector metadata from any existing sidecar .json in
    finance/data/ (defaults to sp500_20240201_20240328.json).
  - ~14 months of daily OHLCV pulled via yfinance (cached to
    finance/data/_sector_diag_cache.npz so reruns are fast).

Outputs:
  - results/finance/pdf/sector_separability_distributions.pdf
        Grid of 8 panels (one per feature), each a violin of the chosen
        sectors vs a "Rest-of-market" bucket.
  - results/finance/pdf/sector_separability_heatmap.pdf
        Heatmap of KS statistic (sector vs rest) per feature.
  - results/finance/json/sector_separability_ks.json
        Raw KS statistics and p-values per (sector, feature).

Features (8):
  1. ann_vol        : annualized std of daily log returns × sqrt(252)
  2. mean_range     : mean (High - Low) / Close
  3. mean_overnight : mean |Open / prev_Close - 1|
  4. mean_turnover  : mean (Volume × Close / shares_outstanding)
  5. beta_spy       : OLS beta of daily returns to SPY
  6. skew_ret       : skewness of daily log returns
  7. kurt_ret       : excess kurtosis of daily log returns
  8. mom_52w        : (last_Close / min_Low_over_period) - 1
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, skew, kurtosis

ROOT = Path(__file__).resolve().parent.parent


SECTORS_OF_INTEREST = [
    "Healthcare",
    "Technology",
    "Energy",
    "Utilities",
    "Real Estate",
    "Consumer Defensive",
    "Financial Services",
]

FEATURES = [
    ("ann_vol",        "Annualised daily-return vol (σ·√252)"),
    ("mean_range",     "Mean intraday range (H−L)/C"),
    ("mean_overnight", "Mean |overnight gap|"),
    ("mean_turnover",  "Mean turnover (Vol·C / shares_out)"),
    ("beta_spy",       "Beta to SPY"),
    ("skew_ret",       "Skewness of daily log returns"),
    ("kurt_ret",       "Excess kurtosis of daily log returns"),
    ("mom_52w",        "Period momentum: last / min(Low) − 1"),
]


def load_metadata(sidecar_json: Path) -> list[dict]:
    m = json.load(open(sidecar_json))
    return m["meta"]


def bulk_download(tickers: list[str], start: str, end: str, cache: Path) -> dict:
    """Return dict[ticker] -> DataFrame with columns Open, High, Low, Close, Volume.

    Uses a local .npz cache keyed by (start, end, ticker-hash).
    """
    import pandas as pd
    import yfinance as yf
    import hashlib

    key = hashlib.md5(f"{start}_{end}_{','.join(sorted(tickers))}".encode()).hexdigest()[:12]
    cache_npz = cache.with_name(cache.name.replace(".npz", f"_{key}.npz"))

    if cache_npz.exists():
        print(f"[cache] loading {cache_npz}")
        z = np.load(cache_npz, allow_pickle=True)
        dates = z["dates"]
        out = {}
        for tk in tickers:
            if tk not in z.files:
                continue
            arr = z[tk]  # shape (n_dates, 5) Open High Low Close Volume
            out[tk] = pd.DataFrame(
                arr, index=pd.to_datetime(dates),
                columns=["Open", "High", "Low", "Close", "Volume"],
            )
        # SPY is stored as special key
        if "__SPY__" in z.files:
            arr = z["__SPY__"]
            out["SPY"] = pd.DataFrame(
                arr, index=pd.to_datetime(dates),
                columns=["Open", "High", "Low", "Close", "Volume"],
            )
        return out

    print(f"[yfinance] downloading {len(tickers)+1} tickers {start} → {end}")
    all_tk = tickers + ["SPY"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(all_tk, start=start, end=end, auto_adjust=True,
                         progress=False, group_by="ticker", threads=True)

    out = {}
    # yfinance returns MultiIndex columns when group_by='ticker'
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0).unique()
        for tk in level0:
            try:
                sub = df[tk][["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(sub) > 0:
                    out[tk] = sub
            except Exception:
                continue
    else:
        # single ticker path
        sub = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        out[tickers[0]] = sub

    # cache
    if out:
        dates = None
        np_save_kwargs = {}
        for tk, sub in out.items():
            if dates is None:
                dates = sub.index.strftime("%Y-%m-%d").to_numpy()
            key_save = "__SPY__" if tk == "SPY" else tk
            # align to common dates; if a ticker is shorter, pad with NaN
            reindexed = sub.reindex(pd.to_datetime(dates))
            np_save_kwargs[key_save] = reindexed.to_numpy(dtype=float)
        np.savez_compressed(cache_npz, dates=dates, **np_save_kwargs)
        print(f"[cache] wrote {cache_npz} ({len(out)} tickers)")
    return out


def per_ticker_features(ohlcv: "pd.DataFrame", spy_ret: "pd.Series", shares_out: float) -> dict:
    import numpy as np
    import pandas as pd
    o, h, l, c, v = [ohlcv[k] for k in ["Open", "High", "Low", "Close", "Volume"]]
    if len(c.dropna()) < 40:
        return None
    logret = np.log(c / c.shift(1)).dropna()
    if len(logret) < 40:
        return None

    ann_vol = float(logret.std() * np.sqrt(252))
    mean_range = float(((h - l) / c).mean())
    overnight = (o / c.shift(1) - 1.0).abs()
    mean_overnight = float(overnight.mean())
    # turnover: dollar-volume / shares_outstanding (if shares_out>0 else NaN)
    if shares_out and shares_out > 0:
        mean_turnover = float(((v * c) / shares_out).mean())
    else:
        mean_turnover = np.nan

    # beta to SPY
    spy_ret_al = spy_ret.reindex(logret.index).dropna()
    common = logret.index.intersection(spy_ret_al.index)
    if len(common) > 30:
        y = logret.loc[common].to_numpy()
        x = spy_ret_al.loc[common].to_numpy()
        if x.std() > 0:
            beta_spy = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
        else:
            beta_spy = np.nan
    else:
        beta_spy = np.nan

    skew_ret = float(skew(logret.to_numpy()))
    kurt_ret = float(kurtosis(logret.to_numpy()))  # excess, default fisher=True

    mom_52w = float(c.iloc[-1] / l.min() - 1.0)

    return {
        "ann_vol": ann_vol,
        "mean_range": mean_range,
        "mean_overnight": mean_overnight,
        "mean_turnover": mean_turnover,
        "beta_spy": beta_spy,
        "skew_ret": skew_ret,
        "kurt_ret": kurt_ret,
        "mom_52w": mom_52w,
    }


def compute_all_features(meta: list[dict], ohlcv_by_tk: dict) -> "pd.DataFrame":
    import pandas as pd
    if "SPY" not in ohlcv_by_tk:
        raise RuntimeError("SPY data missing")
    spy_logret = np.log(ohlcv_by_tk["SPY"]["Close"] /
                        ohlcv_by_tk["SPY"]["Close"].shift(1)).dropna()

    rows = []
    for e in meta:
        tk = e["ticker"]
        if tk not in ohlcv_by_tk:
            continue
        feats = per_ticker_features(ohlcv_by_tk[tk], spy_logret, e.get("shares_outstanding", 0))
        if feats is None:
            continue
        feats["ticker"] = tk
        feats["sector"] = e.get("sector", "Unknown")
        rows.append(feats)
    return pd.DataFrame(rows)


def plot_distributions(df, sectors, features, save_path: Path):
    import pandas as pd
    n_feat = len(features)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    # colors
    palette = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd",
               "#8c564b", "#17becf", "#bcbd22"]
    sector_colors = {s: palette[i % len(palette)] for i, s in enumerate(sectors)}
    sector_colors["Rest-of-market"] = "#888888"

    for j, (key, label) in enumerate(features):
        ax = axes[j]
        plot_data = []
        plot_labels = []
        plot_colors = []
        for s in sectors:
            vals = df.loc[df["sector"] == s, key].dropna().to_numpy()
            if len(vals) == 0:
                continue
            # clip extreme outliers for plotting only (5/95 percentiles)
            lo, hi = np.percentile(df[key].dropna(), [2, 98])
            vals = vals[(vals >= lo) & (vals <= hi)]
            if len(vals) < 3:
                continue
            plot_data.append(vals)
            plot_labels.append(s)
            plot_colors.append(sector_colors[s])
        # add rest-of-market bucket
        rest_mask = ~df["sector"].isin(sectors)
        rest = df.loc[rest_mask, key].dropna().to_numpy()
        lo, hi = np.percentile(df[key].dropna(), [2, 98])
        rest = rest[(rest >= lo) & (rest <= hi)]
        if len(rest) >= 3:
            plot_data.append(rest)
            plot_labels.append("Rest-of-market")
            plot_colors.append(sector_colors["Rest-of-market"])

        parts = ax.violinplot(plot_data, showmeans=False, showmedians=True,
                              showextrema=False, widths=0.85)
        for pc, c in zip(parts["bodies"], plot_colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.6)
            pc.set_edgecolor("black")
        if "cmedians" in parts:
            parts["cmedians"].set_color("black")

        ax.set_xticks(range(1, len(plot_labels) + 1))
        ax.set_xticklabels(plot_labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle(
        "Per-ticker feature distributions by sector (S&P 500, ~14 months)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[plot] wrote {save_path}")
    plt.close(fig)


def plot_ks_heatmap(df, sectors, features, save_path: Path, ks_json_path: Path):
    """KS statistic of each sector's feature distribution vs rest-of-market.

    Higher KS ⇒ more separable. Also writes raw (KS, p) per cell.
    """
    import pandas as pd
    n_s = len(sectors)
    n_f = len(features)
    ks_mat = np.zeros((n_s, n_f))
    p_mat = np.zeros((n_s, n_f))
    raw = {}

    for i, s in enumerate(sectors):
        raw[s] = {}
        sec_mask = df["sector"] == s
        rest_mask = ~df["sector"].isin(sectors)  # strict "other sectors" bucket
        for j, (key, _) in enumerate(features):
            a = df.loc[sec_mask, key].dropna().to_numpy()
            b = df.loc[rest_mask, key].dropna().to_numpy()
            if len(a) < 5 or len(b) < 5:
                ks_mat[i, j] = np.nan
                p_mat[i, j] = np.nan
                raw[s][key] = {"ks": None, "p": None, "n_sector": len(a), "n_rest": len(b)}
                continue
            stat = ks_2samp(a, b)
            ks_mat[i, j] = stat.statistic
            p_mat[i, j] = stat.pvalue
            raw[s][key] = {"ks": float(stat.statistic), "p": float(stat.pvalue),
                           "n_sector": len(a), "n_rest": len(b)}

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(ks_mat, aspect="auto", cmap="Reds", vmin=0, vmax=0.6)
    ax.set_xticks(range(n_f))
    ax.set_xticklabels([f[1] for f in features], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_s))
    ax.set_yticklabels(sectors, fontsize=10)
    ax.set_title("KS statistic vs rest-of-market (higher = more separable)",
                 fontsize=11)
    for i in range(n_s):
        for j in range(n_f):
            if np.isnan(ks_mat[i, j]):
                continue
            star = "***" if p_mat[i, j] < 1e-3 else ("**" if p_mat[i, j] < 1e-2 else
                    ("*" if p_mat[i, j] < 0.05 else ""))
            ax.text(j, i, f"{ks_mat[i,j]:.2f}{star}",
                    ha="center", va="center",
                    color="white" if ks_mat[i, j] > 0.35 else "black",
                    fontsize=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("KS statistic", fontsize=9)
    ax.text(0.0, -0.28, "Significance: * p<0.05   ** p<0.01   *** p<0.001   (n_sector vs n_rest per cell)",
            transform=ax.transAxes, fontsize=8, color="gray")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[plot] wrote {save_path}")
    plt.close(fig)

    ks_json_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(raw, open(ks_json_path, "w"), indent=2)
    print(f"[json] wrote {ks_json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar_json", default=str(
        ROOT / "finance" / "data" / "sp500_20240201_20240328.json"))
    parser.add_argument("--start", default="2024-01-02")
    parser.add_argument("--end",   default="2025-02-28")
    parser.add_argument("--cache", default=str(
        ROOT / "finance" / "data" / "_sector_diag_cache.npz"))
    parser.add_argument("--out_dist", default=str(
        ROOT / "results" / "finance" / "pdf" / "sector_separability_distributions.pdf"))
    parser.add_argument("--out_heatmap", default=str(
        ROOT / "results" / "finance" / "pdf" / "sector_separability_heatmap.pdf"))
    parser.add_argument("--out_json", default=str(
        ROOT / "results" / "finance" / "json" / "sector_separability_ks.json"))
    args = parser.parse_args()

    meta = load_metadata(Path(args.sidecar_json))
    tickers = [e["ticker"] for e in meta]
    print(f"[load] {len(tickers)} tickers, {len(set(e['sector'] for e in meta))} sectors")

    ohlcv = bulk_download(tickers, args.start, args.end, Path(args.cache))
    print(f"[data] {len(ohlcv)} tickers returned from yfinance / cache")

    df = compute_all_features(meta, ohlcv)
    print(f"[features] {len(df)} tickers have usable feature rows")
    # quick sector count summary
    from collections import Counter
    print("[sectors in df]", Counter(df["sector"]).most_common())

    plot_distributions(df, SECTORS_OF_INTEREST, FEATURES, Path(args.out_dist))
    plot_ks_heatmap(df, SECTORS_OF_INTEREST, FEATURES, Path(args.out_heatmap),
                    Path(args.out_json))

    # summary table to stdout
    print("\n=== KS statistic (sector vs rest-of-market) ===")
    print(f"{'sector':20s} " + " ".join(f"{k:>14s}" for k, _ in FEATURES))
    for s in SECTORS_OF_INTEREST:
        import pandas as pd
        sec_mask = df["sector"] == s
        rest_mask = ~df["sector"].isin(SECTORS_OF_INTEREST)
        cells = []
        for key, _ in FEATURES:
            a = df.loc[sec_mask, key].dropna().to_numpy()
            b = df.loc[rest_mask, key].dropna().to_numpy()
            if len(a) >= 5 and len(b) >= 5:
                cells.append(f"{ks_2samp(a, b).statistic:14.3f}")
            else:
                cells.append(f"{'nan':>14s}")
        print(f"{s:20s} " + " ".join(cells))


if __name__ == "__main__":
    main()
