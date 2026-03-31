"""
finance_data.py
==================
Pulls daily S&P 500 (or any ticker list) data from yfinance and formats it
for use with AdaptedCAFHT. Includes save/load utilities for persisting data
to disk without re-downloading.

QUICK START
-----------
  # Pull all S&P 500 stocks for Jan–Mar 2024 and save to disk:
  python finance_data.py --pull

  # Pull a small test batch of 10 stocks:
  python finance_data.py --pull --top_n 10

  # Load saved data and print a summary:
  python finance_data.py --summary

  # Load saved data and filter to Technology sector:
  python finance_data.py --summary --sector Technology

  # Load saved data and filter to Semiconductors industry:
  python finance_data.py --summary --industry Semiconductors

  # List all sectors in saved data:
  python finance_data.py --list_sectors

  # List all industries in saved data:
  python finance_data.py --list_industries

ARRAYS
------
  Y         : (n_series, L, 1)      daily Close price
  X         : (n_series, L, n_cov)  covariates (see lag logic below)
  dates     : (L,) str              trading dates
  tickers   : list[str]             ticker symbols in series order
  meta      : list[dict]            sector, industry, exchange per ticker
  cov_names : list[str]             covariate column names

COVARIATE LAG LOGIC
-------------------
  Open          NOT lagged  today's open is known before today's close
  OvernightGap  NOT lagged  Open_today - Close_yesterday, known at market open
  Volume        lagged 1    only available after prior day's close
  Dividends     lagged 1    ex-dividend date is the prior day
  DailyRange    lagged 1    uses High/Low, final only after close
  VWAPproxy     lagged 1    uses High/Low/Close, final only after close

STORAGE FORMAT
--------------
  <stem>.npz   compressed numpy arrays  (Y, X, dates)
  <stem>.json  tickers, cov_names, metadata (sector/industry per ticker)

  Default stem: sp500_YYYYMMDD_YYYYMMDD  e.g. sp500_20240101_20240301
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
#  Loading from yfinance
# ============================================================

def load_series(
    tickers: list[str],
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
    extra_covariates: bool = True,
) -> dict:
    """
    Download daily OHLCV data for a list of tickers and return arrays
    ready for AdaptedCAFHT.

    Parameters
    ----------
    tickers : list[str]
        E.g. ["AAPL", "MSFT"] or ["^GSPC"] for S&P 500 index.
    start : str
        Start date, e.g. "2018-01-01".
    end : str, optional
        End date, e.g. "2024-03-01". Defaults to today.
    interval : str
        yfinance interval string. "1d" for daily (recommended).
    extra_covariates : bool
        If True, also include DailyRange_lag1 and VWAPproxy_lag1.

    Returns
    -------
    dict with keys: Y, X, dates, tickers, meta, cov_names
    """
    raw = _download_all(tickers, start, end, interval)

    common_dates = _get_common_dates(raw)
    if len(common_dates) == 0:
        raise ValueError("No overlapping trading dates found across tickers.")

    Y_list, X_list, valid_tickers = [], [], []
    cov_df = None

    for ticker in tickers:
        df = raw.get(ticker)
        if df is None or df.empty:
            print(f"[WARNING] No data for {ticker}, skipping.")
            continue

        df = df.reindex(common_dates)

        if df["Close"].isna().all():
            print(f"[WARNING] All-NaN Close for {ticker}, skipping.")
            continue

        df = df.ffill().dropna(subset=["Close"])

        y      = df["Close"].values.reshape(-1, 1)
        cov_df = _build_covariates(df, extra=extra_covariates)
        x      = cov_df.values

        Y_list.append(y)
        X_list.append(x)
        valid_tickers.append(ticker)

    if len(valid_tickers) == 0:
        raise ValueError("No valid ticker data could be loaded.")

    Y         = np.stack(Y_list, axis=0).astype(np.float64)   # (n_series, L, 1)
    X         = np.stack(X_list, axis=0).astype(np.float64)   # (n_series, L, n_cov)
    dates     = np.array(common_dates.astype(str))
    cov_names = list(cov_df.columns)
    meta      = _fetch_metadata(valid_tickers)

    return {
        "Y":         Y,
        "X":         X,
        "dates":     dates,
        "tickers":   valid_tickers,
        "meta":      meta,
        "cov_names": cov_names,
    }


def load_sp500(
    start: str,
    end: Optional[str] = None,
    top_n: Optional[int] = None,
    extra_covariates: bool = True,
) -> dict:
    """
    Load all S&P 500 constituent stocks (scraped from Wikipedia) via yfinance.

    Parameters
    ----------
    start, end : str
        Date range, e.g. start="2024-01-01", end="2024-03-01".
    top_n : int, optional
        Limit to the first N constituents — useful for quick testing.
    extra_covariates : bool
        Whether to include DailyRange_lag1 and VWAPproxy_lag1.

    Returns
    -------
    Same dict as load_series().
    Use filter_by_sector() / filter_by_industry() to slice by group.
    """
    tickers = _get_sp500_constituents()
    if top_n is not None:
        tickers = tickers[:top_n]
    print(f"[INFO] Loading {len(tickers)} S&P 500 constituents...")
    return load_series(tickers, start=start, end=end, extra_covariates=extra_covariates)


# ============================================================
#  Saving & loading from disk
# ============================================================

def save(result: dict, stem: Optional[str] = None, directory: str = ".") -> tuple[Path, Path]:
    """
    Save a result dict to a compressed .npz (arrays) + .json (metadata) pair.

    Parameters
    ----------
    result : dict
        Output from load_series() or load_sp500().
    stem : str, optional
        Base filename without extension.
        Defaults to sp500_<start>_<end> derived from the dates array.
    directory : str
        Directory to write files into. Created if it does not exist.

    Returns
    -------
    (npz_path, json_path) : tuple[Path, Path]
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    if stem is None:
        d0   = str(result["dates"][0]).replace("-", "")
        d1   = str(result["dates"][-1]).replace("-", "")
        stem = f"sp500_{d0}_{d1}"

    npz_path  = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}.json"

    np.savez_compressed(
        npz_path,
        Y     = result["Y"],
        X     = result["X"],
        dates = result["dates"],
    )

    payload = {
        "start":     str(result["dates"][0]),
        "end":       str(result["dates"][-1]),
        "n_series":  len(result["tickers"]),
        "L":         int(result["Y"].shape[1]),
        "tickers":   result["tickers"],
        "cov_names": result["cov_names"],
        "meta":      result["meta"],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[SAVED] Arrays   → {npz_path}  ({npz_path.stat().st_size / 1e6:.2f} MB)")
    print(f"[SAVED] Metadata → {json_path}")
    return npz_path, json_path


def load_stored(npz_path: str | Path, json_path: Optional[str | Path] = None) -> dict:
    """
    Reload a previously saved dataset from disk without re-downloading.

    Parameters
    ----------
    npz_path : str or Path
        Path to the .npz file.
    json_path : str or Path, optional
        Path to the companion .json file.
        If omitted, inferred by replacing .npz with .json.

    Returns
    -------
    dict with keys: Y, X, dates, tickers, cov_names, meta
    """
    npz_path  = Path(npz_path)
    json_path = Path(json_path) if json_path else npz_path.with_suffix(".json")

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    npz  = np.load(npz_path, allow_pickle=False)
    meta = json.load(open(json_path))

    return {
        "Y":         npz["Y"],
        "X":         npz["X"],
        "dates":     npz["dates"],
        "tickers":   meta["tickers"],
        "cov_names": meta["cov_names"],
        "meta":      meta["meta"],
    }


def default_paths(start: str = "2024-01-01", end: str = "2024-03-01",
                  directory: str = ".") -> tuple[Path, Path]:
    """Return the default (npz_path, json_path) for a given date range."""
    d0   = start.replace("-", "")
    d1   = end.replace("-", "")
    stem = f"sp500_{d0}_{d1}"
    base = Path(directory)
    return base / f"{stem}.npz", base / f"{stem}.json"


# ============================================================
#  Filtering
# ============================================================

def filter_by_sector(result: dict, sectors: list[str], case_sensitive: bool = False) -> dict:
    """
    Return a sub-result containing only series whose sector matches.

    Parameters
    ----------
    result : dict
    sectors : list[str]
        E.g. ["Technology", "Healthcare"].
        Call list_sectors(result) to see all available values.
    case_sensitive : bool
        Default False.
    """
    return _filter(result, field="sector", values=sectors, case_sensitive=case_sensitive)


def filter_by_industry(result: dict, industries: list[str], case_sensitive: bool = False) -> dict:
    """
    Return a sub-result containing only series whose industry matches.

    Parameters
    ----------
    result : dict
    industries : list[str]
        E.g. ["Semiconductors", "Software—Application"].
        Call list_industries(result) to see all available values.
    case_sensitive : bool
        Default False.
    """
    return _filter(result, field="industry", values=industries, case_sensitive=case_sensitive)


def list_sectors(result: dict) -> list[str]:
    """Return a sorted list of unique sectors present in the result."""
    return sorted({m["sector"] for m in result["meta"] if m["sector"] != "N/A"})


def list_industries(result: dict) -> list[str]:
    """Return a sorted list of unique industries present in the result."""
    return sorted({m["industry"] for m in result["meta"] if m["industry"] != "N/A"})


def _filter(result: dict, field: str, values: list[str], case_sensitive: bool) -> dict:
    if not case_sensitive:
        match_values = {v.lower() for v in values}
    else:
        match_values = set(values)

    indices = [
        i for i, m in enumerate(result["meta"])
        if (m.get(field, "N/A") if case_sensitive else m.get(field, "N/A").lower())
        in match_values
    ]

    if not indices:
        available = sorted({m.get(field, "N/A") for m in result["meta"]})
        raise ValueError(
            f"No series matched {field}={values}.\n"
            f"Available {field}s: {available}"
        )

    idx = np.array(indices)
    return {
        "Y":         result["Y"][idx],
        "X":         result["X"][idx],
        "dates":     result["dates"],
        "tickers":   [result["tickers"][i] for i in indices],
        "meta":      [result["meta"][i]    for i in indices],
        "cov_names": result["cov_names"],
    }


# ============================================================
#  Summarize
# ============================================================

def summarize(result: dict) -> None:
    """Print a human-readable summary of a result dict."""
    Y, X       = result["Y"], result["X"]
    tickers    = result["tickers"]
    meta       = result["meta"]
    dates      = result["dates"]
    cov_names  = result["cov_names"]
    sectors    = list_sectors(result)
    industries = list_industries(result)

    print("=" * 62)
    print("  yfinance Loader — Data Summary")
    print("=" * 62)
    print(f"  Tickers loaded  : {len(tickers)}")
    print(f"  Date range      : {dates[0]}  →  {dates[-1]}")
    print(f"  Time steps (L)  : {Y.shape[1]}")
    print(f"  Y shape         : {Y.shape}  (n_series, L, 1)")
    print(f"  X shape         : {X.shape}  (n_series, L, n_cov)")
    print(f"  Covariates      : {cov_names}")
    print(f"  Sectors  ({len(sectors):2d})    : {sectors}")
    print(f"  Industries ({len(industries):2d})  : {industries[:8]}"
          f"{'  ...' if len(industries) > 8 else ''}")
    print()
    print("  Ticker metadata (first 10):")
    for m in meta[:10]:
        print(f"    {m['ticker']:8s}  sector={m['sector']:24s}  industry={m['industry']}")
    if len(meta) > 10:
        print(f"    ... and {len(meta) - 10} more.")
    print("=" * 62)


# ============================================================
#  Internal helpers
# ============================================================

def _download_all(tickers, start, end, interval) -> dict[str, pd.DataFrame]:
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                print(f"[WARNING] Empty download for {ticker}.")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).normalize()
            data[ticker] = df
        except Exception as e:
            print(f"[WARNING] Failed to download {ticker}: {e}")
    return data


def _get_common_dates(raw: dict) -> pd.DatetimeIndex:
    if not raw:
        return pd.DatetimeIndex([])
    indices = [df.index for df in raw.values()]
    common  = indices[0]
    for idx in indices[1:]:
        common = common.intersection(idx)
    return common.sort_values()


def _build_covariates(df: pd.DataFrame, extra: bool) -> pd.DataFrame:
    cov = pd.DataFrame(index=df.index)

    # Not lagged — known before today's close
    cov["Open"]         = df["Open"]
    cov["OvernightGap"] = df["Open"] - df["Close"].shift(1)

    # Lagged by 1 day — available only after prior close
    cov["Volume_lag1"]    = df["Volume"].shift(1)
    cov["Dividends_lag1"] = df["Dividends"].shift(1) if "Dividends" in df.columns else 0.0

    if extra:
        cov["DailyRange_lag1"] = (df["High"] - df["Low"]).shift(1)
        cov["VWAPproxy_lag1"]  = ((df["High"] + df["Low"] + df["Close"]) / 3).shift(1)

    return cov.fillna(0.0)


def _fetch_metadata(tickers: list[str]) -> list[dict]:
    meta = []
    for ticker in tickers:
        entry = {"ticker": ticker, "sector": "N/A", "industry": "N/A",
                 "exchange": "N/A", "country": "N/A", "longName": ticker}
        try:
            info = yf.Ticker(ticker).info
            entry["sector"]   = info.get("sector",   "N/A")
            entry["industry"] = info.get("industry", "N/A")
            entry["exchange"] = info.get("exchange", "N/A")
            entry["country"]  = info.get("country",  "N/A")
            entry["longName"] = info.get("longName",  ticker)
        except Exception as e:
            print(f"[WARNING] Could not fetch metadata for {ticker}: {e}")
        meta.append(entry)
    return meta


def _get_sp500_constituents() -> list[str]:
    try:
        tables  = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"[INFO] Found {len(tickers)} S&P 500 constituents on Wikipedia.")
        return tickers
    except Exception as e:
        raise RuntimeError(f"Could not scrape S&P 500 constituents: {e}")


# ============================================================
#  CLI
# ============================================================

def _cli():
    parser = argparse.ArgumentParser(
        description="Pull and/or inspect S&P 500 time series data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python yfinance_loader.py --pull
  python yfinance_loader.py --pull --top_n 10
  python yfinance_loader.py --summary
  python yfinance_loader.py --summary --sector Technology
  python yfinance_loader.py --summary --industry Semiconductors
  python yfinance_loader.py --list_sectors
  python yfinance_loader.py --list_industries
        """,
    )
    parser.add_argument("--pull",            action="store_true", help="Download from yfinance and save to disk.")
    parser.add_argument("--summary",         action="store_true", help="Load saved data and print summary.")
    parser.add_argument("--list_sectors",    action="store_true", help="List all sectors in saved data.")
    parser.add_argument("--list_industries", action="store_true", help="List all industries in saved data.")
    parser.add_argument("--start",  default="2024-01-01", help="Start date (default: 2024-01-01).")
    parser.add_argument("--end",    default="2024-03-01", help="End date   (default: 2024-03-01).")
    parser.add_argument("--top_n",  type=int, default=None, help="Limit to first N tickers (for testing).")
    parser.add_argument("--sector",   default=None, help="Filter --summary to this sector.")
    parser.add_argument("--industry", default=None, help="Filter --summary to this industry.")
    parser.add_argument("--dir",    default=".",  help="Directory for saved files (default: current dir).")
    args = parser.parse_args()

    npz_path, json_path = default_paths(args.start, args.end, args.dir)

    # --pull
    if args.pull:
        print(f"\nPulling S&P 500 data  {args.start} → {args.end}\n")
        t0     = time.time()
        result = load_sp500(start=args.start, end=args.end, top_n=args.top_n)
        print(f"\nDownload complete in {time.time() - t0:.1f}s")
        save(result, directory=args.dir)
        summarize(result)
        return

    # All remaining commands require saved data on disk
    if not npz_path.exists():
        print(f"[ERROR] No saved data found at {npz_path}")
        print( "        Run:  python yfinance_loader.py --pull  first.")
        return

    result = load_stored(npz_path, json_path)

    if args.list_sectors:
        sectors = list_sectors(result)
        print(f"\n{len(sectors)} sectors in saved data:\n")
        for s in sectors:
            print(f"  {s}")
        return

    if args.list_industries:
        industries = list_industries(result)
        print(f"\n{len(industries)} industries in saved data:\n")
        for ind in industries:
            print(f"  {ind}")
        return

    if args.summary:
        if args.sector:
            result = filter_by_sector(result, [args.sector])
        if args.industry:
            result = filter_by_industry(result, [args.industry])
        summarize(result)
        return

    parser.print_help()


if __name__ == "__main__":
    _cli()