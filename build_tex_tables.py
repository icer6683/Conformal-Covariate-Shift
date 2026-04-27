#!/usr/bin/env python3
"""
build_tex_tables.py
-------------------
Produce LaTeX tables (one per setting) from saved experiment JSONs.

Outputs:
    results/synthetic/tables/synthetic_ar07.tex
    results/synthetic/tables/synthetic_ar09.tex
    results/finance/tables/finance.tex
    results/medical/tables/medical.tex

Each table mirrors the column layout used in `LaTeX Paper/main.tex`:
    Setting | Algorithm | Coverage | |Δ̄| | Width

|Δ̄| := mean per-seed (synthetic, medical) or per-window (finance) absolute
deviation from the target coverage 1−α.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import mean
from typing import Iterable

ROOT = Path(__file__).resolve().parent
ALPHA = 0.10
TARGET = 1 - ALPHA

# Canonical algorithm display labels (paper-matching).
ALG_LABELS = {
    "full":    "Weighted CAFHT",
    "uniform": "Uniform weights",
    "zerog":   r"Zero $\gamma$",
}
ALG_ORDER = ["full", "uniform", "zerog"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _abs_dev(seed_coverages: Iterable[float], target: float = TARGET) -> float:
    vals = [abs(c - target) for c in seed_coverages]
    return mean(vals) if vals else float("nan")


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _emit_table(
    out_path: Path,
    caption: str,
    label: str,
    column_spec: str,
    header: str,
    body_rows: list[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{8pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\caption{%",
        f"  {caption}",
        r"}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines))
    print(f"  wrote {out_path}")


# -----------------------------------------------------------------------------
# Synthetic
# -----------------------------------------------------------------------------
SYN_DATA_ORDER = [
    ("static_noshift",  "No shift, static"),
    ("dynamic_noshift", "No shift, dynamic"),
    ("static_shift",    "Static shift"),
    ("dynamic_shift",   "Dynamic shift"),
]


def _load_syn_cell(ar_tag: str, data_tag: str, method: str) -> dict | None:
    p = ROOT / "results/synthetic/json" / f"synth_{ar_tag}_{data_tag}_{method}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def build_synthetic(ar_tag: str, ar_label: str) -> None:
    body: list[str] = []
    have_any = False
    for i, (data_tag, data_label) in enumerate(SYN_DATA_ORDER):
        if i > 0:
            body.append(r"\midrule")
        for method in ALG_ORDER:
            d = _load_syn_cell(ar_tag, data_tag, method)
            if d is None:
                cov = absdev = wid = None
            else:
                have_any = True
                overall = d.get("overall", {})
                cov = overall.get("coverage_mean")
                wid = overall.get("width_mean")
                per_seed = d.get("per_seed_coverage") or []
                absdev = _abs_dev(per_seed) if per_seed else None
            cov_s    = f"{cov:.3f}"    if cov    is not None else "--"
            absdev_s = f"{absdev:.3f}" if absdev is not None else "--"
            wid_s    = f"{wid:.3f}"    if wid    is not None else "--"
            body.append(
                f"{data_label}\n  & {ALG_LABELS[method]:<18} "
                f"& {cov_s} & {absdev_s} & {wid_s} \\\\"
            )

    if not have_any:
        print(f"  [synthetic {ar_tag}] no JSONs found — skipping.")
        return

    out = ROOT / f"results/synthetic/tables/synthetic_{ar_tag}.tex"
    _emit_table(
        out_path=out,
        caption=(
            f"Synthetic results, AR coefficient $\\rho={ar_label}$ "
            f"($T=40$, $\\alpha={ALPHA}$, 30 seeds, target coverage $= {TARGET:.2f}$). "
            r"\textit{Coverage}: pooled empirical coverage across all series and time steps. "
            r"$\overline{|\Delta|}$: mean per-seed absolute deviation from $1-\alpha$. "
            r"\textit{Width}: mean prediction interval width."
        ),
        label=f"tab:synthetic_{ar_tag}",
        column_spec=r"@{} llrrr @{}",
        header=(
            r"{\bfseries Data condition} & {\bfseries Algorithm}"
            r"  & {\bfseries Coverage} & $\boldsymbol{\overline{|\Delta|}}$"
            r" & {\bfseries Width} \\"
        ),
        body_rows=body,
    )


# -----------------------------------------------------------------------------
# Finance
# -----------------------------------------------------------------------------
# For each method we accept multiple legacy stems for cross-version JSONs.
FIN_METHOD_STEMS = {
    "full":    ["full", "shift"],
    "uniform": ["uniform", "noshift"],
    "zerog":   ["zerog", "LRonly"],
}


def _gather_finance_windows(sector_tag: str, method: str, suffix: str) -> list[dict]:
    """Return list of per-window result dicts. `suffix` is "" for tech, "_g10" for util."""
    out: list[dict] = []
    seen_dates: set[str] = set()
    json_dir = ROOT / "results/finance/json"
    # For each method, try suffix-then-empty-suffix so legacy LRonly files
    # (saved without the `_g10` tag) are still picked up.
    suffixes = [suffix, ""] if suffix else [""]
    patterns = [
        f"finance_{sector_tag}_{stem}{sfx}_*.json"
        for stem in FIN_METHOD_STEMS[method]
        for sfx in suffixes
    ]
    for pattern in patterns:
        for p in sorted(json_dir.glob(pattern)):
            # Date tag: tail after the last underscore-separated 2 chunks (YYYYMMDD_YYYYMMDD)
            base = p.stem  # e.g. finance_util_full_g10_20240102_20240229
            tail = base.split("_")
            dates = "_".join(tail[-2:])
            if dates in seen_dates:
                continue
            seen_dates.add(dates)
            out.append(json.loads(p.read_text()))
    return out


def _finance_window_metrics(jsons: list[dict]) -> tuple[float | None, float | None, float | None]:
    if not jsons:
        return None, None, None
    cov_per_win = [d["overall_coverage"] for d in jsons if "overall_coverage" in d]
    width_per_win: list[float] = []
    for d in jsons:
        if "width_by_time" in d and d["width_by_time"]:
            width_per_win.append(mean(d["width_by_time"]))
    if not cov_per_win:
        return None, None, None
    cov   = mean(cov_per_win)
    absdv = _abs_dev(cov_per_win)
    wid   = mean(width_per_win) if width_per_win else None
    return cov, absdv, wid


def _finance_mixed_metrics(method: str) -> tuple[float | None, float | None, float | None]:
    """Mixed has only one window — try several legacy filenames."""
    json_dir = ROOT / "results/finance/json"
    candidates = [
        json_dir / f"finance_mixed_{method}.json",
    ]
    if method == "full":
        candidates.append(json_dir / "finance_mixed_withweighting.json")
    if method == "uniform":
        candidates.append(json_dir / "finance_mixed_noweighting.json")
    p = _first_existing(candidates)
    if p is None:
        return None, None, None
    d = json.loads(p.read_text())
    cov = d.get("overall_coverage")
    wid = mean(d["width_by_time"]) if d.get("width_by_time") else None
    absdv = abs(cov - TARGET) if cov is not None else None
    return cov, absdv, wid


def build_finance() -> None:
    body: list[str] = []
    have_any = False
    sectors = [
        ("tech", "Technology", ""),
        ("util", "Utilities",  "_g10"),
    ]
    for i, (sector_tag, sector_label, suffix) in enumerate(sectors):
        if i > 0:
            body.append(r"\midrule")
        for method in ALG_ORDER:
            wins = _gather_finance_windows(sector_tag, method, suffix)
            cov, absdv, wid = _finance_window_metrics(wins)
            if cov is not None:
                have_any = True
            cov_s    = f"{cov:.3f}"      if cov    is not None else "--"
            absdv_s  = f"{absdv:.3f}"    if absdv  is not None else "--"
            wid_s    = f"{wid:.4f}"      if wid    is not None else "--"
            body.append(
                f"{sector_label}\n  & {ALG_LABELS[method]:<18} "
                f"& {cov_s} & {absdv_s} & {wid_s} \\\\"
            )

    body.append(r"\midrule")
    for method in ALG_ORDER:
        cov, absdv, wid = _finance_mixed_metrics(method)
        if cov is not None:
            have_any = True
        cov_s    = f"{cov:.3f}"   if cov    is not None else "--"
        absdv_s  = f"{absdv:.3f}" if absdv  is not None else "--"
        wid_s    = f"{wid:.4f}"   if wid    is not None else "--"
        body.append(
            f"Mixed (null)\n  & {ALG_LABELS[method]:<18} "
            f"& {cov_s} & {absdv_s} & {wid_s} \\\\"
        )

    if not have_any:
        print("  [finance] no JSONs found — skipping.")
        return

    out = ROOT / "results/finance/tables/finance.tex"
    _emit_table(
        out_path=out,
        caption=(
            r"Finance results (S\&P~500 intraday returns, $\alpha=0.10$, seed~42, target "
            r"coverage $= 0.90$). Technology and Utilities: mean over 13 rolling windows. "
            r"Mixed (null baseline): single window, 15\% random test draw. "
            r"$\overline{|\Delta|}$: mean per-window absolute deviation from $1-\alpha$. "
            r"\textit{Width}: mean interval width (intraday return units). "
            r"Utilities uses $\gamma$-grid $\{0.001, 0.005, 0.01, 0.05, 0.1\}$; "
            r"Technology uses $\{0.001, 0.005, 0.01, 0.05\}$."
        ),
        label="tab:finance",
        column_spec=r"@{} llrrr @{}",
        header=(
            r"{\bfseries Sector} & {\bfseries Algorithm}"
            r"  & {\bfseries Coverage} & $\boldsymbol{\overline{|\Delta|}}$"
            r" & {\bfseries Width} \\"
        ),
        body_rows=body,
    )


# -----------------------------------------------------------------------------
# Medical
# -----------------------------------------------------------------------------
MED_FILES = [
    ("full",    "medical_multi10_full.json"),
    ("uniform", "medical_multi10_aci_only.json"),
    ("zerog",   "medical_multi10_lr_only.json"),
]


def build_medical() -> None:
    body: list[str] = []
    have_any = False
    json_dir = ROOT / "results/medical/json"
    for method, fname in MED_FILES:
        p = json_dir / fname
        if not p.exists():
            cov = absdv = wid = None
        else:
            d = json.loads(p.read_text())
            overall = d.get("overall", {})
            cov = overall.get("coverage_mean")
            wid = overall.get("width_mean")
            per_seed = d.get("per_seed_coverage") or []
            absdv = _abs_dev(per_seed) if per_seed else None
            if cov is not None:
                have_any = True
        cov_s   = f"{cov:.3f}"   if cov   is not None else "--"
        absdv_s = f"{absdv:.3f}" if absdv is not None else "--"
        wid_s   = f"{wid:.1f}"   if wid   is not None else "--"
        body.append(
            f"{ALG_LABELS[method]:<18} & {cov_s} & {absdv_s} & {wid_s} \\\\"
        )

    if not have_any:
        print("  [medical] no JSONs found — skipping.")
        return

    out = ROOT / "results/medical/tables/medical.tex"
    _emit_table(
        out_path=out,
        caption=(
            r"Medical results (MIMIC-III sepsis cohort, NaCl 0.9\% dosage target, "
            r"$\alpha=0.10$, target coverage $= 0.90$, $T = 23$, 10 seeds with "
            r"$n_{\mathrm{traincal}} = 1{,}000$ and $n_{\mathrm{test}} = 500$ subsampled "
            r"per seed). $\overline{|\Delta|}$: mean per-seed absolute deviation from "
            r"$1-\alpha$. \textit{Width}: mean interval width in mL/hr."
        ),
        label="tab:medical",
        column_spec=r"@{} lrrr @{}",
        header=(
            r"{\bfseries Algorithm}"
            r"  & {\bfseries Coverage} & $\boldsymbol{\overline{|\Delta|}}$"
            r" & {\bfseries Width} \\"
        ),
        body_rows=body,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=["synthetic", "finance", "medical"],
        default=["synthetic", "finance", "medical"],
    )
    args = parser.parse_args()

    if "synthetic" in args.sections:
        print("Building synthetic tables...")
        build_synthetic("ar07", "0.7")
        build_synthetic("ar09", "0.9")
    if "finance" in args.sections:
        print("Building finance table...")
        build_finance()
    if "medical" in args.sections:
        print("Building medical table...")
        build_medical()


if __name__ == "__main__":
    main()
