#!/usr/bin/env python3
"""
Build master results table from every saved JSON in
results/synthetic/json and results/finance/json.

Outputs:
  results/audit/master_results_table.csv
  results/audit/master_results_table.md
  results/audit/finance_tech_aggregate.csv
  results/audit/finance_tech_aggregate.md
  results/audit/schema_inconsistencies.md

No interpretation — faithful extraction only.
"""
import json
import hashlib
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent.parent.parent
SYN_DIR = ROOT / "results" / "synthetic" / "json"
FIN_DIR = ROOT / "results" / "finance" / "json"
OUT_DIR = ROOT / "results" / "audit"


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:8]


def paper_method_name(predictor: str, with_shift: bool, aci_stepsize: float | None,
                      mixed: bool = False) -> str:
    if predictor == "algorithm":
        if mixed:
            return "AdaptedCAFHT (LR + ACI, mixed null)" if with_shift else "AdaptedCAFHT (uniform weights + ACI, mixed null)"
        if with_shift:
            if aci_stepsize == 0.0:
                return "AdaptedCAFHT (LR only, γ=0)"
            return "Weighted CAFHT (LR + ACI)"
        else:
            if aci_stepsize == 0.0:
                return "AdaptedCAFHT (uniform weights, γ=0)"
            return "AdaptedCAFHT (uniform weights + ACI)"
    if predictor == "adaptive":
        return "Sliding-window split conformal"
    return predictor


def synthetic_dgp(cfg: dict) -> str:
    mode = cfg.get("covariate_mode", "?")
    if mode == "static":
        base = f"static-X Pois(λ={cfg.get('covar_rate')})"
        if cfg.get("with_shift"):
            base += f"→λ̃={cfg.get('covar_rate_shift')}"
        return base
    elif mode == "dynamic":
        base = f"dynamic-X ρ={cfg.get('x_rate')}"
        if cfg.get("with_shift"):
            base += f"→ρ̃={cfg.get('x_rate_shift')}"
        return base
    return str(mode)


def extract_synthetic(path: Path) -> dict:
    d = json.load(open(path))
    cfg = d.get("config", {}) or {}
    overall = d.get("overall", {}) or {}
    time_steps = d.get("time_steps", [])
    by_time = d.get("by_time", {}) or {}
    available_fields = []
    # by_time entries: check one
    if by_time:
        first_key = next(iter(by_time))
        available_fields = sorted(list(by_time[first_key].keys())) if isinstance(by_time[first_key], dict) else []

    predictor = cfg.get("predictor")
    with_shift = bool(cfg.get("with_shift", False))
    aci = cfg.get("aci_stepsize")

    notes = []
    if predictor == "adaptive":
        notes.append("baseline labeled 'adaptive' but is sliding-window split conformal (no ACI); see CLAUDE.md §Baselines")
    if aci is not None and aci != 0.0 and predictor == "algorithm":
        notes.append(f"ACI stepsize fixed at {aci} (γ-selector not active in this run)")
    if "n_test" not in cfg and "n_series" in cfg:
        notes.append("n_test stored as n_series")
    if not overall.get("coverage_std"):
        pass  # present in all
    else:
        notes.append("coverage_std is pooled per-(series,t), not cross-seed")

    return {
        "domain": "synthetic",
        "experiment_id": path.stem,
        "file_path": str(path.relative_to(ROOT)),
        "method": paper_method_name(predictor, with_shift, aci),
        "shift_condition": "shift" if with_shift else "no_shift",
        "dgp_or_dataset": synthetic_dgp(cfg),
        "n_train": cfg.get("n_train"),
        "n_cal": cfg.get("n_cal"),
        "n_test": cfg.get("n_series"),
        "T": cfg.get("T"),
        "alpha": cfg.get("alpha"),
        "n_seeds": d.get("n_seeds"),
        "seed": None,
        "overall_coverage": overall.get("coverage_mean"),
        "coverage_std_pooled": overall.get("coverage_std"),
        "coverage_se": overall.get("coverage_se"),
        "mean_width": overall.get("width_mean"),
        "width_std_pooled": overall.get("width_std"),
        "early_coverage_mean": overall.get("early_coverage_mean"),
        "late_coverage_mean": overall.get("late_coverage_mean"),
        "coverage_degradation": overall.get("coverage_degradation"),
        "time_resolved_metrics": ";".join(available_fields) if available_fields else "",
        "aci_stepsize": aci,
        "covariate_mode": cfg.get("covariate_mode"),
        "with_shift": with_shift,
        "mixed": None,
        "notes": " | ".join(notes),
    }


def extract_finance_experiment(path: Path) -> dict:
    d = json.load(open(path))
    cfg = d.get("config", {}) or {}

    # method
    test_sector = cfg.get("test_sector") or ""
    with_shift = bool(cfg.get("with_shift", False))
    mixed = bool(cfg.get("mixed", False))

    # Parse shift condition from filename as fallback
    name = path.stem
    if mixed:
        shift_label = "mixed_null_weighted" if with_shift else "mixed_null_unweighted"
    elif "noshift" in name:
        shift_label = "no_shift_unweighted"
    elif "shift" in name:
        shift_label = "shift_weighted"
    else:
        shift_label = "?"

    # detect LR-only ablation (γ=0) via gamma_grid
    gg = cfg.get("gamma_grid") or []
    is_lr_only = (len(gg) == 1 and float(gg[0]) == 0.0 and with_shift)

    # method label
    if mixed:
        method = "Weighted CAFHT (mixed-sector null, LR active)" if with_shift else "AdaptedCAFHT (mixed-sector null, uniform weights + ACI)"
    elif is_lr_only:
        method = "AdaptedCAFHT (LR only, γ=0)"
    elif with_shift:
        method = "Weighted CAFHT (LR + ACI)"
    else:
        method = "AdaptedCAFHT (uniform weights + ACI)"

    # refine shift_condition to reflect LRonly
    if is_lr_only:
        shift_label = "shift_LRonly"

    cov = d.get("overall_coverage")
    w_by_t = d.get("width_by_time") or []
    c_by_t = d.get("coverage_by_time") or []
    mean_width = (sum(w_by_t) / len(w_by_t)) if w_by_t else None

    # time-resolved present
    tr = []
    for k in ["coverage_by_time", "width_by_time", "gamma_opt_history",
              "clf_prob1_mean_by_time", "clf_prob1_std_by_time", "dates"]:
        if k in d:
            tr.append(k)

    notes = []
    if "clf_prob1_mean_by_time" not in d:
        notes.append("missing clf_prob1_mean_by_time")
    if " 2.json" in path.name:
        notes.append("duplicate artifact ('2.json') — likely rerun")
    if cfg.get("gamma_grid") and len(cfg["gamma_grid"]) > 0:
        pass  # γ-grid recorded

    # dates window
    dates = d.get("dates") or []
    date_window = f"{dates[0]}→{dates[-1]}" if dates else ""

    # dataset label: infer from npz pattern in filename
    # e.g. finance_tech_shift_20240201_20240328.json → sp500 2024-02-01..2024-03-28
    tag = path.stem.replace("finance_", "").replace(" 2", "_DUP")
    dataset = f"S&P 500 {test_sector or '?'} · {date_window}" if date_window else f"S&P 500 {test_sector or '?'}"

    return {
        "domain": "finance",
        "experiment_id": tag,
        "file_path": str(path.relative_to(ROOT)),
        "method": method,
        "shift_condition": shift_label,
        "dgp_or_dataset": dataset,
        "n_train": cfg.get("n_train"),
        "n_cal": cfg.get("n_cal"),
        "n_test": cfg.get("n_test"),
        "T": cfg.get("L"),  # L is the horizon in finance files
        "alpha": cfg.get("alpha"),
        "n_seeds": 1,
        "seed": cfg.get("seed"),
        "overall_coverage": cov,
        "coverage_std_pooled": None,
        "coverage_se": None,
        "mean_width": mean_width,
        "width_std_pooled": None,
        "early_coverage_mean": None,
        "late_coverage_mean": None,
        "coverage_degradation": None,
        "time_resolved_metrics": ";".join(tr),
        "aci_stepsize": None,
        "covariate_mode": None,
        "with_shift": with_shift,
        "mixed": mixed,
        "notes": " | ".join(notes),
    }


def extract_featurizer_tuning(path: Path) -> dict:
    d = json.load(open(path))
    name = path.stem
    variants = d.get("variants", [])
    ns = d.get("no_shift", {}) or {}
    notes = [
        f"featurizer tuning artifact (not an experiment run); {len(variants)} variants compared",
        "schema different from experiment JSONs: no config/coverage_by_time/width_by_time",
    ]
    return {
        "domain": "finance",
        "experiment_id": name,
        "file_path": str(path.relative_to(ROOT)),
        "method": "(featurizer grid-search)",
        "shift_condition": "tuning",
        "dgp_or_dataset": f"S&P 500 {d.get('test_sector','?')} · {d.get('npz','?').split('/')[-1]}",
        "n_train": None,
        "n_cal": None,
        "n_test": None,
        "T": None,
        "alpha": d.get("alpha"),
        "n_seeds": 1,
        "seed": d.get("seed"),
        "overall_coverage": ns.get("coverage"),
        "coverage_std_pooled": None,
        "coverage_se": None,
        "mean_width": ns.get("mean_width"),
        "width_std_pooled": None,
        "early_coverage_mean": None,
        "late_coverage_mean": None,
        "coverage_degradation": None,
        "time_resolved_metrics": "variants[]",
        "aci_stepsize": None,
        "covariate_mode": None,
        "with_shift": None,
        "mixed": None,
        "notes": " | ".join(notes),
    }


def main():
    rows = []
    # synthetic
    for p in sorted(SYN_DIR.glob("*.json")):
        rows.append(extract_synthetic(p))
    # finance
    for p in sorted(FIN_DIR.glob("*.json")):
        name = p.name
        if name.startswith("featurizer_tuning"):
            rows.append(extract_featurizer_tuning(p))
        else:
            rows.append(extract_finance_experiment(p))

    columns = [
        "domain", "experiment_id", "file_path", "method",
        "shift_condition", "dgp_or_dataset",
        "n_train", "n_cal", "n_test", "T", "alpha", "n_seeds", "seed",
        "overall_coverage", "coverage_std_pooled", "coverage_se",
        "mean_width", "width_std_pooled",
        "early_coverage_mean", "late_coverage_mean", "coverage_degradation",
        "time_resolved_metrics", "aci_stepsize", "covariate_mode",
        "with_shift", "mixed", "notes",
    ]

    # CSV
    out_csv = OUT_DIR / "master_results_table.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in columns})
    print(f"wrote {out_csv} ({len(rows)} rows)")

    # Markdown
    md_lines = []
    md_lines.append("# Master results table\n")
    md_lines.append(f"Total rows: {len(rows)} (synthetic: {sum(1 for r in rows if r['domain']=='synthetic')}, finance experiment: {sum(1 for r in rows if r['domain']=='finance' and r['shift_condition']!='tuning')}, featurizer tuning: {sum(1 for r in rows if r['shift_condition']=='tuning')})\n")
    md_lines.append("Medical: no saved JSONs in results/medical/json (experiment not yet run).\n")

    # Compact markdown table (not all columns — too wide)
    short_cols = [
        "domain", "experiment_id", "method", "shift_condition",
        "dgp_or_dataset", "n_train", "n_cal", "n_test", "T", "alpha",
        "n_seeds", "seed", "overall_coverage", "mean_width",
        "time_resolved_metrics", "notes",
    ]
    md_lines.append("## Compact view (all experiments)\n")
    md_lines.append("| " + " | ".join(short_cols) + " |")
    md_lines.append("|" + "|".join(["---"] * len(short_cols)) + "|")
    for r in rows:
        row_cells = []
        for c in short_cols:
            v = r.get(c)
            if v is None or v == "":
                cell = "—"
            elif isinstance(v, float):
                cell = f"{v:.4f}"
            else:
                cell = str(v)
            # truncate notes column
            if c == "notes" and len(cell) > 80:
                cell = cell[:77] + "..."
            # escape pipes inside cell values so markdown table columns align
            cell = cell.replace("|", "\\|")
            row_cells.append(cell)
        md_lines.append("| " + " | ".join(row_cells) + " |")
    md_lines.append("")

    # Full column reference
    md_lines.append("## Full column set")
    md_lines.append("See `master_results_table.csv` for all columns: " + ", ".join(columns))
    md_lines.append("")

    out_md = OUT_DIR / "master_results_table.md"
    out_md.write_text("\n".join(md_lines))
    print(f"wrote {out_md}")

    # ==================== Finance tech aggregate ====================
    tech_shift = [r for r in rows if r["domain"] == "finance"
                  and r["experiment_id"].startswith("tech_shift_")
                  and "DUP" not in r["experiment_id"]]
    tech_noshift = [r for r in rows if r["domain"] == "finance"
                    and r["experiment_id"].startswith("tech_noshift_")
                    and "DUP" not in r["experiment_id"]]
    tech_lronly = [r for r in rows if r["domain"] == "finance"
                   and r["experiment_id"].startswith("tech_LRonly_")
                   and "DUP" not in r["experiment_id"]]
    util_shift = [r for r in rows if r["domain"] == "finance"
                  and r["experiment_id"].startswith("util_shift_")]
    util_noshift = [r for r in rows if r["domain"] == "finance"
                    and r["experiment_id"].startswith("util_noshift_")]
    util_lronly = [r for r in rows if r["domain"] == "finance"
                   and r["experiment_id"].startswith("util_LRonly_")]
    hc_shift = [r for r in rows if r["domain"] == "finance"
                and r["experiment_id"].startswith("healthcare_shift_")]
    hc_noshift = [r for r in rows if r["domain"] == "finance"
                  and r["experiment_id"].startswith("healthcare_noshift_")]

    def stats(xs):
        xs = [x for x in xs if x is not None]
        n = len(xs)
        if n == 0:
            return (0, None, None, None, None)
        m = sum(xs) / n
        var = sum((x - m) ** 2 for x in xs) / n
        s = var ** 0.5
        return (n, m, s, min(xs), max(xs))

    agg_rows = []
    for label, rs in [("tech_shift (Weighted CAFHT: LR + ACI)", tech_shift),
                      ("tech_noshift (AdaptedCAFHT: uniform weights + ACI)", tech_noshift),
                      ("tech_LRonly (AdaptedCAFHT: LR only, γ=0)", tech_lronly),
                      ("util_shift (Weighted CAFHT: LR + ACI)", util_shift),
                      ("util_noshift (AdaptedCAFHT: uniform weights + ACI)", util_noshift),
                      ("util_LRonly (AdaptedCAFHT: LR only, γ=0)", util_lronly)]:
        covs = [r["overall_coverage"] for r in rs]
        widths = [r["mean_width"] for r in rs]
        nc, mc, sc, minc, maxc = stats(covs)
        nw, mw, sw, minw, maxw = stats(widths)
        agg_rows.append({
            "condition": label,
            "n_windows": nc,
            "coverage_mean": mc,
            "coverage_std_across_windows": sc,
            "coverage_min": minc,
            "coverage_max": maxc,
            "width_mean": mw,
            "width_std_across_windows": sw,
            "width_min": minw,
            "width_max": maxw,
        })

    def build_per_window(sh, no, lr, prefix):
        out = []
        by_date = {}
        for r in sh + no + lr:
            tag = r["experiment_id"]
            parts = tag.split("_")
            window = "_".join(parts[-2:]) if len(parts) >= 4 else tag
            if tag.startswith(f"{prefix}_LRonly_"):
                cond = "lronly"
            elif tag.startswith(f"{prefix}_noshift_"):
                cond = "noshift"
            elif tag.startswith(f"{prefix}_shift_"):
                cond = "shift"
            else:
                cond = "?"
            by_date.setdefault(window, {})[cond] = r
        for window in sorted(by_date.keys()):
            d = by_date[window]
            out.append({
                "window": window,
                "shift_coverage": d.get("shift", {}).get("overall_coverage"),
                "shift_width": d.get("shift", {}).get("mean_width"),
                "noshift_coverage": d.get("noshift", {}).get("overall_coverage"),
                "noshift_width": d.get("noshift", {}).get("mean_width"),
                "lronly_coverage": d.get("lronly", {}).get("overall_coverage"),
                "lronly_width": d.get("lronly", {}).get("mean_width"),
            })
        return out

    per_window_rows = build_per_window(tech_shift, tech_noshift, tech_lronly, "tech")
    util_window_rows = build_per_window(util_shift, util_noshift, util_lronly, "util")

    # Aggregate CSV
    agg_csv = OUT_DIR / "finance_tech_aggregate.csv"
    with open(agg_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["--- Aggregate across windows ---"])
        w.writerow(["condition", "n_windows", "coverage_mean",
                    "coverage_std_across_windows", "coverage_min", "coverage_max",
                    "width_mean", "width_std_across_windows",
                    "width_min", "width_max"])
        for r in agg_rows:
            w.writerow([r[k] if r[k] is not None else "" for k in
                       ["condition", "n_windows", "coverage_mean",
                        "coverage_std_across_windows", "coverage_min", "coverage_max",
                        "width_mean", "width_std_across_windows",
                        "width_min", "width_max"]])
        def sub(a, b):
            return (a - b) if (a is not None and b is not None) else ""

        def write_pw_block(title, rows):
            w.writerow([])
            w.writerow([f"--- {title} ---"])
            w.writerow(["window",
                        "shift_cov", "shift_width",
                        "noshift_cov", "noshift_width",
                        "lronly_cov", "lronly_width",
                        "Δcov_shift_minus_noshift", "Δwidth_shift_minus_noshift",
                        "Δcov_lronly_minus_noshift", "Δwidth_lronly_minus_noshift",
                        "Δcov_shift_minus_lronly", "Δwidth_shift_minus_lronly"])
            for r in rows:
                w.writerow([r["window"],
                            r["shift_coverage"] if r["shift_coverage"] is not None else "",
                            r["shift_width"] if r["shift_width"] is not None else "",
                            r["noshift_coverage"] if r["noshift_coverage"] is not None else "",
                            r["noshift_width"] if r["noshift_width"] is not None else "",
                            r["lronly_coverage"] if r["lronly_coverage"] is not None else "",
                            r["lronly_width"] if r["lronly_width"] is not None else "",
                            sub(r["shift_coverage"], r["noshift_coverage"]),
                            sub(r["shift_width"], r["noshift_width"]),
                            sub(r["lronly_coverage"], r["noshift_coverage"]),
                            sub(r["lronly_width"], r["noshift_width"]),
                            sub(r["shift_coverage"], r["lronly_coverage"]),
                            sub(r["shift_width"], r["lronly_width"])])

        write_pw_block("Tech per-window, three conditions", per_window_rows)
        write_pw_block("Utilities per-window, three conditions", util_window_rows)

        # Healthcare block
        w.writerow([])
        w.writerow(["--- Healthcare (2 windows) ---"])
        w.writerow(["window", "shift_cov", "shift_width", "noshift_cov", "noshift_width",
                    "Δcov_shift_minus_noshift", "Δwidth_shift_minus_noshift"])
        hc_by_date = {}
        for r in hc_shift:
            parts = r["experiment_id"].split("_")
            window = "_".join(parts[-2:])
            hc_by_date.setdefault(window, {})["shift"] = r
        for r in hc_noshift:
            parts = r["experiment_id"].split("_")
            window = "_".join(parts[-2:])
            hc_by_date.setdefault(window, {})["noshift"] = r
        for window in sorted(hc_by_date.keys()):
            d = hc_by_date[window]
            s = d.get("shift", {}); n = d.get("noshift", {})
            w.writerow([window,
                        s.get("overall_coverage", ""),
                        s.get("mean_width", ""),
                        n.get("overall_coverage", ""),
                        n.get("mean_width", ""),
                        sub(s.get("overall_coverage"), n.get("overall_coverage")),
                        sub(s.get("mean_width"), n.get("mean_width"))])
    print(f"wrote {agg_csv}")

    # Aggregate markdown
    def f(v, n=4): return f"{v:.{n}f}" if isinstance(v, float) else ("—" if v is None else str(v))
    def fv(x, n=4): return f"{x:.{n}f}" if isinstance(x, float) else "—"
    def subf(a, b):
        return fv((a - b)) if (a is not None and b is not None) else "—"

    lines = ["# Finance Technology — aggregate over windows (tech + healthcare)\n",
             "Duplicates (` 2.json`) excluded.\n",
             "## Tech summary (13 windows)\n",
             "| condition | n_windows | coverage_mean | coverage_std | coverage_min | coverage_max | width_mean | width_std | width_min | width_max |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in agg_rows:
        lines.append(f"| {r['condition']} | {r['n_windows']} | {f(r['coverage_mean'])} | {f(r['coverage_std_across_windows'])} | {f(r['coverage_min'])} | {f(r['coverage_max'])} | {f(r['width_mean'])} | {f(r['width_std_across_windows'])} | {f(r['width_min'])} | {f(r['width_max'])} |")
    lines.append("")

    def pw_md_block(title, rows):
        lines.append(f"## {title}\n")
        lines.append("Columns: shift = Weighted CAFHT (LR + ACI); noshift = AdaptedCAFHT (uniform + ACI); LRonly = AdaptedCAFHT (LR + γ=0).\n")
        lines.append("| window | shift_cov | shift_width | noshift_cov | noshift_width | LRonly_cov | LRonly_width | Δcov (shift−noshift) | Δw (shift−noshift) | Δcov (LRonly−noshift) | Δw (LRonly−noshift) | Δcov (shift−LRonly) | Δw (shift−LRonly) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            sc = r["shift_coverage"]; sw = r["shift_width"]
            nsc = r["noshift_coverage"]; nsw = r["noshift_width"]
            lc = r["lronly_coverage"]; lw = r["lronly_width"]
            lines.append(f"| {r['window']} | {fv(sc)} | {fv(sw)} | {fv(nsc)} | {fv(nsw)} | {fv(lc)} | {fv(lw)} | {subf(sc, nsc)} | {subf(sw, nsw)} | {subf(lc, nsc)} | {subf(lw, nsw)} | {subf(sc, lc)} | {subf(sw, lw)} |")
        lines.append("")

    pw_md_block("Tech per-window, three conditions", per_window_rows)
    pw_md_block("Utilities per-window, three conditions", util_window_rows)

    # Healthcare block
    lines.append("## Healthcare (2 windows — preliminary)\n")
    lines.append("Note: the old no-date file `finance_healthcare_shift.json` (same as 20240201_20240328) is NOT included below; only the new dated pairs are shown.\n")
    lines.append("| window | shift_cov | shift_width | noshift_cov | noshift_width | Δcov (shift−noshift) | Δw (shift−noshift) |")
    lines.append("|---|---|---|---|---|---|---|")
    hc_by_date = {}
    for r in hc_shift:
        parts = r["experiment_id"].split("_")
        window = "_".join(parts[-2:])
        hc_by_date.setdefault(window, {})["shift"] = r
    for r in hc_noshift:
        parts = r["experiment_id"].split("_")
        window = "_".join(parts[-2:])
        hc_by_date.setdefault(window, {})["noshift"] = r
    for window in sorted(hc_by_date.keys()):
        d = hc_by_date[window]
        s = d.get("shift", {}); n = d.get("noshift", {})
        sc = s.get("overall_coverage"); sw = s.get("mean_width")
        nc = n.get("overall_coverage"); nw = n.get("mean_width")
        lines.append(f"| {window} | {fv(sc)} | {fv(sw)} | {fv(nc)} | {fv(nw)} | {subf(sc, nc)} | {subf(sw, nw)} |")

    agg_md = OUT_DIR / "finance_tech_aggregate.md"
    agg_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {agg_md}")

    # ==================== Schema inconsistencies note ====================
    # Detect by scanning all paths
    issues = []
    # synthetic: look for version differences
    syn_paths = sorted(SYN_DIR.glob("*.json"))
    fin_paths = sorted(FIN_DIR.glob("*.json"))

    # schemas
    syn_key_sets = {}
    for p in syn_paths:
        syn_key_sets[p.name] = sorted(json.load(open(p)).keys())
    fin_key_sets = {}
    for p in fin_paths:
        fin_key_sets[p.name] = sorted(json.load(open(p)).keys())

    # Find files missing clf_prob1_*
    missing_clf = []
    for n, ks in fin_key_sets.items():
        if n.startswith("featurizer_tuning"):
            continue
        if "clf_prob1_mean_by_time" not in ks:
            missing_clf.append(n)

    # Find duplicates " 2.json"
    dups = [n for n in fin_key_sets if " 2.json" in n]

    # n_series vs n_test naming in synthetic config
    syn_uses_n_series = []
    for p in syn_paths:
        cfg = (json.load(open(p)).get("config") or {})
        if "n_series" in cfg and "n_test" not in cfg:
            syn_uses_n_series.append(p.name)

    # early-style synthetic files vs updated schema
    syn_versions = {}
    for p in syn_paths:
        d = json.load(open(p))
        sig = []
        if "overall" in d:
            sig.append("has `overall`")
            if "coverage_se" in (d["overall"] or {}):
                sig.append("overall.coverage_se present")
        if "by_time" in d:
            sig.append("has `by_time`")
        syn_versions[p.name] = " + ".join(sig)

    lines = ["# Schema inconsistencies across saved JSONs\n"]
    lines.append("## Three distinct schemas\n")
    lines.append("1. **Synthetic multi-seed** (top-level keys: `time_steps`, `n_seeds`, `config`, `by_time`, `overall`). 7 files in `results/synthetic/json/`.")
    lines.append("2. **Finance experiment** (top-level keys: `coverage_by_time`, `width_by_time`, `overall_coverage`, `target_coverage`, `dates`, `gamma_opt_history`, `first_test_ticker`, `first_test_series`, `config`; optional `clf_prob1_mean_by_time`, `clf_prob1_std_by_time`). 37 files in `results/finance/json/`.")
    lines.append("3. **Featurizer tuning** (top-level keys: `npz`, `test_sector`, `alpha`, `seed`, `no_shift`, `variants`). 5 files in `results/finance/json/`.\n")

    lines.append("## Inconsistencies\n")

    lines.append("### Finance experiment files missing `clf_prob1_mean_by_time` / `clf_prob1_std_by_time`\n")
    if missing_clf:
        for n in missing_clf:
            lines.append(f"- `{n}`")
    else:
        lines.append("- None.")
    lines.append("")

    lines.append("### Duplicate artifacts (` 2.json` suffix)\n")
    if dups:
        for n in sorted(dups):
            lines.append(f"- `{n}`")
    else:
        lines.append("- None.")
    lines.append("")

    lines.append("### Synthetic config field naming\n")
    lines.append("- All 7 synthetic files use `config.n_series` (not `n_test`). Interpreted as test-set size.")
    lines.append("- `n_train` and `n_cal` are stored as top-level config fields.")
    lines.append("")

    lines.append("### Synthetic `overall` structure\n")
    for n, sig in sorted(syn_versions.items()):
        lines.append(f"- `{n}`: {sig}")
    lines.append("")

    lines.append("### Missing fields across all experiment JSONs (never populated)\n")
    lines.append("- Joint coverage (per-series `prod_t 1[covered]`) — never computed.")
    lines.append("- Cross-seed std of per-seed means (synthetic `overall.coverage_std` is pooled per-(series,t), not across seeds).")
    lines.append("- Raw per-series coverage arrays are stored inside `by_time[t].coverage_history` (synthetic) and `coverage_by_time` aggregates only the fraction (finance), so reconstructing joint coverage is possible on synthetic but not directly on finance without reloading.")
    lines.append("")

    lines.append("### Horizon field name\n")
    lines.append("- Synthetic: `config.T`.")
    lines.append("- Finance: `config.L` (number of time steps = 40; `coverage_by_time`/`dates`/`width_by_time` have length 39 = L−1).")
    lines.append("")

    lines.append("### Predictor identifier\n")
    lines.append("- Synthetic: `config.predictor` ∈ {`algorithm`, `adaptive`}. Files with `predictor=adaptive` are sliding-window split conformal (no ACI); CLAUDE.md is now updated to reflect this.")
    lines.append("- Finance: no `config.predictor` field — method is inferred from filename (`shift`/`noshift`/`mixed`) and `config.with_shift` / `config.mixed` flags.")
    lines.append("")

    lines.append("### Seeds\n")
    lines.append("- Synthetic: `n_seeds` top-level (all 4 of the `20260414_*` files have `n_seeds=30`). Older `20260206`/`20260210` files also multi-seed.")
    lines.append("- Finance: single `config.seed=42` (no multi-seed wrapper for finance).")
    lines.append("")

    lines.append("### Width and coverage time series representation\n")
    lines.append("- Synthetic: `by_time[t]` is a dict per time step with `coverage_rate`, `coverage_std`, `interval_width`, `width_std`, `alpha_mean`, `alpha_std`, `gamma_opt`, etc. Keyed by stringified t.")
    lines.append("- Finance: flat lists `coverage_by_time[i]` / `width_by_time[i]` / `gamma_opt_history[i]` / `dates[i]`, length L−1.")
    lines.append("")

    lines.append("### Sample-size drift across synthetic runs\n")
    lines.append("The 7 synthetic files do NOT share common sample sizes:\n")
    for p in syn_paths:
        cfg = (json.load(open(p)).get("config") or {})
        nseed = json.load(open(p)).get("n_seeds")
        lines.append(f"- `{p.name}`: n_seeds={nseed}, n_train={cfg.get('n_train')}, n_cal={cfg.get('n_cal')}, n_series={cfg.get('n_series')}, T={cfg.get('T')}")
    lines.append("")
    lines.append("- Older `20260206` files: n_train=1000, n_cal=1000, n_series=500, n_seeds=10.")
    lines.append("- Single-seed outlier `20260210_121930`: n_seeds=1 (not a multi-seed run).")
    lines.append("- Newer `20260414` files (the current reference runs): n_train=600, n_cal=600, n_series=300, n_seeds=30.")
    lines.append("- CLAUDE.md states the `20260414` runs supersede earlier ones, but all 7 files are still present in `results/synthetic/json/` without being moved or deleted.")
    lines.append("")

    lines.append("### Finance horizon (`config.L`) varies across windows\n")
    lines.append("`L` ranges from 39 to 44 across the 13 tech windows because trading-day counts differ per 2-month window (holidays, short months). `coverage_by_time`/`width_by_time`/`dates`/`gamma_opt_history` have length `L − 1`. This means 'mean width across windows' averages over different horizon lengths; per-time-step panels across windows do not line up in index.")
    lines.append("")

    lines.append("### Units / semantics\n")
    lines.append("- Synthetic target Y is on the raw AR(1) scale; widths (~1.28) are comparable to AR(1) residual std × q_{0.9} ≈ 2·0.65 = 1.30 when noise_std≈0.2, ar_coef=0.7.")
    lines.append("- Finance target Y is intraday return (Close/Open − 1) in decimal units; widths (~0.04–0.06) are comparable across tickers but are absolute returns, not bps.")
    lines.append("")

    out_issues = OUT_DIR / "schema_inconsistencies.md"
    out_issues.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_issues}")


if __name__ == "__main__":
    main()
