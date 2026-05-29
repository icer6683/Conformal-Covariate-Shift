"""
build_tex_tables_v2.py — emit one LaTeX table per (domain × regime) = 6 tables.

Extends OLD_build_tex_tables.py (logic copied, NOT imported — § B.3): reuses the
_emit_table / _abs_dev helpers and the per-domain builders, but produces six
tables instead of four — {synthetic, finance, medical} × {whole_trajectory,
last_step}.

Layout (see WEIGHTED_CAFHT_PLAN.md § 5.4): the shared 4-column block
(Algorithm × Coverage × |Δ̄| × Width) plus a leading data-condition column where
the domain has one (synthetic: noshift/static/dynamic; finance: tech/util/mixed).
The medical last-step table has only 2 rows (no zerog). Reads JSONs from
results/{domain}/{regime}/json/ and writes
results/{domain}/{regime}/tables/{domain}_{regime}.tex.

TODO: implement per § 4 step 9.
"""
