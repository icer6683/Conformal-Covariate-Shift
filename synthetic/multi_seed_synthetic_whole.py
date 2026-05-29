"""
synthetic/multi_seed_synthetic_whole.py — 30-seed wrapper around
synthetic_runner_whole (Algorithm 1, whole-trajectory).

Runs synthetic_runner_whole across n_seeds independent seeds, aggregates
coverage + width (mean ± std across seeds; inter-seed IQR for the
coverage-over-time profile), and writes a single aggregated JSON whose schema
matches the other multi-seed outputs.

Defaults: n_seeds=30, base_seed=1000. CLI / details: see WEIGHTED_CAFHT_PLAN.md
§ 4 step 8.

TODO: implement per § 4 step 8.
"""
