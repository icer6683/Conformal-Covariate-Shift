"""
synthetic/multi_seed_synthetic_last.py — 30-seed wrapper around
synthetic_runner_last (Algorithm 2, last-step).

Runs synthetic_runner_last across n_seeds independent seeds, aggregates
coverage + width across seeds, and writes a single aggregated JSON whose schema
matches the other multi-seed outputs.

Defaults: n_seeds=30, base_seed=1000. CLI / details: see WEIGHTED_CAFHT_PLAN.md
§ 4 step 8.

TODO: implement per § 4 step 8.
"""
