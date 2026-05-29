"""
medical/multi_seed_medical_last.py — 10-seed wrapper around
medical_runner_last (Algorithm 2, last-step).

Subsamples n_traincal / n_test patients per seed from the full pool, runs
medical_runner_last, and aggregates coverage + width across seeds. Output
schema matches the other multi-seed JSONs. Defaults: n_seeds=10, base_seed=1000.

CLI / details: see WEIGHTED_CAFHT_PLAN.md § 4 step 6.

TODO: implement per § 4 step 6.
"""
