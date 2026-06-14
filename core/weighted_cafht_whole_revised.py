"""
core/weighted_cafht_whole_revised.py — EXPERIMENTAL revised Algorithm 1.

Per-step likelihood-ratio reweighting (see REVISED_WHOLE_TRAJECTORY.md). Instead
of one classifier on X_1 giving a single correction η_j that inflates every step,
we train a classifier p̂_t on the covariate prefix X_{1:t} for each step t,
producing per-step weights Ŵ_{i,t} and a per-step correction η_{j,t} that
inflates only step t. The conformity scores ε_i (max-over-trajectory) are reused;
only the weights change with t.

Experimental: this file IMPORTS the base class + helpers (relaxing the §0
self-contained rule) so it stays small and easy to delete on revert. The
original core/weighted_cafht_whole.py is untouched.
"""

import numpy as np

from core.weighted_cafht_whole import (
    WeightedCAFHTWholeTrajectory, ACI,
    weighted_quantile_with_inf, density_ratio_weights, _resid_inf_norm,
)


class WeightedCAFHTWholeTrajectoryRevised(WeightedCAFHTWholeTrajectory):
    """Revised whole-trajectory method with per-step (prefix X_{1:t}) classifiers.

    Inherits __init__, select_gamma, calibration_scores unchanged. The
    `featurize_fn` given to __init__ is NOT used here (kept only for constructor
    compatibility) — the per-step classifier features are passed directly to
    predict_bands as (n, horizon, d) arrays.
    """

    def predict_bands(self, tr_data, cal_data, test_data, aci_data,
                      Xc_tr, Xc_cal, Xc_test, y_trim=None, seed=123):
        """Per-step-reweighted deployment.

        Parameters
        ----------
        tr_data, cal_data, test_data, aci_data : as in the base class — the
            (pred, true) pairs for D_tr / D_cal / D_test / D_ACI.
        Xc_tr, Xc_cal, Xc_test : (n, horizon, d) PER-STEP classifier features;
            slice [:, t, :] summarizes the covariate prefix X_{1:t}.

        Returns
        -------
        (n_test, horizon, 2, ndim) prediction bands.
        """
        tr_pred, tr_true = tr_data
        cal_pred, cal_true = cal_data
        test_pred = np.asarray(test_data[0], float)
        test_true = np.asarray(test_data[1], float)
        aci_pred, aci_true = aci_data

        n_test, horizon, ndim = test_pred.shape
        Xc_tr = np.asarray(Xc_tr, float)
        Xc_cal = np.asarray(Xc_cal, float)
        Xc_test = np.asarray(Xc_test, float)
        assert Xc_test.shape[1] == horizon, "per-step features must match horizon"

        # (1) frozen D_ACI per-step score bank (unchanged from base).
        score_bank = _resid_inf_norm(np.asarray(aci_pred, float),
                                     np.asarray(aci_true, float))
        self.score_bank_shape_ = tuple(score_bank.shape)

        # (2) γ_opt and (3) calibration ε_i — inherited, unchanged.
        gamma_opt = self.select_gamma(tr_pred, tr_true, seed=seed)
        self.gamma_opt_ = gamma_opt
        eps = self.calibration_scores(cal_pred, cal_true, score_bank,
                                      gamma_opt, seed=seed)
        self.eps_ = eps

        # (4) cross-half deployment with PER-STEP classifiers.
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_test)
        cut = n_test // 2
        half_a, half_b = perm[:cut], perm[cut:]

        out = np.empty((n_test, horizon, 2, ndim))
        level = 1.0 - self.alpha
        self.n_inf_ = 0

        for pos_idx, deploy_idx in ((half_a, half_b), (half_b, half_a)):
            if deploy_idx.size == 0:
                continue

            # ACI bands for the deploy half (frozen bank, γ_opt) — unchanged.
            aci_bands = ACI(self.alpha, verbose=False).predict_intervals(
                score_bank, test_pred[deploy_idx], test_true[deploy_idx],
                gamma=gamma_opt, seed=seed)

            # Per-step correction η_{j,t}: fit p̂_t once per step on the
            # prefix-X_{1:t} features (negatives = D_tr, positives = the
            # opposite test half), then reweight the SAME ε scores.
            eta = np.empty((deploy_idx.size, horizon))
            for t in range(horizon):
                dr = density_ratio_weights(
                    Xc_tr[:, t, :], Xc_test[pos_idx, t, :], Xc_cal[:, t, :],
                    clip_factor=self.weight_clip)
                W_cal_t = dr["w_eval"]                      # clipped cal weights
                W_dep_t = dr["weight_fn"](Xc_test[deploy_idx, t, :])  # raw
                for local in range(deploy_idx.size):
                    eta[local, t] = weighted_quantile_with_inf(
                        eps, W_cal_t, float(W_dep_t[local]), level)

            for local, j in enumerate(deploy_idx):
                band = aci_bands[local]                     # (horizon, 2, ndim)
                e = eta[local]                              # (horizon,)
                inf_mask = ~np.isfinite(e)
                low = band[:, 0, :] - e[:, None]
                high = band[:, 1, :] + e[:, None]
                if inf_mask.any():
                    low[inf_mask, :] = -np.inf
                    high[inf_mask, :] = np.inf
                    self.n_inf_ += 1                  # series with ≥1 δ_∞ step
                if y_trim is not None:
                    low = np.maximum(low, y_trim[0])
                    high = np.minimum(high, y_trim[1])
                out[j, :, 0, :] = low
                out[j, :, 1, :] = high

        return out
