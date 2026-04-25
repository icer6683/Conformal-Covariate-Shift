import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False


class AdaptedCAFHT:
    def __init__(self, alpha=0.1, logistic_kwargs=None,
                 weight_mode="estimated",
                 lambda_source=None, lambda_target=None,
                 dynamic_source_params=None, dynamic_target_params=None):
        """
        weight_mode:
            "estimated"      — fit a logistic classifier on Y (and optionally X)
                               summary features to estimate the LR (default).
            "oracle_poisson" — closed-form Poisson likelihood ratio for static-X
                               (uses lambda_source / lambda_target).
            "oracle_dynamic" — closed-form prefix likelihood ratio for the AR(1)
                               Gaussian dynamic-X DGP. Source params:
                               {x_rate, x_trend, x_noise_std, x0_lambda}; target
                               params: {x_rate_shift, x_trend_shift,
                               x_noise_std_shift, x0_lambda_shift}.
                               Prefix LR is the online-safe version: at outer
                               step t we use only the observed prefix x_{0:t}.
            "uniform"        — uniform weights (sets _is_shifted_ctx=False).
        """
        self.alpha = alpha
        self.logistic_kwargs = logistic_kwargs or {}
        self.weight_mode = weight_mode
        self.lambda_source = lambda_source
        self.lambda_target = lambda_target
        self.dynamic_source_params = dynamic_source_params  # for oracle_dynamic
        self.dynamic_target_params = dynamic_target_params  # for oracle_dynamic
        self.ar_intercept = 0.0
        self.ar_coef = 0.0
        self.noise_std = 1.0
        self._scores = None
        self._weights = None
        self._q = None
        self._is_shifted_ctx = False
        self._train_feat_t = None
        self._test_feat_t = None
        self._t_ctx = None
        self._clf = None
        self._last_cal_prob1 = None   # prob(test class) on cal set, set by _compute_density_ratio_weights
        self._last_ess = None         # effective sample size of last weight vector
        self._cal_X = None            # static covariate values for cal set (oracle path only)
        self._oracle_inf_count = 0    # how often the per-test-point quantile fell back to s_max
        self._cal_log_lr_unnorm = None  # unnormalized log-LR per cal series (oracle_dynamic)
        self._cal_X_paths = None        # full cal X paths (n_cal, T+1) for oracle_dynamic
        self._t_pred = None             # outer-loop t (prefix length used = t+1) for oracle_dynamic

    def reset_adaptation(self):
        self._scores = None
        self._weights = None
        self._q = None
        self._is_shifted_ctx = False
        self._train_feat_t = None
        self._test_feat_t = None
        self._t_ctx = None
        self._clf = None
        self._last_cal_prob1 = None
        self._last_ess = None
        self._cal_X = None
        self._oracle_inf_count = 0
        self._cal_log_lr_unnorm = None
        self._cal_X_paths = None
        self._t_pred = None

    def fit_ar_model(self, Y_subset):
        n, L, _ = Y_subset.shape
        if L < 2:
            return
        Y = Y_subset[..., 0]
        x_list = [Y[:, t] for t in range(L - 1)]
        y_list = [Y[:, t + 1] for t in range(L - 1)]
        X = np.concatenate([x[:, None] for x in x_list], axis=0)
        y = np.concatenate(y_list, axis=0)
        X_design = np.column_stack([np.ones_like(X), X])
        theta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        self.ar_intercept = float(theta[0])
        self.ar_coef = float(theta[1])
        resid = y - X_design @ theta
        self.noise_std = (float(np.std(resid, ddof=1)) if resid.size > 1
                          else float(np.abs(resid).mean() + 1e-6))

    def update_weighting_context(self, train_prefixes, test_prefixes,
                                  is_shifted: bool,
                                  train_X_prefixes=None,
                                  test_X_prefixes=None):
        """
        Update context for density-ratio weighting.

        Args:
            train_prefixes:   (n_train, t+1, 1)        Y history, train tickers
            test_prefixes:    (n_half,  t+1, 1)        Y history, test-ctx tickers
            is_shifted:       bool
            train_X_prefixes: (n_train, t+1, n_cov)   optional X history, train
            test_X_prefixes:  (n_half,  t+1, n_cov)   optional X history, test-ctx

        If X prefixes are provided they are passed alongside Y into
        _featurize_prefixes, which computes summary statistics over both.
        """
        if train_prefixes is None or test_prefixes is None:
            self._is_shifted_ctx = False
            self._train_feat_t = None
            self._test_feat_t = None
            self._t_ctx = None
            self._clf = None
            return

        t = train_prefixes.shape[1] - 1
        self._t_ctx = t
        if (not is_shifted) or t <= 0:
            self._is_shifted_ctx = False
            self._train_feat_t = None
            self._test_feat_t = None
            self._clf = None
            return

        self._is_shifted_ctx = True
        self._train_feat_t = self._featurize_prefixes(
            train_prefixes, train_X_prefixes)
        self._test_feat_t = self._featurize_prefixes(
            test_prefixes, test_X_prefixes)
        self._clf = None

    def calibrate(self, cal_Y_subset, cal_X=None):
        """
        cal_X: per-series covariate data, used only in oracle modes.
               - oracle_poisson: shape (n_cal,), the static Poisson draw.
               - oracle_dynamic: shape (n_cal, T+1), the full X path. Only the
                 prefix x_{0:t} is consumed at each calibrate call where
                 t = cal_Y_subset.shape[1] - 2 (online-safe).
        Ignored when weight_mode is "estimated" or "uniform".
        """
        n_cal, L, _ = cal_Y_subset.shape
        if L < 2:
            self._scores = np.array([])
            self._weights = np.array([])
            self._q = None
            self._last_ess = None
            return
        Y = cal_Y_subset[..., 0]
        y_prev = Y[:, -2]
        y_true = Y[:, -1]
        y_pred = self.ar_intercept + self.ar_coef * y_prev
        scores = np.abs(y_true - y_pred)
        if self.weight_mode == "oracle_poisson" and cal_X is not None:
            self._cal_X = np.asarray(cal_X, dtype=float).copy()
            # Stored normalized cal-only weights are kept for ESS reporting and
            # the legacy predict_with_interval path; the corrected per-test-point
            # quantile (predict_with_interval_oracle) re-normalizes including
            # the test point's own weight.
            weights = self._oracle_poisson_weights(
                self._cal_X,
                lam_src=self.lambda_source,
                lam_tgt=self.lambda_target,
            )
        elif self.weight_mode == "oracle_dynamic" and cal_X is not None:
            # Online-safe prefix LR: at outer step t (predicting Y at index t+1),
            # we have observed x_{0:t} (length t+1). cal_Y_subset.shape[1] = t+2.
            self._cal_X_paths = np.asarray(cal_X, dtype=float).copy()
            t_pred = int(cal_Y_subset.shape[1]) - 2
            self._t_pred = t_pred
            self._cal_log_lr_unnorm = self._oracle_dynamic_log_weights(
                self._cal_X_paths, t_use=t_pred,
                src=self.dynamic_source_params,
                tgt=self.dynamic_target_params,
            )
            # Normalized weights for ESS / legacy path; predict_with_interval_oracle_dynamic
            # re-normalizes per test point.
            log_w = self._cal_log_lr_unnorm - np.max(self._cal_log_lr_unnorm)
            w = np.exp(log_w)
            s = np.sum(w)
            weights = (w / s) if (np.isfinite(s) and s > 0) else np.ones_like(scores)
        elif (not self._is_shifted_ctx) or (self._t_ctx is None) or (self._t_ctx <= 0):
            weights = np.ones_like(scores, dtype=float)
        else:
            cal_prefixes = cal_Y_subset[:, :-1, :]
            calX = self._featurize_prefixes(cal_prefixes)
            weights = self._compute_density_ratio_weights(
                trainX=self._train_feat_t,
                testX=self._test_feat_t,
                evalX=calX,
            )
        self._scores = np.asarray(scores, dtype=float)
        self._weights = np.asarray(weights, dtype=float)
        self._last_ess = self._effective_sample_size(self._weights)
        self._q = None

    def predict_with_interval(self, input_series, alpha_level=None):
        if input_series.ndim == 2 and input_series.shape[1] == 1:
            last_y = float(input_series[-1, 0])
        elif input_series.ndim == 1:
            last_y = float(input_series[-1])
        else:
            last_y = float(np.ravel(input_series)[-1])
        pred = self.ar_intercept + self.ar_coef * last_y
        a = float(np.clip(
            self.alpha if alpha_level is None else float(alpha_level),
            1e-6, 1.0 - 1e-6))
        if self._scores is None or self._weights is None or np.asarray(self._scores).size == 0:
            q = 2.0 * self.noise_std
        else:
            q = self._weighted_quantile(self._scores, self._weights, 1.0 - a)
        return float(pred), float(pred - q), float(pred + q)

    def predict_with_interval_oracle(self, input_series, test_x, alpha_level=None):
        """
        Oracle-Poisson interval that includes the test point's own LR weight in
        the normalization denominator (Tibshirani et al. 2019, weighted split
        conformal). For each test point with covariate value `test_x`, the
        normalized weights are
            w_i = w(Z_i)   / (Σ_j w(Z_j) + w(Z_test))     for i ≤ n
            w_test = w(Z_test) / (...)                    pseudo-point at +∞
        and the (1−α)-quantile is taken over scores [s_1, …, s_n, +∞].

        When the test weight exceeds α·D the ideal quantile is +∞ (unbounded
        interval). We instead return the maximum cal score in that case so that
        downstream width statistics stay finite, and increment _oracle_inf_count
        for diagnostic purposes.
        """
        if input_series.ndim == 2 and input_series.shape[1] == 1:
            last_y = float(input_series[-1, 0])
        elif input_series.ndim == 1:
            last_y = float(input_series[-1])
        else:
            last_y = float(np.ravel(input_series)[-1])
        pred = self.ar_intercept + self.ar_coef * last_y
        a = float(np.clip(
            self.alpha if alpha_level is None else float(alpha_level),
            1e-6, 1.0 - 1e-6))

        if (self._scores is None or np.asarray(self._scores).size == 0
                or self._cal_X is None
                or self.lambda_source is None or self.lambda_target is None):
            q = 2.0 * self.noise_std
            return float(pred), float(pred - q), float(pred + q)

        lam_s = float(self.lambda_source)
        lam_t = float(self.lambda_target)
        log_lr_cal = (lam_s - lam_t) + self._cal_X * np.log(lam_t / lam_s)
        log_lr_test = (lam_s - lam_t) + float(test_x) * np.log(lam_t / lam_s)

        m = max(float(np.max(log_lr_cal)), log_lr_test)
        w_cal = np.exp(log_lr_cal - m)
        w_test = float(np.exp(log_lr_test - m))
        D = float(np.sum(w_cal)) + w_test
        if not np.isfinite(D) or D <= 0:
            q = 2.0 * self.noise_std
            return float(pred), float(pred - q), float(pred + q)
        w_cal = w_cal / D                # cal mass; sum < 1, remainder = w_test/D at +∞

        scores = np.asarray(self._scores, dtype=float)
        sorter = np.argsort(scores)
        s_sorted = scores[sorter]
        w_sorted = w_cal[sorter]
        cdf = np.cumsum(w_sorted)        # cdf[-1] = 1 - w_test/D < 1

        q_target = 1.0 - a
        if cdf[-1] >= q_target:
            idx = int(np.searchsorted(cdf, q_target, side="left"))
            idx = min(idx, len(s_sorted) - 1)
            if idx == 0:
                q_val = float(s_sorted[0])
            else:
                v1, v2 = s_sorted[idx - 1], s_sorted[idx]
                c1, c2 = cdf[idx - 1], cdf[idx]
                if c2 == c1:
                    q_val = float(v1)
                else:
                    q_val = float(v1 + (q_target - c1) / (c2 - c1) * (v2 - v1))
        else:
            self._oracle_inf_count += 1
            q_val = float(s_sorted[-1])  # finite stand-in for +∞

        return float(pred), float(pred - q_val), float(pred + q_val)

    def predict_with_interval_oracle_dynamic(self, input_series, test_x_path,
                                             alpha_level=None):
        """
        Oracle dynamic-X interval. Uses the prefix log-LR for the test point
        (online-safe, prefix length = self._t_pred + 1) and includes that test
        weight in the normalization denominator (Tibshirani 2019).

        Args:
            input_series : (t+1, 1) Y prefix.
            test_x_path  : (T+1,)   full X path for this test series; only the
                                    prefix x_{0:self._t_pred} is consumed.
        """
        if input_series.ndim == 2 and input_series.shape[1] == 1:
            last_y = float(input_series[-1, 0])
        elif input_series.ndim == 1:
            last_y = float(input_series[-1])
        else:
            last_y = float(np.ravel(input_series)[-1])
        pred = self.ar_intercept + self.ar_coef * last_y
        a = float(np.clip(
            self.alpha if alpha_level is None else float(alpha_level),
            1e-6, 1.0 - 1e-6))

        if (self._scores is None or np.asarray(self._scores).size == 0
                or self._cal_log_lr_unnorm is None
                or self._t_pred is None
                or self.dynamic_source_params is None
                or self.dynamic_target_params is None):
            q = 2.0 * self.noise_std
            return float(pred), float(pred - q), float(pred + q)

        test_x = np.asarray(test_x_path, dtype=float).reshape(1, -1)
        log_lr_test = float(self._oracle_dynamic_log_weights(
            test_x, t_use=self._t_pred,
            src=self.dynamic_source_params,
            tgt=self.dynamic_target_params,
        )[0])

        log_lr_cal = self._cal_log_lr_unnorm
        m = max(float(np.max(log_lr_cal)), log_lr_test)
        if not np.isfinite(m):
            q = 2.0 * self.noise_std
            return float(pred), float(pred - q), float(pred + q)
        w_cal = np.exp(log_lr_cal - m)
        w_test = float(np.exp(log_lr_test - m))
        D = float(np.sum(w_cal)) + w_test
        if not np.isfinite(D) or D <= 0:
            q = 2.0 * self.noise_std
            return float(pred), float(pred - q), float(pred + q)
        w_cal = w_cal / D

        scores = np.asarray(self._scores, dtype=float)
        sorter = np.argsort(scores)
        s_sorted = scores[sorter]
        w_sorted = w_cal[sorter]
        cdf = np.cumsum(w_sorted)

        q_target = 1.0 - a
        if cdf[-1] >= q_target:
            idx = int(np.searchsorted(cdf, q_target, side="left"))
            idx = min(idx, len(s_sorted) - 1)
            if idx == 0:
                q_val = float(s_sorted[0])
            else:
                v1, v2 = s_sorted[idx - 1], s_sorted[idx]
                c1, c2 = cdf[idx - 1], cdf[idx]
                q_val = float(v1) if c2 == c1 else float(
                    v1 + (q_target - c1) / (c2 - c1) * (v2 - v1))
        else:
            self._oracle_inf_count += 1
            q_val = float(s_sorted[-1])

        return float(pred), float(pred - q_val), float(pred + q_val)

    def _featurize_prefixes(self, Y_prefixes, X_prefixes=None):
        """
        Default featurizer — uses only the last time step's value of Y.
        Monkey-patched in finance_conformal.py with the richer Y+X summary
        version when with_shift=True.

        Y_prefixes: (n, t+1, 1)
        X_prefixes: (n, t+1, n_cov)  optional, ignored in default version
        Returns:    (n, 1)
        """
        Y = Y_prefixes[..., 0]
        return Y[:, -1].reshape(-1, 1)

    def _compute_density_ratio_weights(self, trainX, testX, evalX):
        if trainX is None or testX is None or trainX.size == 0 or testX.size == 0:
            return np.ones(evalX.shape[0], dtype=float)
        self._clf = None
        N0 = trainX.shape[0]
        N1 = testX.shape[0]
        X  = np.vstack([trainX, testX])
        y  = np.concatenate([np.zeros(N0, dtype=int), np.ones(N1, dtype=int)])

        if _SKLEARN_AVAILABLE:
            kw = dict(max_iter=1000, solver="lbfgs",
                      warm_start=False, class_weight="balanced")
            user_kw = dict(self.logistic_kwargs) if self.logistic_kwargs else {}
            user_kw.pop("warm_start", None)
            kw.update(user_kw)
            self._clf = LogisticRegression(**kw)
            self._clf.fit(X, y)
            prob1 = self._clf.predict_proba(evalX)[:, 1]
        else:
            w = self._logreg_fit_gd(X, y)
            z = evalX @ w[1:] + w[0]
            prob1 = 1.0 / (1.0 + np.exp(-z))

        eps   = 1e-6
        prob1 = np.clip(prob1, eps, 1.0 - eps)
        self._last_cal_prob1 = prob1.copy()   # expose for diagnostics

        # class_weight='balanced' removes the prior from prob1, so the ratio
        # prob1/(1-prob1) is already a pure likelihood ratio — no prior_ratio
        # multiplication needed.
        ratio = prob1 / (1.0 - prob1)
        ratio = np.maximum(ratio, eps)

        # Clip at 5x mean to prevent weight degeneracy
        ratio = np.minimum(ratio, 5.0 * np.mean(ratio))

        s = np.sum(ratio)
        if not np.isfinite(s) or s <= 0:
            return np.ones(evalX.shape[0], dtype=float)
        return ratio / s

    @staticmethod
    def _oracle_poisson_weights(x_values, lam_src, lam_tgt):
        """
        Closed-form likelihood ratio for Poisson source vs Poisson target:
            w(x) = P_target(X=x) / P_source(X=x)
                 = exp(lam_src - lam_tgt) * (lam_tgt / lam_src) ** x
        Computed in log-space, max-centered for numerical stability, then
        exponentiated and normalized to sum to 1. Returns ones if params
        are missing or degenerate.
        """
        x = np.asarray(x_values, dtype=float)
        if lam_src is None or lam_tgt is None:
            return np.ones_like(x)
        lam_s = float(lam_src)
        lam_t = float(lam_tgt)
        if not (lam_s > 0 and lam_t > 0):
            return np.ones_like(x)
        log_w = (lam_s - lam_t) + x * np.log(lam_t / lam_s)
        log_w = log_w - np.max(log_w)
        w = np.exp(log_w)
        s = np.sum(w)
        if not np.isfinite(s) or s <= 0:
            return np.ones_like(x)
        return w / s

    @staticmethod
    def _oracle_dynamic_log_weights(X_paths, t_use, src, tgt):
        """
        Online-safe prefix log-likelihood-ratio for the AR(1) Gaussian dynamic-X
        DGP. For each row x_{0:t_use} of X_paths returns

            log w_t = log p̃_0(x_0) − log p_0(x_0)
                      + Σ_{s=0}^{t_use-1} [
                            log N(x_{s+1}; ρ̃·x_s + κ̃·s, σ̃²)
                          − log N(x_{s+1}; ρ·x_s + κ·s, σ²) ]

        Initial term: source X_0 ~ Pois(λ), target X_0 ~ Pois(λ̃).
        log Pois(k; λ̃) − log Pois(k; λ) = (λ − λ̃) + k · log(λ̃/λ).

        Args:
            X_paths : (n, T+1) realized covariate paths (cal or test).
            t_use   : prefix length minus one (we use x_{0:t_use}; t_use=0 means
                      only x_0 is observed).
            src,tgt : dicts with keys x_rate, x_trend, x_noise_std, x0_lambda
                      (and the *_shift variants on tgt).
        Returns: (n,) unnormalized log weights. Subtract max before exp().
        """
        X = np.asarray(X_paths, dtype=float)
        n, _ = X.shape
        log_w = np.zeros(n, dtype=float)

        # Initial Poisson PMF ratio at X_0
        lam_s = float(src.get("x0_lambda"))
        lam_t = float(tgt.get("x0_lambda_shift", tgt.get("x0_lambda")))
        if lam_s > 0 and lam_t > 0:
            x0 = X[:, 0]
            log_w += (lam_s - lam_t) + x0 * np.log(lam_t / lam_s)
        # else: degenerate Poisson params → leave initial term at 0 (treat as ratio 1)

        rho_s = float(src["x_rate"])
        rho_t = float(tgt.get("x_rate_shift", src["x_rate"]))
        kap_s = float(src.get("x_trend", 0.0))
        kap_t = float(tgt.get("x_trend_shift", kap_s))
        sig_s = float(src["x_noise_std"])
        sig_t = float(tgt.get("x_noise_std_shift", sig_s))
        if sig_s <= 0 or sig_t <= 0:
            # degenerate → return what we have so far (initial term only)
            return log_w

        # Sum log-LR over transitions s = 0..t_use-1 :  x_{s+1} | x_s
        for s in range(int(t_use)):
            xs   = X[:, s]
            xs1  = X[:, s + 1]
            mu_s = rho_s * xs + kap_s * s
            mu_t = rho_t * xs + kap_t * s
            # log N(x; μ, σ) = -0.5*log(2πσ²) - (x-μ)²/(2σ²)
            log_w += (
                -np.log(sig_t) - 0.5 * ((xs1 - mu_t) / sig_t) ** 2
                + np.log(sig_s) + 0.5 * ((xs1 - mu_s) / sig_s) ** 2
            )

        # Guard against nan/inf
        log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
        return log_w

    @staticmethod
    def _effective_sample_size(weights):
        w = np.asarray(weights, dtype=float)
        w = w[np.isfinite(w) & (w >= 0)]
        if w.size == 0:
            return 0.0
        s1 = float(np.sum(w))
        s2 = float(np.sum(w * w))
        if s2 <= 0:
            return 0.0
        return (s1 * s1) / s2

    def _logreg_fit_gd(self, X, y, lr=0.1, iters=500, l2=1e-4):
        n, d = X.shape
        w  = np.zeros(d + 1)
        Xb = np.column_stack([np.ones(n), X])
        for _ in range(iters):
            z    = Xb @ w
            p    = 1.0 / (1.0 + np.exp(-z))
            grad = Xb.T @ (p - y) / n + l2 * np.r_[0.0, w[1:]]
            w   -= lr * grad
        return w

    def _weighted_quantile(self, values, weights, q):
        v = np.asarray(values, dtype=float)
        w = np.asarray(weights, dtype=float)
        mask = np.isfinite(v) & np.isfinite(w)
        v, w = v[mask], w[mask]
        if v.size == 0:
            return 2.0 * self.noise_std
        w = np.clip(w, 0.0, np.inf)
        total_w = np.sum(w)
        if total_w <= 0 or not np.isfinite(total_w):
            try:
                return float(np.quantile(v, q))
            except Exception:
                return float(np.mean(v)) if v.size else 2.0 * self.noise_std
        w /= total_w
        sorter   = np.argsort(v)
        v_sorted = v[sorter]
        w_sorted = w[sorter]
        cdf      = np.cumsum(w_sorted)
        cdf[-1]  = 1.0
        q        = float(np.clip(q, 0.0, 1.0))
        idx = np.searchsorted(cdf, q, side="left")
        if idx == 0:
            return float(v_sorted[0])
        if idx >= len(v_sorted):
            return float(v_sorted[-1])
        v1, v2 = v_sorted[idx - 1], v_sorted[idx]
        c1, c2 = cdf[idx - 1], cdf[idx]
        if c2 == c1:
            return float(v1)
        return float(v1 + (q - c1) / (c2 - c1) * (v2 - v1))