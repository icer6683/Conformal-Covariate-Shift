import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False


class AdaptedCAFHT:
    def __init__(self, alpha=0.1, logistic_kwargs=None):
        self.alpha = alpha
        self.logistic_kwargs = logistic_kwargs or {}
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

    def reset_adaptation(self):
        self._scores = None
        self._weights = None
        self._q = None
        self._is_shifted_ctx = False
        self._train_feat_t = None
        self._test_feat_t = None
        self._t_ctx = None
        self._clf = None

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

    def calibrate(self, cal_Y_subset):
        n_cal, L, _ = cal_Y_subset.shape
        if L < 2:
            self._scores = np.array([])
            self._weights = np.array([])
            self._q = None
            return
        Y = cal_Y_subset[..., 0]
        y_prev = Y[:, -2]
        y_true = Y[:, -1]
        y_pred = self.ar_intercept + self.ar_coef * y_prev
        scores = np.abs(y_true - y_pred)
        if (not self._is_shifted_ctx) or (self._t_ctx is None) or (self._t_ctx <= 0):
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