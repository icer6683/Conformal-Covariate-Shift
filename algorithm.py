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
        # Y_subset: (n_series, L, 1), with transitions from 0..L-2 -> 1..L-1
        n, L, _ = Y_subset.shape
        if L < 2:
            # Not enough to fit; keep previous params
            return
        x_list = []
        y_list = []
        Y = Y_subset[..., 0]
        for t in range(L - 1):
            x_list.append(Y[:, t])       # y_t
            y_list.append(Y[:, t + 1])   # y_{t+1}
        X = np.concatenate([x[:, None] for x in x_list], axis=0)
        y = np.concatenate([y for y in y_list], axis=0)
        X_design = np.column_stack([np.ones_like(X), X])  # [1, y_t]

        theta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        self.ar_intercept = float(theta[0])
        self.ar_coef = float(theta[1])

        y_hat = X_design @ theta
        resid = y - y_hat
        self.noise_std = float(np.std(resid, ddof=1)) if resid.size > 1 else float(np.abs(resid).mean() + 1e-6)

    def update_weighting_context(self, train_prefixes, test_prefixes, is_shifted: bool):
        # train_prefixes: (n_train, t+1, 1), test_prefixes: (n_half, t+1, 1)
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
        self._train_feat_t = self._featurize_prefixes(train_prefixes)
        self._test_feat_t = self._featurize_prefixes(test_prefixes)
        self._clf = None

    def calibrate(self, cal_Y_subset):
        # cal_Y_subset: (n_cal, t+2, 1); predict last step using previous value
        n_cal, L, _ = cal_Y_subset.shape
        if L < 2:
            self._scores = np.array([])
            self._weights = np.array([])
            self._q = 2.0 * self.noise_std
            return

        Y = cal_Y_subset[..., 0]
        y_prev = Y[:, -2]
        y_true = Y[:, -1]
        y_pred = self.ar_intercept + self.ar_coef * y_prev
        scores = np.abs(y_true - y_pred)

        if (not self._is_shifted_ctx) or (self._t_ctx is None) or (self._t_ctx <= 0):
            weights = np.ones_like(scores, dtype=float)
        else:
            cal_prefixes = cal_Y_subset[:, : -1, :]
            calX = self._featurize_prefixes(cal_prefixes)

            weights = self._compute_density_ratio_weights(
                trainX=self._train_feat_t,
                testX=self._test_feat_t,
                evalX=calX,
            )

        self._scores = np.asarray(scores, dtype=float)
        self._weights = np.asarray(weights, dtype=float)
        self._q = self._weighted_quantile(self._scores, self._weights, 1.0 - self.alpha)

    def predict_with_interval(self, input_series):
        # input_series: (<=t+1, 1), predict next value
        if input_series.ndim == 2 and input_series.shape[1] == 1:
            last_y = float(input_series[-1, 0])
        elif input_series.ndim == 1:
            last_y = float(input_series[-1])
        else:
            last_y = float(np.ravel(input_series)[-1])

        pred = self.ar_intercept + self.ar_coef * last_y
        q = self._q if self._q is not None else 2.0 * self.noise_std
        lower = pred - q
        upper = pred + q
        return float(pred), float(lower), float(upper)

    def _featurize_prefixes(self, prefixes):
        # prefixes: (n, t+1, 1)
        Y = prefixes[..., 0]
        y_t = Y[:, -1]
        # Use only the last time step's value as the classifier feature
        return y_t.reshape(-1, 1)

    def _compute_density_ratio_weights(self, trainX, testX, evalX):
        if trainX is None or testX is None or trainX.size == 0 or testX.size == 0:
            return np.ones(evalX.shape[0], dtype=float)

        # Always reset classifier before training
        self._clf = None

        N0 = trainX.shape[0]
        N1 = testX.shape[0]
        X = np.vstack([trainX, testX])
        y = np.concatenate([np.zeros(N0, dtype=int), np.ones(N1, dtype=int)])

        if _SKLEARN_AVAILABLE:
            kw = dict(max_iter=1000, solver="lbfgs", warm_start=False)
            # user kwargs may include random_state, C, etc., but we ensure warm_start=False for reset
            user_kw = dict(self.logistic_kwargs) if self.logistic_kwargs is not None else {}
            if "warm_start" in user_kw:
                user_kw.pop("warm_start")
            kw.update(user_kw)
            self._clf = LogisticRegression(**kw)
            self._clf.fit(X, y)
            prob1 = self._clf.predict_proba(evalX)[:, 1]
        else:
            w = self._logreg_fit_gd(X, y)
            z = evalX @ w[1:] + w[0]
            prob1 = 1.0 / (1.0 + np.exp(-z))

        eps = 1e-6
        prob1 = np.clip(prob1, eps, 1.0 - eps)
        prior_ratio = (N1 / max(N0, 1))
        ratio = prior_ratio * (prob1 / (1.0 - prob1))
        ratio = np.maximum(ratio, eps)
        s = np.sum(ratio)
        if not np.isfinite(s) or s <= 0:
            return np.ones_like(ratio)
        return ratio / s

    def _logreg_fit_gd(self, X, y, lr=0.1, iters=500, l2=1e-4):
        # Simple logistic regression with bias term via gradient descent (fallback)
        n, d = X.shape
        w = np.zeros(d + 1)
        Xb = np.column_stack([np.ones(n), X])
        for _ in range(iters):
            z = Xb @ w
            p = 1.0 / (1.0 + np.exp(-z))
            grad = Xb.T @ (p - y) / n + l2 * np.r_[0.0, w[1:]]
            w -= lr * grad
        return w

    def _weighted_quantile(self, values, weights, q):
        v = np.asarray(values, dtype=float)
        w = np.asarray(weights, dtype=float)

        # Clean NaN or invalid entries
        mask = np.isfinite(v) & np.isfinite(w)
        v = v[mask]
        w = w[mask]

        # Handle empty case
        if v.size == 0:
            return 2.0 * self.noise_std

        # Ensure non-negative weights
        w = np.clip(w, 0.0, np.inf)
        total_w = np.sum(w)
        if total_w <= 0 or not np.isfinite(total_w):
            # fallback to unweighted quantile
            try:
                return float(np.quantile(v, q))
            except Exception:
                return float(np.mean(v)) if v.size else 2.0 * self.noise_std

        # Normalize weights
        w /= total_w

        # Sort by values
        sorter = np.argsort(v)
        v_sorted = v[sorter]
        w_sorted = w[sorter]

        # Compute CDF
        cdf = np.cumsum(w_sorted)
        cdf[-1] = 1.0  # ensure proper normalization
        q = np.clip(q, 0.0, 1.0)

        # Find quantile by interpolation
        idx = np.searchsorted(cdf, q, side="left")
        if idx == 0:
            return float(v_sorted[0])
        elif idx >= len(v_sorted):
            return float(v_sorted[-1])
        else:
            # Linear interpolation between adjacent points
            v1, v2 = v_sorted[idx - 1], v_sorted[idx]
            c1, c2 = cdf[idx - 1], cdf[idx]
            if c2 == c1:
                return float(v1)
            frac = (q - c1) / (c2 - c1)
            return float(v1 + frac * (v2 - v1))