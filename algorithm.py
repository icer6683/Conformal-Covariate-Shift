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
        self.covar_coef = 0.0
        self.noise_std = 1.0
        self._scores = None
        self._weights = None
        self._q = None
        self._is_shifted_ctx = False
        self._train_feat_t = None
        self._test_feat_t = None
        self._t_ctx = None
        self._clf = None
        # Stability controls
        self._std_epsilon = 1e-8          # feature standardization epsilon
        self._dr_temperature = 1.5        # >1 softens odds to avoid extreme weights
        self._ratio_floor = 1e-6          # min ratio after odds
        self._ratio_ceiling = 1e6         # max ratio after odds
        self._ess_tau = 20.0              # ESS threshold for weight mixing
        self._mix_base = 0.8              # upper bound on mixing toward learned weights

    def reset_adaptation(self):
        self._scores = None
        self._weights = None
        self._q = None
        self._is_shifted_ctx = False
        self._train_feat_t = None
        self._test_feat_t = None
        self._t_ctx = None
        self._clf = None

    def fit_ar_model(self, Y_subset, X_subset=None):
        # Y_subset: (n_series, L, 1), with transitions from 0..L-2 -> 1..L-1
        n, L, _ = Y_subset.shape
        if L < 2:
            # Not enough to fit; keep previous params
            return
        Y = Y_subset[..., 0]
        Y_prev = Y[:, :-1].reshape(-1)
        Y_next = Y[:, 1:].reshape(-1)

        ones = np.ones_like(Y_prev)
        cols = [ones, Y_prev]

        if X_subset is not None:
            X_arr = np.asarray(X_subset)
            if X_arr.ndim == 3 and X_arr.shape[2] == 1:
                X_arr = X_arr[:, :, 0]
            X_prev = X_arr[:, :-1].reshape(-1)
            cols.append(X_prev)
        X_design = np.column_stack(cols)  # [1, y_t] or [1, y_t, x_t]

        theta, *_ = np.linalg.lstsq(X_design, Y_next, rcond=None)
        self.ar_intercept = float(theta[0])
        self.ar_coef = float(theta[1])
        self.covar_coef = float(theta[2]) if len(theta) > 2 else 0.0

        y_hat = X_design @ theta
        resid = Y_next - y_hat
        self.noise_std = float(np.std(resid, ddof=1)) if resid.size > 1 else float(np.abs(resid).mean() + 1e-6)

    def update_weighting_context(self, train_prefixes, test_prefixes, is_shifted: bool, train_X_prefixes=None, test_X_prefixes=None):
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
        self._train_feat_t = self._featurize_prefixes(train_prefixes, train_X_prefixes)
        self._test_feat_t = self._featurize_prefixes(test_prefixes, test_X_prefixes)
        self._clf = None

    def calibrate(self, cal_Y_subset, cal_X_subset=None):
        # cal_Y_subset: (n_cal, t+2, 1); predict last step using previous value (and X if provided)
        n_cal, L, _ = cal_Y_subset.shape
        if L < 2:
            self._scores = np.array([])
            self._weights = np.array([])
            self._q = 2.0 * self.noise_std
            return

        Y = cal_Y_subset[..., 0]
        y_prev = Y[:, -2]
        y_true = Y[:, -1]
        if cal_X_subset is not None:
            X_arr = np.asarray(cal_X_subset)
            if X_arr.ndim == 3 and X_arr.shape[2] == 1:
                X_arr = X_arr[:, :, 0]
            x_prev = X_arr[:, -2]
        else:
            x_prev = np.zeros_like(y_prev)

        y_pred = self.ar_intercept + self.ar_coef * y_prev + self.covar_coef * x_prev
        scores = np.abs(y_true - y_pred)

        # If calibration is tiny, fall back to uniform weights
        if scores.size < 5:
            weights = np.ones_like(scores, dtype=float)
        else:
            if (not self._is_shifted_ctx) or (self._t_ctx is None) or (self._t_ctx <= 0):
                weights = np.ones_like(scores, dtype=float)
            else:
                cal_prefixes_Y = cal_Y_subset[:, : -1, :]
                cal_prefixes_X = None
                if cal_X_subset is not None:
                    cal_prefixes_X = cal_X_subset[:, : -1]
                calX = self._featurize_prefixes(cal_prefixes_Y, cal_prefixes_X)

                weights = self._compute_density_ratio_weights(
                    trainX=self._train_feat_t,
                    testX=self._test_feat_t,
                    evalX=calX,
                )

        self._scores = np.asarray(scores, dtype=float)
        self._weights = np.asarray(weights, dtype=float)
        self._q = self._weighted_quantile(self._scores, self._weights, 1.0 - self.alpha)

    def predict_with_interval(self, input_series, input_x_series=None):
        # input_series: (<=t+1, 1), predict next value; input_x_series aligned (<=t+1,) or (<=t+1,1)
        if input_series.ndim == 2 and input_series.shape[1] == 1:
            last_y = float(input_series[-1, 0])
        elif input_series.ndim == 1:
            last_y = float(input_series[-1])
        else:
            last_y = float(np.ravel(input_series)[-1])

        last_x = 0.0
        if input_x_series is not None:
            x_arr = np.asarray(input_x_series)
            if x_arr.ndim == 2 and x_arr.shape[1] == 1:
                x_arr = x_arr[:, 0]
            last_x = float(x_arr[-1])

        pred = self.ar_intercept + self.ar_coef * last_y + self.covar_coef * last_x
        q = self._q if self._q is not None else 2.0 * self.noise_std
        lower = pred - q
        upper = pred + q
        return float(pred), float(lower), float(upper)

    def _featurize_prefixes(self, prefixes, x_prefixes=None):
        # prefixes: (n, t+1, 1); x_prefixes: (n, t+1) or (n, t+1, 1)
        Y = prefixes[..., 0]
        y_t = Y[:, -1]
        if x_prefixes is None:
            return y_t.reshape(-1, 1)
        X = np.asarray(x_prefixes)
        if X.ndim == 3 and X.shape[2] == 1:
            X = X[:, :, 0]
        x_t = X[:, -1]
        return np.column_stack([y_t, x_t])

    def _compute_density_ratio_weights(self, trainX, testX, evalX):
        if trainX is None or testX is None or trainX.size == 0 or testX.size == 0:
            return np.ones(evalX.shape[0], dtype=float)

        # Always reset classifier before training
        self._clf = None

        N0 = trainX.shape[0]
        N1 = testX.shape[0]

        # --- Standardize features for stability (fit on combined train+test) ---
        XY = np.vstack([trainX, testX])
        mean = XY.mean(axis=0, keepdims=True)
        std = XY.std(axis=0, keepdims=True)
        std = np.where(std < self._std_epsilon, 1.0, std)
        trainZ = (trainX - mean) / std
        testZ = (testX - mean) / std
        evalZ = (evalX - mean) / std

        X = np.vstack([trainZ, testZ])
        y = np.concatenate([np.zeros(N0, dtype=int), np.ones(N1, dtype=int)])

        # --- Train logistic (sklearn if available; otherwise GD) ---
        if _SKLEARN_AVAILABLE:
            kw = dict(max_iter=10000, solver="lbfgs", warm_start=True, C=0.5, class_weight=None)
            user_kw = dict(self.logistic_kwargs) if self.logistic_kwargs is not None else {}
            user_kw.pop("warm_start", None)
            kw.update(user_kw)
            self._clf = LogisticRegression(**kw)
            self._clf.fit(X, y)
            prob1 = self._clf.predict_proba(evalZ)[:, 1]
        else:
            w = self._logreg_fit_gd(X, y)
            z = evalZ @ w[1:] + w[0]
            prob1 = 1.0 / (1.0 + np.exp(-z))

        # --- Convert to odds with temperature to soften extremes ---
        eps = 1e-4
        p = np.clip(prob1, eps, 1.0 - eps)
        log_odds = np.log(p) - np.log(1.0 - p)
        log_odds /= max(1.0, float(self._dr_temperature))
        odds = np.exp(np.clip(log_odds, np.log(self._ratio_floor), np.log(self._ratio_ceiling)))

        # Prior correction
        prior_ratio = (N1 / max(N0, 1))
        ratio = prior_ratio * odds

        # --- Normalize to a probability vector ---
        ratio = np.clip(ratio, self._ratio_floor, self._ratio_ceiling)
        s = np.sum(ratio)
        if not np.isfinite(s) or s <= 0:
            return np.ones(evalX.shape[0], dtype=float)
        w_norm = ratio / s

        # --- Effective sample size & adaptive mixing with uniform to stabilize ---
        ess = 1.0 / np.sum(w_norm ** 2)
        lam = min(self._mix_base, float(ess) / (float(ess) + self._ess_tau))
        u = np.full_like(w_norm, 1.0 / w_norm.size)
        w_mixed = lam * w_norm + (1.0 - lam) * u

        # Renormalize (for numerical cleanliness)
        w_mixed /= np.sum(w_mixed)
        return w_mixed

    def _logreg_fit_gd(self, X, y, lr=0.05, iters=1000, l2=1e-4, max_grad_norm=10.0):
        # Simple logistic regression with bias term via gradient descent (fallback)
        n, d = X.shape
        w = np.zeros(d + 1)
        Xb = np.column_stack([np.ones(n), X])
        for _ in range(iters):
            z = Xb @ w
            # Clip logits to avoid overflow
            z = np.clip(z, -20.0, 20.0)
            p = 1.0 / (1.0 + np.exp(-z))
            grad = Xb.T @ (p - y) / n + l2 * np.r_[0.0, w[1:]]
            # Gradient clipping for stability
            gnorm = np.linalg.norm(grad)
            if np.isfinite(gnorm) and gnorm > max_grad_norm:
                grad = grad * (max_grad_norm / gnorm)
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

        # Find left-continuous quantile (smallest v with CDF >= q)
        idx = np.searchsorted(cdf, q, side="left")
        if idx <= 0:
            return float(v_sorted[0])
        elif idx >= len(v_sorted):
            return float(v_sorted[-1])
        else:
            return float(v_sorted[idx])