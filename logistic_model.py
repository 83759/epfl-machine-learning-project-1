"""NumPy logistic regression and survey feature encoding used by run.ipynb."""

import time

import numpy as np


class SurveyFeatures:
    """Fit the encoding on training rows, then reuse it for unseen rows.

    Survey response codes remain separate categories. Continuous features are
    median-imputed and clipped to training quantiles. Optional bins or linear
    splines add nonlinear terms. All statistics come from the fitting rows.
    """

    DROP = {"SEQNO", "_PSU", "IDATE", "IDAY", "IYEAR", "_STSTR"}
    CATEGORICAL = {"_STATE", "EXRACT11", "EXRACT21", "IMONTH", "FMONTH"}
    HEALTH_DAYS = {"PHYSHLTH", "MENTHLTH", "POORHLTH"}
    INTERACTION_COLUMNS = {
        "_AGE80",
        "SEX",
        "GENHLTH",
        "BPHIGH4",
        "TOLDHI2",
        "CVDSTRK3",
        "CHCCOPD1",
        "DIABETE3",
        "_SMOKER3",
        "DIFFWALK",
        "_BMI5CAT",
        "HAREHAB1",
        "CHCKIDNY",
    }

    def __init__(
        self,
        mode="linear",
        interactions=False,
        categorical_columns=(),
        n_knots=10,
        nonlinear_columns=None,
        nonlinear_interactions=False,
        clean_health_days=False,
    ):
        if mode not in {"linear", "bins", "spline", "rank"}:
            raise ValueError("Unknown feature encoding")
        if n_knots < 1:
            raise ValueError("n_knots must be positive")
        self.mode = mode
        self.interactions = interactions
        self.categorical_columns = self.CATEGORICAL | set(categorical_columns)
        self.n_knots = n_knots
        self.nonlinear_columns = (
            None if nonlinear_columns is None else set(nonlinear_columns)
        )
        self.nonlinear_interactions = nonlinear_interactions
        self.clean_health_days = clean_health_days

    def fit_transform(self, x, columns):
        self.columns = tuple(columns)
        self.specs = []
        self.feature_names = []
        self.pairs = []
        self.means = None
        self.scales = None
        if x.shape[1] != len(columns):
            raise ValueError("Feature names do not match the input matrix")

        def add(j, kind, parameters, name):
            self.specs.append((j, kind, parameters))
            self.feature_names.append(name)

        for j, name in enumerate(columns):
            if name in self.DROP:
                continue
            column = self._column(x, j)
            observed = column[np.isfinite(column)]
            if len(observed) < 100:
                continue
            values, counts = np.unique(observed, return_counts=True)
            if len(values) <= 1:
                continue
            numeric_days = self.clean_health_days and name in self.HEALTH_DAYS
            if not numeric_days and (
                len(values) <= 32 or name in self.categorical_columns
            ):
                for value in values[counts >= 20]:
                    add(j, "eq", float(value), f"{name}={value:g}")
            else:
                low, median, high = np.quantile(observed, [0.005, 0.5, 0.995])
                span = max(high - low, 1e-6)
                if self.mode == "rank":
                    quantiles = np.linspace(0, 1, 101)
                    edges = np.quantile(observed, quantiles)
                    knots, first, knot_counts = np.unique(
                        edges, return_index=True, return_counts=True
                    )
                    ranks = (quantiles[first] + quantiles[first + knot_counts - 1]) / 2
                    add(j, "rank", (knots, ranks, median), f"{name}_rank")
                else:
                    add(j, "linear", (low, median, high, span), name)
                expand = (
                    self.nonlinear_columns is None or name in self.nonlinear_columns
                )
                if self.mode in {"bins", "spline"} and expand:
                    knots = np.unique(
                        np.quantile(observed, np.linspace(0.05, 0.95, self.n_knots))
                    )
                    kind = "gt" if self.mode == "bins" else "hinge"
                    for knot in knots:
                        add(j, kind, (knot, span, high), f"{name}>{knot:g}")
                if self.mode == "rank" or (self.mode in {"bins", "spline"} and expand):
                    for value in values[counts >= 0.02 * len(observed)]:
                        add(j, "eq", float(value), f"{name}={value:g}")
            if np.isnan(column).any():
                add(j, "missing", None, f"{name}=missing")

        encoded = self._encode(x)
        self.means = encoded.mean(axis=0, dtype=np.float64).astype(np.float32)
        # Compute one column at a time to avoid a large float64 temporary matrix.
        stds = np.array(
            [encoded[:, j].std(dtype=np.float64) for j in range(encoded.shape[1])]
        )
        self.scales = np.maximum(stds, 0.1).astype(np.float32)
        encoded -= self.means
        encoded /= self.scales

        if self.interactions:
            chosen = [
                i
                for i, (j, kind, _) in enumerate(self.specs)
                if columns[j] in self.INTERACTION_COLUMNS
                and (
                    kind in {"linear", "rank", "eq"}
                    or (self.nonlinear_interactions and kind in {"gt", "hinge"})
                )
                and (kind != "eq" or self.means[i] >= 0.01)
            ]
            self.pairs = [
                (i, j)
                for k, i in enumerate(chosen)
                for j in chosen[k + 1 :]
                if self.specs[i][0] != self.specs[j][0]
            ]
        return self._add_interactions(encoded)

    def fit(self, x, columns):
        self.fit_transform(x, columns)
        return self

    def _encode(self, x):
        encoded = np.empty((len(x), len(self.specs)), dtype=np.float32, order="F")
        cleaned = {
            j: self._column(x, j)
            for j, name in enumerate(self.columns)
            if self.clean_health_days and name in self.HEALTH_DAYS
        }
        for i, (j, kind, parameters) in enumerate(self.specs):
            column = cleaned.get(j, x[:, j])
            if kind == "eq":
                encoded[:, i] = column == parameters
            elif kind == "missing":
                encoded[:, i] = np.isnan(column)
            elif kind == "linear":
                low, median, high, span = parameters
                filled = np.nan_to_num(column, nan=median)
                encoded[:, i] = (np.clip(filled, low, high) - median) / span
            elif kind == "rank":
                knots, ranks, median = parameters
                filled = np.nan_to_num(column, nan=median)
                encoded[:, i] = np.interp(filled, knots, ranks)
            elif kind == "gt":
                encoded[:, i] = column > parameters[0]
            else:
                knot, span, high = parameters
                filled = np.nan_to_num(column, nan=knot)
                encoded[:, i] = np.maximum(np.minimum(filled, high) - knot, 0) / span
        return encoded

    def _column(self, x, j):
        """Decode health-day responses using the 2015 BRFSS codebook.

        https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf
        """
        name = self.columns[j]
        if not self.clean_health_days or name not in self.HEALTH_DAYS:
            return x[:, j]
        raw = x[:, j]
        column = raw.copy()
        column[raw == 88] = 0
        column[(raw == 77) | (raw == 99)] = np.nan
        if name == "POORHLTH" and all(
            field in self.columns for field in ("PHYSHLTH", "MENTHLTH")
        ):
            physical = x[:, self.columns.index("PHYSHLTH")]
            mental = x[:, self.columns.index("MENTHLTH")]
            skipped = np.isnan(raw) & (physical == 88) & (mental == 88)
            column[skipped] = 0
        return column

    def _add_interactions(self, encoded):
        if not self.pairs:
            return encoded
        width = encoded.shape[1]
        result = np.empty(
            (len(encoded), width + len(self.pairs)), dtype=np.float32, order="F"
        )
        result[:, :width] = encoded
        for k, (i, j) in enumerate(self.pairs):
            result[:, width + k] = encoded[:, i] * encoded[:, j]
        return result

    def transform(self, x):
        if x.shape[1] != len(self.columns):
            raise ValueError("Input has a different number of survey columns")
        encoded = self._encode(x)
        encoded -= self.means
        encoded /= self.scales
        return self._add_interactions(encoded)


def sigmoid(scores):
    """Stable sigmoid, including for large positive or negative scores."""
    return np.exp(-np.logaddexp(0.0, -scores))


def fit_logistic(
    x,
    y,
    lambda_=1e-3,
    pos_weight=1.0,
    max_iters=400,
    tolerance=2e-5,
    initial_w=None,
    verbose=False,
):
    """Minimize weighted logistic loss + lambda_ * ||w||^2 using L-BFGS.

    The last coefficient is an unpenalized intercept. Labels must be 0 or 1.
    Backtracking accepts steps only when the penalized objective decreases.
    Float32 matrix products keep memory use practical; optimizer state and
    objective accumulation use float64. No external optimization library is used.
    """
    n, d = x.shape
    y = np.asarray(y, dtype=np.float32)
    if y.shape != (n,) or not np.isin(y, [0, 1]).all() or not 0 < y.mean() < 1:
        raise ValueError("Training needs both classes and a one-dimensional 0/1 target")
    if lambda_ < 0 or pos_weight <= 0 or max_iters < 1:
        raise ValueError("Invalid optimizer parameters")
    sample_weights = np.where(y == 1, pos_weight, 1.0).astype(np.float32)

    def objective(weights):
        scores = x @ weights[:-1].astype(np.float32) + weights[-1]
        losses = sample_weights * (np.logaddexp(0.0, scores) - y * scores)
        loss = float(np.mean(losses, dtype=np.float64))
        loss += lambda_ * np.dot(weights[:-1], weights[:-1])
        errors = sample_weights * (sigmoid(scores) - y) / n
        gradient = np.r_[x.T @ errors.astype(np.float32), errors.sum(dtype=np.float64)]
        gradient[:-1] += 2 * lambda_ * weights[:-1]
        return loss, gradient

    w = (
        np.zeros(d + 1)
        if initial_w is None
        else np.asarray(initial_w, dtype=float).copy()
    )
    if w.shape != (d + 1,):
        raise ValueError("initial_w must include the intercept")
    if initial_w is None:
        w[-1] = np.log(y.mean() * pos_weight / (1 - y.mean()))
    loss, gradient = objective(w)
    history = []
    started = time.perf_counter()
    evaluations = 1
    iterations = 0
    status = "iteration limit"

    for iteration in range(max_iters):
        if np.max(np.abs(gradient)) < tolerance:
            status = "converged"
            break
        q = gradient.copy()
        alphas = []
        # Apply the inverse-Hessian approximation with the two-loop recursion.
        for delta_w, delta_g, rho in reversed(history):
            alpha = rho * np.dot(delta_w, q)
            alphas.append(alpha)
            q -= alpha * delta_g
        scale = 1.0
        if history:
            delta_w, delta_g, _ = history[-1]
            scale = np.dot(delta_w, delta_g) / np.dot(delta_g, delta_g)
        direction = scale * q
        for (delta_w, delta_g, rho), alpha in zip(history, reversed(alphas)):
            direction += delta_w * (alpha - rho * np.dot(delta_g, direction))
        direction = -direction
        slope = np.dot(gradient, direction)
        if slope >= 0:
            direction = -gradient
            slope = -np.dot(gradient, gradient)
            history = []

        step = 1.0
        for _ in range(25):
            candidate = w + step * direction
            new_loss, new_gradient = objective(candidate)
            evaluations += 1
            if np.isfinite(new_loss) and new_loss <= loss + 1e-4 * step * slope + 1e-10:
                break
            step *= 0.5
        else:
            status = "line search stopped"
            break

        delta_w = candidate - w
        delta_g = new_gradient - gradient
        curvature = np.dot(delta_w, delta_g)
        if curvature > 1e-12:
            history.append((delta_w, delta_g, 1.0 / curvature))
            history = history[-12:]
        w, loss, gradient = candidate, new_loss, new_gradient
        iterations = iteration + 1
        if verbose and iterations % 50 == 0:
            print(f"  Iteration {iterations}: objective={loss:.6f}", flush=True)

    gradient_norm = float(np.max(np.abs(gradient)))
    if gradient_norm < tolerance:
        status = "converged"
    scores = x @ w[:-1].astype(np.float32) + w[-1]
    info = {
        "objective": loss,
        "log_loss": float(
            np.mean(np.logaddexp(0.0, scores) - y * scores, dtype=np.float64)
        ),
        "iterations": iterations,
        "gradient_norm": gradient_norm,
        "status": status,
        "seconds": time.perf_counter() - started,
        "evaluations": evaluations,
    }
    return w, info


def predict_probabilities(x, w):
    return sigmoid(x @ w[:-1].astype(np.float32) + w[-1])


def blend_logistic_probabilities(probabilities, weights):
    """Average logistic scores, then map back to probabilities.

    Rows contain component models and columns contain observations. Nonnegative
    weights are normalized to sum to one. Clipping near zero and one keeps
    log-odds finite, including when probabilities equal the endpoints.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] == 0
        or weights.shape != (probabilities.shape[0],)
        or not np.isfinite(probabilities).all()
        or not np.isfinite(weights).all()
        or np.any((probabilities < 0) | (probabilities > 1))
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError("Expected component probabilities and nonnegative weights")
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probabilities, epsilon, 1 - epsilon)
    scores = np.log(clipped) - np.log1p(-clipped)
    normalized = weights / weights.sum()
    return sigmoid(np.sum(normalized[:, None] * scores, axis=0))


def fit_elastic_logistic(
    x, y, l1=3e-4, lambda_=1e-4, max_iters=800, initial_w=None, verbose=False
):
    """Fit logistic regression with L1 and L2 penalties using proximal FISTA.

    Soft thresholding creates exact zero coefficients. Backtracking checks the
    smooth objective's quadratic upper bound; an adaptive momentum restart
    limits oscillation. The intercept is exempt from both penalties.
    """
    n, d = x.shape
    y = np.asarray(y, dtype=np.float32)
    if y.shape != (n,) or not np.isin(y, [0, 1]).all() or not 0 < y.mean() < 1:
        raise ValueError("Training needs both classes and a one-dimensional 0/1 target")
    if l1 < 0 or lambda_ < 0 or max_iters < 1:
        raise ValueError("Invalid optimizer parameters")

    def smooth(weights, gradient=False):
        scores = x @ weights[:-1].astype(np.float32) + weights[-1]
        value = np.mean(np.logaddexp(0.0, scores) - y * scores, dtype=np.float64)
        value += lambda_ * np.dot(weights[:-1], weights[:-1])
        if not gradient:
            return value
        errors = (sigmoid(scores) - y) / n
        grad = np.r_[x.T @ errors.astype(np.float32), errors.sum(dtype=np.float64)]
        grad[:-1] += 2 * lambda_ * weights[:-1]
        return value, grad

    def proximal_step(weights, gradient, step):
        candidate = weights - step * gradient
        candidate[:-1] = np.sign(candidate[:-1]) * np.maximum(
            np.abs(candidate[:-1]) - step * l1, 0.0
        )
        return candidate

    w = (
        np.zeros(d + 1)
        if initial_w is None
        else np.asarray(initial_w, dtype=float).copy()
    )
    if w.shape != (d + 1,):
        raise ValueError("initial_w must include the intercept")
    if initial_w is None:
        w[-1] = np.log(y.mean() / (1 - y.mean()))
    extrapolated = w.copy()
    momentum, step, previous = 1.0, 1.0, np.inf
    started = time.perf_counter()
    status = "iteration limit"
    iterations = 0
    for iteration in range(max_iters):
        value, gradient = smooth(extrapolated, True)
        for _ in range(30):
            candidate = proximal_step(extrapolated, gradient, step)
            delta = candidate - extrapolated
            new_value = smooth(candidate)
            upper = value + np.dot(gradient, delta) + np.dot(delta, delta) / (2 * step)
            if new_value <= upper + 1e-9:
                break
            step *= 0.5
        else:
            status = "line search stopped"
            break
        objective = new_value + l1 * np.abs(candidate[:-1]).sum()
        if np.dot(extrapolated - candidate, candidate - w) > 0:
            next_momentum = 1.0
            extrapolated = candidate.copy()
        else:
            next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
            extrapolated = candidate + (momentum - 1) / next_momentum * (candidate - w)
        w, momentum = candidate, next_momentum
        iterations = iteration + 1
        if verbose and iterations % 100 == 0:
            print(
                f"  Iteration {iterations}: objective={objective:.6f}, "
                f"nonzero={np.count_nonzero(w[:-1])}",
                flush=True,
            )
        if abs(previous - objective) < 1e-9 and iteration > 100:
            status = "objective tolerance"
            break
        previous = objective

    value, gradient = smooth(w, True)
    gradient_norm = float(np.max(np.abs((w - proximal_step(w, gradient, step)) / step)))
    if gradient_norm < 2e-5:
        status = "converged"
    info = {
        "objective": float(value + l1 * np.abs(w[:-1]).sum()),
        "log_loss": float(value - lambda_ * np.dot(w[:-1], w[:-1])),
        "iterations": iterations,
        "gradient_norm": gradient_norm,
        "status": status,
        "nonzero": int(np.count_nonzero(w[:-1])),
        "seconds": time.perf_counter() - started,
    }
    return w, info


def classification_metrics(y, predicted):
    positive = np.asarray(y) == 1
    predicted = np.asarray(predicted) == 1
    tp = int(np.sum(positive & predicted))
    denominator = int(positive.sum() + predicted.sum())
    return {
        "f1": 2 * tp / denominator if denominator else 0.0,
        "precision": tp / max(int(predicted.sum()), 1),
        "recall": tp / max(int(positive.sum()), 1),
        "accuracy": float(np.mean(positive == predicted)),
        "positive_rate": float(predicted.mean()),
    }


def best_f1_threshold(y, probabilities):
    """Search every distinct probability, keeping tied predictions together."""
    y = np.asarray(y) == 1
    probabilities = np.asarray(probabilities)
    if (
        y.shape != probabilities.shape
        or not len(y)
        or not np.isfinite(probabilities).all()
    ):
        raise ValueError(
            "Expected nonempty, finite, equally sized labels and probabilities"
        )
    order = np.argsort(-probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    true_positives = np.cumsum(y[order])
    predicted_positives = np.arange(1, len(y) + 1)
    f1 = 2 * true_positives / (y.sum() + predicted_positives)
    distinct = np.r_[sorted_probabilities[:-1] != sorted_probabilities[1:], True]
    f1[~distinct] = -1
    best = np.argmax(f1)
    return float(sorted_probabilities[best]), float(f1[best])


def stratified_split(y, seed=42, validation_ratio=0.2):
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between zero and one")
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        count = int(len(indices) * validation_ratio)
        train.extend(indices[count:])
        validation.extend(indices[:count])
    return rng.permutation(train), rng.permutation(validation)


def stratified_folds(y, n_folds=5, seed=2026):
    """Return disjoint validation folds covering every training row once."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        for fold, part in zip(folds, np.array_split(indices, n_folds)):
            fold.extend(part)
    return [rng.permutation(fold) for fold in folds]
