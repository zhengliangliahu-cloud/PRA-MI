from __future__ import annotations

import json

import numpy as np

from autoeegencoder.protocols.transforms import MethodArrays
from autoeegencoder.utils.metrics import balanced_accuracy


def _list_param(method: dict, key: str, default: list) -> list:
    value = method.get(key, default)
    if isinstance(value, list):
        return value
    return [value]


def _fit_predict_csp(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    *,
    baseline: str,
    n_components: int,
    c_value: float,
    run_seed: int,
) -> np.ndarray:
    try:
        from mne.decoding import CSP
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC
    except ImportError as exc:
        raise RuntimeError(
            "CSP baselines require mne and scikit-learn. Install the AEC environment first."
        ) from exc

    steps: list[tuple[str, object]] = [
        ("csp", CSP(n_components=n_components, log=True, norm_trace=False)),
    ]
    if baseline == "csp_logreg":
        steps.append(
            (
                "clf",
                LogisticRegression(
                    C=float(c_value),
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=int(run_seed),
                ),
            )
        )
    elif baseline == "csp_svm":
        steps.append(("clf", SVC(kernel="linear", C=float(c_value))))
    else:
        raise NotImplementedError(f"Unknown CSP baseline: {baseline}")
    clf = Pipeline(steps)
    clf.fit(train_X, train_y)
    return np.asarray(clf.predict(test_X), dtype="int64")


def _run_csp_pipeline(arrays: MethodArrays, method: dict, baseline: str, run_seed: int) -> tuple[np.ndarray, dict]:
    n_components = int(method.get("n_components", min(6, arrays.train_X.shape[1])))
    c_value = float(method.get("c", 1.0))
    n_components = min(n_components, arrays.train_X.shape[1])
    pred = _fit_predict_csp(
        arrays.train_X,
        arrays.train_y,
        arrays.test_X,
        baseline=baseline,
        n_components=n_components,
        c_value=c_value,
        run_seed=run_seed,
    )
    return pred, {"n_components": n_components, "c": c_value}


def _fit_predict_riemann(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    *,
    baseline: str,
    cov_estimator: str,
    c_value: float,
    mdm_metric: str,
    run_seed: int,
) -> np.ndarray:
    try:
        from pyriemann.classification import MDM
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Riemannian baselines require pyriemann and scikit-learn. Install the AEC environment first."
        ) from exc

    if baseline == "riemann_mdm":
        clf = Pipeline(
            [
                ("cov", Covariances(estimator=cov_estimator)),
                ("clf", MDM(metric=mdm_metric)),
            ]
        )
    elif baseline == "riemann_tangent_lr":
        clf = Pipeline(
            [
                ("cov", Covariances(estimator=cov_estimator)),
                ("ts", TangentSpace(metric="riemann")),
                (
                    "clf",
                    LogisticRegression(
                        C=float(c_value),
                        max_iter=1000,
                        solver="lbfgs",
                        random_state=int(run_seed),
                    ),
                ),
            ]
        )
    else:
        raise NotImplementedError(f"Unknown Riemannian baseline: {baseline}")
    clf.fit(train_X, train_y)
    return np.asarray(clf.predict(test_X), dtype="int64")


def _run_riemann_pipeline(arrays: MethodArrays, method: dict, baseline: str, run_seed: int) -> tuple[np.ndarray, dict]:
    cov_estimator = str(method.get("cov_estimator", "oas"))
    c_value = float(method.get("c", 1.0))
    mdm_metric = str(method.get("mdm_metric", "riemann"))
    pred = _fit_predict_riemann(
        arrays.train_X,
        arrays.train_y,
        arrays.test_X,
        baseline=baseline,
        cov_estimator=cov_estimator,
        c_value=c_value,
        mdm_metric=mdm_metric,
        run_seed=run_seed,
    )
    return pred, {"cov_estimator": cov_estimator, "c": c_value, "mdm_metric": mdm_metric}


def _tune_classical_method(arrays: MethodArrays, method: dict, baseline: str, run_seed: int) -> tuple[np.ndarray, dict, float]:
    best_score = -np.inf
    best_params: dict | None = None
    failures: list[str] = []

    if baseline in {"csp_logreg", "csp_svm"}:
        for n_components in _list_param(method, "n_components_grid", [2, 4, 6, 8]):
            for c_value in _list_param(method, "c_grid", [0.1, 1.0, 10.0]):
                params = {
                    "n_components": int(min(int(n_components), arrays.train_X.shape[1])),
                    "c": float(c_value),
                }
                try:
                    val_pred = _fit_predict_csp(
                        arrays.train_X,
                        arrays.train_y,
                        arrays.val_X,
                        baseline=baseline,
                        n_components=params["n_components"],
                        c_value=params["c"],
                        run_seed=run_seed,
                    )
                    score = balanced_accuracy(arrays.val_y, val_pred)
                except Exception as exc:  # pragma: no cover - depends on estimator internals.
                    failures.append(f"{params}: {exc}")
                    continue
                if score > best_score:
                    best_score = score
                    best_params = params
        if best_params is None:
            raise RuntimeError(f"All CSP tuning candidates failed: {failures[:3]}")
        train_X = np.concatenate([arrays.train_X, arrays.val_X], axis=0)
        train_y = np.concatenate([arrays.train_y, arrays.val_y], axis=0)
        test_pred = _fit_predict_csp(
            train_X,
            train_y,
            arrays.test_X,
            baseline=baseline,
            n_components=best_params["n_components"],
            c_value=best_params["c"],
            run_seed=run_seed,
        )
        return test_pred, best_params, float(best_score)

    if baseline in {"riemann_mdm", "riemann_tangent_lr"}:
        cov_grid = _list_param(method, "cov_estimator_grid", ["oas", "lwf", "scm"])
        c_grid = _list_param(method, "c_grid", [0.1, 1.0, 10.0])
        mdm_grid = _list_param(method, "mdm_metric_grid", ["riemann", "logeuclid"])
        if baseline == "riemann_tangent_lr":
            mdm_grid = ["riemann"]
        for cov_estimator in cov_grid:
            for c_value in c_grid:
                for mdm_metric in mdm_grid:
                    params = {
                        "cov_estimator": str(cov_estimator),
                        "c": float(c_value),
                        "mdm_metric": str(mdm_metric),
                    }
                    try:
                        val_pred = _fit_predict_riemann(
                            arrays.train_X,
                            arrays.train_y,
                            arrays.val_X,
                            baseline=baseline,
                            cov_estimator=params["cov_estimator"],
                            c_value=params["c"],
                            mdm_metric=params["mdm_metric"],
                            run_seed=run_seed,
                        )
                        score = balanced_accuracy(arrays.val_y, val_pred)
                    except Exception as exc:  # pragma: no cover - depends on estimator internals.
                        failures.append(f"{params}: {exc}")
                        continue
                    if score > best_score:
                        best_score = score
                        best_params = params
        if best_params is None:
            raise RuntimeError(f"All Riemannian tuning candidates failed: {failures[:3]}")
        train_X = np.concatenate([arrays.train_X, arrays.val_X], axis=0)
        train_y = np.concatenate([arrays.train_y, arrays.val_y], axis=0)
        test_pred = _fit_predict_riemann(
            train_X,
            train_y,
            arrays.test_X,
            baseline=baseline,
            cov_estimator=best_params["cov_estimator"],
            c_value=best_params["c"],
            mdm_metric=best_params["mdm_metric"],
            run_seed=run_seed,
        )
        return test_pred, best_params, float(best_score)

    raise NotImplementedError(f"Unknown classical baseline: {baseline}")


def run_classical_method(arrays: MethodArrays, method: dict, run_seed: int) -> dict[str, float]:
    baseline = str(method.get("baseline", "csp_logreg"))
    selected_params: dict = {}
    val_score = np.nan
    if bool(method.get("source_validation_tune", False)):
        y_pred, selected_params, val_score = _tune_classical_method(arrays, method, baseline, run_seed)
    elif baseline in {"csp_logreg", "csp_svm"}:
        y_pred, selected_params = _run_csp_pipeline(arrays, method, baseline, run_seed)
    elif baseline in {"riemann_mdm", "riemann_tangent_lr"}:
        y_pred, selected_params = _run_riemann_pipeline(arrays, method, baseline, run_seed)
    else:
        raise NotImplementedError(f"Unknown classical baseline: {baseline}")

    y_true = np.asarray(arrays.test_y, dtype="int64")
    acc = float(np.mean(y_pred == y_true))
    bacc = balanced_accuracy(y_true, y_pred)
    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "classical_val_balanced_accuracy": float(val_score) if np.isfinite(val_score) else "",
        "classical_selected_params": json.dumps(selected_params, sort_keys=True),
    }
