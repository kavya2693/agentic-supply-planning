"""Global LightGBM demand forecaster (the 'existing model' the agents build on).

Direct multi-horizon: from a forecast origin week, one model predicts each of the next
HORIZON weeks, with the horizon as a feature. Two quantile heads give P50 (expected
demand) and P90 (demand to cover for the service level)."""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import DATA, HORIZON, SEED

CATS = ["family", "channel", "pack", "source"]


def _panel(sales: pd.DataFrame):
    Y = sales.pivot(index="sku", columns="week", values="units").sort_index()
    P = sales.pivot(index="sku", columns="week", values="promo").reindex(Y.index)
    return Y.to_numpy(float), P.to_numpy(float), Y.index.to_numpy(), Y.columns


def _features(Y, P, skus_meta, origin, weeks, future_promo=None):
    """Rows for every SKU x horizon from one origin. Only data up to `origin` is used,
    except the promo plan and calendar of the target week, which are known in advance."""
    n = Y.shape[0]
    hist = Y[:, : origin + 1]
    base = {
        "last1": hist[:, -1],
        "mean4": hist[:, -4:].mean(1),
        "mean13": hist[:, -13:].mean(1),
        "mean52": hist[:, -52:].mean(1),
        "std13": hist[:, -13:].std(1),
        "zero13": (hist[:, -13:] == 0).mean(1),
    }
    frames = []
    for k in range(1, HORIZON + 1):
        t = origin + k
        f = pd.DataFrame(base)
        f["h"] = k
        f["lag52_target"] = Y[:, t - 52]
        f["woy"] = (weeks[0] + pd.Timedelta(weeks=t)).isocalendar()[1]
        f["promo"] = P[:, t] if future_promo is None else future_promo
        f["row"] = np.arange(n)
        frames.append(f)
    X = pd.concat(frames, ignore_index=True)
    for c in CATS:
        X[c] = pd.Categorical(skus_meta[c].to_numpy()[X.row.to_numpy()])
    return X


def _training_set(Y, P, meta, weeks, last_target):
    Xs, ys = [], []
    for o in range(52, last_target - HORIZON + 1, 2):
        X = _features(Y, P, meta, o, weeks)
        Xs.append(X)
        ys.append(Y[X.row.to_numpy(), o + X.h.to_numpy()])
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


def _fit(X, y, alpha):
    m = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        random_state=SEED,
        verbose=-1,
    )
    return m.fit(X.drop(columns="row"), y)


def wape(actual, pred):
    return float(np.abs(actual - pred).sum() / max(actual.sum(), 1e-9))


def train_and_forecast(skus: pd.DataFrame, sales: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    Y, P, sku_ids, weeks = _panel(sales)
    meta = skus.set_index("sku").loc[sku_ids].reset_index()
    T = Y.shape[1]

    # 1) Backtest: forecast the last HORIZON weeks from an origin the model never saw past.
    o_h = T - 1 - HORIZON
    X, y = _training_set(Y, P, meta, weeks, o_h)
    p50, p90 = _fit(X, y, 0.5), _fit(X, y, 0.9)
    Xh = _features(Y, P, meta, o_h, weeks)
    actual = Y[Xh.row.to_numpy(), o_h + Xh.h.to_numpy()]
    pred50 = p50.predict(Xh.drop(columns="row")).clip(0)
    pred90 = p90.predict(Xh.drop(columns="row")).clip(0)
    metrics = {
        "backtest_weeks": HORIZON,
        "skus": len(sku_ids),
        "wape_model_p50": round(wape(actual, pred50), 3),
        "wape_seasonal_naive": round(wape(actual, Xh.lag52_target.to_numpy()), 3),
        "wape_moving_avg_4wk": round(wape(actual, Xh.mean4.to_numpy()), 3),
        "p90_coverage": round(float((actual <= pred90).mean()), 3),
    }

    # 2) Production forecast from the latest week (no promotions planned yet).
    X, y = _training_set(Y, P, meta, weeks, T - 1)
    p50, p90 = _fit(X, y, 0.5), _fit(X, y, 0.9)
    # Horizon rows need target-week columns that do not exist yet; pad with the seasonal proxy.
    Ypad = np.hstack([Y, Y[:, T - 52 : T - 52 + HORIZON]])
    Ppad = np.hstack([P, np.zeros((P.shape[0], HORIZON))])
    Xf = _features(Ypad, Ppad, meta, T - 1, weeks, future_promo=0)
    Xf["lag52_target"] = Y[Xf.row.to_numpy(), T - 1 + Xf.h.to_numpy() - 52]
    fc = pd.DataFrame(
        {
            "sku": sku_ids[Xf.row.to_numpy()],
            "week_ahead": Xf.h.to_numpy(),
            "p50": p50.predict(Xf.drop(columns="row")).clip(0).round(1),
            "p90": p90.predict(Xf.drop(columns="row")).clip(0).round(1),
        }
    )
    fc["p90"] = np.maximum(fc.p90, fc.p50)
    fc.to_csv(DATA / "forecast.csv", index=False)
    (DATA / "forecast_metrics.json").write_text(json.dumps(metrics, indent=2))
    return fc, metrics


def load_forecast(skus, sales, refresh=False):
    if refresh or not (DATA / "forecast.csv").exists():
        return train_and_forecast(skus, sales)
    return pd.read_csv(DATA / "forecast.csv"), json.loads((DATA / "forecast_metrics.json").read_text())
