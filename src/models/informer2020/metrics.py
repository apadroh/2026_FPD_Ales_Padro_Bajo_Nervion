import numpy as np


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    denom = np.where(np.abs(true) < 1e-6, np.nan, true)
    return np.nanmean(np.abs((pred - true) / denom))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    return mae, mse, rmse, mape


def moving_average_24h(series: np.ndarray) -> np.ndarray:
    """
    Mean over the forecast horizon (axis=1).

    For pred_len == 24 this is MA24 (h+1 ... h+24).
    For pred_len > 24 use moving_average_first_24h / moving_average_second_24h.
    """
    return np.mean(series, axis=1, keepdims=True)


def moving_average_first_24h(series: np.ndarray) -> np.ndarray:
    """MA24_D1: mean of hours h+1 ... h+24 (first 24 columns)."""
    return np.mean(series[:, :24], axis=1, keepdims=True)


def moving_average_second_24h(series: np.ndarray) -> np.ndarray:
    """MA24_D2: mean of hours h+25 ... h+48 (columns 24:48)."""
    return np.mean(series[:, 24:48], axis=1, keepdims=True)


def _metric_block(pred: np.ndarray, true: np.ndarray, definition: str) -> dict:
    mae, mse, rmse, mape = metric(pred, true)
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
        "definition": definition,
    }


def metric_hourly_and_ma24(pred: np.ndarray, true: np.ndarray) -> dict:
    """
    Hourly metrics on the full horizon plus MA24 metrics.

    At pred_len == 24: ma24 = mean(y_{t+1}, ..., y_{t+24}).
    At pred_len >= 48: ma24_d1 (day 1) and ma24_d2 (day 2); ma24 aliases ma24_d1.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    mae, mse, rmse, mape = metric(pred, true)
    out: dict = {
        "hourly": {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape),
        },
    }

    pred_len = int(pred.shape[1]) if pred.ndim >= 2 else 1
    if pred_len >= 48:
        pred_d1 = moving_average_first_24h(pred)
        true_d1 = moving_average_first_24h(true)
        pred_d2 = moving_average_second_24h(pred)
        true_d2 = moving_average_second_24h(true)
        out["ma24_d1"] = _metric_block(
            pred_d1,
            true_d1,
            "mean(pred[h+1], ..., pred[h+24])",
        )
        out["ma24_d2"] = _metric_block(
            pred_d2,
            true_d2,
            "mean(pred[h+25], ..., pred[h+48])",
        )
        out["ma24"] = dict(out["ma24_d1"])
        out["ma24"]["definition"] = "alias of ma24_d1 when horizon >= 48"
    else:
        pred_ma = moving_average_24h(pred)
        true_ma = moving_average_24h(true)
        out["ma24"] = _metric_block(
            pred_ma,
            true_ma,
            "mean of the 24 hourly forecasts (h+1 ... h+24)",
        )
    return out
