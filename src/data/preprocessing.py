import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_dataset(df):

    df = df.copy()

    # =====================================
    # TIME
    # =====================================
    df["time"] = pd.to_datetime(df["time"])

    df = df.sort_values(["time", "ID"])

    # =====================================
    # INTERPOLACIÓN
    # =====================================
    numeric_cols = [
        c for c in df.columns
        if c not in [
            "time",
            "ID",
            "station_name"
        ]
    ]

    df[numeric_cols] = (
        df.groupby("ID")[numeric_cols]
        .transform(lambda x: x.interpolate(limit=3))
    )

    # =====================================
    # FEATURES TEMPORALES
    # =====================================
    df["hour"] = df["time"].dt.hour
    df["dow"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)

    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)

    # =====================================
    # VIENTO (DV -> sin/cos)
    # =====================================
    if "DV" in df.columns:
        df["wind_sin"] = np.sin(np.deg2rad(df["DV"]))
        df["wind_cos"] = np.cos(np.deg2rad(df["DV"]))

    # =====================================
    # ESCALADO
    # =====================================
    scaler = StandardScaler()

    exclude_scale = {
        "Sah", "time", "ID", "station_name", "lat", "lon", "zone",
        "hour", "dow", "month", "DV",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "wind_sin", "wind_cos",
    }
    scale_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_scale
    ]

    df[scale_cols] = scaler.fit_transform(
        df[scale_cols]
    )

    return df, scaler