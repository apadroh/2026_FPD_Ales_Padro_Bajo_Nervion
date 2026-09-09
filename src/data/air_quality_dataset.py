"""
PyTorch dataset compatible with Informer2020 for air-quality series.

Temporal split 70/10/20 (train/val/test); scaler fit on train only.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.models.informer2020.timefeatures import time_features
from src.models.informer2020.tools import StandardScaler


class AirQualityDataset(Dataset):
    """
    Read exported wide CSV: date + features + target (target in last column).

    Compatible with Informer2020 feature modes S / M / MS.
    """

    def __init__(
        self,
        root_path: str | Path,
        flag: str = "train",
        size: tuple[int, int, int] | None = None,
        features: str = "MS",
        data_path: str = "data.csv",
        target: str = "PM10",
        scale: bool = True,
        inverse: bool = False,
        timeenc: int = 1,
        freq: str = "h",
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
    ):
        if size is None:
            self.seq_len = 48
            self.label_len = 24
            self.pred_len = 24
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in {"train", "val", "test"}
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.root_path = Path(root_path)
        self.data_path = data_path
        self.scaler = StandardScaler()
        self._read_data()

    def _read_data(self) -> None:
        df_raw = pd.read_csv(self.root_path / self.data_path)
        df_raw = df_raw.rename(columns={"date": "date"})
        if "date" not in df_raw.columns and "time" in df_raw.columns:
            df_raw = df_raw.rename(columns={"time": "date"})

        cols = list(df_raw.columns)
        cols.remove("date")
        if self.target in cols:
            cols.remove(self.target)
        df_raw = df_raw[["date"] + cols + [self.target]]

        n = len(df_raw)
        num_train = int(n * self.train_ratio)
        num_test = int(n * (1 - self.train_ratio - self.val_ratio))
        num_vali = n - num_train - num_test

        border1s = [
            0,
            num_train - self.seq_len,
            n - num_test - self.seq_len,
        ]
        border2s = [num_train, num_train + num_vali, n]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in {"M", "MS"}:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        else:
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2].copy()
        df_stamp["date"] = pd.to_datetime(df_stamp["date"])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.border1 = border1
        self.border2 = border2
        self.borders = {
            "train": (border1s[0], border2s[0]),
            "val": (border1s[1], border2s[1]),
            "test": (border1s[2], border2s[2]),
        }

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
