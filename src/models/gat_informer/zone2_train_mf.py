"""GAT-Informer multi-feature (GAT-MF) training for MASTER zone packs.

Parallel to zone2_train.py (PM10-only GAT-v1). Does not modify v1.

Packs: X [B,N,L,F], Y [B,N,H]
- GAT branch: flatten L×F → GraphAttentionLayer(in_c=L*F)
- Informer branch: target channel 0 only (same as v1 Informer path)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.informer2020.metrics import (
    metric_hourly_and_ma24,
    moving_average_first_24h,
    moving_average_second_24h,
)
from src.utils.paths import gat_informer_upstream


def _import_upstream():
    up = gat_informer_upstream()
    if str(up) not in sys.path:
        sys.path.insert(0, str(up))
    from block.GAT import GraphAttentionLayer
    from block.cross import cross_att
    from block.informer_arch import Informer, InformerStack
    from block.revin import RevIN
    from metric.mask_metric import masked_mae

    return GraphAttentionLayer, cross_att, Informer, InformerStack, RevIN, masked_mae


def Inverse_normalization(x, vmax, vmin):
    return x * (vmax - vmin) + vmin


class XYDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class GATINFORMER_MF(nn.Module):
    def __init__(
        self,
        IF_STACK,
        num_layer,
        enc_in,
        dec_in,
        c_out,
        seq_len,
        label_len,
        out_len,
        n_features,
        time_of_day_size,
        day_of_week_size,
        day_of_month_size,
        day_of_year_size,
        factor,
        d_model,
        n_heads,
        e_layers,
        d_layers,
        d_ff,
        dropout,
        attn,
        embed,
        freq,
        activation,
        output_attention,
        distil,
        mix,
        num_time_features,
        IF_cross,
        GraphAttentionLayer,
        cross_att,
        Informer,
        InformerStack,
        RevIN,
    ):
        super().__init__()
        self.IF_STACK = IF_STACK
        self.num_layer = num_layer
        self.seq_len = seq_len
        self.n_features = n_features
        self.lay_norm = nn.LayerNorm([out_len])
        self.IF_cross = IF_cross
        self.RevIN = RevIN(enc_in)
        gat_in = seq_len * n_features
        self.GAT1 = GraphAttentionLayer(gat_in, out_len, dropout)
        self.GAT2 = GraphAttentionLayer(out_len, out_len, dropout)
        if self.IF_STACK:
            self.Informer = InformerStack(
                enc_in,
                dec_in,
                c_out,
                seq_len,
                label_len,
                out_len,
                time_of_day_size,
                day_of_week_size,
                day_of_month_size,
                day_of_year_size,
                factor,
                d_model,
                n_heads,
                e_layers,
                d_layers,
                d_ff,
                dropout=dropout,
                attn=attn,
                embed=embed,
                freq=freq,
                activation=activation,
                output_attention=output_attention,
                distil=distil,
                mix=mix,
                num_time_features=num_time_features,
            )
        else:
            self.Informer = Informer(
                enc_in,
                dec_in,
                c_out,
                seq_len,
                label_len,
                out_len,
                time_of_day_size,
                day_of_week_size,
                day_of_month_size=day_of_month_size,
                day_of_year_size=day_of_year_size,
                factor=factor,
                d_model=d_model,
                n_heads=n_heads,
                e_layers=e_layers,
                d_layers=d_layers,
                d_ff=d_ff,
                dropout=dropout,
                attn=attn,
                embed=embed,
                freq=freq,
                activation=activation,
                output_attention=output_attention,
                distil=distil,
                mix=mix,
                num_time_features=num_time_features,
            )
        self.cross = cross_att(out_len, n_heads, dropout)
        self.decoder = nn.Conv1d(
            in_channels=out_len, out_channels=out_len, kernel_size=1
        )

    def forward(self, x, y, graph_data, device):
        """
        x: [B, N, L, F] or [B, N, L] (F=1)
        y: [B, N, H]
        """
        graph_data = graph_data.to(device)
        graph_data = self.calculate_laplacian_with_self_loop(graph_data)

        if x.dim() == 3:
            x_pm10 = x
            x_gat = x
        else:
            # [B, N, L, F]
            b, n, l, f = x.shape
            x_pm10 = x[..., 0]
            x_gat = x.reshape(b, n, l * f)

        # RevIN on target series only (Informer path; F1 matches v1)
        x_pm10 = self.RevIN(x_pm10.transpose(-2, -1), "norm").transpose(-2, -1)
        if self.n_features == 1:
            x_gat = x_pm10

        for i in range(self.num_layer):
            if i == 0:
                prediction_GAT = F.gelu(self.GAT1(x_gat, graph_data))
            else:
                prediction_GAT = F.gelu(self.GAT2(prediction_GAT, graph_data))
        prediction_In = self.Informer(x_pm10, y)
        if self.IF_cross:
            out = self.cross(prediction_In, prediction_GAT).transpose(-2, -1)
        else:
            out = prediction_GAT + prediction_In
            out = self.lay_norm(out).transpose(-2, -1)
        out = self.decoder(out)
        out = self.RevIN(out, "denorm")
        return out.transpose(-2, -1)

    @staticmethod
    def calculate_laplacian_with_self_loop(matrix):
        row_sum = matrix.sum(1)
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)


def find_npz(data_dir: Path, seq_len: int) -> Path:
    preferred = data_dir / f"data{seq_len}.npz"
    if preferred.exists():
        return preferred
    cands = sorted(data_dir.glob("data*.npz"))
    if not cands:
        raise FileNotFoundError(f"No data*.npz under {data_dir}")
    return cands[0]


def _target_scale(max_min: np.ndarray) -> tuple[float, float]:
    """Support v1 [2] and MF [F,2] max_min layouts."""
    arr = np.asarray(max_min)
    if arr.ndim == 1:
        return float(arr[0]), float(arr[1])
    return float(arr[0, 0]), float(arr[0, 1])


def train_and_eval(
    *,
    data_dir: Path,
    out_dir: Path,
    seq_len: int = 48,
    horizon: int = 24,
    batch_size: int = 32,
    max_epochs: int = 50,
    seed: int = 3407,
    d_model: int = 32,
    n_heads: int = 4,
    dropout: float = 0.15,
    lr: float = 0.002,
    weight_decay: float = 0.0005,
    num_layer: int = 2,
    if_cross: bool = True,
) -> dict:
    GraphAttentionLayer, cross_att, Informer, InformerStack, RevIN, masked_mae = (
        _import_upstream()
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    npz_path = find_npz(data_dir, seq_len)
    raw = np.load(npz_path, allow_pickle=True)
    meta = {}
    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    train_x = torch.tensor(raw["train_x_raw"]).float()
    train_y = torch.tensor(raw["train_y"]).float()
    valid_x = torch.tensor(raw["vail_x_raw"]).float()
    valid_y = torch.tensor(raw["vail_y"]).float()
    test_x = torch.tensor(raw["test_x_raw"]).float()
    test_y = torch.tensor(raw["test_y"]).float()
    graph_data = torch.tensor(raw["graph"]).float()
    vmax, vmin = _target_scale(raw["max_min"])

    num_nodes = int(train_x.shape[1])
    if train_x.dim() == 4:
        input_len = int(train_x.shape[2])
        n_features = int(train_x.shape[3])
    else:
        input_len = int(train_x.shape[-1])
        n_features = 1
    output_len = int(train_y.shape[-1])
    label_len = input_len // 2

    train_loader = DataLoader(
        XYDataset(train_x, train_y), batch_size=batch_size, shuffle=False
    )
    valid_loader = DataLoader(
        XYDataset(valid_x, valid_y), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        XYDataset(test_x, test_y), batch_size=batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"GAT-MF device={device} nodes={num_nodes} "
        f"seq={input_len} F={n_features} horizon={output_len} npz={npz_path}"
    )

    net = GATINFORMER_MF(
        False,
        num_layer,
        num_nodes,
        num_nodes,
        num_nodes,
        input_len,
        label_len,
        output_len,
        n_features,
        24,
        7,
        31,
        366,
        3,
        d_model,
        n_heads,
        2,
        1,
        d_model,
        dropout,
        "prob",
        "timeF",
        "h",
        "gelu",
        False,
        True,
        True,
        2,
        if_cross,
        GraphAttentionLayer,
        cross_att,
        Informer,
        InformerStack,
        RevIN,
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    milestone = {1, 4, 10, 15, 30, 50, 70, 90}
    gamme = 0.5
    history = []

    for epoch in range(max_epochs):
        net.train()
        loss_sum, n_b = 0.0, 0
        for feat, tgt in train_loader:
            net.zero_grad()
            feat = feat.to(device)
            tgt = tgt.to(device)
            pred = net(feat, tgt, graph_data, device)
            loss = masked_mae(pred, tgt, 0.0)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            n_b += 1
        train_loss = loss_sum / max(n_b, 1)

        net.eval()
        v_sum, v_n = 0.0, 0
        with torch.no_grad():
            for feat, tgt in valid_loader:
                feat = feat.to(device)
                tgt = tgt.to(device)
                pred = net(feat, tgt, graph_data, device)
                v_sum += float(masked_mae(pred, tgt, 0.0).cpu())
                v_n += 1
        val_loss = v_sum / max(v_n, 1)
        if (epoch + 1) in milestone:
            for g in optimizer.param_groups:
                g["lr"] *= gamme
        history.append(
            {"epoch": epoch + 1, "train_mae": train_loss, "val_mae": val_loss}
        )
        print(f"epoch {epoch+1}/{max_epochs} train={train_loss:.4f} val={val_loss:.4f}")

    net.eval()
    net = net.to("cpu")
    graph_cpu = graph_data
    preds, trues = [], []
    with torch.no_grad():
        for feat, tgt in test_loader:
            pred = net(feat, tgt, graph_cpu, torch.device("cpu"))
            preds.append(pred)
            trues.append(tgt)
    all_pre = torch.cat(preds, dim=0)
    all_true = torch.cat(trues, dim=0)
    final_pred = Inverse_normalization(all_pre, vmax, vmin).numpy()
    final_true = Inverse_normalization(all_true, vmax, vmin).numpy()
    pred_m = np.transpose(final_pred, (0, 2, 1))[..., None]
    true_m = np.transpose(final_true, (0, 2, 1))[..., None]

    overall = metric_hourly_and_ma24(pred_m, true_m)
    stations = meta.get("stations") or [f"node_{i}" for i in range(num_nodes)]
    by_station = []
    for i, name in enumerate(stations[:num_nodes]):
        m = metric_hourly_and_ma24(
            pred_m[:, :, i : i + 1, :], true_m[:, :, i : i + 1, :]
        )
        row = {
            "station": name,
            "hourly_mae": m["hourly"]["mae"],
            "hourly_rmse": m["hourly"]["rmse"],
            "ma24_mae": m["ma24"]["mae"],
            "ma24_rmse": m["ma24"]["rmse"],
        }
        if "ma24_d1" in m:
            row["ma24_d1_rmse"] = m["ma24_d1"]["rmse"]
            row["ma24_d2_rmse"] = m["ma24_d2"]["rmse"]
        by_station.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "test_preds_hourly.npy", pred_m)
    np.save(out_dir / "test_trues_hourly.npy", true_m)
    if pred_m.shape[1] >= 48:
        np.save(out_dir / "test_preds_ma24.npy", moving_average_first_24h(pred_m))
        np.save(out_dir / "test_trues_ma24.npy", moving_average_first_24h(true_m))
        np.save(out_dir / "test_preds_ma24_d2.npy", moving_average_second_24h(pred_m))
        np.save(out_dir / "test_trues_ma24_d2.npy", moving_average_second_24h(true_m))
    else:
        np.save(out_dir / "test_preds_ma24.npy", pred_m.mean(axis=1, keepdims=True))
        np.save(out_dir / "test_trues_ma24.npy", true_m.mean(axis=1, keepdims=True))
    torch.save(net.state_dict(), out_dir / "model.pt")

    results = {
        "model": "gat_informer_mf",
        "zone": meta.get("zone"),
        "feature_config": meta.get("feature_config"),
        "graph_tag": meta.get("graph_tag"),
        "graph_csv": meta.get("graph_csv"),
        "num_nodes": num_nodes,
        "n_features": n_features,
        "feature_cols": meta.get("feature_cols"),
        "seq_len": input_len,
        "horizon": output_len,
        "stations": stations[:num_nodes],
        "npz": str(npz_path.as_posix()),
        "device_train": str(device),
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "note_features": meta.get("note_features"),
        "forecast_design": {
            "at_hour_h": "model outputs hours h+1 ... h+horizon for all nodes",
            "ma24": (
                "mean(pred[h+1], ..., pred[h+24]); at horizon>=48 also ma24_d2 "
                "in test_metrics_raw"
            ),
            "gat_input": "flatten(lookback × n_features) per node",
            "informer_input": "target channel 0 only",
            "teacher_forcing": "upstream passes future y into Informer branch (train+test)",
        },
        "test_metrics_raw": {
            "hourly": overall["hourly"],
            "ma24": overall["ma24"],
            **(
                {"ma24_d1": overall["ma24_d1"], "ma24_d2": overall["ma24_d2"]}
                if "ma24_d1" in overall
                else {}
            ),
        },
        "per_station": by_station,
        "history": history,
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    with open(out_dir / "per_station_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "station",
                "hourly_mae",
                "hourly_rmse",
                "ma24_mae",
                "ma24_rmse",
            ],
        )
        w.writeheader()
        w.writerows(by_station)

    print(
        f"Overall hourly MAE={overall['hourly']['mae']:.4f} "
        f"RMSE={overall['hourly']['rmse']:.4f}"
    )
    print(
        f"Overall MA24   MAE={overall['ma24']['mae']:.4f} "
        f"RMSE={overall['ma24']['rmse']:.4f}"
    )
    return results
