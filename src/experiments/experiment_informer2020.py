"""
Train the original Informer (Informer2020) on air-quality data.

Requires wide CSV exported with export_informer2020_csv.py.
70/10/20 split, early stopping, MAE/MSE/RMSE metrics on test.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.air_quality_dataset import AirQualityDataset
from src.data.feature_configs import ABLATION_LEVELS
from src.models.informer2020.metrics import (
    metric,
    metric_hourly_and_ma24,
    moving_average_first_24h,
    moving_average_second_24h,
)
from src.models.informer2020.model import Informer
from src.models.informer2020.plots import save_run_plots
from src.models.informer2020.tools import EarlyStopping, adjust_learning_rate
from src.utils.paths import informer2020_results_dir, informer2020_station_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Informer2020 (official)")
    parser.add_argument("--station", required=True)
    parser.add_argument("--zone", type=int, default=2)
    parser.add_argument("--contaminant", default="PM10")
    parser.add_argument(
        "--feature-config",
        default="F5",
        choices=ABLATION_LEVELS,
        help="Feature set from feature_selection (default F5 for PM10)",
    )
    parser.add_argument(
        "--data-root",
        default="data/training",
        help="Root directory of training CSVs",
    )
    parser.add_argument(
        "--results-root",
        default="results/informer2020",
        help="Root directory for checkpoints and metrics",
    )
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--target", default="PM10")
    parser.add_argument(
        "--features",
        default="MS",
        choices=["S", "M", "MS"],
        help="S=univariate, M=multivariate, MS=multivariate→univariate",
    )
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--label-len", type=int, default=24)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-layers", type=int, default=1)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--attn", default="prob", choices=["prob", "full"])
    parser.add_argument("--embed", default="timeF")
    parser.add_argument("--freq", default="h")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lradj", default="type1")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, fix RNG and write under .../{F*}/seed_{N}/",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def station_dir(args: argparse.Namespace) -> Path:
    return informer2020_station_dir(args.zone, args.station, args.feature_config)


def results_dir(args: argparse.Namespace) -> Path:
    base = informer2020_results_dir(
        args.zone,
        args.station,
        args.feature_config,
        seq_len=args.seq_len,
        horizon=args.pred_len,
    )
    if args.seed is not None:
        return base / f"seed_{args.seed}"
    return base


def ensure_export(args: argparse.Namespace, data_dir: Path) -> None:
    if (data_dir / "data.csv").exists() and not args.export:
        return
    cmd = [
        sys.executable,
        str(ROOT / "src/data/export_informer2020_csv.py"),
        "--station",
        args.station,
        "--zone",
        str(args.zone),
        "--contaminant",
        args.contaminant,
        "--feature-config",
        args.feature_config,
        "--output-root",
        args.data_root,
    ]
    print("Exportando CSV wide...")
    subprocess.run(cmd, check=True, cwd=ROOT)


def build_loaders(args: argparse.Namespace, data_dir: Path):
    size = (args.seq_len, args.label_len, args.pred_len)
    common = {
        "root_path": data_dir,
        "size": size,
        "features": args.features,
        "target": args.target,
        "scale": True,
        "timeenc": 1,
        "freq": args.freq,
    }

    train_set = AirQualityDataset(flag="train", **common)
    val_set = AirQualityDataset(flag="val", **common)
    test_set = AirQualityDataset(flag="test", **common)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_set, val_set, test_set, train_loader, val_loader, test_loader


def n_channels(features: str, n_cols: int) -> tuple[int, int, int]:
    if features == "S":
        return 1, 1, 1
    if features == "MS":
        return n_cols, n_cols, 1
    return n_cols, n_cols, n_cols


def process_batch(
    model,
    dataset,
    batch_x,
    batch_y,
    batch_x_mark,
    batch_y_mark,
    device,
    pred_len,
    label_len,
    features,
):
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float()
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    dec_inp = torch.zeros(
        [batch_y.shape[0], pred_len, batch_y.shape[-1]]
    ).float()
    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)

    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    f_dim = -1 if features == "MS" else 0
    outputs = outputs[:, -pred_len:, f_dim:]
    batch_y = batch_y[:, -pred_len:, f_dim:].to(device)
    return outputs, batch_y


def evaluate(model, loader, dataset, device, args, criterion):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            pred, true = process_batch(
                model,
                dataset,
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                device,
                args.pred_len,
                args.label_len,
                args.features,
            )
            losses.append(criterion(pred, true).item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def collect_predictions(model, loader, dataset, device, args):
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            pred, true = process_batch(
                model,
                dataset,
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                device,
                args.pred_len,
                args.label_len,
                args.features,
            )
            pred = pred.detach().cpu().numpy()
            true = true.detach().cpu().numpy()
            preds.append(pred)
            trues.append(true)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return preds, trues


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
        print(f"RNG seed={args.seed}")
    data_dir = station_dir(args)
    out_dir = results_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_export(args, data_dir)

    with open(data_dir / "metadata.json", encoding="utf-8") as f:
        export_meta = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set, test_set, train_loader, val_loader, test_loader = (
        build_loaders(args, data_dir)
    )

    export_cols = (
        export_meta.get("export_cols")
        or export_meta.get("features")
        or [
            c
            for c in (export_meta.get("columns") or [])
            if c not in {"date", "time"}
        ]
    )
    if not export_cols:
        raise KeyError(
            "metadata.json must contain export_cols, features, or columns"
        )
    n_cols = len(export_cols)
    enc_in, dec_in, c_out = n_channels(args.features, n_cols)

    model = Informer(
        enc_in=enc_in,
        dec_in=dec_in,
        c_out=c_out,
        seq_len=args.seq_len,
        label_len=args.label_len,
        out_len=args.pred_len,
        factor=args.factor,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        attn=args.attn,
        embed=args.embed,
        freq=args.freq,
        device=device,
    ).float().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    checkpoint_dir = out_dir / "checkpoints"
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    print(
        f"Informer2020 | {args.station} | {args.feature_config} | "
        f"{args.features} | device={device}"
    )
    print(
        f"  train={len(train_set)} val={len(val_set)} test={len(test_set)} windows"
    )
    print(f"  borders: {train_set.borders}")

    history = []
    for epoch in range(args.train_epochs):
        t0 = time.time()
        train_losses = []
        model.train()
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            optimizer.zero_grad()
            pred, true = process_batch(
                model,
                train_set,
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                device,
                args.pred_len,
                args.label_len,
                args.features,
            )
            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = evaluate(model, val_loader, val_set, device, args, criterion)
        test_loss = evaluate(model, test_loader, test_set, device, args, criterion)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "elapsed_s": round(time.time() - t0, 1),
            }
        )
        print(
            f"Epoch {epoch + 1}/{args.train_epochs} | "
            f"train={train_loss:.4f} val={val_loss:.4f} test={test_loss:.4f} | "
            f"{history[-1]['elapsed_s']}s"
        )

        early_stopping(val_loss, model, checkpoint_dir)
        adjust_learning_rate(
            optimizer, epoch + 1, args.learning_rate, args.lradj
        )
        if early_stopping.early_stop:
            print("Early stopping")
            break

    best_path = checkpoint_dir / "checkpoint.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    preds, trues = collect_predictions(model, test_loader, test_set, device, args)
    # Keep scaled metrics for training parity
    mae_s, mse_s, rmse_s, mape_s = metric(preds, trues)
    metrics_scaled = metric_hourly_and_ma24(preds, trues)

    # Original units (ug/m3 etc.): invert last channel of the fitted scaler
    preds_raw = test_set.inverse_transform(preds)
    trues_raw = test_set.inverse_transform(trues)
    metrics_raw = metric_hourly_and_ma24(preds_raw, trues_raw)

    np.save(out_dir / "test_preds_hourly.npy", preds_raw)
    np.save(out_dir / "test_trues_hourly.npy", trues_raw)
    if preds_raw.shape[1] >= 48:
        preds_ma24 = moving_average_first_24h(preds_raw)
        trues_ma24 = moving_average_first_24h(trues_raw)
        np.save(out_dir / "test_preds_ma24_d2.npy", moving_average_second_24h(preds_raw))
        np.save(out_dir / "test_trues_ma24_d2.npy", moving_average_second_24h(trues_raw))
    else:
        preds_ma24 = preds_raw.mean(axis=1, keepdims=True)
        trues_ma24 = trues_raw.mean(axis=1, keepdims=True)
    np.save(out_dir / "test_preds_ma24.npy", preds_ma24)
    np.save(out_dir / "test_trues_ma24.npy", trues_ma24)

    full_target = None
    csv_path = informer2020_station_dir(
        args.zone, args.station, args.feature_config
    ) / "data.csv"
    if not csv_path.exists():
        csv_path = (
            Path(args.data_root)
            / f"zone_{args.zone}"
            / "informer2020"
            / args.station.replace(" ", "_").replace("(", "").replace(")", "")
            / args.feature_config
            / "data.csv"
        )
    if csv_path.exists():
        import pandas as pd

        full_target = pd.read_csv(csv_path)[args.target].to_numpy()

    plot_arts: dict = {}
    try:
        plot_arts = save_run_plots(
            out_dir,
            history=history,
            preds_ma24=preds_ma24,
            trues_ma24=trues_ma24,
            target_series=full_target,
            borders=train_set.borders,
            station=args.station,
            feature_config=args.feature_config,
            contaminant=args.contaminant,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: plots skipped ({exc})")

    results = {
        "station": args.station,
        "zone": args.zone,
        "contaminant": args.contaminant,
        "feature_config": args.feature_config,
        "features_mode": args.features,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "label_len": args.label_len,
        "pred_len": args.pred_len,
        "forecast_design": {
            "at_hour_h": "model outputs hours h+1 ... h+pred_len in one forward pass",
            "ma24": (
                "mean(pred[h+1], ..., pred[h+24]); at pred_len>=48 also ma24_d2 "
                "in test_metrics_raw"
            ),
        },
        "input_cols": export_meta.get("input_cols", export_meta.get("features")),
        "borders": train_set.borders,
        "n_train_windows": len(train_set),
        "n_val_windows": len(val_set),
        "n_test_windows": len(test_set),
        "history": history,
        "test_metrics_scaled": {
            "mae": float(mae_s),
            "mse": float(mse_s),
            "rmse": float(rmse_s),
            "mape": float(mape_s),
            "hourly": metrics_scaled["hourly"],
            "ma24": metrics_scaled["ma24"],
        },
        "test_metrics_raw": {
            "hourly": metrics_raw["hourly"],
            "ma24": metrics_raw["ma24"],
            **(
                {
                    "ma24_d1": metrics_raw["ma24_d1"],
                    "ma24_d2": metrics_raw["ma24_d2"],
                }
                if "ma24_d1" in metrics_raw
                else {}
            ),
        },
        "artifacts": {
            "test_preds_hourly": str((out_dir / "test_preds_hourly.npy").as_posix()),
            "test_trues_hourly": str((out_dir / "test_trues_hourly.npy").as_posix()),
            "test_preds_ma24": str((out_dir / "test_preds_ma24.npy").as_posix()),
            "test_trues_ma24": str((out_dir / "test_trues_ma24.npy").as_posix()),
            **plot_arts,
        },
        "best_val_loss": early_stopping.val_loss_min,
        "checkpoint": str(best_path.as_posix()),
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(
        f"\nTest hourly (raw): MAE={metrics_raw['hourly']['mae']:.4f} "
        f"RMSE={metrics_raw['hourly']['rmse']:.4f}"
    )
    print(
        f"Test MA24   (raw): MAE={metrics_raw['ma24']['mae']:.4f} "
        f"RMSE={metrics_raw['ma24']['rmse']:.4f}"
    )
    print(f"Resultados: {results_path}")


if __name__ == "__main__":
    main()
