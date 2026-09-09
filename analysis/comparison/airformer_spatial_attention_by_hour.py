"""
Extract AirFormer DS-MSA spatial attention by hour-of-day and compare to GAT Adj.

AirFormer attends over dartboard *sectors* (zone2: self / neighbors / rest).
We project sector attention to an approximate N×N station matrix via the
assignment tensor, then average by forecast-origin hour (0–23).

Usage (from repo root, CPU OK for F3):
  python analysis/comparison/airformer_spatial_attention_by_hour.py
  python analysis/comparison/airformer_spatial_attention_by_hour.py --max-batches 20  # smoke

Outputs:
  results/comparison/zone_2/spatial_attention/
    A_by_hour.npy          # (24, N, N)
    A_mean.npy             # (N, N)
    A_gat.npy              # (N, N)
    sector_attn_by_hour.npy
    meta.json
    spatial_attention_interactive.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "src" / "models" / "airformer" / "_upstream"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AirFormer spatial attn by hour")
    p.add_argument("--feature-config", default="F3", choices=["F1", "F2", "F3", "F4", "F5"])
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--zone", type=int, default=2)
    p.add_argument("--dataset", default="ZONE2_PM10")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=0, help="0 = all test")
    p.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: results/comparison/zone_2/spatial_attention",
    )
    return p.parse_args()


def hour_from_sincos(sin_h: np.ndarray, cos_h: np.ndarray) -> np.ndarray:
    """Recover hour-of-day 0..23 from hour_sin / hour_cos (unscaled)."""
    ang = np.arctan2(sin_h, cos_h)  # [-pi, pi]
    h = np.mod(np.round(ang * 24 / (2 * np.pi)), 24).astype(np.int64)
    return h


def sectors_to_adjacency(
    attn_ns: np.ndarray, assignment: np.ndarray
) -> np.ndarray:
    """attn [N, S], assignment [N, N, S] -> A [N, N] row-stochastic."""
    # A[i,j] = sum_s attn[i,s] * assignment[i,j,s]
    a = np.einsum("is,ijs->ij", attn_ns, assignment)
    row = a.sum(axis=1, keepdims=True)
    row = np.where(row > 1e-8, row, 1.0)
    return a / row


def short_label(name: str, max_len: int = 14) -> str:
    s = (
        str(name)
        .replace("(Monte)", "")
        .replace("(Puerto)", "")
        .replace("(BBIZI2)", "BB")
    )
    s = " ".join(s.split())
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def load_gat_adj(stations: list[str]) -> np.ndarray:
    npz = (
        ROOT
        / "data"
        / "training"
        / "zone_2"
        / "gat_informer_mf_pm10graph"
        / "F1"
        / "data48.npz"
    )
    if not npz.exists():
        # try F3
        npz = npz.parent.parent / "F3" / "data48.npz"
    raw = np.load(npz, allow_pickle=True)
    adj = np.asarray(raw["graph"], dtype=float)
    if adj.shape[0] != len(stations):
        raise SystemExit(f"GAT adj N={adj.shape[0]} vs stations={len(stations)}")
    # row-normalize off-diag+diag for fairer heatmap vs AF softmax rows
    row = adj.sum(axis=1, keepdims=True)
    row = np.where(row > 1e-8, row, 1.0)
    return adj / row


def build_interactive_html(
    A_by_hour: np.ndarray,
    A_gat: np.ndarray,
    A_mean: np.ndarray,
    stations: list[str],
    out_html: Path,
    title: str,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise SystemExit(
            "plotly required for interactive HTML: pip install plotly"
        ) from e

    labels = [short_label(s) for s in stations]
    n_hours = A_by_hour.shape[0]

    # Diff vs GAT (same row-norm scale)
    diff0 = A_by_hour[0] - A_gat
    zmax = float(max(A_by_hour.max(), A_gat.max(), 1e-6))
    dmax = float(max(abs(diff0).max(), abs(A_by_hour - A_gat).max(), 1e-6))

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "AirFormer Ā(hour) (projected)",
            "GAT Adj (row-norm)",
            "AF − GAT",
        ),
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Heatmap(
            z=A_by_hour[0],
            x=labels,
            y=labels,
            coloraxis="coloraxis",
            name="AF",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=A_gat,
            x=labels,
            y=labels,
            coloraxis="coloraxis",
            name="GAT",
            showscale=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            z=A_by_hour[0] - A_gat,
            x=labels,
            y=labels,
            coloraxis="coloraxis2",
            name="diff",
        ),
        row=1,
        col=3,
    )

    frames = []
    for h in range(n_hours):
        frames.append(
            go.Frame(
                name=str(h),
                data=[
                    go.Heatmap(z=A_by_hour[h], x=labels, y=labels, coloraxis="coloraxis"),
                    go.Heatmap(z=A_gat, x=labels, y=labels, coloraxis="coloraxis"),
                    go.Heatmap(
                        z=A_by_hour[h] - A_gat,
                        x=labels,
                        y=labels,
                        coloraxis="coloraxis2",
                    ),
                ],
                layout=go.Layout(
                    title_text=f"{title} · hour={h:02d}:00  (mean AF also in meta)"
                ),
            )
        )
    fig.frames = frames

    sliders = [
        {
            "active": 0,
            "currentvalue": {"prefix": "Hour of day: ", "suffix": ":00"},
            "pad": {"t": 50},
            "steps": [
                {
                    "args": [
                        [str(h)],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                        },
                    ],
                    "label": f"{h:02d}",
                    "method": "animate",
                }
                for h in range(n_hours)
            ],
        }
    ]

    fig.update_layout(
        title=f"{title} · hour=00:00",
        height=620,
        width=1400,
        sliders=sliders,
        coloraxis=dict(colorscale="YlOrRd", cmin=0, cmax=zmax, colorbar=dict(title="weight", x=0.30)),
        coloraxis2=dict(
            colorscale="RdBu_r",
            cmin=-dmax,
            cmax=dmax,
            colorbar=dict(title="Δ", x=1.0),
        ),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "y": 1.15,
                "x": 0.0,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 400, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        annotations=[
            dict(
                text=(
                    "AF: DS-MSA sector attn projected to stations via dartboard assignment. "
                    "GAT: fixed PM10 top-5 Adj (row-normalized). "
                    "Slider = forecast-origin hour-of-day (test set)."
                ),
                xref="paper",
                yref="paper",
                x=0,
                y=-0.18,
                showarrow=False,
                font=dict(size=11, color="#444"),
                align="left",
            )
        ],
        margin=dict(b=100),
    )
    # Also dump mean AF as a static companion note in HTML title attribute via meta file
    _ = A_mean  # saved separately; heatmap focuses on hourly
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", auto_play=False)
    print(f"wrote {out_html}")


def main() -> None:
    args = parse_args()
    print("start airformer_spatial_attention_by_hour", flush=True)
    out_dir = args.out_dir or (
        ROOT / "results" / "comparison" / f"zone_{args.zone}" / "spatial_attention"
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not UPSTREAM.is_dir():
        raise SystemExit(f"Missing upstream: {UPSTREAM}")

    # Upstream `src` must win
    sys.path.insert(0, str(UPSTREAM.resolve()))
    os.chdir(UPSTREAM)

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.airformer import AirFormer  # noqa: E402
    from src.utils.helper import check_device, get_num_nodes  # noqa: E402
    from src.utils.scaler import StandardScaler  # noqa: E402

    data_dir = (
        ROOT
        / "data"
        / "training"
        / f"zone_{args.zone}"
        / "airformer"
        / args.dataset
        / args.feature_config
    )
    if args.seq_len != 48:
        data_dir = (
            ROOT
            / "data"
            / "training"
            / f"zone_{args.zone}"
            / "airformer"
            / args.dataset
            / f"seq_len_{args.seq_len}"
            / args.feature_config
        )
    ckpt = (
        ROOT
        / "results"
        / "airformer"
        / f"zone_{args.zone}"
        / args.dataset
        / args.feature_config
        / "logs"
        / "final_model_0.pt"
    )
    if args.seq_len != 48:
        ckpt = (
            ROOT
            / "results"
            / "airformer"
            / f"zone_{args.zone}"
            / args.dataset
            / f"seq_len_{args.seq_len}"
            / args.feature_config
            / "logs"
            / "final_model_0.pt"
        )
    if not ckpt.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt}")

    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    stations = list(meta["stations"])
    features = list(meta["features"])
    hour_sin_i = features.index("hour_sin")
    hour_cos_i = features.index("hour_cos")
    input_dim = len(features)
    n_nodes = get_num_nodes(args.dataset)

    # Raw hours from unscaled test x (last lookback step = forecast origin)
    raw_test = np.load(data_dir / "test.npz")
    x_raw = raw_test["x"]  # [W, L, N, F]
    # use node 0 calendar (shared across nodes)
    sin_h = x_raw[:, -1, 0, hour_sin_i]
    cos_h = x_raw[:, -1, 0, hour_cos_i]
    hours = hour_from_sincos(sin_h, cos_h)

    # Scaled tensors like training
    train_x = np.load(data_dir / "train.npz")["x"]
    scalers = []
    x_test = x_raw.copy()
    y_test = raw_test["y"].copy()
    for i in range(1):  # output_dim=1 scales channel 0 only (same as helper)
        sc = StandardScaler(mean=train_x[..., i].mean(), std=train_x[..., i].std())
        scalers.append(sc)
        x_test[..., i] = sc.transform(x_test[..., i])
        y_test[..., i] = sc.transform(y_test[..., i])

    device = check_device(args.device)
    model = AirFormer(
        dropout=0.3,
        spatial_flag=True,
        stochastic_flag=False,
        hidden_channels=32,
        dartboard=4,
        end_channels=32 * 8,
        num_heads=2,
        name="airformer",
        dataset=args.dataset,
        device=device,
        num_nodes=n_nodes,
        seq_len=args.seq_len,
        horizon=24,
        input_dim=input_dim,
        output_dim=1,
    )
    state = torch.load(ckpt, map_location=device)
    try:
        model.load_state_dict(state)
    except TypeError:
        model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Collect SpatialAttention modules
    attn_modules = []
    for m in model.modules():
        if m.__class__.__name__ == "SpatialAttention":
            attn_modules.append(m)
    if not attn_modules:
        raise SystemExit("No SpatialAttention modules found (spatial_flag?)")

    assignment = attn_modules[0].assignment.detach().cpu().numpy()  # [N,N,S]
    n_s = assignment.shape[-1]

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Accumulators per hour
    sum_A = np.zeros((24, n_nodes, n_nodes), dtype=np.float64)
    sum_sec = np.zeros((24, n_nodes, n_s), dtype=np.float64)
    counts = np.zeros(24, dtype=np.int64)

    idx0 = 0
    seq_len = args.seq_len
    with torch.no_grad():
        for b_i, (xb, _) in enumerate(loader):
            if args.max_batches and b_i >= args.max_batches:
                break
            B = xb.shape[0]
            xb = xb.to(device)
            # AirFormer expects [B, L, N, C] — check forward
            _ = model(xb)

            # Each SpatialAttention saw [B*T, N, C]; last_attn [B*T, N, heads, S]
            # Average over layers & heads; take last lookback timestep only
            layer_atts = []
            for mod in attn_modules:
                a = mod.last_attn  # [B*T, N, H, S] — wait we reshaped to [B,N,H,S]
                # BUG: reshape used B from SpatialAttention which is B*T from DS_MSA!
                # So last_attn is actually [B*T, N, H, S]
                layer_atts.append(a.mean(dim=2).cpu().numpy())  # [B*T, N, S]
            attn_bt = np.mean(np.stack(layer_atts, axis=0), axis=0)  # [B*T, N, S]

            # Reshape to [B, T, N, S] and take t = -1
            attn_last = attn_bt.reshape(B, seq_len, n_nodes, n_s)[:, -1, :, :]

            batch_hours = hours[idx0 : idx0 + B]
            for i in range(B):
                h = int(batch_hours[i])
                sec = attn_last[i]  # [N, S]
                A = sectors_to_adjacency(sec, assignment)
                sum_A[h] += A
                sum_sec[h] += sec
                counts[h] += 1
            idx0 += B
            if (b_i + 1) % 20 == 0:
                print(f"  batch {b_i+1}/{len(loader)}  samples={idx0}")

    if counts.sum() == 0:
        raise SystemExit("No samples processed")

    A_by_hour = np.zeros_like(sum_A)
    sec_by_hour = np.zeros_like(sum_sec)
    for h in range(24):
        if counts[h] > 0:
            A_by_hour[h] = sum_A[h] / counts[h]
            sec_by_hour[h] = sum_sec[h] / counts[h]
        else:
            A_by_hour[h] = np.nan
            sec_by_hour[h] = np.nan

    # Overall mean (weighted by hour counts)
    A_mean = sum_A.sum(axis=0) / max(counts.sum(), 1)
    A_gat = load_gat_adj(stations)

    np.save(out_dir / "A_by_hour.npy", A_by_hour)
    np.save(out_dir / "A_mean.npy", A_mean)
    np.save(out_dir / "A_gat.npy", A_gat)
    np.save(out_dir / "sector_attn_by_hour.npy", sec_by_hour)
    np.save(out_dir / "hour_counts.npy", counts)

    # Correlation AF mean vs GAT (flatten, exclude nan)
    flat_af = A_mean.ravel()
    flat_gat = A_gat.ravel()
    corr = float(np.corrcoef(flat_af, flat_gat)[0, 1])

    meta_out = {
        "feature_config": args.feature_config,
        "seq_len": args.seq_len,
        "stations": stations,
        "n_samples_used": int(counts.sum()),
        "hour_counts": counts.tolist(),
        "corr_A_mean_vs_gat": corr,
        "checkpoint": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
        "note": (
            "A_by_hour[h] = mean projected DS-MSA attention at forecast origin hour h. "
            "Projection: A_ij = sum_s attn_is * assignment_ijs (row-normalized). "
            "GAT = PM10 top-5 graph row-normalized."
        ),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"samples={counts.sum()}  corr(AF_mean, GAT)={corr:.3f}")
    print(f"hour_counts min/max = {counts.min()}/{counts.max()}")

    build_interactive_html(
        A_by_hour=np.nan_to_num(A_by_hour, nan=0.0),
        A_gat=A_gat,
        A_mean=A_mean,
        stations=stations,
        out_html=out_dir / "spatial_attention_interactive.html",
        title=f"AirFormer {args.feature_config} L={args.seq_len} · spatial attn vs GAT",
    )
    print(f"Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
