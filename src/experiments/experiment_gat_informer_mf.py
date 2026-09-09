"""
Train / evaluate GAT-Informer multi-feature (GAT-MF) on zone-2 PM10 packs.

Parallel to experiment_gat_informer.py (v1, target-only). Does not touch v1.

Usage:
  python src/experiments/experiment_gat_informer_mf.py --feature-config F3 --max-epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.feature_configs import ABLATION_LEVELS
from src.models.gat_informer.zone2_train_mf import train_and_eval
from src.utils.paths import (
    gat_informer_mf_pm10graph_data_dir,
    gat_informer_mf_pm10graph_results_dir,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GAT-Informer-MF zone-2 experiment")
    p.add_argument("--zone", type=int, default=2)
    p.add_argument(
        "--feature-config",
        default="F1",
        choices=list(ABLATION_LEVELS),
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Override pack dir (default: gat_informer_mf/{F*})",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Override results dir (default: results/gat_informer_mf/...)",
    )
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--num-layer", type=int, default=2)
    p.add_argument("--no-cross", action="store_true", help="Disable cross-attn fusion")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = (
        Path(args.data_dir)
        if args.data_dir
        else gat_informer_mf_pm10graph_data_dir(
            args.zone,
            args.feature_config,
            seq_len=args.seq_len,
            horizon=args.horizon,
        )
    )
    if not any(data_dir.glob("data*.npz")):
        raise SystemExit(
            f"Missing GAT-MF pack: {data_dir}\n"
            f"Export with: python src/data/export_gat_informer_mf_zone2.py "
            f"--feature-configs {args.feature_config}"
        )
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else gat_informer_mf_pm10graph_results_dir(
            args.zone,
            args.feature_config,
            seq_len=args.seq_len,
            horizon=args.horizon,
        )
    )
    print("datapath :", data_dir)
    print("results  :", out_dir)
    train_and_eval(
        data_dir=data_dir,
        out_dir=out_dir,
        seq_len=args.seq_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        seed=args.seed,
        d_model=args.d_model,
        n_heads=args.n_heads,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_layer=args.num_layer,
        if_cross=not args.no_cross,
    )
    print(f"Results: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
