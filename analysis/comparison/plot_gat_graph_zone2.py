"""Visualize the meteo top-5 adjacency used by GAT / GAT-MF / GNN→Informer (zone 2).

Reads the graph baked into a gat_informer_mf NPZ (same Adj for all GAT lines).

Usage:
  python analysis/comparison/plot_gat_graph_zone2.py
  python analysis/comparison/plot_gat_graph_zone2.py --out-dir results/comparison/zone_2/all_models/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _short_label(name: str, max_len: int = 12) -> str:
    s = str(name).replace("(Monte)", "").replace("(Puerto)", "").replace("(BBIZI2)", "BB")
    s = " ".join(s.split())
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def load_graph_from_npz(npz_path: Path) -> tuple[np.ndarray, list[str]]:
    raw = np.load(npz_path, allow_pickle=True)
    adj = np.asarray(raw["graph"], dtype=float)
    meta_path = npz_path.parent / "metadata.json"
    stations: list[str]
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stations = list(meta.get("stations") or [])
    else:
        stations = [f"n{i}" for i in range(adj.shape[0])]
    if len(stations) != adj.shape[0]:
        stations = [f"n{i}" for i in range(adj.shape[0])]
    return adj, stations


def plot_adjacency_heatmap(
    adj: np.ndarray,
    stations: list[str],
    out_path: Path,
    title: str = "GAT graph · adjacency (meteo top-5, zone 2)",
) -> Path:
    labels = [_short_label(s) for s in stations]
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(adj, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("peso (Pearson / self=1)")
    # off-diagonal edge count
    n = adj.shape[0]
    n_edges = int(((adj > 0) & ~np.eye(n, dtype=bool)).sum() // 2)
    ax.text(
        0.0,
        -0.12,
        f"N={n} nodos · ~{n_edges} aristas undirected (diag=1)",
        transform=ax.transAxes,
        fontsize=9,
        color="#444",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_graph_network(
    adj: np.ndarray,
    stations: list[str],
    out_path: Path,
    title: str = "GAT graph · network (edges w>0, excl. self)",
    min_weight: float = 1e-6,
) -> Path:
    n = adj.shape[0]
    # Spring layout from weighted Laplacian-ish distances
    try:
        import networkx as nx

        g = nx.Graph()
        for i, name in enumerate(stations):
            g.add_node(i, label=_short_label(name, 14))
        for i in range(n):
            for j in range(i + 1, n):
                w = float(adj[i, j])
                if w > min_weight:
                    g.add_edge(i, j, weight=w)
        pos = nx.spring_layout(g, weight="weight", seed=42, k=1.8 / max(n**0.5, 1))
        fig, ax = plt.subplots(figsize=(10, 8))
        weights = [g[u][v]["weight"] for u, v in g.edges()]
        w_min, w_max = (min(weights), max(weights)) if weights else (0.0, 1.0)
        widths = [
            0.5 + 3.0 * ((w - w_min) / (w_max - w_min + 1e-9)) for w in weights
        ]
        nx.draw_networkx_edges(
            g, pos, ax=ax, width=widths, edge_color="#888888", alpha=0.7
        )
        nx.draw_networkx_nodes(
            g, pos, ax=ax, node_color="#264653", node_size=520, alpha=0.95
        )
        nx.draw_networkx_labels(
            g,
            pos,
            labels={i: g.nodes[i]["label"] for i in g.nodes},
            font_size=7,
            font_color="white",
            ax=ax,
        )
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return out_path
    except ImportError:
        # Fallback: circular layout without networkx
        fig, ax = plt.subplots(figsize=(10, 8))
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xy = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        for i in range(n):
            for j in range(i + 1, n):
                w = float(adj[i, j])
                if w <= min_weight:
                    continue
                lw = 0.4 + 2.5 * w
                ax.plot(
                    [xy[i, 0], xy[j, 0]],
                    [xy[i, 1], xy[j, 1]],
                    color="#888888",
                    lw=lw,
                    alpha=0.65,
                    zorder=1,
                )
        ax.scatter(xy[:, 0], xy[:, 1], s=420, c="#264653", zorder=2)
        for i, name in enumerate(stations):
            ax.text(
                xy[i, 0] * 1.12,
                xy[i, 1] * 1.12,
                _short_label(name, 14),
                ha="center",
                va="center",
                fontsize=7,
            )
        ax.set_title(title + " (circular; instala networkx para spring layout)")
        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zone", type=int, default=2)
    p.add_argument("--feature-config", default="F1")
    p.add_argument(
        "--npz",
        default=None,
        help="Default: data/training/zone_{z}/gat_informer_mf/{F}/data48.npz",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Default: results/comparison/zone_{z}/all_models/figures",
    )
    args = p.parse_args()

    npz = (
        Path(args.npz)
        if args.npz
        else ROOT
        / f"data/training/zone_{args.zone}/gat_informer_mf/{args.feature_config}/data48.npz"
    )
    if not npz.exists():
        # fallback v1 pack
        npz = (
            ROOT
            / f"data/training/zone_{args.zone}/gat_informer/{args.feature_config}/data48.npz"
        )
    if not npz.exists():
        raise SystemExit(f"Missing NPZ with graph: {npz}")

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else ROOT / f"results/comparison/zone_{args.zone}/all_models/figures"
    )
    adj, stations = load_graph_from_npz(npz)
    h = plot_adjacency_heatmap(
        adj, stations, out_dir / "fig_gat_graph_adjacency.png"
    )
    n = plot_graph_network(adj, stations, out_dir / "fig_gat_graph_network.png")
    print("stations:", stations)
    print("wrote", h)
    print("wrote", n)


if __name__ == "__main__":
    main()
