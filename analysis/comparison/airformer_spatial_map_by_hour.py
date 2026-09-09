"""
Geographic map of Bajo Nervión: link stations by AirFormer spatial attention per hour.

Reads A_by_hour.npy from spatial_attention/ (run airformer_spatial_attention_by_hour.py first).

Usage:
  python analysis/comparison/airformer_spatial_map_by_hour.py
  python analysis/comparison/airformer_spatial_map_by_hour.py --top-k 3 --min-weight 0.05
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AF spatial attention geo map by hour")
    p.add_argument(
        "--attn-dir",
        type=Path,
        default=ROOT / "results/comparison/zone_2/spatial_attention",
    )
    p.add_argument("--zone", type=int, default=2)
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Keep top-k outgoing neighbors per station (excl. self)",
    )
    p.add_argument(
        "--min-weight",
        type=float,
        default=0.04,
        help="Drop edges below this attention weight",
    )
    p.add_argument(
        "--out-html",
        type=Path,
        default=None,
        help="Default: <attn-dir>/spatial_map_interactive.html",
    )
    return p.parse_args()


def station_key(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_").upper()
    if "DIAZ" in s and "HARO" in s:
        return "M_DIAZ_HARO"
    return s


def short_label(name: str, max_len: int = 16) -> str:
    s = (
        str(name)
        .replace("(Monte)", "")
        .replace("(Puerto)", "")
        .replace("(BBIZI2)", "BB")
    )
    s = " ".join(s.split())
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def load_coords(zone: int, stations: list[str]) -> pd.DataFrame:
    meta = pd.read_csv(
        ROOT / "data" / "metadata" / "stations_metadata.csv",
        sep=";",
        dtype=str,
    )
    meta["zone_int"] = pd.to_numeric(meta["zone"], errors="coerce")
    meta = meta[meta["zone_int"] == zone].copy()
    meta["lat"] = meta["lat"].str.replace(",", ".", regex=False).astype(float)
    meta["lon"] = meta["lon"].str.replace(",", ".", regex=False).astype(float)
    meta["sk"] = meta["name"].map(station_key)

    rows = []
    for i, name in enumerate(stations):
        sk = station_key(name)
        hit = meta[meta["sk"] == sk]
        if hit.empty:
            # fuzzy: startswith
            hit = meta[meta["sk"].str.startswith(sk[:6])]
        if hit.empty:
            raise SystemExit(f"No coords for station {name!r} (key={sk})")
        r = hit.iloc[0]
        rows.append(
            {
                "idx": i,
                "station": name,
                "label": short_label(name),
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "town": r.get("town", ""),
            }
        )
    return pd.DataFrame(rows)


def edges_for_hour(
    A: np.ndarray, top_k: int, min_weight: float
) -> list[tuple[int, int, float]]:
    """Directed edges i→j from attention row i (excl. self)."""
    n = A.shape[0]
    edges: list[tuple[int, int, float]] = []
    for i in range(n):
        row = A[i].copy()
        row[i] = -1.0  # exclude self
        order = np.argsort(row)[::-1]
        kept = 0
        for j in order:
            w = float(row[j])
            if w < min_weight or kept >= top_k:
                break
            if j == i:
                continue
            edges.append((i, int(j), w))
            kept += 1
    return edges


def build_html(
    A_by_hour: np.ndarray,
    sector_by_hour: np.ndarray | None,
    coords: pd.DataFrame,
    top_k: int,
    min_weight: float,
    out_html: Path,
    title: str,
) -> None:
    """Build HTML with a working hour slider (visibility toggle, not frames).

    Mapbox+frames often fails in browsers. We overlay 24 edge traces and
    24 node-color traces and toggle visibility. Node color encodes the
    'neighbors' sector weight (varies by hour); edge topology is nearly
    stable under dartboard projection — widths still update per hour.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise SystemExit("pip install plotly") from e

    lats = coords["lat"].tolist()
    lons = coords["lon"].tolist()
    labels = coords["label"].tolist()
    n_hours = A_by_hour.shape[0]
    n_nodes = len(labels)

    if sector_by_hour is None:
        # fallback: mean off-diag outgoing mass
        sector_by_hour = np.zeros((n_hours, n_nodes, 3))
        for h in range(n_hours):
            for i in range(n_nodes):
                row = A_by_hour[h, i].copy()
                row[i] = 0.0
                sector_by_hour[h, i, 1] = float(row.sum())

    # shared color scale for neighbor-sector attention
    neigh = sector_by_hour[:, :, 1]
    cmin, cmax = float(neigh.min()), float(neigh.max())
    if cmax - cmin < 1e-6:
        cmax = cmin + 1e-3

    def edge_xy(h: int) -> tuple[list, list, list, list]:
        edges = edges_for_hour(A_by_hour[h], top_k=top_k, min_weight=min_weight)
        elat: list = []
        elon: list = []
        etext: list = []
        widths: list = []
        for i, j, w in edges:
            elat += [lats[i], lats[j], None]
            elon += [lons[i], lons[j], None]
            etext += [
                f"{labels[i]} → {labels[j]}  w={w:.3f}",
                f"{labels[i]} → {labels[j]}  w={w:.3f}",
                None,
            ]
            widths.append(w)
        # single width for the trace (mean of edges that hour)
        mean_w = float(np.mean(widths)) if widths else 0.05
        line_w = 1.5 + 40.0 * mean_w  # amplify so hour diffs are visible
        return elat, elon, etext, line_w

    traces: list = []
    # traces 0..23 = edges per hour; 24..47 = nodes per hour
    for h in range(n_hours):
        elat, elon, etext, line_w = edge_xy(h)
        traces.append(
            go.Scattermapbox(
                lat=elat,
                lon=elon,
                mode="lines",
                line=dict(width=line_w, color="rgba(30, 90, 180, 0.7)"),
                hoverinfo="text",
                text=etext,
                name=f"edges_{h:02d}",
                visible=(h == 0),
                showlegend=False,
            )
        )
    for h in range(n_hours):
        cvals = neigh[h].tolist()
        hover = [
            (
                f"{labels[i]} @ {h:02d}:00<br>"
                f"sector self={sector_by_hour[h, i, 0]:.3f} · "
                f"neighbors={sector_by_hour[h, i, 1]:.3f} · "
                f"rest={sector_by_hour[h, i, 2]:.3f}"
            )
            for i in range(n_nodes)
        ]
        traces.append(
            go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode="markers+text",
                marker=dict(
                    size=16,
                    color=cvals,
                    colorscale="YlOrRd",
                    cmin=cmin,
                    cmax=cmax,
                    colorbar=dict(title="attn<br>neighbors", x=1.02)
                    if h == 0
                    else None,
                    showscale=(h == 0),
                ),
                text=labels,
                textposition="top right",
                textfont=dict(size=10, color="#222"),
                hovertext=hover,
                hoverinfo="text",
                name=f"nodes_{h:02d}",
                visible=(h == 0),
                showlegend=False,
            )
        )

    fig = go.Figure(data=traces)

    # visibility: for hour h → edges[h]=True, nodes[24+h]=True, rest False
    steps = []
    for h in range(n_hours):
        vis = [False] * (2 * n_hours)
        vis[h] = True
        vis[n_hours + h] = True
        steps.append(
            dict(
                method="update",
                args=[
                    {"visible": vis},
                    {
                        "title": (
                            f"{title} · {h:02d}:00 · top-{top_k} "
                            f"(color = atención a vecinos)"
                        )
                    },
                ],
                label=f"{h:02d}",
            )
        )

    fig.update_layout(
        title=f"{title} · 00:00 · top-{top_k} (color = atención a vecinos)",
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=float(np.mean(lats)), lon=float(np.mean(lons))),
            zoom=10.5,
        ),
        margin=dict(l=0, r=80, t=60, b=80),
        height=740,
        width=1050,
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Hora: ", "suffix": ":00"},
                "pad": {"t": 40, "b": 10},
                "steps": steps,
            }
        ],
        annotations=[
            dict(
                text=(
                    "Mueve el slider: el color de cada estación cambia con la hora "
                    "(peso del sector «neighbors»). Las líneas (top-k) casi no cambian "
                    "de destino — el dartboard agrupa vecinos con el mismo peso."
                ),
                xref="paper",
                yref="paper",
                x=0,
                y=-0.12,
                showarrow=False,
                font=dict(size=11, color="#444"),
                align="left",
            )
        ],
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", auto_play=False)
    print(f"wrote {out_html}")


def main() -> None:
    args = parse_args()
    attn_dir = Path(args.attn_dir)
    A_path = attn_dir / "A_by_hour.npy"
    meta_path = attn_dir / "meta.json"
    if not A_path.exists() or not meta_path.exists():
        raise SystemExit(
            f"Missing {A_path} or {meta_path}. "
            "Run airformer_spatial_attention_by_hour.py first."
        )
    A_by_hour = np.load(A_path)
    sector_path = attn_dir / "sector_attn_by_hour.npy"
    sector_by_hour = np.load(sector_path) if sector_path.exists() else None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stations = list(meta["stations"])
    coords = load_coords(args.zone, stations)
    out_html = args.out_html or (attn_dir / "spatial_map_interactive.html")
    build_html(
        A_by_hour=np.nan_to_num(A_by_hour, nan=0.0),
        sector_by_hour=sector_by_hour,
        coords=coords,
        top_k=args.top_k,
        min_weight=args.min_weight,
        out_html=Path(out_html),
        title="AirFormer F3 · Bajo Nervión · atención espacial",
    )
    edges = edges_for_hour(A_by_hour[23], args.top_k, args.min_weight)
    rows = [
        {"hour": 23, "src": stations[i], "dst": stations[j], "weight": w}
        for i, j, w in edges
    ]
    pd.DataFrame(rows).to_csv(attn_dir / "edges_hour_23.csv", index=False)
    print(f"example edges @23:00 -> {attn_dir / 'edges_hour_23.csv'}")


if __name__ == "__main__":
    main()
