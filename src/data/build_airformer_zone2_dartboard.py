"""Build a trivial dartboard partition for zone-2 (N=18).

assignment: (N, N, S) soft assignment of nodes into S sectors
mask: (N, S) True = sector masked out for that node

Uses adjacency rings: sector0=self, sector1=neighbors, sector2=rest.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DATA = (
    ROOT / "src" / "models" / "airformer" / "_upstream" / "data" / "local_partition" / "zone2_18"
)
DEFAULT_ADJ = (
    ROOT
    / "data"
    / "training"
    / "zone_2"
    / "airformer"
    / "ZONE2_PM10"
    / "F1"
    / "adj_mx.pkl"
)


def build_zone2_dartboard(adj_path: Path | None = None, out_dir: Path | None = None) -> Path:
    adj_path = adj_path or DEFAULT_ADJ
    out_dir = out_dir or UPSTREAM_DATA
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(adj_path, "rb") as f:
        _ids, _id_map, adj = pickle.load(f)
    adj = np.asarray(adj, dtype=np.float64)
    n = adj.shape[0]
    assert n == 18, f"Expected 18 nodes, got {n}"

    # binary neighbors (ignore self)
    neigh = (adj > 0).astype(np.float64)
    np.fill_diagonal(neigh, 0.0)

    s = 3
    assignment = np.zeros((n, n, s), dtype=np.float64)
    mask = np.zeros((n, s), dtype=np.float64)

    for i in range(n):
        # sector 0: self
        assignment[i, i, 0] = 1.0
        # sector 1: graph neighbors
        row = neigh[i]
        if row.sum() > 0:
            assignment[i, :, 1] = row / row.sum()
        else:
            mask[i, 1] = 1.0
        # sector 2: everyone else (non-neighbor, non-self)
        rest = np.ones(n, dtype=np.float64)
        rest[i] = 0.0
        rest = rest * (1.0 - (row > 0).astype(np.float64))
        if rest.sum() > 0:
            assignment[i, :, 2] = rest / rest.sum()
        else:
            mask[i, 2] = 1.0

    np.save(out_dir / "assignment.npy", assignment)
    np.save(out_dir / "mask.npy", mask)
    meta = out_dir / "README.txt"
    meta.write_text(
        "Trivial zone-2 dartboard (N=18, S=3): self / neighbors / rest.\n"
        "Not the China km-rings from the AirFormer paper.\n",
        encoding="utf-8",
    )
    print(f"Wrote dartboard to {out_dir}")
    print(f"  assignment {assignment.shape}  mask {mask.shape}")
    return out_dir


if __name__ == "__main__":
    build_zone2_dartboard()
