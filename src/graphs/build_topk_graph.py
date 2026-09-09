import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================

GRAPHS_DIR = Path("data/graphs")
TARGET_ZONES = [1, 3, 4, 5, 6, 7, 8]
K = 5


def build_topk(zone: int, k: int = K) -> Path:
    input_graph = GRAPHS_DIR / f"meteo_graph_zone_{zone}.csv"
    output_graph = GRAPHS_DIR / f"meteo_graph_zone_{zone}_top5.csv"

    if not input_graph.exists():
        raise FileNotFoundError(f"No existe {input_graph}")

    graph = pd.read_csv(input_graph)

    if graph.empty:
        graph_topk = pd.DataFrame(columns=["source", "target", "weight"])
        graph_topk.to_csv(output_graph, index=False)
        print(f"Zona {zone}: grafo vacio -> {output_graph} (0 aristas)")
        return output_graph

    # Undirected -> bidirectional
    reverse_graph = graph.rename(
        columns={
            "source": "target",
            "target": "source",
        }
    )
    graph = pd.concat([graph, reverse_graph], ignore_index=True)

    topk_edges = []
    for node in graph["source"].unique():
        neighbors = (
            graph[graph["source"] == node]
            .sort_values("weight", ascending=False)
            .head(k)
        )
        topk_edges.append(neighbors)

    graph_topk = pd.concat(topk_edges, ignore_index=True)
    graph_topk.to_csv(output_graph, index=False)

    print(f"Zona {zone}: K={k} -> {output_graph} ({len(graph_topk)} aristas)")
    return output_graph


if __name__ == "__main__":
    print()
    print("===================================")
    print("TOP-K GRAPHS")
    print("===================================")
    print()

    for zone in TARGET_ZONES:
        build_topk(zone)

    print()
    print("Hecho.")
