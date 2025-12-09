import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np

ROOT = Path("..").resolve()  # your notebook folder
PATH_TRAIN = ROOT / "dataset" / "train.txt"
PATH_VAL = ROOT / "dataset" / "val.txt"
PATH_TEST = ROOT / "dataset" / "test.txt"


def load_split_whole_file(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        content = f.read()

    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        print(f"Failed to parse {path} as JSON.")
        exit(1)

    # we expect a *list of dicts*
    if not isinstance(obj, list):
        raise TypeError(f"Expected a list, got {type(obj)} from {path}")

    # one row per paper
    df = pd.json_normalize(obj)  # or pd.DataFrame(obj)
    return df


def build_citation_graph(df):
    """build directed citation graph from dataframe"""
    G = nx.DiGraph()

    paper_ids = set(df["publication_ID"].astype(int))

    # track which papers have in-network citations
    papers_with_in_network_citations = set()
    source_ids = set()
    target_ids = set()
    # add citation edges
    edge_count = 0
    for _, row in df.iterrows():
        source = int(row["publication_ID"])
        source_ids.add(source)
        if type(row["Citations"]) != str:
            continue

        citations = str(row["Citations"]).split(";")
        for cite in citations:
            try:
                target = int(cite)
                # only add edge if target is in our dataset
                if target in paper_ids:
                    G.add_edge(source, target)
                    papers_with_in_network_citations.add(source)

                    target_ids.add(target)
                    papers_with_in_network_citations.add(target)
                    edge_count += 1
            except ValueError:
                continue
    df = df[df["publication_ID"].astype(int).isin(papers_with_in_network_citations)]
    print(df["split"].value_counts())
    print("intersection", len(source_ids.intersection(target_ids)))
    print(f"papers with in-network citations: {len(papers_with_in_network_citations)}")
    print(
        f"papers filtered out: {len(paper_ids) - len(papers_with_in_network_citations)}"
    )

    return G


def analyze_graph(G):
    """print graph statistics"""
    print(f"\n=== graph statistics ===")
    print(f"total nodes: {G.number_of_nodes()}")
    print(f"total edges: {G.number_of_edges()}")
    print(f"is connected: {nx.is_weakly_connected(G)}")
    print(
        f"number of weakly connected components: {nx.number_weakly_connected_components(G)}"
    )

    # degree distribution
    degrees = [d for n, d in G.degree()]
    print(f"\navg degree: {np.mean(degrees):.2f}")
    print(f"median degree: {np.median(degrees):.2f}")
    print(f"max degree: {max(degrees)}")
    print(f"nodes with degree 0: {sum(1 for d in degrees if d == 0)}")

    # largest component
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    print(
        f"\nlargest component size: {len(largest_cc)} ({len(largest_cc)/G.number_of_nodes()*100:.1f}%)"
    )

    return degrees, largest_cc


def plot_full_graph(G):
    """visualize entire citation network"""

    subgraph = G

    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(subgraph, k=1.0, iterations=50, seed=42)

    # color by degree
    degrees = dict(subgraph.degree())
    node_colors = [degrees[node] for node in subgraph.nodes()]

    nx.draw_networkx_nodes(
        subgraph, pos, node_color=node_colors, node_size=30, cmap="viridis", alpha=0.7
    )
    nx.draw_networkx_edges(
        subgraph, pos, alpha=0.5, arrows=True, arrowsize=8, width=0.5
    )

    plt.title(
        f"full citation network ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(ROOT / "full_citation_network.png", dpi=300, bbox_inches="tight")
    print(f"saved full network visualization to {ROOT / 'full_citation_network.png'}")


if __name__ == "__main__":
    # load data
    print("loading data...")
    df_train = load_split_whole_file(PATH_TRAIN)
    df_val = load_split_whole_file(PATH_VAL)
    df_test = load_split_whole_file(PATH_TEST)

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    df = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # build graph
    print("\nbuilding citation graph...")
    G = build_citation_graph(df)

    # analyze
    degrees, largest_cc = analyze_graph(G)

    plot_full_graph(G)

    print("\ndone!")
