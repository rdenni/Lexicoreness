#!/usr/bin/env python3
"""
Extract the largest connected component (GCC) of a graph and save it
in the same format so all pipeline scripts can be run on it unchanged.

Output is saved to {graphs_dir}/{output_name}/ with files:
  {output_name}_edges.txt  — space-separated edge list
  {output_name}_groups.txt — space-separated node-group pairs

Example:
  python src/extract_gcc.py --graph iphone_samsung --graphs-dir data/graphs/
  # produces data/graphs/iphone_samsung_gcc/iphone_samsung_gcc_edges.txt
  #          data/graphs/iphone_samsung_gcc/iphone_samsung_gcc_groups.txt
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))
from methods import load_graph


def extract_gcc(graph_name: str, graphs_dir: Path, output_name: str) -> None:
    graph_dir = graphs_dir / graph_name
    edge_file = graph_dir / f"{graph_name}_edges.txt"
    group_file = graph_dir / f"{graph_name}_groups.txt"

    if not edge_file.exists():
        print(f"Edge file not found: {edge_file}")
        sys.exit(1)
    if not group_file.exists():
        print(f"Group file not found: {group_file}")
        sys.exit(1)

    print(f"Loading graph: {graph_name}")
    G, group_assignments = load_graph(str(edge_file), str(group_file))

    n_orig = G.number_of_nodes()
    m_orig = G.number_of_edges()
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    n_components = len(components)

    print(f"  Nodes:      {n_orig}")
    print(f"  Edges:      {m_orig}")
    print(f"  Components: {n_components}")
    if n_components > 1:
        sizes = [len(c) for c in components]
        print(f"  Component sizes: {sizes[:10]}{'...' if len(sizes) > 10 else ''}")

    gcc_nodes = components[0]
    gcc = G.subgraph(gcc_nodes).copy()

    n_gcc = gcc.number_of_nodes()
    m_gcc = gcc.number_of_edges()
    n_dropped = n_orig - n_gcc
    m_dropped = m_orig - m_gcc

    print(f"\nGCC:")
    print(f"  Nodes: {n_gcc} ({n_gcc / n_orig:.1%} of original, {n_dropped} dropped)")
    print(f"  Edges: {m_gcc} ({m_gcc / m_orig:.1%} of original, {m_dropped} dropped)")

    # Group distribution in GCC
    gcc_groups = Counter(group_assignments[node] for node in gcc_nodes)
    print(f"  Groups: {dict(sorted(gcc_groups.items()))}")

    if n_components == 1:
        print("\nGraph is already fully connected — GCC equals the full graph.")

    # Output paths
    out_dir = graphs_dir / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_edge_file = out_dir / f"{output_name}_edges.txt"
    out_group_file = out_dir / f"{output_name}_groups.txt"

    # Save edges
    with open(out_edge_file, 'w', encoding='utf-8') as f:
        for u, v in gcc.edges():
            f.write(f"{u} {v}\n")

    # Save groups (only nodes in GCC, sorted for reproducibility)
    with open(out_group_file, 'w', encoding='utf-8') as f:
        for node in sorted(gcc.nodes()):
            f.write(f"{node} {group_assignments[node]}\n")

    print(f"\nSaved to: {out_dir}")
    print(f"  {out_edge_file.name}")
    print(f"  {out_group_file.name}")
    print(f"\nRun pipeline with: --graph {output_name} --graphs-dir {graphs_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract the largest connected component of a graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_gcc.py --graph iphone_samsung --graphs-dir data/graphs/
  python extract_gcc.py --graph blogs --graphs-dir data/graphs/ --output-name blogs_gcc
        """
    )
    parser.add_argument(
        "--graph",
        required=True,
        help="Graph name (e.g. iphone_samsung)"
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=Path("data/graphs"),
        help="Directory containing graph subdirectories (default: data/graphs)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Name for the output graph (default: {graph}_gcc)"
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    graphs_dir = project_root / args.graphs_dir
    output_name = args.output_name or f"{args.graph}_gcc"

    extract_gcc(args.graph, graphs_dir, output_name)


if __name__ == "__main__":
    main()
