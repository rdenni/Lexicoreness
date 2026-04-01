#!/usr/bin/env python3
"""
Graph characteristics analysis tool.

Prints and saves useful statistics about a graph:
- Basic stats (nodes, edges, density, avg degree)
- Degree distribution with horizontal bar visualization
- Core decomposition distribution with horizontal bar visualization
- Connected components info
- Multipartiteness check
- Group statistics
- Clustering and diameter info
"""

import argparse
import sys
import os
import math
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, Optional

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))
from methods import load_graph


def bar(count: int, max_count: int, width: int = 40) -> str:
    """Create a horizontal bar like tqdm."""
    if max_count == 0:
        return ''
    filled = round(count / max_count * width)
    return '\u2588' * filled + '\u2591' * (width - filled)


def format_distribution(dist: Dict[int, int], label: str, max_rows: int = 40) -> str:
    """Format a value->count distribution with horizontal bars."""
    if not dist:
        return f"  No {label} data\n"

    lines = []
    sorted_keys = sorted(dist.keys())
    max_count = max(dist.values())
    total = sum(dist.values())

    # If too many unique values, bucket them
    if len(sorted_keys) > max_rows:
        lines.append(f"  ({len(sorted_keys)} unique values, showing bucketed view)\n")
        # Create logarithmic-ish buckets
        n_buckets = max_rows
        min_val, max_val = sorted_keys[0], sorted_keys[-1]
        if max_val - min_val < n_buckets:
            # Few enough values, show all
            pass
        else:
            bucket_size = max(1, (max_val - min_val + 1) // n_buckets)
            bucketed = Counter()
            for val, cnt in dist.items():
                bucket_start = ((val - min_val) // bucket_size) * bucket_size + min_val
                bucketed[bucket_start] += cnt
            # Re-format as ranges
            max_count = max(bucketed.values())
            for bstart in sorted(bucketed.keys()):
                bend = min(bstart + bucket_size - 1, max_val)
                cnt = bucketed[bstart]
                pct = cnt / total * 100
                if bstart == bend:
                    key_str = f"{bstart:>6}"
                else:
                    key_str = f"{bstart:>3}-{bend:<3}"
                lines.append(f"  {key_str} | {bar(cnt, max_count)} {cnt:>6} ({pct:5.1f}%)")
            lines.append(f"  {'Total':>7}: {total}")
            return '\n'.join(lines) + '\n'

    # Show all values
    for val in sorted_keys:
        cnt = dist[val]
        pct = cnt / total * 100
        lines.append(f"  {val:>6} | {bar(cnt, max_count)} {cnt:>6} ({pct:5.1f}%)")

    lines.append(f"  {'Total':>7}: {total}")
    return '\n'.join(lines) + '\n'


def check_multipartite(G: nx.Graph, group_assignments: Dict[int, int]) -> Tuple[bool, int, int]:
    """
    Check if the graph is multipartite with respect to the group assignments.
    Returns (is_multipartite, intra_group_edges, inter_group_edges).
    """
    intra = 0
    inter = 0
    for u, v in G.edges():
        if u in group_assignments and v in group_assignments:
            if group_assignments[u] == group_assignments[v]:
                intra += 1
            else:
                inter += 1
        else:
            inter += 1  # Count edges with unknown groups as inter

    return intra == 0, intra, inter


def analyze_graph(edge_file: str, group_file: str) -> str:
    """Analyze a graph and return a formatted report string."""
    G, group_assignments = load_graph(edge_file, group_file)
    lines = []

    graph_name = Path(edge_file).stem.replace('_edges', '')
    lines.append(f"{'=' * 70}")
    lines.append(f"GRAPH: {graph_name}")
    lines.append(f"{'=' * 70}")

    # Basic stats
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    min_degree = min(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0
    median_degree = sorted(degrees)[len(degrees) // 2] if degrees else 0

    lines.append(f"\n--- BASIC STATISTICS ---")
    lines.append(f"  Nodes:          {n_nodes}")
    lines.append(f"  Edges:          {n_edges}")
    lines.append(f"  Density:        {density:.6f}")
    lines.append(f"  Avg degree:     {avg_degree:.2f}")
    lines.append(f"  Median degree:  {median_degree}")
    lines.append(f"  Min degree:     {min_degree}")
    lines.append(f"  Max degree:     {max_degree}")

    # Connected components
    components = list(nx.connected_components(G))
    lines.append(f"\n--- CONNECTED COMPONENTS ---")
    lines.append(f"  Number of components:  {len(components)}")
    if len(components) > 1:
        comp_sizes = sorted([len(c) for c in components], reverse=True)
        lines.append(f"  Largest component:     {comp_sizes[0]} nodes ({comp_sizes[0]/n_nodes*100:.1f}%)")
        lines.append(f"  Component sizes:       {comp_sizes[:10]}{'...' if len(comp_sizes) > 10 else ''}")
    else:
        lines.append(f"  Graph is connected")

    # Diameter and avg path length (only for connected graphs or largest component)
    if len(components) == 1 and n_nodes <= 10000:
        try:
            diameter = nx.diameter(G)
            avg_path = nx.average_shortest_path_length(G)
            lines.append(f"  Diameter:              {diameter}")
            lines.append(f"  Avg shortest path:     {avg_path:.2f}")
        except Exception:
            pass
    elif n_nodes <= 10000:
        largest_cc = G.subgraph(max(components, key=len)).copy()
        try:
            diameter = nx.diameter(largest_cc)
            avg_path = nx.average_shortest_path_length(largest_cc)
            lines.append(f"  Diameter (largest CC): {diameter}")
            lines.append(f"  Avg shortest path (largest CC): {avg_path:.2f}")
        except Exception:
            pass

    # Clustering
    avg_clustering = nx.average_clustering(G)
    transitivity = nx.transitivity(G)
    lines.append(f"\n--- CLUSTERING ---")
    lines.append(f"  Avg clustering coefficient:  {avg_clustering:.4f}")
    lines.append(f"  Transitivity (global):       {transitivity:.4f}")

    # Group statistics
    groups = Counter(group_assignments.values())
    n_groups = len(groups)
    lines.append(f"\n--- GROUP STATISTICS ---")
    lines.append(f"  Number of groups:  {n_groups}")
    for gid in sorted(groups.keys()):
        lines.append(f"  Group {gid}: {groups[gid]} nodes ({groups[gid]/n_nodes*100:.1f}%)")

    # Multipartiteness check
    is_multipartite, intra, inter = check_multipartite(G, group_assignments)
    lines.append(f"\n--- MULTIPARTITENESS ---")
    lines.append(f"  Is multipartite (w.r.t. groups):  {'YES' if is_multipartite else 'NO'}")
    lines.append(f"  Intra-group edges:  {intra} ({intra/n_edges*100:.1f}%)")
    lines.append(f"  Inter-group edges:  {inter} ({inter/n_edges*100:.1f}%)")
    if not is_multipartite:
        lines.append(f"  Intra/Inter ratio:  {intra/inter:.4f}" if inter > 0 else "  Intra/Inter ratio:  inf")

    # Assortativity
    try:
        degree_assortativity = nx.degree_assortativity_coefficient(G)
        lines.append(f"\n--- ASSORTATIVITY ---")
        lines.append(f"  Degree assortativity:  {degree_assortativity:.4f}")
        if group_assignments:
            nx.set_node_attributes(G, group_assignments, 'group')
            group_assortativity = nx.attribute_assortativity_coefficient(G, 'group')
            lines.append(f"  Group assortativity:   {group_assortativity:.4f}")
    except Exception:
        pass

    # Degree distribution
    degree_dist = Counter(degrees)
    lines.append(f"\n--- DEGREE DISTRIBUTION ---")
    lines.append(format_distribution(degree_dist, "degree"))

    # Core decomposition
    core_numbers = nx.core_number(G)
    core_dist = Counter(core_numbers.values())
    max_core = max(core_numbers.values()) if core_numbers else 0
    lines.append(f"\n--- CORE DECOMPOSITION ---")
    lines.append(f"  Degeneracy (max core number): {max_core}")
    lines.append(format_distribution(core_dist, "core number"))

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze graph characteristics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python graph_info.py --graph books --graphs-dir data/graphs
  python graph_info.py --edge-file data/graphs/books/books_edges.txt --group-file data/graphs/books/books_groups.txt
  python graph_info.py --graph books --graphs-dir data/graphs --output results/graph_info_books.txt
        """
    )
    parser.add_argument('--graph', type=str, help='Graph name (e.g., books)')
    parser.add_argument('--graphs-dir', type=Path, default=Path('data/graphs'),
                        help='Directory containing graph subdirectories (default: data/graphs)')
    parser.add_argument('--edge-file', type=Path, help='Direct path to edge file')
    parser.add_argument('--group-file', type=Path, help='Direct path to group file')
    parser.add_argument('--output', type=Path, help='Save output to file (also prints to terminal)')
    parser.add_argument('--all', action='store_true', help='Analyze all graphs in graphs-dir')

    args = parser.parse_args()

    if args.all:
        # Find all graph subdirectories
        graphs = []
        for subdir in sorted(args.graphs_dir.iterdir()):
            if subdir.is_dir():
                edge_files = list(subdir.glob('*_edges.txt'))
                group_files = list(subdir.glob('*_groups.txt'))
                if edge_files and group_files:
                    graphs.append((str(edge_files[0]), str(group_files[0])))

        if not graphs:
            print(f"No graphs found in {args.graphs_dir}")
            return 1

        all_output = []
        for edge_file, group_file in graphs:
            report = analyze_graph(edge_file, group_file)
            all_output.append(report)
            print(report)
            print()

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(all_output))
            print(f"\nSaved to: {args.output}")

    elif args.edge_file and args.group_file:
        report = analyze_graph(str(args.edge_file), str(args.group_file))
        print(report)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nSaved to: {args.output}")

    elif args.graph:
        graph_dir = args.graphs_dir / args.graph
        if not graph_dir.exists():
            print(f"Graph directory not found: {graph_dir}")
            return 1

        edge_file = graph_dir / f"{args.graph}_edges.txt"
        group_file = graph_dir / f"{args.graph}_groups.txt"

        if not edge_file.exists():
            print(f"Edge file not found: {edge_file}")
            return 1
        if not group_file.exists():
            print(f"Group file not found: {group_file}")
            return 1

        report = analyze_graph(str(edge_file), str(group_file))
        print(report)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nSaved to: {args.output}")
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
