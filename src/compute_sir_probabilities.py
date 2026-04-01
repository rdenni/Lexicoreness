#!/usr/bin/env python3
"""
Compute SIR infection probabilities based on the epidemic threshold.

Uses the Molloy-Reed criterion to compute the epidemic threshold:
    threshold = <k> / (<k²> - <k>)

where <k> is the average degree and <k²> is the average squared degree.

Two modes:
  1. Threshold-based: generates h uniformly-spaced probabilities in [l*threshold, u*threshold]
  2. Explicit range: generates h uniformly-spaced probabilities in [min, max]
"""

import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from methods import load_graph


def compute_epidemic_threshold(G):
    """
    Compute the epidemic threshold using the Molloy-Reed criterion.

    Args:
        G: NetworkX graph

    Returns:
        Tuple of (avg_degree, avg_squared_degree, threshold)
    """
    degrees = np.array([d for _, d in G.degree()])
    avg_degree = degrees.mean()
    avg_squared_degree = (degrees ** 2).mean()

    denominator = avg_squared_degree - avg_degree
    if denominator <= 0:
        raise ValueError(
            f"Cannot compute threshold: <k²> - <k> = {denominator:.4f} <= 0. "
            f"This can happen for very regular or very sparse graphs."
        )

    threshold = avg_degree / denominator
    return float(avg_degree), float(avg_squared_degree), float(threshold)


def compute_probabilities(threshold, l, u, h):
    """
    Generate h uniformly-spaced probabilities in [l*threshold, u*threshold].

    Args:
        threshold: Epidemic threshold
        l: Lower multiplier (default 0.8)
        u: Upper multiplier (default 2)
        h: Number of probabilities to generate (default 20)

    Returns:
        numpy array of probabilities
    """
    low = l * threshold
    high = u * threshold
    return np.linspace(low, high, h)


def main():
    parser = argparse.ArgumentParser(
        description="Compute SIR infection probabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Two modes:

  1. Threshold-based (default): computes the Molloy-Reed epidemic threshold and
     generates probabilities as multiples of it.

       threshold = <k> / (<k²> - <k>)
       probabilities = linspace(l * threshold, u * threshold, h)

  2. Explicit range: generates probabilities directly between --min and --max,
     without computing the threshold. Requires --min and --max.

       probabilities = linspace(min, max, h)

Examples:
  # Threshold-based
  python compute_sir_probabilities.py --graph books --graphs-dir data/graphs
  python compute_sir_probabilities.py --graph books --graphs-dir data/graphs --l 0.5 --u 3 --h 30

  # Explicit range
  python compute_sir_probabilities.py --graph books --graphs-dir data/graphs --min 0.003 --max 0.05 --h 10
  python compute_sir_probabilities.py --min 0.003 --max 0.05 --h 10 --output data/graphs/books/books_sir_probabilities.txt
        """
    )

    # Graph input: either --graph + --graphs-dir, or --edge-file + --group-file
    graph_group = parser.add_argument_group("Graph input (choose one mode)")
    graph_group.add_argument(
        "--graph",
        type=str,
        help="Graph name (e.g., 'books'). Used with --graphs-dir."
    )
    graph_group.add_argument(
        "--graphs-dir",
        type=Path,
        help="Directory containing graph files. Used with --graph."
    )
    graph_group.add_argument(
        "--edge-file",
        type=Path,
        help="Path to edge file (alternative to --graph + --graphs-dir)"
    )
    graph_group.add_argument(
        "--group-file",
        type=Path,
        help="Path to group file (alternative to --graph + --graphs-dir)"
    )

    # Threshold-based mode parameters
    threshold_group = parser.add_argument_group("Threshold-based mode (default)")
    threshold_group.add_argument(
        "--l",
        type=float,
        default=0.8,
        help="Lower multiplier for threshold (default: 0.8)"
    )
    threshold_group.add_argument(
        "--u",
        type=float,
        default=2.0,
        help="Upper multiplier for threshold (default: 2.0)"
    )

    # Explicit range mode parameters
    explicit_group = parser.add_argument_group("Explicit range mode")
    explicit_group.add_argument(
        "--min",
        type=float,
        default=None,
        help="Lower bound probability. Used with --max to skip threshold computation."
    )
    explicit_group.add_argument(
        "--max",
        type=float,
        default=None,
        help="Upper bound probability. Used with --min to skip threshold computation."
    )

    # Common parameters
    parser.add_argument(
        "--h",
        type=int,
        default=20,
        help="Number of probabilities to generate (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: same directory as graph files)"
    )

    args = parser.parse_args()

    # Determine mode
    use_explicit = args.min is not None or args.max is not None
    if use_explicit:
        if args.min is None or args.max is None:
            parser.error("--min and --max must be used together")
        if args.min >= args.max:
            parser.error("--min must be less than --max")

    # Resolve graph file paths (needed for output dir in both modes)
    graph_name = None
    graph_dir = None
    edge_file = None
    group_file = None

    if args.graph and args.graphs_dir:
        graphs_dir = args.graphs_dir
        if not graphs_dir.is_absolute():
            graphs_dir = Path(__file__).parent.parent / graphs_dir
        edge_file = graphs_dir / f"{args.graph}_edges.txt"
        group_file = graphs_dir / f"{args.graph}_groups.txt"
        if not edge_file.exists():
            edge_file = graphs_dir / args.graph / f"{args.graph}_edges.txt"
            group_file = graphs_dir / args.graph / f"{args.graph}_groups.txt"
        graph_name = args.graph
        graph_dir = edge_file.parent
    elif args.edge_file and args.group_file:
        edge_file = args.edge_file
        group_file = args.group_file
        graph_name = edge_file.stem.replace("_edges", "")
        graph_dir = edge_file.parent
    elif not use_explicit or not args.output:
        parser.error(
            "Provide graph input (--graph + --graphs-dir, or --edge-file + --group-file), "
            "or use explicit range mode (--min + --max) with --output"
        )

    # Determine output path
    if args.output:
        output_file = args.output
    elif graph_dir and graph_name:
        if use_explicit:
            output_file = graph_dir / f"{graph_name}_sir_probabilities_min-max-refinement.txt"
        else:
            output_file = graph_dir / f"{graph_name}_sir_probabilities_starting-point-lower-upper.txt"
    else:
        parser.error("Cannot determine output path: provide --output or graph input")

    # -------------------------------------------------------------------------
    # Explicit range mode
    # -------------------------------------------------------------------------
    if use_explicit:
        probabilities = np.linspace(args.min, args.max, args.h)

        print(f"Explicit range mode")
        if graph_name:
            print(f"  Graph: {graph_name}")
        print(f"  Range: [{args.min:.6f}, {args.max:.6f}]")
        print(f"  Count: {args.h}")
        print(f"\nInfection probabilities ({args.h} values):")
        for i, p in enumerate(probabilities):
            print(f"  {i+1:3d}. {p:.6f}")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(f"# min = {args.min:.6f}\n")
            f.write(f"# max = {args.max:.6f}\n")
            f.write(f"# count = {args.h}\n")
            f.write(f"# probability\n")
            for p in probabilities:
                f.write(f"{p:.6f}\n")

        print(f"\nProbabilities saved to: {output_file}")
        print(f"\nYAML format (copy into experiment config):")
        print(f"sir:")
        print(f"  infection_probabilities:")
        for p in probabilities:
            print(f"  - {p:.6f}")

        return 0

    # -------------------------------------------------------------------------
    # Threshold-based mode
    # -------------------------------------------------------------------------

    # Validate files exist
    if not edge_file.exists():
        print(f"Error: Edge file not found: {edge_file}")
        return 1
    if not group_file.exists():
        print(f"Error: Group file not found: {group_file}")
        return 1

    # Load graph
    print(f"Loading graph: {graph_name}")
    G, _ = load_graph(str(edge_file), str(group_file))

    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"  Nodes: {n}")
    print(f"  Edges: {m}")

    # Compute threshold
    avg_degree, avg_squared_degree, threshold = compute_epidemic_threshold(G)

    print(f"\nDegree statistics:")
    print(f"  <k>  = {avg_degree:.4f}")
    print(f"  <k²> = {avg_squared_degree:.4f}")
    print(f"  <k²> - <k> = {avg_squared_degree - avg_degree:.4f}")

    print(f"\nEpidemic threshold:")
    print(f"  threshold = <k> / (<k²> - <k>) = {threshold:.6f}")

    # Compute probabilities
    probabilities = compute_probabilities(threshold, args.l, args.u, args.h)

    # Compute coefficients (prob / threshold)
    coefficients = probabilities / threshold

    print(f"\nInfection probabilities ({args.h} values in [{args.l}*threshold, {args.u}*threshold]):")
    print(f"  Range: [{args.l * threshold:.6f}, {args.u * threshold:.6f}]")
    for i, (p, c) in enumerate(zip(probabilities, coefficients)):
        print(f"  {i+1:3d}. {p:.6f}  (coeff={c:.4f})")

    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(f"# threshold = {threshold:.6f}\n")
        f.write(f"# probability  coefficient\n")
        for p, c in zip(probabilities, coefficients):
            f.write(f"{p:.6f}  {c:.4f}\n")

    print(f"\nProbabilities saved to: {output_file}")

    # Print YAML-ready format for easy copy-paste
    print(f"\nYAML format (copy into experiment config):")
    print(f"sir:")
    print(f"  infection_probabilities:")
    for p in probabilities:
        print(f"  - {p:.6f}")

    return 0


if __name__ == "__main__":
    exit(main())
