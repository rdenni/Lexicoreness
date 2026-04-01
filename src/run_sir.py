#!/usr/bin/env python3
"""
Run SIR Simulations (Component 3)

Pure SIR simulation runner - evaluates nodes and saves RAW results.
Does NOT rank results - that's Component 4 (Comparison).

Flexible input modes:
  --all-nodes: Evaluate all nodes in graph
  --nodes N1 N2 N3: Evaluate specific nodes
  --from-methods FILE --top-k K: Top K from methods results
  --from-methods FILE --rank-1-only: All rank-1 nodes from methods
  --from-file FILE: Read node IDs from file
"""

import json
import yaml
import pandas as pd
from pathlib import Path
import sys
import time
import argparse
from typing import List, Dict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from methods import load_graph
from sir_simulator import SIRSimulator
from compute_sir_probabilities import compute_epidemic_threshold


def load_config(config_path: Path) -> dict:
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_single_node(args_tuple):
    """
    Evaluate a single node (designed for parallel processing).

    Args:
        args_tuple: A tuple containing (node_id, edge_file, group_file, infection_prob, num_sims, seed).

    Returns:
        A tuple of (node_id, results_dict).
    """
    node, edge_file, group_file, infection_prob, num_sims, seed = args_tuple

    # Load graph in this process (can't pickle NetworkX graphs easily)
    G, group_assignments = load_graph(edge_file, group_file)
    simulator = SIRSimulator(G, group_assignments)

    # Run SIR simulations
    results = simulator.run_multiple_simulations(
        seed_nodes=[node],
        infection_prob=infection_prob,
        num_simulations=num_sims,
        random_seed=seed
    )

    return node, results

def run_sir_for_graph(
    graph_name: str,
    replicate_id: int,
    infection_prob: float,
    num_simulations: int,
    nodes_to_evaluate: List[int],
    graphs_dir: Path,
    output_dir: Path,
    num_processes: int = None,
    spread_min_pct: float = 0.01,
    spread_degree_coeff: float = 1.5
) -> Dict:
    """
    Run SIR evaluation for a specified list of nodes.

    Args:
        graph_name: The full name of the graph (e.g., 'ER_2groups_rep0').
        replicate_id: Replicate ID
        infection_prob: The probability of infection for the simulation.
        num_simulations: The number of Monte Carlo simulations to run per node.
        nodes_to_evaluate: A list of node IDs to run the simulation on.
        graphs_dir: The directory containing the graph edge and group files.
        output_dir: The directory where the results CSV will be saved.
        num_processes: The number of parallel processes to use.

    Returns:
        A summary dictionary of the simulation run.
    """
    # File paths - use graph_name directly (already includes _rep if needed)
    prefix = graph_name
    edge_file = graphs_dir / f"{prefix}_edges.txt"
    group_file = graphs_dir / f"{prefix}_groups.txt"

    print(f"\n{'='*80}")
    print(f"SIR SIMULATION: {prefix}")
    print(f"  Infection prob: {infection_prob:.4f}")
    print(f"{'='*80}")

    # Load graph
    print("Loading graph...")
    G, group_assignments = load_graph(str(edge_file), str(group_file))

    n = G.number_of_nodes()
    m = G.number_of_edges()

    print(f"  Nodes: {n}")
    print(f"  Edges: {m}")
    avg_degree = 2 * m / n
    print(f"  Avg degree: {avg_degree:.2f}")
    threshold_by_degree = spread_degree_coeff * avg_degree / n
    spread_significance = max(spread_min_pct, threshold_by_degree)
    print(f"  Significant spreader threshold: {spread_significance:.4f} ({spread_significance:.1%})")
    print(f"    [max(min_pct={spread_min_pct:.1%}, {spread_degree_coeff}×avg_deg/n={threshold_by_degree:.4f})]")
    print(f"  Nodes to evaluate: {len(nodes_to_evaluate)}")
    print(f"  Simulations per node: {num_simulations}")

    # Determine number of processes
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    print(f"  Parallel processes: {num_processes}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for parallel processing
    start_time = time.time()

    args_list = [
        (node, str(edge_file), str(group_file), infection_prob,
         num_simulations, 42 + node * 1000)
        for node in nodes_to_evaluate
    ]


    # Run in parallel with progress bar
    print(f"\n  Running SIR simulations...")
    if num_processes > 1 and len(nodes_to_evaluate) > 1:
        with Pool(num_processes) as pool:
            results_list = list(tqdm(
                pool.imap_unordered(evaluate_single_node, args_list),
                total=len(args_list),
                desc="  Nodes completed",
                unit="node"
            ))
    else:
        results_list = [
            evaluate_single_node(args)
            for args in tqdm(args_list, desc="  Nodes completed", unit="node")
        ]

    # Convert to dict
    node_results = {node: results for node, results in results_list}

    elapsed = time.time() - start_time
    sims_per_sec = (len(nodes_to_evaluate) * num_simulations) / elapsed

    print(f"  Completed in {elapsed:.1f}s ({sims_per_sec:.0f} sims/sec)")

    # Save RAW results (NO ranking in filename - ranking is applied during comparison!)
    output_file = output_dir / f"{prefix}_sir_p{infection_prob:.4f}.parquet"

    # Prepare data
    rows = []
    groups = sorted(set(group_assignments.values()))

    for node in nodes_to_evaluate:
        results = node_results[node]
        row = {
            'node_id': node,
        }

        # Add group-specific reaches
        for group_id in groups:
            row[f'group_{group_id}_mean'] = results['group_reach_mean'][group_id]
            row[f'group_{group_id}_std'] = results['group_reach_std'][group_id]

        # Add total reach
        row['total_mean'] = results['total_reach_mean']
        row['total_std'] = results['total_reach_std']
        row['num_simulations'] = num_simulations

        rows.append(row)

    # Save to Parquet
    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(output_file, index=False, engine='pyarrow')

        print(f"\n  Saved RAW SIR results: {output_file}")
        print(f"  Note: Ranking will be applied during comparison, not here!")

        # Compute and save spread summary
        total_means = df['total_mean']
        avg_spread = float(total_means.mean())
        median_spread = float(total_means.median())
        min_spread = float(total_means.min())
        max_spread = float(total_means.max())
        p25_spread = float(total_means.quantile(0.25))
        p75_spread = float(total_means.quantile(0.75))

        avg_spread_pct = avg_spread / n
        median_spread_pct = median_spread / n
        min_spread_pct = min_spread / n
        max_spread_pct = max_spread / n
        p25_spread_pct = p25_spread / n
        p75_spread_pct = p75_spread / n

        # Significant spreaders: nodes whose avg spread reaches >= spread_significance of the graph
        spread_pcts = total_means / n
        significant_count = int((spread_pcts >= spread_significance).sum())
        significant_ratio = significant_count / n

        # Compute threshold coefficient: infection_prob / threshold
        try:
            _, _, threshold = compute_epidemic_threshold(G)
            threshold_coefficient = infection_prob / threshold
        except ValueError:
            threshold = None
            threshold_coefficient = None

        summary = {
            'avg_spread': avg_spread,
            'median_spread': median_spread,
            'min_spread': min_spread,
            'max_spread': max_spread,
            'p25_spread': p25_spread,
            'p75_spread': p75_spread,
            'avg_spread_pct': avg_spread_pct,
            'median_spread_pct': median_spread_pct,
            'min_spread_pct': min_spread_pct,
            'max_spread_pct': max_spread_pct,
            'p25_spread_pct': p25_spread_pct,
            'p75_spread_pct': p75_spread_pct,
            'num_nodes_graph': n,
            'num_nodes_evaluated': len(nodes_to_evaluate),
            'avg_degree': avg_degree,
            'significant_spread_threshold': spread_significance,
            'significant_count': significant_count,
            'significant_ratio': significant_ratio,
            'infection_prob': infection_prob,
            'threshold': threshold,
            'threshold_coefficient': threshold_coefficient,
            'num_simulations': num_simulations,
        }
        summary_file = output_dir / f"{prefix}_sir_p{infection_prob:.4f}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        coeff_str = f", coeff={threshold_coefficient:.4f}" if threshold_coefficient is not None else ""
        print(f"  Spread summary (p={infection_prob:.4f}{coeff_str}):")
        print(f"    spread:     avg={avg_spread:.2f}  median={median_spread:.2f}  "
              f"min={min_spread:.2f}  max={max_spread:.2f}  "
              f"p25={p25_spread:.2f}  p75={p75_spread:.2f}")
        print(f"    spread_pct: avg={avg_spread_pct:.1%}  median={median_spread_pct:.1%}  "
              f"min={min_spread_pct:.1%}  max={max_spread_pct:.1%}  "
              f"p25={p25_spread_pct:.1%}  p75={p75_spread_pct:.1%}")
        print(f"    significant spreaders (>= {spread_significance:.4f} = {spread_significance:.1%}): "
              f"{significant_count}/{n} nodes (ratio={significant_ratio:.1%})")
        print(f"    ({len(nodes_to_evaluate)} seeds, {n} total nodes)")

    return {
        'graph_name': graph_name,
        'replicate_id': replicate_id,
        'num_nodes_evaluated': len(nodes_to_evaluate),
        'total_time': elapsed
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run SIR simulations on graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input modes (choose ONE):
  --all-nodes                       Evaluate all nodes in graph
  --nodes N1 N2 N3                  Evaluate specific nodes
  --from-methods FILE --top-k K     Top K nodes from methods results
  --from-methods FILE --rank-1-only All rank-1 nodes from methods
  --from-methods FILE --max-rank N  All nodes with rank <= N
  --from-file FILE                  Read node IDs from file (one per line)

Examples:
  # Evaluate all nodes (expensive!)
  python run_sir.py --graph ER_2groups_rep0 --all-nodes

  # Evaluate specific nodes
  python run_sir.py --graph ER_2groups_rep0 --nodes 10 25 37

  # Top 20 nodes from methods results
  python run_sir.py --graph ER_2groups_rep0 \\
    --from-methods results/methods/ER_2groups_rep0_methods.parquet --top-k 20

  # All rank-1 nodes from methods
  python run_sir.py --graph ER_2groups_rep0 \\
    --from-methods results/methods/ER_2groups_rep0_methods.parquet --rank-1-only
        """
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiment_config.yaml"),
        help="Path to experiment configuration"
    )
    parser.add_argument(
        "--graph",
        required=True,
        help="Graph to process (e.g., ER_2groups_rep0)"
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=Path("data/graphs"),
        help="Directory containing graph files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sir"),
        help="Output directory"
    )

    input_mode = parser.add_mutually_exclusive_group(required=True)
    input_mode.add_argument(
        "--all-nodes",
        action="store_true",
        help="Evaluate all nodes in graph"
    )
    input_mode.add_argument(
        "--nodes",
        nargs='+',
        type=int,
        help="Specific node IDs to evaluate"
    )
  
    input_mode.add_argument(
        "--from-file",
        type=Path,
        help="Load node IDs from file (one per line)"
    )

    parser.add_argument(
        "--infection-prob",
        type=float,
        help="Infection probability (overrides config)"
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        help="Number of SIR simulations per node (overrides config)"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: all CPUs - 1)"
    )
    parser.add_argument(
        "--spread-min-pct",
        type=float,
        default=0.01,
        help="Minimum fraction of graph nodes a node must infect to be a significant spreader (default: 0.01 = 1%%)"
    )
    parser.add_argument(
        "--spread-degree-coeff",
        type=float,
        default=1.5,
        help="Coefficient for avg-degree component of significant spreader threshold: "
             "threshold = max(spread-min-pct, coeff * avg_degree / n) (default: 1.5)"
    )

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / args.config
    graphs_dir = project_root / args.graphs_dir
    output_dir = project_root / args.output_dir

    # Load config
    config = load_config(config_path)

    # Get SIR parameters
    if args.num_simulations:
        num_simulations = args.num_simulations
    else:
        num_simulations = config['global']['sir_simulations']

    # Parse graph name (e.g., "ER_2groups_rep0" -> "ER_2groups", 0)
    # This is for display purposes and output file naming
    parts = args.graph.rsplit('_rep', 1)
    if len(parts) == 2:
        graph_name_base = parts[0]
        replicate_id = int(parts[1])
    else:
        graph_name_base = args.graph
        replicate_id = 0

    # Use args.graph directly for file paths (already has _rep if needed)
    # Load graph early to compute actual avg_degree (needed for infection_prob)
    edge_file = graphs_dir / f"{args.graph}_edges.txt"
    group_file = graphs_dir / f"{args.graph}_groups.txt"
    G, group_assignments = load_graph(str(edge_file), str(group_file))

    # Compute actual average degree from loaded graph
    actual_avg_degree = 2 * len(G.edges()) / len(G.nodes())

    # Calculate infection probability
    if args.infection_prob:
        infection_prob = args.infection_prob
    else:
        infection_probabilities = config['sir']['infection_probabilities']
        infection_prob = infection_probabilities[0]

    # Determine nodes to evaluate
    print("="*80)
    print("SIR SIMULATION (Component 3)")
    print("="*80)
    print(f"Graph: {args.graph}")
    print(f"  Nodes: {len(G.nodes())}")
    print(f"  Edges: {len(G.edges())}")
    print(f"  Actual avg degree: {actual_avg_degree:.2f}")

    if args.all_nodes:
        nodes_to_evaluate = list(G.nodes())
        print(f"Mode: All nodes ({len(nodes_to_evaluate)} nodes)")

    elif args.nodes:
        nodes_to_evaluate = args.nodes
        print(f"Mode: Specific nodes ({len(nodes_to_evaluate)} nodes)")

    # elif args.from_methods:
    #     print(f"Mode: From methods results")
    #     print(f"  File: {args.from_methods}")
    #     nodes_to_evaluate = list(get_nodes_from_methods_csv(
    #         args.from_methods,
    #         args.ranking,
    #         top_k=args.top_k,
    #         rank_1_only=args.rank_1_only,
    #         max_rank=args.max_rank
    #     ))

    elif args.from_file:
        print(f"Mode: From file ({args.from_file})")
        with open(args.from_file, 'r') as f:
            nodes_to_evaluate = [int(line.strip()) for line in f]
        print(f"  Loaded {len(nodes_to_evaluate)} nodes")

    # Run SIR
    # summary = run_sir_for_graph(
    #     graph_name=args.graph,  # Use full graph name directly (includes _rep if needed)
    #     replicate_id=replicate_id,  # Still used for output file naming
    #     ranking=args.ranking,
    #     infection_prob=infection_prob,
    #     num_simulations=num_simulations,
    #     nodes_to_evaluate=nodes_to_evaluate,
    #     graphs_dir=graphs_dir,
    #     output_dir=output_dir,
    #     num_processes=args.num_processes
    # )
    # Run SIR
    summary = run_sir_for_graph(
        graph_name=args.graph,
        replicate_id=replicate_id,
        infection_prob=infection_prob,
        num_simulations=num_simulations,
        nodes_to_evaluate=nodes_to_evaluate,
        graphs_dir=graphs_dir,
        output_dir=output_dir,
        num_processes=args.num_processes,
        spread_min_pct=args.spread_min_pct,
        spread_degree_coeff=args.spread_degree_coeff
    )

    print(f"\n{'='*80}")
    print(f"SIR SIMULATION COMPLETED")
    print(f"  Nodes evaluated: {summary['num_nodes_evaluated']}")
    print(f"  Time: {summary['total_time']:.1f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
