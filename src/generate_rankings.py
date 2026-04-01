#!/usr/bin/env python3
"""
Generate Graph Rankings

This script generates group rankings for a specified graph instance (e.g., ER_2groups_rep0)
and saves them into dedicated JSON files with names encoding the input parameters:

1. GRAPH_NAME_rankings_L{lengths}_S{strategies}_R{random_count}.json
   A list of List[int] rankings, as they would appear in config.

2. GRAPH_NAME_ranking_strategies_L{lengths}_S{strategies}_R{random_count}.json
   A mapping from a ranking's string representation to its strategy name(s).

Filename encoding:
  - L{lengths}: ranking lengths joined by '-' (e.g., L2-3-5 for lengths 2, 3, 5)
  - S{strategies}: strategy names joined by '-' (e.g., Sall or Stop_size-random)
  - R{random_count}: number of random rankings per length (e.g., R5 for 5 random rankings)

Example filenames:
  - email-eu-core_rankings_L2-3-5_Sall_R5.json
  - email-eu-core_ranking_strategies_L2-3-5_Sall_R5.json

This script is intended to be run after graphs are generated (by graph_generator.py)
and before methods/SIR simulations are run. It allows 'rankings: auto' in config
to reference these generated files.

The helper functions build_ranking_filename_suffix() and parse_ranking_filename_suffix()
can be imported by other scripts (run_methods.py, run_sir_batch.py, aggregate scripts)
to identify and parse ranking files matching specific parameters.
"""

import argparse
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))

from methods import load_graph  # To load graph for metrics
from ranking_strategies import compute_group_metrics, generate_rankings


def build_ranking_filename_suffix(ranking_lengths: list, strategies: list, random_count: int = None) -> str:
    """
    Build a filename suffix encoding the ranking parameters.

    Format: L{lengths}_S{strategies}_R{random_count}
    Where:
        - lengths are joined with '-' (e.g., "2-3-5")
        - strategies are joined with '-' (e.g., "top_size-random" or "all")
        - random_count is the number of random rankings per length (only included if provided)

    Example outputs:
        - "L2-3-5_Sall_R5"
        - "L2-3_Stop_size-random_R10"
        - "L2-3_Stop_size" (if random_count is None and no random strategy)
    """
    lengths_str = '-'.join(map(str, sorted(ranking_lengths)))
    strategies_str = '-'.join(sorted(strategies))
    suffix = f"L{lengths_str}_S{strategies_str}"
    if random_count is not None:
        suffix += f"_R{random_count}"
    return suffix


def parse_ranking_filename_suffix(suffix: str) -> dict:
    """
    Parse a ranking filename suffix to extract parameters.

    Args:
        suffix: String like "L2-3-5_Sall_R5" or "L2-3_Stop_size-random_R10"
                Also handles legacy format without _R: "L2-3-5_Sall"

    Returns:
        dict with 'lengths' (list of int), 'strategies' (list of str),
        and 'random_count' (int or None)
    """
    result = {'lengths': [], 'strategies': [], 'random_count': None}

    if not suffix.startswith('L'):
        return result

    # Check for _R{count} at the end
    random_count = None
    if '_R' in suffix:
        # Find the last _R and extract the number
        r_idx = suffix.rfind('_R')
        random_part = suffix[r_idx + 2:]  # Everything after '_R'
        try:
            random_count = int(random_part)
            suffix = suffix[:r_idx]  # Remove the _R{count} part for further parsing
        except ValueError:
            pass  # Not a valid _R{count}, might be part of strategy name

    # Split by _S to get lengths and strategies parts
    parts = suffix.split('_S')
    if len(parts) != 2:
        return result

    lengths_part = parts[0][1:]  # Remove leading 'L'
    strategies_part = parts[1]

    # Parse lengths
    try:
        result['lengths'] = [int(x) for x in lengths_part.split('-')]
    except ValueError:
        pass

    # Parse strategies
    result['strategies'] = strategies_part.split('-')
    result['random_count'] = random_count

    return result


def load_config(config_path: Path) -> dict:
    """Load YAML configuration"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_ranking_params_from_graph_config(graph_config: dict) -> tuple:
    """
    Extract ranking parameters from graph config's generate_rankings_params.

    Returns:
        Tuple of (lengths, strategies, random_count) or (None, None, None) if not found
    """
    params = graph_config.get('generate_rankings_params', {})
    if not params:
        return None, None, None

    lengths = params.get('lengths')
    strategies = params.get('strategies')
    random_count = params.get('random_count')

    # Handle 'all' strategy
    if strategies == 'all':
        strategies = ['all']
    elif isinstance(strategies, str):
        strategies = [strategies]

    return lengths, strategies, random_count


def main():
    parser = argparse.ArgumentParser(
        description="Generate group rankings for a graph and save them to JSON files."
    )
    parser.add_argument(
        "graph_full_name",  # e.g., 'email-eu-core', 'ER_2groups_rep0'
        type=str,
        nargs='?', # Make it optional
        help="Full name of a specific graph instance (e.g., 'email-eu-core' or 'ER_2groups_rep0'). If not provided, process all"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to experiment config YAML. If provided, reads ranking parameters from generate_rankings_params."
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=Path("data/graphs"),
        help="Directory containing graph files (edges.txt, groups.txt)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for the generated ranking JSON files. Defaults to graphs-dir.",
        default=None  # Will default to graphs_dir later
    )
    parser.add_argument(
        "--ranking-lengths",
        nargs='+',
        type=int,
        help="List of ranking lengths for which to generate strategies (e.g., 2 3 5). "
             "Can be omitted if --config is provided with generate_rankings_params."
    )
    parser.add_argument(
        "--strategies",
        nargs='+',
        type=str,
        default=['all'],  # Default to 'all' strategies
        help="List of strategy types to generate (e.g., top_size random). Use 'all' for all standard strategies."
    )
    parser.add_argument(
        "--random-rankings-per-length",
        type=int,
        default=None,
        help="Number of random rankings to generate per length if 'random' strategy is included. Default: 5"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing ranking JSON files even if they are already present."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output."
    )

    args = parser.parse_args()

    # resolve paths
    project_root = Path(__file__).parent.parent
    graphs_dir = project_root / args.graphs_dir
    output_dir = project_root / (args.output_dir if args.output_dir else args.graphs_dir)

    # Load config if provided to get default ranking parameters
    config = None
    config_graph_params = {}  # {graph_name: (lengths, strategies, random_count)}
    if args.config:
        config_path = project_root / args.config
        if config_path.exists():
            config = load_config(config_path)
            # Build mapping of graph name to ranking params
            for tier in ['sir_optimal', 'fast_only']:
                for gc in config.get('graphs', {}).get(tier, []):
                    if gc.get('rankings') == 'auto':
                        lengths, strategies, random_count = get_ranking_params_from_graph_config(gc)
                        if lengths:
                            config_graph_params[gc['name']] = (lengths, strategies, random_count)
            if args.verbose:
                print(f"Loaded config from {config_path}")
                print(f"Found ranking params for {len(config_graph_params)} graphs")
        else:
            print(f"Warning: Config file not found: {config_path}")

    # Determine ranking lengths and strategies
    # Command line takes precedence over config
    ranking_lengths = args.ranking_lengths
    strategies = args.strategies
    random_count = args.random_rankings_per_length if args.random_rankings_per_length else 5

    # If no ranking lengths from command line, we'll get them per-graph from config
    if not ranking_lengths and not config_graph_params:
        print("Error: --ranking-lengths is required when not using --config with generate_rankings_params")
        sys.exit(1)

    # ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which graph instances to process
    graphs_to_process = {} # {graph_full_name: edge_file_path}
    if args.graph_full_name:
        # Process only the specified graph
        edge_file_path = graphs_dir / f"{args.graph_full_name}_edges.txt"
        if not edge_file_path.exists():
            print(f"Error: Edge file not found for specified graph '{args.graph_full_name}' at {edge_file_path}.")
            sys.exit(1)
        graphs_to_process[args.graph_full_name] = edge_file_path
    else:
        # Process all graphs in graphs_dir
        edge_files = list(graphs_dir.glob("*_edges.txt"))
        if not edge_files:
            print(f"Error: No graph edge files (*_edges.txt) found in {graphs_dir}. Exiting.")
            sys.exit(1)
        for edge_file in edge_files:
            graph_full_name = edge_file.stem.replace('_edges', '')
            graphs_to_process[graph_full_name] = edge_file

    if not graphs_to_process:
        print(f"Error: No graph instances to process. Exiting.")
        sys.exit(1)
    if args.verbose:
        print(f"Found {len(graphs_to_process)} unique graph instances to process.")

    # Track skipped lengths across all graphs for summary warning
    all_skipped_lengths = {}  # {graph_name: [skipped_lengths]}

    # Process each graph instance
    for graph_full_name, edge_file_path in graphs_to_process.items():
        group_file = graphs_dir / f"{graph_full_name}_groups.txt"
        edge_file = graphs_dir / f"{graph_full_name}_edges.txt"

        if not edge_file.exists() or not group_file.exists():
            print(f"Error: Graph files not found for {graph_full_name} at {graphs_dir}")
            print(f"Please ensure graph is generated first.")
            sys.exit(1)

        G, group_assignments = load_graph(str(edge_file), str(group_file))
        if args.verbose:
            print(f"\n--- Generating Rankings for Graph: {graph_full_name} ---")
            print(f"Loaded graph '{graph_full_name}' (Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()})")

        # generate rankings
        num_groups = len(set(group_assignments.values()))
        group_metrics = compute_group_metrics(G, group_assignments, num_groups)

        # Determine ranking params for this graph
        # Priority: command line > config > error
        graph_ranking_lengths = ranking_lengths
        graph_strategies = strategies
        graph_random_count = random_count

        # Check if this graph has config params (match by base name without _repX)
        base_graph_name = graph_full_name.rsplit('_rep', 1)[0] if '_rep' in graph_full_name else graph_full_name
        if not graph_ranking_lengths and base_graph_name in config_graph_params:
            cfg_lengths, cfg_strategies, cfg_random = config_graph_params[base_graph_name]
            graph_ranking_lengths = cfg_lengths
            if cfg_strategies:
                graph_strategies = cfg_strategies
            if cfg_random:
                graph_random_count = cfg_random
            if args.verbose:
                print(f"  Using params from config: lengths={graph_ranking_lengths}, strategies={graph_strategies}, random_count={graph_random_count}")

        if not graph_ranking_lengths:
            print(f"  Warning: No ranking lengths specified for {graph_full_name}. Skipping.")
            continue

        all_standard_strategies = [
            'top_size', 'bottom_size', 'top_coreness', 'bottom_coreness',
            'top_degree', 'bottom_degree',
            'alt_size', 'alt_coreness', 'alt_degree',
            'high_size_high_core', 'high_size_low_core',
            'low_size_high_core', 'low_size_low_core',
            'high_size_high_degree', 'high_size_low_degree',
            'low_size_high_degree', 'low_size_low_degree',
            'high_core_high_degree', 'high_core_low_degree',
            'low_core_high_degree', 'low_core_low_degree',
            'random'
        ]
        strategies_to_generate = graph_strategies if graph_strategies else ['all']
        if 'all' in strategies_to_generate:
            strategies_to_generate = all_standard_strategies

        generated_rankings_by_strategy = {}  # {ranking_str: strategy_name}
        lengths_actually_used = []  # Track which lengths were actually used
        lengths_skipped = []  # Track which lengths were skipped
        skipped_strategies = []  # Track strategies skipped due to repeated groups

        # Generate rankings for each length
        for length in graph_ranking_lengths:
            if length > num_groups:
                print(f"Warning: Cannot generate ranking of length {length} for {num_groups} groups. Skipping length {length}.")
                lengths_skipped.append(length)
                continue

            # generate_rankings returns {strategy_name: ranking_list} for a single length
            generated_for_this_length = generate_rankings(group_metrics, length, num_random=graph_random_count)

            length_produced_rankings = False
            for strategy_name, ranking_list in generated_for_this_length.items():
                base_strategy_name = strategy_name.split('_')[0]
                if base_strategy_name in strategies_to_generate or strategy_name in strategies_to_generate:
                    # Check for repeated groups in the ranking
                    if len(ranking_list) != len(set(ranking_list)):
                        skipped_strategies.append(strategy_name)
                        if args.verbose:
                            print(f"Skipping strategy '{strategy_name}': ranking {ranking_list} has repeated groups")
                        continue

                    # store the ranking as a string to use as a dictionary key
                    ranking_str = '_'.join(map(str, ranking_list))
                    # Now store a list of strategy names for each unique ranking
                    if ranking_str not in generated_rankings_by_strategy:
                        generated_rankings_by_strategy[ranking_str] = []
                    generated_rankings_by_strategy[ranking_str].append(strategy_name)
                    length_produced_rankings = True

            if length_produced_rankings:
                lengths_actually_used.append(length)

        if skipped_strategies:
            print(f"Skipped {len(skipped_strategies)} strategies due to repeated groups: {skipped_strategies}")

        # Track skipped lengths for this graph
        if lengths_skipped:
            all_skipped_lengths[graph_full_name] = lengths_skipped

        if not generated_rankings_by_strategy:
            print(f"Warning: No rankings generated for {graph_full_name} based on specified parameters.")
            continue

        # Build filename suffix encoding the parameters
        # Use 'all' in filename if all standard strategies were selected, otherwise list them
        # Use only lengths that actually produced rankings
        # Include random_count if 'random' strategy is being used
        strategies_for_filename = ['all'] if set(strategies_to_generate) == set(all_standard_strategies) else strategies_to_generate
        include_random_count = 'random' in strategies_to_generate or 'all' in strategies_for_filename
        param_suffix = build_ranking_filename_suffix(
            lengths_actually_used,
            strategies_for_filename,
            random_count=graph_random_count if include_random_count else None
        )

        # prepare files for saving
        rankings_json_path = output_dir / f"{graph_full_name}_rankings_{param_suffix}.json"
        ranking_strategies_json_path = output_dir / f"{graph_full_name}_ranking_strategies_{param_suffix}.json"

        # save GRAPH_NAME_rankings.json (pure List[List[int]])
        if rankings_json_path.exists() and not args.force:
            print(f"  Skipping {rankings_json_path}: file exists. Use --force to overwrite.")
        else:
            rankings_list_of_lists = [json.loads(f"[{r_str.replace('_', ', ')}]") for r_str in generated_rankings_by_strategy.keys()]
            with open(rankings_json_path, 'w') as f:
                json.dump(rankings_list_of_lists, f, indent=2)
            if args.verbose:
                print(f"  Generated rankings saved to {rankings_json_path}")

        # save GRAPH_NAME_ranking_strategies.json (ranking_str -> strategy_name)
        if ranking_strategies_json_path.exists() and not args.force:
            print(f"Skipping {ranking_strategies_json_path}: file exists. Use --force to overwrite.")
        else:
            with open(ranking_strategies_json_path, 'w') as f:
                json.dump(generated_rankings_by_strategy, f, indent=2)
            if args.verbose:
                print(f"Generated ranking strategies saved to {ranking_strategies_json_path}")

        if args.verbose:
            print(f"--- Rankings generation complete for {graph_full_name} ---")

    # Final summary: warn about skipped lengths
    if all_skipped_lengths:
        print(f"\n{'='*70}")
        print(f"WARNING: Some ranking lengths could not be generated")
        print(f"{'='*70}")
        for graph_name, skipped in all_skipped_lengths.items():
            print(f"  {graph_name}: skipped lengths {skipped}")

        # Collect all unique skipped lengths
        all_unique_skipped = set()
        for skipped in all_skipped_lengths.values():
            all_unique_skipped.update(skipped)

        print(f"\n  o avoid errors in run_methods.py and run_sir_batch.py, consider")
        print(f"removing these lengths from your config file's generate_rankings_params:")
        print(f"Lengths to potentially remove: {sorted(all_unique_skipped)}")
        print(f"\n Or ensure the ranking files generated match the lengths expected")
        print(f"by downstream scripts (run_methods.py, run_sir_batch.py).")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()

