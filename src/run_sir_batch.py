#!/usr/bin/env python3
"""
Run SIR simulations on multiple graphs with automatic config-driven execution

The script reads the experiment config to automatically determine:
  - Which graphs need full SIR (sir_optimal): runs on ALL nodes
  - Which graphs need fast methods (fast_only): runs on selected nodes from methods
  - ALL infection probabilities: loops over all infection_probabilities from config

Ranking resolution for fast_only graphs:
  - If 'rankings: auto' in config, looks for {graph}_rankings_L{lengths}_S{strategies}.json
  - Use --ranking-lengths and --strategies to specify which ranking file to load
  - Or specify explicit rankings in config as List[List[int]]

Parallelism modes:
  outer: Parallel jobs, 1 core per job (good for many graphs/infection_probs)
  inner: Sequential jobs, all cores per job (good for few large graphs)
  none:  No parallelism 

Node selection strategies (for fast_only graphs):
  rank-1:   All nodes ranked #1 by any method
  top-k:    Top K nodes by minimum rank (specify K with --k)
  max-rank: All nodes with rank =< N (specify N with --k)
"""

import argparse
import os
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import cpu_count
#import csv
import yaml
import json
import sys
from typing import List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from methods import load_graph
from generate_rankings import build_ranking_filename_suffix


def load_config(config_path: Path) -> dict:
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_metadata(metadata_path: Path) -> dict:
    """Load graph metadata from JSON"""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def find_ranking_file(graphs_dir: Path, graph_full_name: str,
                      ranking_lengths: List[int] = None,
                      strategies: List[str] = None,
                      random_count: int = None) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find ranking files for a graph.

    If ranking_lengths and strategies are specified, looks for exact match.
    Otherwise, finds the first available ranking file.

    Returns:
        Tuple of (rankings_file_path, strategies_file_path) or (None, None) if not found
    """
    if ranking_lengths and strategies:
        # Look for exact match
        suffix = build_ranking_filename_suffix(ranking_lengths, strategies, random_count)
        rankings_file = graphs_dir / f"{graph_full_name}_rankings_{suffix}.json"
        strategies_file = graphs_dir / f"{graph_full_name}_ranking_strategies_{suffix}.json"

        if rankings_file.exists():
            return rankings_file, strategies_file if strategies_file.exists() else None
        return None, None
    else:
        # Find first available ranking file for this graph
        pattern = f"{graph_full_name}_rankings_L*_S*.json"
        matches = list(graphs_dir.glob(pattern))

        if matches:
            rankings_file = matches[0]
            # Derive strategies file name from rankings file name
            strategies_file = Path(str(rankings_file).replace('_rankings_', '_ranking_strategies_'))
            return rankings_file, strategies_file if strategies_file.exists() else None

        return None, None


def load_rankings_from_file(rankings_file: Path) -> List[List[int]]:
    """Load rankings from a JSON file."""
    with open(rankings_file, 'r') as f:
        return json.load(f)


def get_ranking_params_from_config(graph_config: dict) -> Tuple[Optional[List[int]], Optional[List[str]], Optional[int]]:
    """
    Extract ranking parameters from graph config's generate_rankings_params.

    Args:
        graph_config: Graph configuration dict from YAML

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

def compute_actual_avg_degree(graph_name, graphs_dir):
    """Load graph and compute actual average degree"""
    edge_file = graphs_dir / f"{graph_name}_edges.txt"
    group_file = graphs_dir / f"{graph_name}_groups.txt"

    try:
        G, _ = load_graph(str(edge_file), str(group_file))
        actual_avg_degree = 2 * len(G.edges()) / len(G.nodes())
        return actual_avg_degree
    except Exception as e:
        print(f"Warning: Could not load graph {graph_name}: {e}")
        return None


def check_sir_result_exists(output_dir: Path, graph_name: str, infection_prob: float) -> bool:
    """
    Check if SIR results already exist for a given graph and infection probability.

    The output file format is: {graph}_sir_p{infection_prob:.4f}.parquet
    """
    output_file = output_dir / f"{graph_name}_sir_p{infection_prob:.4f}.parquet"
    return output_file.exists() and output_file.stat().st_size > 0

def run_sir_job(job, config_path, graphs_dir, output_dir, cores_per_job, stream_output=False):
    """
    Execute a single SIR job.

    If job['mode'] == 'from-methods', it:
        1. Calls select_nodes.py once per ranking to get node lists.
        2. Takes the union of all selected nodes across all rankings.
        3. Calls run_sir.py with that combined list of nodes.

    Args:
        stream_output: If True, stream stdout/stderr to console (for progress bars).
                       If False, capture output (for parallel execution).

    Returns:
        dict with keys:
            'success': bool
            'elapsed': float (seconds)
            'error': str or None
            plus all keys from job
    """

    temp_nodes_file = None

    try:
        if job['mode'] == 'from-methods':
            temp_nodes_file = output_dir / f"temp_nodes_{job['graph']}_{job['infection_prob']:.4f}.txt"

            # Collect the union of nodes selected by ALL rankings
            all_nodes: set = set()
            for ranking in job['rankings']:
                ranking_key = '_'.join(map(str, ranking))
                tmp = output_dir / f"temp_nodes_{job['graph']}_{ranking_key}_{job['infection_prob']:.4f}.txt"
                select_cmd = [
                    sys.executable, 'src/select_nodes.py',
                    '--methods-csv', str(job['methods_csv']),
                    '--ranking', *[str(r) for r in ranking],
                    '--output-file', str(tmp)
                ]
                if job['node_selection'] == 'rank-1':
                    select_cmd.append('--rank-1-only')
                elif job['node_selection'] == 'top-k':
                    select_cmd.extend(['--top-k', str(job['k_value'])])
                elif job['node_selection'] == 'max-rank':
                    select_cmd.extend(['--max-rank', str(job['k_value'])])

                select_result = subprocess.run(select_cmd, capture_output=True, text=True, encoding='utf-8',
                                               env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}, timeout=600)
                if select_result.returncode != 0:
                    return {'success': False, 'elapsed': 0, 'error': f"Node selection failed for ranking {ranking}:\n{select_result.stderr}", **job}

                if tmp.exists() and tmp.stat().st_size > 0:
                    with open(tmp, 'r', encoding='utf-8') as f:
                        all_nodes.update(line.strip() for line in f if line.strip())
                    tmp.unlink()

            # Write the union of all selected nodes
            with open(temp_nodes_file, 'w', encoding='utf-8') as f:
                for node in sorted(all_nodes):
                    f.write(f"{node}\n")

            # If no nodes were selected across all rankings, skip SIR
            if not all_nodes:
                print(f"Note: No nodes selected for {job['graph']} across all rankings. Skipping SIR.")
                return {'success': True, 'elapsed': 0, 'error': "No nodes selected", **job}

        sir_cmd = [
            sys.executable, 'src/run_sir.py',
            '--config', str(config_path),
            '--graph', job['graph'],
            '--graphs-dir', str(graphs_dir),
            '--output-dir', str(output_dir),
            '--infection-prob', str(job['infection_prob']),
            '--num-processes', str(cores_per_job)
        ]

        if job['mode'] == 'all-nodes':
            sir_cmd.append('--all-nodes')
        elif job['mode'] == 'from-methods':
            sir_cmd.extend(['--from-file', str(temp_nodes_file)])

        start_time = datetime.now()

        if stream_output:
            # Stream output to console (shows progress bar)
            sir_result = subprocess.run(sir_cmd)
        else:
            # Capture output (for parallel execution to avoid mixed output)
            sir_result = subprocess.run(sir_cmd, capture_output=True, text=True, encoding='utf-8',
                                        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})

        elapsed = (datetime.now() - start_time).total_seconds()

        if sir_result.returncode != 0:
            error_msg = sir_result.stderr if hasattr(sir_result, 'stderr') and sir_result.stderr else "Unknown error"
            return {'success': False, 'elapsed': elapsed, 'error': f"SIR simulation failed:\n{error_msg}", **job}

        return {'success': True, 'elapsed': elapsed, 'error': None, **job}

    except Exception as e:
        return {'success': False, 'elapsed': 0, 'error': str(e), **job}

    finally:
        if temp_nodes_file and temp_nodes_file.exists(): # Check if it was created and exists
            temp_nodes_file.unlink()

def main():
    parser = argparse.ArgumentParser(
        description='Run SIR simulations on multiple graphs (config-driven)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The script automatically determines which graphs need full SIR vs fast methods
by reading the experiment config file.

Examples:
  # All sir_optimal graphs (no methods needed)
  python3 run_sir_batch.py --config exp/config.yaml --graphs-dir data --output-dir results/sir --mode outer

  # Mix of sir_optimal and fast_only
  python3 run_sir_batch.py --config exp/config.yaml --graphs-dir data --output-dir results/sir \
    --methods-dir results/methods --node-selection rank-1 --mode outer
        """
    )

    # Required paths
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Experiment configuration file'
    )
    parser.add_argument(
        '--graphs-dir',
        type=Path,
        required=True,
        help='Directory containing graph files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for SIR results'
    )

    # Optional (for fast_only graphs)
    parser.add_argument(
        '--methods-dir',
        type=Path,
        help='Directory containing method results CSV files (required for fast_only graphs)'
    )
    parser.add_argument(
        '--node-selection',
        type=str,
        default='rank-1',
        choices=['rank-1', 'top-k', 'max-rank'],
        help='Node selection strategy for fast_only graphs (default: rank-1)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='K value for top-k or max-rank (default: 10)'
    )

    # Parallelism
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['outer', 'inner', 'none'],
        help='Parallelism mode'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of workers (default: auto = num_cores - 1)'
    )
    parser.add_argument(
        '--ranking-lengths',
        nargs='+',
        type=int,
        help="Ranking lengths to look for when using 'auto' rankings (e.g., 2 3 5). "
             "Must match the parameters used in generate_graph_ranking_metadata.py."
    )
    parser.add_argument(
        '--strategies',
        nargs='+',
        type=str,
        default=['all'],
        help="Strategies to look for when using 'auto' rankings (e.g., 'all' or 'top_size random'). "
             "Must match the parameters used in generate_graph_ranking_metadata.py."
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help="Skip simulations where output file already exists (default: True)"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force re-run of all simulations, even if output files exist"
    )

    args = parser.parse_args()

    # Handle --force overriding --skip-existing
    if args.force:
        args.skip_existing = False

    # Load config
    config = load_config(args.config)

    # Get graphs
    sir_optimal_graphs = config['graphs'].get('sir_optimal', [])
    fast_only_graphs = config['graphs'].get('fast_only', [])

    # Validate: if fast_only graphs exist, require methods-dir
    if fast_only_graphs and not args.methods_dir:
        print(f"Error: --methods-dir is required because config has fast_only graphs")
        return 1

    # Get infection probabilities (used directly, not divided by avg_degree)
    infection_probabilities = config['sir']['infection_probabilities']

    # Get number of replicates (default to 1 if not specified)
    replicates = config['global'].get('replicates_per_config', 1)
    if replicates == 0:
        replicates = 1

    # Auto-detect workers if not specified
    total_cores = cpu_count()
    if args.workers is None:
        args.workers = max(1, total_cores - 1)

    # Set parallelism based on mode
    if args.mode == 'outer':
        parallel_jobs = args.workers
        cores_per_job = 1
    elif args.mode == 'inner':
        parallel_jobs = 1
        cores_per_job = args.workers
    else:  # none
        parallel_jobs = 1
        cores_per_job = 1

    # Build job list
    jobs = []

    # Process sir_optimal graphs
    for graph_config in sir_optimal_graphs:
        graph_base_name = graph_config['name']
        graph_type = graph_config['type']

        # Determine replicate range based on graph type
        if graph_type == 'real':
            num_reps = 1
        else:
            num_reps = replicates

        for rep in range(num_reps):
            # Construct graph name based on type
            if graph_type == 'real':
                graph_name = graph_base_name
            else:
                graph_name = f"{graph_base_name}_rep{rep}"

            # Create job for each infection probability
            for infection_prob in infection_probabilities:
                jobs.append({
                    'graph': graph_name,
                    'infection_prob': infection_prob,
                    'mode': 'all-nodes',
                })

    # Process fast_only graphs
    for graph_config in fast_only_graphs:
        graph_base_name = graph_config['name']
        graph_type = graph_config.get('type', 'random')  # Default to random if not specified

        # Determine replicate range based on graph type
        if graph_type == 'real':
            num_reps = 1
        else:
            num_reps = replicates

        for rep in range(num_reps):
            # Construct graph name based on type
            if graph_type == 'real':
                graph_name = graph_base_name
            else:
                graph_name = f"{graph_base_name}_rep{rep}"

            # Find corresponding methods file (parquet first, then CSV)
            methods_file = args.methods_dir / f"{graph_name}_methods.parquet"
            if not methods_file.exists():
                methods_file = args.methods_dir / f"{graph_name}_methods.csv"
            if not methods_file.exists():
                print(f"Skipping {graph_name}: methods file not found")
                continue

            # --- RANKING RESOLUTION LOGIC ---
            rankings_to_test: List[List[int]] = []
            rankings_source_in_config = graph_config.get('rankings')

            # Case 1: 'auto' specified, load from ranking JSON files
            if rankings_source_in_config == 'auto':
                # Get ranking params from config or command line
                # Command line takes precedence
                ranking_lengths = args.ranking_lengths
                strategies = args.strategies
                random_count = None

                if not ranking_lengths:
                    # Try to get from config's generate_rankings_params
                    config_lengths, config_strategies, config_random_count = get_ranking_params_from_config(graph_config)
                    if config_lengths:
                        ranking_lengths = config_lengths
                        if config_strategies and args.strategies == ['all']:
                            strategies = config_strategies
                        random_count = config_random_count
                        print(f"  Using ranking params from config: lengths={ranking_lengths}, strategies={strategies}, random_count={random_count}")

                rankings_file, _ = find_ranking_file(
                    args.graphs_dir, graph_name,
                    ranking_lengths, strategies, random_count
                )

                if rankings_file and rankings_file.exists():
                    rankings_to_test = load_rankings_from_file(rankings_file)
                    print(f"  Loaded {len(rankings_to_test)} rankings from {rankings_file.name} for {graph_name}")
                else:
                    # Try to find any ranking file if specific params not found
                    if ranking_lengths:
                        print(f"  Warning: 'rankings: auto' but no ranking file found for "
                              f"{graph_name} with lengths={ranking_lengths}, strategies={strategies}")
                        # Fall back to finding any available ranking file
                        rankings_file, _ = find_ranking_file(args.graphs_dir, graph_name)
                        if rankings_file:
                            rankings_to_test = load_rankings_from_file(rankings_file)
                            print(f"  Fallback: Loaded {len(rankings_to_test)} rankings from {rankings_file.name}")
                    else:
                        print(f"  Warning: 'rankings: auto' but no ranking file found for {graph_name}")
                        print(f"    Run generate_graph_ranking_metadata.py first, or specify --ranking-lengths")

            # Case 2: load rankings from config explicitly (List[List[int]])
            elif isinstance(rankings_source_in_config, list) and \
                all(isinstance(r, list) and all(isinstance(i, int) for i in r) for r in rankings_source_in_config):
                rankings_to_test = rankings_source_in_config
                print(f"  Using {len(rankings_to_test)} explicit rankings from config for {graph_name}")

            # Case 3: No rankings defined, or invalid format
            else:
                print(f"  Warning: No valid explicit or 'auto' rankings defined in config for {graph_name}.")

            # If no rankings were resolved, skip this graph
            if not rankings_to_test:
                print(f"⚠️ Skipping {graph_name}: no rankings resolved.")
                continue

            # Create ONE job per infection probability; nodes are the union across all rankings
            for infection_prob in infection_probabilities:
                jobs.append({
                    'graph': graph_name,
                    'infection_prob': infection_prob,
                    'mode': 'from-methods',
                    'rankings': rankings_to_test,  # ALL rankings → union of nodes
                    'methods_csv': methods_file,
                    'node_selection': args.node_selection,
                    'k_value': args.k,
                })

    if not jobs:
        print(f"No jobs to run (check config and graph files)")
        return 1

    # Filter out already completed simulations if --skip-existing is enabled
    original_job_count = len(jobs)
    skipped_jobs = []
    if args.skip_existing:
        remaining_jobs = []
        for job in jobs:
            if check_sir_result_exists(args.output_dir, job['graph'], job['infection_prob']):
                skipped_jobs.append(job)
            else:
                remaining_jobs.append(job)
        jobs = remaining_jobs

        if skipped_jobs:
            print(f"Skipping {len(skipped_jobs)} jobs with existing results (use --force to re-run)")

    if not jobs:
        print(f"All {original_job_count} simulations already completed. Nothing to do.")
        print(f"Use --force to re-run all simulations.")
        return 0

    # Print configuration
    num_sir_optimal = sum(1 for j in jobs if j['mode'] == 'all-nodes')
    num_fast_only = sum(1 for j in jobs if j['mode'] == 'from-methods')
    unique_graphs = len(set(j['graph'] for j in jobs))

    print(f"{'='*80}")
    print(f"SIR BATCH PROCESSING (Config-Driven)")
    print(f"{'='*80}")
    print(f"Graphs: {unique_graphs} ({len(sir_optimal_graphs)} sir_optimal x {replicates} reps, "
          f"{len(fast_only_graphs)} fast_only x {replicates} reps)")
    print(f"Infection probabilities: {len(infection_probabilities)} ({infection_probabilities})")
    print(f"Total jobs: {len(jobs)} ({num_sir_optimal} all-nodes, {num_fast_only} from-methods)")
    print(f"Parallelism mode: {args.mode}")
    if args.mode == 'outer':
        print(f"  -> {parallel_jobs} jobs in parallel, {cores_per_job} core(s) each")
    elif args.mode == 'inner':
        print(f"  -> 1 job at a time, {cores_per_job} cores per job")
    else:
        print(f"  -> Sequential (no parallelism)")
    print(f"Total CPU cores: {total_cores}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    # Execute jobs
    results = []
    completed = 0

    if parallel_jobs == 1:
        # Sequential execution - stream output to show progress bar
        for job in jobs:
            result = run_sir_job(job, args.config, args.graphs_dir, args.output_dir, cores_per_job, stream_output=True)
            results.append(result)
            completed += 1

            status = "✓" if result['success'] else "✗"
            mode_str = "all" if result['mode'] == 'all-nodes' else "sel"
            print(f"[{completed}/{len(jobs)}] {status} {result['graph']:<45} "
                  f"p={result['infection_prob']:.4f} [{mode_str}] ({result['elapsed']:.1f}s)")
            if not result['success'] and result['error']:
                for line in result['error'].strip().splitlines():
                    print(f"    | {line}")
    else:
        # Parallel execution - capture output to avoid mixed progress bars
        with ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
            futures = {
                executor.submit(run_sir_job, job, args.config, args.graphs_dir,
                                args.output_dir, cores_per_job, False): job
                for job in jobs
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1

                status = "ok!" if result['success'] else "oh no!"
                mode_str = "all" if result['mode'] == 'all-nodes' else "sel"
                print(f"[{completed}/{len(jobs)}] {status} {result['graph']:<45} "
                      f"p={result['infection_prob']:.4f} [{mode_str}] ({result['elapsed']:.1f}s)")

    # Summary
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    total_time = sum(r['elapsed'] for r in results)

    print(f"\n{'='*80}")
    print(f"COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"Success: {len(successful)}/{len(results)}")
    if skipped_jobs:
        print(f"Skipped (already completed): {len(skipped_jobs)}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    if failed:
        print(f"\n :( Failed jobs ({len(failed)}):")
        for result in failed:
            print(f"  - {result['graph']} (p={result['infection_prob']:.4f})")
            if result['error']:
                # Show last line of error (full message)
                error_lines = result['error'].strip().split('\n')
                print(f"    {error_lines[-1]}")
    else:
        print(f"\n :) All jobs completed successfully!")

    print(f"\nResults saved to: {args.output_dir}")
    print(f"  SIR files: {len(successful)} (format: graph_sir_pX.XXXX.parquet)")
    print(f"{'='*80}")

    return 0 if not failed else 1


if __name__ == '__main__':
    exit(main())
