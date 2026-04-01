#!/usr/bin/env python3
"""
Run Methods

Runs centrality methods on graphs and saves results.

Output: Parquet file with method results, with the following structure:
        graph_name,replicate_id,ranking,method,node_id,rank,value,runtime

Ranking resolution:
  - If 'rankings: auto' in config, looks for {graph}_rankings_L{lengths}_S{strategies}.json
  - Use --ranking-lengths and --strategies to specify which ranking file to load
  - Or specify explicit rankings in config as List[List[int]]
"""

import yaml
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path
import sys
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from methods import CentralityMethods, load_graph, compute_betas_configs, FAIRGD_AVAILABLE, FAIRLAR_AVAILABLE
from generate_rankings import build_ranking_filename_suffix, parse_ranking_filename_suffix


def load_config(config_path: Path) -> dict:
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_methods_from_file(methods_file: Path) -> List[str]:
    """
    Load methods from a text file (one method per line).

    Lines starting with '#' are treated as comments and ignored.
    Empty lines are also ignored.

    Returns:
        List of method names
    """
    methods = []
    with open(methods_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                methods.append(line)
    return methods


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


def load_metadata(metadata_path: Path) -> dict:
    """Load graph metadata"""
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


def load_ranking_strategies_from_file(strategies_file: Path) -> Dict[str, List[str]]:
    """Load ranking-to-strategy mapping from a JSON file."""
    with open(strategies_file, 'r') as f:
        return json.load(f)


def run_methods_for_graph(
    graph_name: str,
    graph_type: str,
    replicate_id: int,
    rankings: List[List[int]],
    graphs_dir: Path,
    output_dir: Path,
    methods_to_run: List[str] = None,
    append_mode: bool = False,
    timeout: int = 3600,
    betas_config: dict = None
) -> Dict[str, Any]:
    """
    Run centrality methods for a single graph

    Args:
        graph_name: Name of graph configuration
        replicate_id: Replicate ID
        rankings: List of rankings to test
        graphs_dir: Directory containing graph files
        output_dir: Output directory
        methods_to_run: List of method names to run (None = all)
        timeout: Maximum time in seconds for each method (default: 3600)
        betas_config: Optional list of beta values for beta-blended variants,
                      e.g. [0.99, 0.5, 0.1]. Each value defines a strategy where all
                      betas are equal to that value.

    Returns:
        Dict with summary statistics
    """
    # File paths
    if graph_type == 'real':
        prefix = f"{graph_name}"
    else:
        prefix = f"{graph_name}_rep{replicate_id}"
    edge_file = graphs_dir / f"{prefix}_edges.txt"
    group_file = graphs_dir / f"{prefix}_groups.txt"
    #metadata_file = graphs_dir / f"{prefix}_metadata.json"

    print(f"\n{'='*80}")
    print(f"METHODS: {prefix}")
    print(f"{'='*80}")

    # Load graph and metadata
    print("Loading graph...")
    G, group_assignments = load_graph(str(edge_file), str(group_file))
    #metadata = load_metadata(metadata_file)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    avg_deg = 2 * m / n

    print(f"Nodes: {n}")
    print(f"Edges: {m}")
    print(f"Avg degree: {avg_deg:.2f}")
    print(f"Groups: {len(set(group_assignments.values()))}")
    print(f"Rankings: {len(rankings)}")

    # Initialize methods
    methods = CentralityMethods(G, group_assignments)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results storage
    all_results = []
    betas_metadata = {}  # ranking_str -> label -> metadata
    output_file = output_dir / f"{prefix}_methods.parquet"
    _writer = [None]  # ParquetWriter kept open across flushes

    def flush_results():
        """Append all_results as a new row group and clear the list to free memory."""
        if not all_results:
            return
        table = pa.Table.from_pandas(pd.DataFrame(all_results), preserve_index=False)
        if _writer[0] is None:
            if append_mode and output_file.exists():
                # Read existing data once, then open writer (which truncates the file)
                existing = pq.read_table(str(output_file))
                _writer[0] = pq.ParquetWriter(str(output_file), existing.schema)
                _writer[0].write_table(existing)
                del existing
            else:
                _writer[0] = pq.ParquetWriter(str(output_file), table.schema)
        _writer[0].write_table(table)
        all_results.clear()

    overall_start = time.time()

    # Methods with vector values (tuples) - for output formatting
    vector_methods = [
        'lexipeeling', 'lexipeeling_extended', 'degree_lexicographic',
        'lexipeeling_coreness', 'lexipeeling_degree', 'lexipeeling_degree_lexicographic',
        'degree_lexicographic_lexipeeling',
        'restricted_degree_lexicographic'
    ]

    # Helper function to store results for a method
    def store_method_results(method_name, method_data, ranking_str):
        print(f"{method_name}: top node = {method_data['top_spreader']}")
        for node, (value, rank) in method_data['result'].items():
            result_row = {
                'graph_name': graph_name,
                'replicate_id': replicate_id,
                'ranking': ranking_str,
                'method': method_name,
                'node_id': node,
                'rank': rank,
                'runtime': method_data['runtime'],
            }
            # Handle vector vs scalar values - convert all to string for consistent parquet typing
            if isinstance(value, (list, tuple)):
                result_row['value'] = '_'.join(map(str, value))
            else:
                result_row['value'] = str(value)
            all_results.append(result_row)

    # Determine which methods to run
    if methods_to_run:
        # Filter into ranking-independent and ranking-dependent
        ranking_independent_to_run = [
            m for m in methods_to_run
            if m in CentralityMethods.RANKING_INDEPENDENT_METHODS
        ]
        ranking_dependent_to_run = [
            m for m in methods_to_run
            if m in CentralityMethods.RANKING_DEPENDENT_METHODS
        ]
    else:
        ranking_independent_to_run = CentralityMethods.RANKING_INDEPENDENT_METHODS
        ranking_dependent_to_run = CentralityMethods.RANKING_DEPENDENT_METHODS

    # Run ranking-independent methods ONCE (if any requested)
    ranking_independent_results = {}
    timed_out_independent = set()
    if ranking_independent_to_run:
        print(f"\n  Running ranking-independent methods...")
        method_start = time.time()
        method_funcs = {
            'coreness': methods.coreness,
            'degree': methods.degree_centrality,
            'betweenness': methods.betweenness_centrality,
            'pagerank': methods.pagerank,
            'h1_index': methods.h1_index,
            'h2_index': methods.h2_index,
            'h3_index': methods.h3_index,
            'h4_index': methods.h4_index,
            'h5_index': methods.h5_index,
            'closeness_centrality': methods.closeness_centrality,
            'gravity_centrality': methods.gravity_centrality,
            'extended_gravity_centrality': methods.extended_gravity_centrality,
            'local_gravity': methods.local_gravity,
            'gravity_h_index': methods.gravity_h_index,
            'clustering_coefficient_centrality': methods.clustering_coefficient_centrality,
            'local_global_centrality': methods.local_global_centrality,
            'entropy_degree_distance_combination': methods.entropy_degree_distance_combination,
        }
        for method_name in ranking_independent_to_run:
            print(f"Running {method_name}...")
            output, was_timeout = methods._run_with_timeout(
                method_name, method_funcs[method_name], timeout)
            if was_timeout:
                timed_out_independent.add(method_name)
                print(f"  {method_name} timed out, excluding from all rankings")
            elif output is not None:
                result, runtime = output
                ranking_independent_results[method_name] = {
                    'result': result,
                    'runtime': runtime,
                    'top_spreader': methods.get_top_spreader(result)
                }
        method_elapsed = time.time() - method_start
        print(f"Ranking-independent methods completed in {method_elapsed:.2f}s")

    # Track methods that timed out on this graph across rankings.
    # If a ranking-dependent method times out on one ranking, it is skipped for all
    # remaining rankings, and any partial results already collected are purged at the end
    # to ensure an all-or-nothing policy per (graph, method).
    timed_out_methods = set()

    # For each ranking
    for ranking_idx, ranking in enumerate(rankings):
        ranking_str = '_'.join(map(str, ranking))
        print(f"\n  Ranking {ranking_idx + 1}/{len(rankings)}: {ranking}")

        # Store ranking-independent results for this ranking (reusing cached results)
        if ranking_independent_results:
            print("Storing ranking-independent results...")
            for method_name, method_data in ranking_independent_results.items():
                store_method_results(method_name, method_data, ranking_str)

        # Run ranking-dependent methods for this ranking (if any requested)
        if ranking_dependent_to_run:
            print("Running ranking-dependent methods...")
            method_start = time.time()
            if methods_to_run:
                # Run only selected ranking-dependent methods
                # Pass cached ranking-independent results for reuse in composite/restricted methods
                ranking_dependent_results, newly_timed_out = methods.run_selected_methods(
                    ranking, ranking_dependent_to_run,
                    ranking_independent_results=ranking_independent_results,
                    timeout=timeout,
                    skip_methods=timed_out_methods | timed_out_independent
                )
            else:
                # Pass cached ranking-independent results for reuse in composite/restricted methods
                ranking_dependent_results, newly_timed_out = methods.run_ranking_dependent_methods(
                    ranking, ranking_independent_results=ranking_independent_results,
                    timeout=timeout,
                    skip_methods=timed_out_methods | timed_out_independent
                )
            method_elapsed = time.time() - method_start
            print(f"Ranking-dependent methods completed in {method_elapsed:.2f}s")

            # Update the set of timed-out methods for subsequent rankings
            if newly_timed_out:
                timed_out_methods.update(newly_timed_out)
                print(f"Methods timed out (will be skipped for remaining rankings): {sorted(newly_timed_out)}")

            # Store ranking-dependent results
            for method_name, method_data in ranking_dependent_results.items():
                store_method_results(method_name, method_data, ranking_str)

        # Run beta-blended variants if configured
        if betas_config:
            beta_configs = compute_betas_configs(ranking, betas_config)
            if beta_configs:
                print(f"Running betas variants ({len(beta_configs)} configs)...")
                betas_metadata[ranking_str] = {}

                # Get ranking-independent results for composite/restricted methods
                core_result_for_betas = ranking_independent_results.get('coreness', {}).get('result') if ranking_independent_results else None
                deg_result_for_betas = ranking_independent_results.get('degree', {}).get('result') if ranking_independent_results else None

                for label, betas_vector, meta in beta_configs:
                    betas_metadata[ranking_str][label] = meta

                    # --- Base methods ---

                    # betas_degree_lexicographic
                    bdl_name = f"betas_degree_lexicographic_{label}"
                    bdl_result = None  # cache for composites/restricted
                    if bdl_name not in timed_out_methods:
                        print(f"Running {bdl_name}...")
                        output, was_timeout = methods._run_with_timeout(
                            bdl_name,
                            lambda bv=betas_vector: methods.betas_degree_lexicographic(ranking, bv),
                            timeout)
                        if was_timeout:
                            timed_out_methods.add(bdl_name)
                            print(f"{bdl_name} timed out (will be skipped for remaining rankings)")
                        elif output is not None:
                            bdl_result, method_time = output
                            method_data = {
                                'result': bdl_result,
                                'runtime': method_time,
                                'top_spreader': methods.get_top_spreader(bdl_result)
                            }
                            store_method_results(bdl_name, method_data, ranking_str)
                    else:
                        print(f"Skipping {bdl_name} (timed out on this graph)")

                    # betas_lexipeeling
                    blp_name = f"betas_lexipeeling_{label}"
                    blp_result = None  # cache for composites
                    if blp_name not in timed_out_methods:
                        print(f"Running {blp_name}...")
                        output, was_timeout = methods._run_with_timeout(
                            blp_name,
                            lambda bv=betas_vector: methods.betas_lexipeeling(ranking, bv),
                            timeout)
                        if was_timeout:
                            timed_out_methods.add(blp_name)
                            print(f"{blp_name} timed out (will be skipped for remaining rankings)")
                        elif output is not None:
                            blp_result, method_time = output
                            method_data = {
                                'result': blp_result,
                                'runtime': method_time,
                                'top_spreader': methods.get_top_spreader(blp_result)
                            }
                            store_method_results(blp_name, method_data, ranking_str)
                    else:
                        print(f"Skipping {blp_name} (timed out on this graph)")

                    # --- Composite methods (using _combine_with_tiebreak) ---
                    # These are essentially free since they reuse cached results.

                    if blp_result is not None:
                        # betas_lexipeeling + coreness tiebreaker
                        if core_result_for_betas is not None:
                            name = f"betas_lexipeeling_coreness_{label}"
                            combined = methods._combine_with_tiebreak(blp_result, core_result_for_betas)
                            store_method_results(name, {
                                'result': combined, 'runtime': 0.0,
                                'top_spreader': methods.get_top_spreader(combined)
                            }, ranking_str)

                        # betas_lexipeeling + degree tiebreaker
                        if deg_result_for_betas is not None:
                            name = f"betas_lexipeeling_degree_{label}"
                            combined = methods._combine_with_tiebreak(blp_result, deg_result_for_betas)
                            store_method_results(name, {
                                'result': combined, 'runtime': 0.0,
                                'top_spreader': methods.get_top_spreader(combined)
                            }, ranking_str)

                        # betas_lexipeeling + betas_degree_lexicographic tiebreaker
                        if bdl_result is not None:
                            name = f"betas_lexipeeling_betas_degree_lexicographic_{label}"
                            combined = methods._combine_with_tiebreak(blp_result, bdl_result)
                            store_method_results(name, {
                                'result': combined, 'runtime': 0.0,
                                'top_spreader': methods.get_top_spreader(combined)
                            }, ranking_str)

                    # Reverse composites: standard primary + betas_lexipeeling tiebreaker
                    if blp_result is not None:
                        if core_result_for_betas is not None:
                            name = f"coreness_betas_lexipeeling_{label}"
                            combined = methods._combine_with_tiebreak(core_result_for_betas, blp_result)
                            store_method_results(name, {
                                'result': combined, 'runtime': 0.0,
                                'top_spreader': methods.get_top_spreader(combined)
                            }, ranking_str)

                        if deg_result_for_betas is not None:
                            name = f"degree_betas_lexipeeling_{label}"
                            combined = methods._combine_with_tiebreak(deg_result_for_betas, blp_result)
                            store_method_results(name, {
                                'result': combined, 'runtime': 0.0,
                                'top_spreader': methods.get_top_spreader(combined)
                            }, ranking_str)

                    # betas_degree_lexicographic + betas_lexipeeling tiebreaker
                    if bdl_result is not None and blp_result is not None:
                        name = f"betas_degree_lexicographic_betas_lexipeeling_{label}"
                        combined = methods._combine_with_tiebreak(bdl_result, blp_result)
                        store_method_results(name, {
                            'result': combined, 'runtime': 0.0,
                            'top_spreader': methods.get_top_spreader(combined)
                        }, ranking_str)

                    # --- Restricted methods ---

                    if bdl_result is not None:
                        name = f"restricted_betas_degree_lexicographic_{label}"
                        restricted = methods._restrict_to_ranking_groups(bdl_result, ranking)
                        store_method_results(name, {
                            'result': restricted, 'runtime': 0.0,
                            'top_spreader': methods.get_top_spreader(restricted)
                        }, ranking_str)

                        name = f"restricted_first_betas_degree_lexicographic_{label}"
                        restricted_first = methods._restrict_to_first_group(bdl_result, ranking)
                        store_method_results(name, {
                            'result': restricted_first, 'runtime': 0.0,
                            'top_spreader': methods.get_top_spreader(restricted_first)
                        }, ranking_str)

                    # --- Betas PageRank-based methods ---

                    beta_value = meta['beta_value']

                    # betas_pagerank_exponential_jump
                    bpej_name = f"betas_pagerank_exponential_jump_{label}"
                    bpej_result = None
                    if bpej_name not in timed_out_methods:
                        print(f"Running {bpej_name}...")
                        output, was_timeout = methods._run_with_timeout(
                            bpej_name,
                            lambda bv=beta_value: methods.betas_pagerank_exponential_jump(ranking, bv),
                            timeout)
                        if was_timeout:
                            timed_out_methods.add(bpej_name)
                            print(f"{bpej_name} timed out (will be skipped for remaining rankings)")
                        elif output is not None:
                            bpej_result, method_time = output
                            store_method_results(bpej_name, {
                                'result': bpej_result, 'runtime': method_time,
                                'top_spreader': methods.get_top_spreader(bpej_result)
                            }, ranking_str)
                    else:
                        print(f"Skipping {bpej_name} (timed out on this graph)")

                    if bpej_result is not None:
                        name = f"restricted_betas_pagerank_exponential_jump_{label}"
                        restricted = methods._restrict_to_ranking_groups(bpej_result, ranking)
                        store_method_results(name, {
                            'result': restricted, 'runtime': 0.0,
                            'top_spreader': methods.get_top_spreader(restricted)
                        }, ranking_str)

                        name = f"restricted_first_betas_pagerank_exponential_jump_{label}"
                        restricted_first = methods._restrict_to_first_group(bpej_result, ranking)
                        store_method_results(name, {
                            'result': restricted_first, 'runtime': 0.0,
                            'top_spreader': methods.get_top_spreader(restricted_first)
                        }, ranking_str)

                    # betas_fairgd
                    if FAIRGD_AVAILABLE:
                        bfgd_name = f"betas_fairgd_{label}"
                        bfgd_result = None
                        if bfgd_name not in timed_out_methods:
                            print(f"Running {bfgd_name}...")
                            output, was_timeout = methods._run_with_timeout(
                                bfgd_name,
                                lambda bv=beta_value: methods.betas_fairgd(ranking, bv),
                                timeout)
                            if was_timeout:
                                timed_out_methods.add(bfgd_name)
                                print(f"{bfgd_name} timed out (will be skipped for remaining rankings)")
                            elif output is not None:
                                bfgd_result, method_time = output
                                store_method_results(bfgd_name, {
                                    'result': bfgd_result, 'runtime': method_time,
                                    'top_spreader': methods.get_top_spreader(bfgd_result)
                                }, ranking_str)
                        else:
                            print(f"Skipping {bfgd_name} (timed out on this graph)")

                        if bfgd_result is not None:
                            name = f"restricted_betas_fairgd_{label}"
                            restricted = methods._restrict_to_ranking_groups(bfgd_result, ranking)
                            store_method_results(name, {
                                'result': restricted, 'runtime': 0.0,
                                'top_spreader': methods.get_top_spreader(restricted)
                            }, ranking_str)

                            name = f"restricted_first_betas_fairgd_{label}"
                            restricted_first = methods._restrict_to_first_group(bfgd_result, ranking)
                            store_method_results(name, {
                                'result': restricted_first, 'runtime': 0.0,
                                'top_spreader': methods.get_top_spreader(restricted_first)
                            }, ranking_str)

                    # betas_adaptgd
                    if FAIRGD_AVAILABLE:
                        bagd_name = f"betas_adaptgd_{label}"
                        bagd_result = None
                        if bagd_name not in timed_out_methods:
                            print(f"Running {bagd_name}...")
                            output, was_timeout = methods._run_with_timeout(
                                bagd_name,
                                lambda bv=beta_value: methods.betas_adaptgd(ranking, bv),
                                timeout)
                            if was_timeout:
                                timed_out_methods.add(bagd_name)
                                print(f"{bagd_name} timed out (will be skipped for remaining rankings)")
                            elif output is not None:
                                bagd_result, method_time = output
                                store_method_results(bagd_name, {
                                    'result': bagd_result, 'runtime': method_time,
                                    'top_spreader': methods.get_top_spreader(bagd_result)
                                }, ranking_str)
                        else:
                            print(f"Skipping {bagd_name} (timed out on this graph)")

                        if bagd_result is not None:
                            name = f"restricted_betas_adaptgd_{label}"
                            restricted = methods._restrict_to_ranking_groups(bagd_result, ranking)
                            store_method_results(name, {
                                'result': restricted, 'runtime': 0.0,
                                'top_spreader': methods.get_top_spreader(restricted)
                            }, ranking_str)

                            name = f"restricted_first_betas_adaptgd_{label}"
                            restricted_first = methods._restrict_to_first_group(bagd_result, ranking)
                            store_method_results(name, {
                                'result': restricted_first, 'runtime': 0.0,
                                'top_spreader': methods.get_top_spreader(restricted_first)
                            }, ranking_str)

                    # betas_fspr (binary only)
                    if FAIRLAR_AVAILABLE and FAIRGD_AVAILABLE and len(ranking) == 2:
                        bfspr_name = f"betas_fspr_{label}"
                        bfspr_result = None
                        if bfspr_name not in timed_out_methods:
                            print(f"Running {bfspr_name}...")
                            output, was_timeout = methods._run_with_timeout(
                                bfspr_name,
                                lambda bv=beta_value: methods.betas_fspr(ranking, bv),
                                timeout)
                            if was_timeout:
                                timed_out_methods.add(bfspr_name)
                                print(f"{bfspr_name} timed out (will be skipped for remaining rankings)")
                            elif output is not None:
                                bfspr_result, method_time = output
                                store_method_results(bfspr_name, {
                                    'result': bfspr_result, 'runtime': method_time,
                                    'top_spreader': methods.get_top_spreader(bfspr_result)
                                }, ranking_str)
                        else:
                            print(f"Skipping {bfspr_name} (timed out on this graph)")

                        if bfspr_result is not None:
                            name = f"restricted_betas_fspr_{label}"
                            restricted = methods._restrict_to_ranking_groups(bfspr_result, ranking)
                            store_method_results(name, {
                                'result': restricted, 'runtime': 0.0,
                                'top_spreader': methods.get_top_spreader(restricted)
                            }, ranking_str)

                            name = f"restricted_first_betas_fspr_{label}"
                            restricted_first = methods._restrict_to_first_group(bfspr_result, ranking)
                            store_method_results(name, {
                                'result': restricted_first, 'runtime': 0.0,
                                'top_spreader': methods.get_top_spreader(restricted_first)
                            }, ranking_str)

                    # betas_lfpr_u (binary only)
                    if FAIRGD_AVAILABLE and len(ranking) == 2:
                        blu_name = f"betas_lfpr_u_{label}"
                        blu_result = None
                        if blu_name not in timed_out_methods:
                            print(f"Running {blu_name}...")
                            output, was_timeout = methods._run_with_timeout(
                                blu_name,
                                lambda bv=beta_value: methods.betas_lfpr_u(ranking, bv),
                                timeout)
                            if was_timeout:
                                timed_out_methods.add(blu_name)
                                print(f"{blu_name} timed out (will be skipped for remaining rankings)")
                            elif output is not None:
                                blu_result, method_time = output
                                store_method_results(blu_name, {
                                    'result': blu_result, 'runtime': method_time,
                                    'top_spreader': methods.get_top_spreader(blu_result)
                                }, ranking_str)
                        else:
                            print(f"Skipping {blu_name} (timed out on this graph)")

                        if blu_result is not None:
                            name = f"restricted_betas_lfpr_u_{label}"
                            restricted = methods._restrict_to_ranking_groups(blu_result, ranking)
                            store_method_results(name, {
                                'result': restricted, 'runtime': 0.0,
                                'top_spreader': methods.get_top_spreader(restricted)
                            }, ranking_str)

                            name = f"restricted_first_betas_lfpr_u_{label}"
                            restricted_first = methods._restrict_to_first_group(blu_result, ranking)
                            store_method_results(name, {
                                'result': restricted_first, 'runtime': 0.0,
                                'top_spreader': methods.get_top_spreader(restricted_first)
                            }, ranking_str)

                    # betas_lfpr_n (binary only)
                    if FAIRGD_AVAILABLE and len(ranking) == 2:
                        bln_name = f"betas_lfpr_n_{label}"
                        bln_result = None
                        if bln_name not in timed_out_methods:
                            print(f"Running {bln_name}...")
                            output, was_timeout = methods._run_with_timeout(
                                bln_name,
                                lambda bv=beta_value: methods.betas_lfpr_n(ranking, bv),
                                timeout)
                            if was_timeout:
                                timed_out_methods.add(bln_name)
                                print(f"{bln_name} timed out (will be skipped for remaining rankings)")
                            elif output is not None:
                                bln_result, method_time = output
                                store_method_results(bln_name, {
                                    'result': bln_result, 'runtime': method_time,
                                    'top_spreader': methods.get_top_spreader(bln_result)
                                }, ranking_str)
                        else:
                            print(f"Skipping {bln_name} (timed out on this graph)")

                        if bln_result is not None:
                            name = f"restricted_betas_lfpr_n_{label}"
                            restricted = methods._restrict_to_ranking_groups(bln_result, ranking)
                            store_method_results(name, {
                                'result': restricted, 'runtime': 0.0,
                                'top_spreader': methods.get_top_spreader(restricted)
                            }, ranking_str)

                            name = f"restricted_first_betas_lfpr_n_{label}"
                            restricted_first = methods._restrict_to_first_group(bln_result, ranking)
                            store_method_results(name, {
                                'result': restricted_first, 'runtime': 0.0,
                                'top_spreader': methods.get_top_spreader(restricted_first)
                            }, ranking_str)

        # Flush results for this ranking to disk and free memory
        flush_results()

    # Flush any remaining results (e.g. if betas_config was empty, no per-ranking flush ran)
    flush_results()

    # Close the writer so the file is finalised
    if _writer[0] is not None:
        _writer[0].close()
        _writer[0] = None

    # Purge partial results for methods that timed out on this graph.
    # This ensures all-or-nothing: a method either has results for ALL rankings or NONE.
    # Results are already on disk, so we filter the parquet file directly using pyarrow
    # (avoids loading into a pandas DataFrame which would double peak memory).
    if timed_out_methods and output_file.exists():
        table = pq.read_table(str(output_file))
        before_count = len(table)
        mask = ~pc.is_in(table['method'], value_set=pa.array(list(timed_out_methods)))
        filtered = table.filter(mask)
        purged_count = before_count - len(filtered)
        pq.write_table(filtered, str(output_file))
        print(f"\n  Purged {purged_count} partial result rows for timed-out methods: {sorted(timed_out_methods)}")

    if output_file.exists():
        print(f"\n  {'Appended to' if append_mode else 'Saved'} results: {output_file}")

    # Save betas metadata if any betas variants were computed
    if betas_metadata:
        betas_meta_file = output_dir / f"{prefix}_betas_metadata.json"
        betas_meta_output = {
            'rankings': betas_metadata,
        }
        with open(betas_meta_file, 'w') as f:
            json.dump(betas_meta_output, f, indent=2)
        print(f"Saved betas metadata: {betas_meta_file}")

    overall_elapsed = time.time() - overall_start
    num_methods = len(ranking_independent_to_run) + len(ranking_dependent_to_run)
    print(f"\n  Total time for {prefix}: {overall_elapsed:.1f}s")
    print(f"Methods run: {num_methods} ({len(ranking_independent_to_run)} ranking-independent + {len(ranking_dependent_to_run)} ranking-dependent)")

    return {
        'graph_name': graph_name,
        'replicate_id': replicate_id,
        'num_nodes': n,
        'num_methods': num_methods,
        'num_rankings': len(rankings),
        'total_time': overall_elapsed
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run centrality methods on graphs"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiment_config.yaml"),
        help="Path to experiment configuration"
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
        default=Path("results/methods"),
        help="Output directory"
    )
    parser.add_argument(
        "--tier",
        choices=['sir_optimal', 'fast_only', 'all'],
        default='all',
        help="Which tier of graphs to process"
    )
    parser.add_argument(
        "--graphs",
        nargs='+',
        help="Specific graphs to process"
    )
    parser.add_argument(
        "--methods",
        nargs='+',
        help="Specific methods to run (default: all)"
    )
    parser.add_argument(
        "--methods-file",
        type=Path,
        help="Path to a text file with methods to run (one method per line). "
             "Lines starting with '#' are treated as comments. "
             "Can also be specified in config under global.methods_file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help="Append results to existing CSV files instead of overwriting."
    )
    parser.add_argument(
        "--ranking-lengths",
        nargs='+',
        type=int,
        help="Ranking lengths to look for when using 'auto' rankings (e.g., 2 3 5). "
             "Must match the parameters used in generate_graph_ranking_metadata.py."
    )
    parser.add_argument(
        "--strategies",
        nargs='+',
        type=str,
        default=['all'],
        help="Strategies to look for when using 'auto' rankings (e.g., 'all' or 'top_size random'). "
             "Must match the parameters used in generate_graph_ranking_metadata.py."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Maximum time in seconds for each method (default: 3600 = 1 hour). "
             "Methods exceeding this time will be skipped."
    )

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / args.config
    graphs_dir = project_root / args.graphs_dir
    output_dir = project_root / args.output_dir

    # Load config
    config = load_config(config_path)
    num_replicates = config['global'].get('replicates_per_config', 0)
    if num_replicates == 0:
        num_replicates = 1
        if args.verbose:
            print("Note: replicates_per_config not set, defaulting to 1")

    # Determine which graphs to process
    if args.graphs:
        # Specific graphs requested
        if args.tier == 'sir_optimal':
            graph_configs = [
                gc for gc in config['graphs']['sir_optimal']
                if gc['name'] in args.graphs
            ]
        elif args.tier == 'fast_only':
            graph_configs = [
                gc for gc in config['graphs']['fast_only']
                if gc['name'] in args.graphs
            ]
        else:  # all
            graph_configs = [
                gc for gc in config['graphs']['sir_optimal'] + config['graphs']['fast_only']
                if gc['name'] in args.graphs
            ]
    else:
        # All graphs in tier
        if args.tier == 'sir_optimal':
            graph_configs = config['graphs']['sir_optimal']
        elif args.tier == 'fast_only':
            graph_configs = config['graphs']['fast_only']
        else:  # all
            graph_configs = config['graphs']['sir_optimal'] + config['graphs']['fast_only']

    # Load methods from file
    # Priority: CLI --methods > CLI --methods-file > config methods_file
    methods_to_run = args.methods
    if not methods_to_run:
        # Check CLI --methods-file first, then config (top-level or under global)
        methods_file_rel = (args.methods_file or
                           config.get('methods_file') or
                           config.get('global', {}).get('methods_file'))
        if methods_file_rel:
            methods_file_path = project_root / methods_file_rel
            if methods_file_path.exists():
                methods_to_run = load_methods_from_file(methods_file_path)
                print(f"Loaded {len(methods_to_run)} methods from {methods_file_rel}")
            else:
                print(f"Error: Methods file not found: {methods_file_path}")
                sys.exit(1)

    # Load betas config (global parameter, optional)
    betas_config = config.get('betas')

    print("="*80)
    print("CENTRALITY METHODS EVALUATION")
    print("="*80)
    print(f"Graphs to process: {len(graph_configs)}")
    print(f"Replicates per graph: {num_replicates}")
    if methods_to_run:
        print(f"Methods: {methods_to_run}")
    else:
        print(f"Methods: all")
    if betas_config:
        print(f"Betas: {betas_config}")
    print(f"Output directory: {output_dir}")

    # Process each graph
    total_start = time.time()
    summaries = []

    for graph_config in graph_configs:
        graph_name = graph_config['name']
        graph_type = graph_config['type']

        for rep_id in range(num_replicates):
            # --- RANKING RESOLUTION LOGIC ---
            resolved_rankings: List[List[int]] = []
            rankings_source_in_config = graph_config.get('rankings')

            # Build graph full name for file lookups
            if graph_type == 'real':
                current_graph_full_name = graph_name
            else:
                current_graph_full_name = f"{graph_name}_rep{rep_id}"

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
                        if args.verbose:
                            print(f"Using ranking params from config: lengths={ranking_lengths}, strategies={strategies}, random_count={random_count}")

                rankings_file, strategies_file = find_ranking_file(
                    graphs_dir, current_graph_full_name,
                    ranking_lengths, strategies, random_count
                )

                if rankings_file and rankings_file.exists():
                    resolved_rankings = load_rankings_from_file(rankings_file)
                    print(f"Loaded {len(resolved_rankings)} rankings from {rankings_file.name}")

                    # Optionally log strategy info
                    if strategies_file and strategies_file.exists() and args.verbose:
                        strategies_map = load_ranking_strategies_from_file(strategies_file)
                        print(f"Strategy mapping available for {len(strategies_map)} unique rankings")
                else:
                    # Try to find any ranking file if specific params not found
                    if ranking_lengths:
                        print(f"Warning: 'rankings: auto' specified but no ranking file found for "
                              f"{current_graph_full_name} with lengths={ranking_lengths}, strategies={strategies}")
                        # Fall back to finding any available ranking file
                        rankings_file, strategies_file = find_ranking_file(graphs_dir, current_graph_full_name)
                        if rankings_file:
                            resolved_rankings = load_rankings_from_file(rankings_file)
                            print(f"Fallback: Loaded {len(resolved_rankings)} rankings from {rankings_file.name}")
                    else:
                        print(f"Warning: 'rankings: auto' specified but no ranking file found for {current_graph_full_name}")
                        print(f"Run generate_graph_ranking_metadata.py first, or specify --ranking-lengths")

            # Case 2: load rankings from config explicitly (List[List[int]])
            elif isinstance(rankings_source_in_config, list) and \
                all(isinstance(r, list) and all(isinstance(i, int) for i in r) for r in rankings_source_in_config):
                resolved_rankings = rankings_source_in_config
                print(f"Using {len(resolved_rankings)} explicit rankings from config for {graph_name} (Rep {rep_id})")

            # Case 3: No rankings defined, or invalid format
            else:
                print(f"Warning: No valid explicit or 'auto' rankings defined in config for {graph_name} (Rep {rep_id}).")

            # If no rankings were resolved, skip this graph replicate
            if not resolved_rankings:
                print(f"Warning: skipping {graph_name} (rep {rep_id}) due to no rankings resolved.")
                continue

            summary = run_methods_for_graph(
                graph_name=graph_name,
                graph_type=graph_type,
                replicate_id=rep_id,
                rankings=resolved_rankings,
                graphs_dir=graphs_dir,
                output_dir=output_dir,
                methods_to_run=methods_to_run,
                append_mode=args.append,
                timeout=args.timeout,
                betas_config=betas_config
            )
            summaries.append(summary)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"ALL GRAPHS COMPLETED")
    print(f"Total graphs processed: {len(summaries)}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
