#!/usr/bin/env python3
"""
Batch comparison of method results vs SIR ground truth

Automatically processes all graphs in an experiment:
1. Finds all methods files (parquet or CSV)
2. Extracts ALL rankings from each file (supports multiple rankings)
3. Finds ALL corresponding SIR results (auto-detects infection probabilities)
4. Runs comparison for each (graph, ranking, infection_prob) combination

Output files: {graph}_r{ranking}_p{infection_prob}_comparison.parquet
Example: email-eu-core_r8_0_p0.0250_comparison.parquet
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import re
import yaml
from tqdm import tqdm

# In-process import so methods_df is loaded once per graph (not per subprocess)
sys.path.insert(0, str(Path(__file__).parent))
from compare_results import compare_all_methods, print_tie_statistics, rank_sir_results


def _print_ranking_tie_summary(ranking_tie_stats: list, ranking_str: str, verbose: bool):
    """
    Print one summary line per ranking, averaging tie fractions across probabilities.

    With verbose=True, also prints the full per-probability tables.
    """
    if not ranking_tie_stats:
        return

    if verbose:
        # Print full table per probability
        for ts in ranking_tie_stats:
            if ts:
                prob = ts[0].get('infection_prob')
                ranking = [int(x) for x in ranking_str.split('_')]
                print_tie_statistics(ts, ranking, prob, verbose=True)
        return

    # Aggregate: for each (variant, component_index), average fraction_tied across probs
    from collections import defaultdict
    sums = defaultdict(list)
    for ts in ranking_tie_stats:
        for stat in ts:
            key = (stat['variant'], stat['component_index'])
            sums[key].append(stat['fraction_tied'])

    if not sums:
        return

    # Group by variant
    by_variant = defaultdict(list)
    for (variant, comp_idx), fracs in sorted(sums.items()):
        mean_frac = sum(fracs) / len(fracs)
        by_variant[variant].append((comp_idx, mean_frac))

    n_probs = len(ranking_tie_stats)
    parts = []
    for variant, comps in by_variant.items():
        comp_str = '  '.join(f"comp{ci}={f:.1%}" for ci, f in sorted(comps))
        parts.append(f"{variant}: {comp_str}")
    print(f"  Ties  r=[{ranking_str}]  ({n_probs} probs avg)  |  " + '  |  '.join(parts))


def extract_all_rankings_from_file(methods_file):
    """
    Extract ALL unique rankings from methods file (parquet or CSV)

    Returns:
        List of (ranking_tuple, ranking_str) tuples
        Example: [([0, 1], "0_1"), ([1, 0], "1_0")]
    """
    try:
        # Load file based on extension
        if str(methods_file).endswith('.parquet'):
            df = pd.read_parquet(methods_file, columns=['ranking'], engine='pyarrow')
        else:
            df = pd.read_csv(methods_file, usecols=['ranking'])

        rankings = df['ranking'].unique()

        # Convert to list of (ranking_list, ranking_str) tuples
        result = []
        for ranking_str in sorted(rankings):
            ranking = [int(x) for x in ranking_str.split('_')]
            result.append((ranking, ranking_str))

        return result
    except Exception as e:
        print(f"⚠️  Warning: Could not extract rankings from {methods_file}: {e}")
        return []


def find_all_sir_files(graph_name, sir_dir, ranking):
    """
    Find ALL SIR results files for this graph
    (auto-detects all infection probabilities)

    NOTE: Ranking is NO LONGER in the filename! SIR results are ranking-independent.
    Ranking will be applied during comparison.

    Returns:
        List of (sir_file_path, infection_prob) tuples
    """
    # Try parquet first, then CSV for backward compatibility
    # NEW pattern (optimized): {graph_name}_sir_p{prob}.parquet
    pattern_parquet = f"{graph_name}_sir_p*.parquet"
    pattern_csv = f"{graph_name}_sir_p*.csv"

    # Try parquet first
    matching_files = list(sir_dir.glob(pattern_parquet))

    # Fallback to CSV for backward compatibility
    if not matching_files:
        matching_files = list(sir_dir.glob(pattern_csv))

    if not matching_files:
        return []

    # Extract infection probabilities from filenames
    results = []
    prob_pattern = re.compile(r'_p([0-9.]+)\.(parquet|csv)$')

    for sir_file in matching_files:
        match = prob_pattern.search(sir_file.name)
        if match:
            infection_prob = float(match.group(1))
            results.append((sir_file, infection_prob))

    # Sort by infection probability for consistent ordering
    results.sort(key=lambda x: x[1])

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Batch comparison of methods vs SIR results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Automatically processes all graphs in an experiment.
- Auto-detects ALL rankings from methods files (parquet or CSV)
- Auto-detects ALL infection probabilities from SIR result files
- Generates comparison for each (graph, ranking, infection_prob) combination

Output naming: {graph}_r{ranking}_p{infection_prob}_comparison.parquet

SNR discretization:
  --sem-percentile sets a single percentile value (default: 50).
  --config reads sem_percentile from a YAML config file. If it is a list
  (e.g., [0, 10, 25, 50]), results are saved in sem_{p}/ subdirectories.
  Use 0 for no discretization (raw mean ordering).

Example:
  python3 src/compare_batch.py \\
    --methods-dir results/toy_experiment_2026-01-10_20.27 \\
    --sir-dir results/sir/toy_experiment \\
    --output-dir results/comparison/toy_experiment \\
    --config experiment_config.yaml

  With sem_percentile: [0, 25, 50] in config, this creates:
    results/comparison/toy_experiment/sem_0/...
    results/comparison/toy_experiment/sem_25/...
    results/comparison/toy_experiment/sem_50/...
        """
    )

    parser.add_argument(
        '--methods-dir',
        type=Path,
        required=True,
        help='Directory containing method results files (parquet or CSV)'
    )
    parser.add_argument(
        '--sir-dir',
        type=Path,
        required=True,
        help='Directory containing SIR results files (parquet or CSV)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for comparison results'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output (only show progress bar and summary)'
    )
    parser.add_argument(
        '--k-values',
        nargs='+',
        type=int,
        default=[1, 5, 10, 20],
        help='K values for legacy Precision@K (default: 1 5 10 20)'
    )
    parser.add_argument(
        '--k-precision-snr',
        nargs='+',
        type=int,
        default=[1, 3, 5, 10],
        help='K values for SNR-based Precision@K (default: 1 3 5 10)'
    )
    parser.add_argument(
        '--k-arhr-abs',
        nargs='+',
        type=int,
        default=[1, 3, 5, 10],
        help='Absolute k values for arhr@k (default: 1 3 5 10)'
    )
    parser.add_argument(
        '--k-arhr-pct',
        nargs='+',
        type=float,
        default=[0.01, 0.03, 0.05, 0.10],
        help='Fractional k values for arhr@k (default: 0.01 0.03 0.05 0.10)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Save ground truth detail for each method\'s top-k ranked nodes (default: 3, 0 to disable)'
    )
    parser.add_argument(
        '--spread-lower',
        type=float,
        default=0.01,
        help='Lower bound for significant spreader ratio (default: 0.01 = 1%%). Skip if below.'
    )
    parser.add_argument(
        '--spread-upper',
        type=float,
        default=0.20,
        help='Upper bound for significant spreader ratio (default: 0.20 = 20%%). Skip if above.'
    )
    parser.add_argument(
        '--spread-min-pct',
        type=float,
        default=0.01,
        help='Minimum fraction of graph nodes a node must infect to be a significant spreader (default: 0.01 = 1%%)'
    )
    parser.add_argument(
        '--spread-degree-coeff',
        type=float,
        default=1.5,
        help='Coefficient for avg-degree component of significant spreader threshold: '
             'threshold = max(spread-min-pct, coeff * avg_degree / n) (default: 1.5)'
    )
    parser.add_argument(
        '--no-spread-filter',
        action='store_true',
        help='Disable spread-based filtering'
    )
    parser.add_argument(
        '--verbose-ties',
        action='store_true',
        help='Print full tie-statistics table per (ranking, probability). '
             'Default: one summary line per (ranking, probability).'
    )
    parser.add_argument(
        '--sem-percentile',
        type=float,
        default=50.0,
        help='Percentile of SEM distribution for SNR discretization '
             '(default: 50). Lower → fewer ties, higher → more ties.'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to experiment config YAML. If sem_percentile is a list, '
             'runs once per value saving to {output_dir}/sem_{p}/ subdirectories.'
    )

    args = parser.parse_args()

    # Resolve sem_percentile list from config
    sem_percentile_values = [args.sem_percentile]
    if args.config and args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        sp = config.get('sem_percentile')
        if sp is not None:
            sem_percentile_values = sp if isinstance(sp, list) else [sp]
    use_subdirs = len(sem_percentile_values) > 1

    # Find all method files (parquet first, then CSV for backward compatibility)
    methods_files = sorted(args.methods_dir.glob('*_methods.parquet'))
    if not methods_files:
        methods_files = sorted(args.methods_dir.glob('*_methods.csv'))

    if not methods_files:
        print(f"❌ No method files found in {args.methods_dir}")
        return 1

    # Pre-create output directories for all sem_percentile values
    out_dirs = {}
    for sem_p in sem_percentile_values:
        out_dir = args.output_dir / f'sem_{int(sem_p)}' if use_subdirs else args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs[sem_p] = out_dir

    if not args.quiet:
        print(f"{'='*70}")
        print(f"BATCH COMPARISON")
        if use_subdirs:
            print(f"SEM percentiles: {[int(v) for v in sem_percentile_values]}")
        print(f"{'='*70}")
        print(f"Methods directory: {args.methods_dir}")
        print(f"SIR directory: {args.sir_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Graphs found: {len(methods_files)}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

    # Per-sem_p tracking
    per_sem = {sp: {'results': [], 'total': 0} for sp in sem_percentile_values}
    any_failed = False
    completed = 0

    for methods_file in methods_files:
        graph_name = methods_file.stem.replace('_methods', '')

        # Load betas metadata ONCE per graph
        betas_meta_path = methods_file.parent / methods_file.name.replace(
            '_methods.parquet', '_betas_metadata.json'
        ).replace('_methods.csv', '_betas_metadata.json')
        betas_meta = None
        if betas_meta_path.exists():
            with open(betas_meta_path) as f:
                betas_meta = json.load(f)
            print(f"\n{graph_name}: found betas metadata ({betas_meta_path.name})")

        # Read only the 'ranking' column to get unique rankings (avoids loading 5+ GB into RAM)
        t_load = time.time()
        if str(methods_file).endswith('.parquet'):
            ranking_col_df = pd.read_parquet(methods_file, columns=['ranking'], engine='pyarrow')
        else:
            ranking_col_df = pd.read_csv(methods_file, usecols=['ranking'])
        ranking_strs = sorted(ranking_col_df['ranking'].unique())
        del ranking_col_df
        rankings = [([int(x) for x in r.split('_')], r) for r in ranking_strs]
        print(f"\n{graph_name}: found {len(rankings)} ranking(s) in {time.time()-t_load:.1f}s")

        if not rankings:
            print(f"[{completed + 1}/{len(methods_files)}] ✗ {graph_name} (no rankings found)")
            for sp in sem_percentile_values:
                per_sem[sp]['results'].append({'graph': graph_name, 'ranking': None, 'infection_prob': None,
                                              'success': False, 'error': 'No rankings found'})
            completed += 1
            continue

        # --- Pre-compute spread filter ONCE per graph (same SIR files for all rankings) ---
        all_sir_files = find_all_sir_files(graph_name, args.sir_dir, rankings[0][0])
        if not all_sir_files:
            print(f"  ✗  no SIR files found for {graph_name}")
            for sp in sem_percentile_values:
                per_sem[sp]['results'].append({'graph': graph_name, 'ranking': None, 'infection_prob': None,
                                              'success': False, 'error': 'No SIR files found'})
            completed += 1
            continue

        # Get num_nodes from methods file (ranking-independent)
        if not args.no_spread_filter:
            _first_ranking_str = rankings[0][1]
            if str(methods_file).endswith('.parquet'):
                _sample = pd.read_parquet(methods_file,
                                          filters=[('ranking', '==', _first_ranking_str)],
                                          columns=['method', 'node_id'], engine='pyarrow')
            else:
                _sample = pd.read_csv(methods_file, usecols=['ranking', 'method', 'node_id'])
                _sample = _sample[_sample['ranking'] == _first_ranking_str]
            num_nodes = _sample[_sample['method'] == _sample['method'].iloc[0]]['node_id'].nunique()
            del _sample

        allowed_sir = []   # (sir_file, infection_prob) that pass the filter
        skipped_sir = []   # (infection_prob, skip_reason, spread_info)

        for sir_file, infection_prob in all_sir_files:
            if args.no_spread_filter:
                allowed_sir.append((sir_file, infection_prob))
                continue

            summary_file = sir_file.parent / sir_file.name.replace(
                '.parquet', '_summary.json').replace('.csv', '_summary.json')
            significant_ratio = None
            max_spread_pct = None
            median_spread_pct = None
            summary = {}
            if summary_file.exists():
                with open(summary_file) as sf:
                    summary = json.load(sf)
                if 'significant_ratio' in summary:
                    significant_ratio = summary['significant_ratio']
                max_spread_pct = summary.get('max_spread_pct')
                median_spread_pct = summary.get('median_spread_pct')

            avg_degree_summary = summary.get('avg_degree', 0)
            if avg_degree_summary > 0:
                threshold = max(args.spread_min_pct,
                                args.spread_degree_coeff * avg_degree_summary / num_nodes)
            else:
                threshold = args.spread_min_pct

            if significant_ratio is None or max_spread_pct is None:
                sir_df_check = (pd.read_parquet(sir_file) if str(sir_file).endswith('.parquet')
                                else pd.read_csv(sir_file))
                spread_pcts = sir_df_check['total_mean'] / num_nodes
                if significant_ratio is None:
                    significant_ratio = float((spread_pcts >= threshold).sum()) / num_nodes
                if max_spread_pct is None:
                    max_spread_pct = float(spread_pcts.max())
                    median_spread_pct = float(spread_pcts.median())
                del sir_df_check

            spread_ratio = (max_spread_pct / median_spread_pct
                            if median_spread_pct and median_spread_pct > 0 else float('inf'))
            spread_info = (f"sig={significant_ratio:.1%} thr={threshold:.1%}"
                           f" max/med={spread_ratio:.1f}")

            if significant_ratio < args.spread_lower or significant_ratio > args.spread_upper:
                reason = (f"sig={significant_ratio:.1%} outside "
                          f"[{args.spread_lower:.0%},{args.spread_upper:.0%}]")
                skipped_sir.append((infection_prob, reason, spread_info))
            else:
                allowed_sir.append((sir_file, infection_prob))

        # Print spread-filter summary once for this graph
        print(f"  SIR: {len(allowed_sir)} kept, {len(skipped_sir)} skipped"
              + (f"  —  SKIP: "
                 + ', '.join(f"p={p:.4f} ({r})" for p, r, _ in skipped_sir)
                 if skipped_sir else ''))

        # Record skipped probabilities in results (for all sem_p)
        for prob, reason, _ in skipped_sir:
            for _, ranking_str in rankings:
                for sp in sem_percentile_values:
                    per_sem[sp]['results'].append({'graph': graph_name, 'ranking': ranking_str,
                                                  'infection_prob': prob, 'success': True,
                                                  'error': f'Skipped: {reason}'})

        graph_success = True
        graph_comparisons = 0
        n_sem = len(sem_percentile_values)

        for ranking, ranking_str in rankings:
            # Per-sem_p tie stats for this ranking
            ranking_tie_stats = {sp: [] for sp in sem_percentile_values}

            if not allowed_sir:
                print(f"  ✗  r=[{ranking_str}]  — all probabilities skipped")
                graph_success = False
                continue

            # Load only rows for this ranking ONCE (predicate pushdown for parquet)
            t_cache = time.time()
            if str(methods_file).endswith('.parquet'):
                methods_df_r = pd.read_parquet(
                    methods_file,
                    filters=[('ranking', '==', ranking_str)],
                    engine='pyarrow',
                )
            else:
                methods_df_r = pd.read_csv(methods_file)
                methods_df_r = methods_df_r[methods_df_r['ranking'] == ranking_str]
            method_ranks_cache = {
                m: methods_df_r[methods_df_r['method'] == m].set_index('node_id')['rank']
                for m in methods_df_r['method'].unique()
            }
            n_methods_r = len(method_ranks_cache)
            sem_label = f" x{n_sem} sem" if n_sem > 1 else ""
            print(f"\n  r=[{ranking_str}]  {n_methods_r} methods  {len(allowed_sir)} probabilities{sem_label}"
                  f"  (loaded in {time.time()-t_cache:.1f}s)")

            pbar = tqdm(allowed_sir, desc=f"  r=[{ranking_str}]", unit="prob", leave=True)
            for sir_file, infection_prob in pbar:
                pbar.set_postfix_str(f"p={infection_prob:.4f}")

                # Pre-load SIR data ONCE per (ranking, sir_file)
                if str(sir_file).endswith('.parquet'):
                    sir_df_raw = pd.read_parquet(sir_file, engine='pyarrow')
                else:
                    sir_df_raw = pd.read_csv(sir_file)

                # Pre-compute raw-mean ranking ONCE (independent of sem_percentile)
                sir_df_ranked = rank_sir_results(sir_df_raw, ranking, verbose=False)

                # Run comparison for each sem_percentile value
                for sem_p in sem_percentile_values:
                    out_dir = out_dirs[sem_p]
                    output_file = out_dir / f"{graph_name}_r{ranking_str}_p{infection_prob:.4f}_comparison.parquet"
                    _tie_stats_dir = out_dir / 'tie_statistics'

                    success = True
                    error = None
                    try:
                        _, tie_stats = compare_all_methods(
                            methods_file=methods_file,
                            sir_file=sir_file,
                            ranking=ranking,
                            output_file=output_file,
                            k_values=args.k_values,
                            k_precision_snr=args.k_precision_snr,
                            k_arhr_abs=args.k_arhr_abs,
                            k_arhr_pct=args.k_arhr_pct,
                            top_k=args.top_k,
                            tie_stats_dir=_tie_stats_dir,
                            verbose=False,
                            methods_df=methods_df_r,
                            betas_meta=betas_meta,
                            method_ranks_cache=method_ranks_cache,
                            sem_percentile=sem_p,
                            sir_df_raw=sir_df_raw,
                            sir_df_ranked=sir_df_ranked,
                        )
                        if tie_stats:
                            ranking_tie_stats[sem_p].append(tie_stats)
                    except Exception as e:
                        success = False
                        error = str(e)

                    if error:
                        sem_label = f" sem={int(sem_p)}" if n_sem > 1 else ""
                        tqdm.write(f"    ✗  p={infection_prob:.4f}{sem_label}  ERROR: {error}")
                        graph_success = False

                    per_sem[sem_p]['total'] += 1
                    graph_comparisons += 1
                    per_sem[sem_p]['results'].append({
                        'graph': graph_name,
                        'ranking': ranking_str,
                        'infection_prob': infection_prob,
                        'success': success,
                        'error': error,
                    })

                # Free SIR data after all sem_p values processed
                del sir_df_raw, sir_df_ranked

            # Print one tie-statistics summary line per (ranking, sem_p)
            for sem_p in sem_percentile_values:
                sem_label = f" sem={int(sem_p)}" if n_sem > 1 else ""
                stats = ranking_tie_stats[sem_p]
                if stats:
                    _print_ranking_tie_summary(stats, f"{ranking_str}{sem_label}", args.verbose_ties)

        completed += 1
        status = "✓" if graph_success else "✗"
        print(f"\n[{completed}/{len(methods_files)}] {status} {graph_name}"
              f"  ({len(rankings)} rankings, {graph_comparisons} comparisons done)")

    # Print summary per sem_percentile
    for sem_p in sem_percentile_values:
        info = per_sem[sem_p]
        results = info['results']
        total_comparisons = info['total']
        out_dir = out_dirs[sem_p]

        comparison_results = [r for r in results if r.get('ranking') and r['infection_prob'] != 'all']
        successful = [r for r in comparison_results if r['success']]
        failed = [r for r in comparison_results if not r['success']]

        unique_graphs = len(methods_files)
        unique_rankings = len(set(r.get('ranking', 'unknown') for r in comparison_results if r.get('ranking')))

        print(f"\n{'='*70}")
        if use_subdirs:
            print(f"COMPLETED — SEM percentile: {int(sem_p)}")
        else:
            print(f"COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        print(f"Graphs processed: {unique_graphs}")
        print(f"Rankings per graph: {unique_rankings}")
        print(f"Total comparisons: {total_comparisons}")
        print(f"Successful: {len(successful)}/{total_comparisons}")

        if failed:
            print(f"\n❌ Failed comparisons ({len(failed)}):")
            for result in failed:
                ranking_str = f"r={result.get('ranking', 'unknown')}" if result.get('ranking') else ""
                prob_str = f"p={result['infection_prob']:.4f}" if result['infection_prob'] else ""
                info_str = f"{ranking_str} {prob_str}".strip()
                print(f"  - {result['graph']} ({info_str})")
                if result['error']:
                    error_lines = result['error'].strip().split('\n')
                    print(f"    {error_lines[-1]}")
            any_failed = True
        else:
            print(f"\n✓ All comparisons completed successfully!")

        print(f"\nResults saved to: {out_dir}")
        print(f"  Output files: {total_comparisons} comparison files")
        print(f"  Naming: {{graph}}_r{{ranking}}_p{{infection_prob}}_comparison.parquet")
        print(f"{'='*70}")

    return 1 if any_failed else 0


if __name__ == '__main__':
    exit(main())
