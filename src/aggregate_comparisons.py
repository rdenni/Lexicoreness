#!/usr/bin/env python3
"""
Multi-level aggregation of comparison results

Provides multiple views of the data at different granularities:
1. Overall (all graphs combined)
2. By graph type (ER vs SBM-aligned vs SBM-random)
3. By size distribution (uniform vs imbalanced)
4. By graph configuration (mean across replicates)
5. Combined (type × distribution)
+  By ranking (performance across different group ranking strategies)
+  By ranking strategy
+  By ranking length
+  By ranking strategy AND length
+  By SIR infection probability
+  By ranking strategy AND SIR probability
+  By lexicographic degeneracy (grouped by lexicoreness vector)

Also saves full detailed data for custom analysis.
"""

import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import ast
import json
import sys
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent))


def read_data_file(file_path: Path) -> pd.DataFrame:
    """Read a data file, supporting both parquet and CSV formats."""
    if file_path.suffix == '.parquet':
        return pd.read_parquet(file_path)
    else:
        return pd.read_csv(file_path)


def write_data_file(df: pd.DataFrame, file_path: Path, index: bool = False):
    """Write a data file, supporting both parquet and CSV formats."""
    if file_path.suffix == '.parquet':
        df.to_parquet(file_path, index=index)
    else:
        df.to_csv(file_path, index=index)


def find_data_file(directory: Path, base_name: str, prefer_parquet: bool = True) -> Optional[Path]:
    """Find a data file with the given base name, preferring parquet over CSV."""
    if prefer_parquet:
        parquet_file = directory / f"{base_name}.parquet"
        if parquet_file.exists():
            return parquet_file
    csv_file = directory / f"{base_name}.csv"
    if csv_file.exists():
        return csv_file
    if not prefer_parquet:
        parquet_file = directory / f"{base_name}.parquet"
        if parquet_file.exists():
            return parquet_file
    return None


def load_ranking_strategies_mapping(graphs_dir: Path, graph_name: str, quiet: bool = False) -> Dict[str, List[str]]:
    """
    Load ranking-to-strategy mapping from JSON files.

    Searches for *_ranking_strategies_*.json files for the given graph.
    Returns a dict mapping ranking strings (e.g., "4_14") to list of strategy names.
    """
    if not graphs_dir:
        return {}

    # Find ranking strategies files for this graph
    pattern = f"{graph_name}_ranking_strategies_*.json"
    matches = list(graphs_dir.glob(pattern))

    # If not found, try searching recursively
    if not matches:
        matches = list(graphs_dir.rglob(pattern))

    # If still not found, try a broader search for any ranking_strategies file
    if not matches:
        all_strategy_files = list(graphs_dir.glob("*_ranking_strategies_*.json"))
        if not all_strategy_files:
            all_strategy_files = list(graphs_dir.rglob("*_ranking_strategies_*.json"))
        # Filter to files that contain the graph name
        matches = [f for f in all_strategy_files if graph_name in f.stem]

    if not matches:
        if not quiet:
            print(f"    Warning: No ranking_strategies file found for '{graph_name}' in {graphs_dir}")
        return {}

    # Load the first match (usually there's only one per graph)
    try:
        with open(matches[0], 'r') as f:
            data = json.load(f)
            if not quiet:
                print(f"    Loaded strategies from {matches[0].name} ({len(data)} rankings)")
            return data
    except Exception as e:
        if not quiet:
            print(f"    Warning: Error loading {matches[0]}: {e}")
        return {}


def get_ranking_length(ranking_str: str) -> Optional[int]:
    """Extract ranking length from ranking string like '4_14' -> 2"""
    if not ranking_str or pd.isna(ranking_str):
        return None
    try:
        return len(ranking_str.split('_'))
    except Exception:
        return None


def get_ranking_strategy(ranking_str: str, strategies_map: Dict[str, List[str]]) -> Optional[str]:
    """
    Get the strategy name for a ranking.

    If multiple strategies produce the same ranking, joins them with '+'.
    Returns the first/primary strategy name.
    """
    if not ranking_str or pd.isna(ranking_str) or not strategies_map:
        return None

    strategies = strategies_map.get(ranking_str, [])
    if not strategies:
        return None

    # Return first strategy (primary), or join if multiple
    if len(strategies) == 1:
        return strategies[0]
    else:
        # Multiple strategies produce same ranking - use first one
        return strategies[0]


def count_rank1_nodes_from_methods(methods_dir: Path, graph_name: str,
                                    ranking_str: str, method: str) -> Optional[int]:
    """
    Count the number of rank-1 nodes for a method from the methods CSV.

    Returns the count of nodes with rank=1 for the given graph/ranking/method.
    """
    if not methods_dir:
        return None

    methods_file = find_data_file(methods_dir, f"{graph_name}_methods")
    if not methods_file:
        return None

    try:
        df = read_data_file(methods_file)
        # Filter to this ranking and method
        filtered = df[(df['ranking'] == ranking_str) & (df['method'] == method)]
        # Count nodes with rank == 1
        return int((filtered['rank'] == 1).sum())
    except Exception:
        return None


def enrich_dataframe_with_ranking_info(df: pd.DataFrame, graphs_dir: Path = None,
                                        methods_dir: Path = None, quiet: bool = False,
                                        strategies_cache: dict = None,
                                        methods_df_cache: dict = None) -> pd.DataFrame:
    """
    Add ranking_length, ranking_strategy, and num_rank1_nodes columns to the dataframe.

    Args:
        strategies_cache: Optional pre-populated cache (persists across calls to avoid re-reading graph files)
        methods_df_cache: Optional pre-populated cache (persists across calls to avoid re-reading methods files)
    """
    df = df.copy()

    # Add ranking_length (can be computed directly)
    df['ranking_length'] = df['ranking'].apply(get_ranking_length)

    # Load strategies mapping for each unique graph
    if graphs_dir:
        if strategies_cache is None:
            strategies_cache = {}
        loaded_graphs = set(strategies_cache.keys())

        if not quiet:
            print("  Loading ranking strategy mappings...")

        def get_strategy_for_row(row):
            graph = row['graph_name']
            ranking = row['ranking']

            if graph not in strategies_cache:
                # Only print loading message once per graph
                show_msg = not quiet and graph not in loaded_graphs
                strategies_cache[graph] = load_ranking_strategies_mapping(graphs_dir, graph, quiet=not show_msg)
                loaded_graphs.add(graph)

            return get_ranking_strategy(ranking, strategies_cache[graph])

        df['ranking_strategy'] = df.apply(get_strategy_for_row, axis=1)

        # Print summary of strategy loading
        if not quiet:
            loaded_count = sum(1 for g in strategies_cache if strategies_cache[g])
            print(f"  Loaded strategies for {loaded_count}/{len(strategies_cache)} graphs")
            if df['ranking_strategy'].isna().all():
                print(f"  WARNING: No strategies were mapped. Check --graphs-dir path.")
    else:
        df['ranking_strategy'] = None
        if not quiet:
            print("  Skipping strategy mapping (--graphs-dir not provided)")

    # Count rank-1 nodes from methods CSV
    if methods_dir:
        if not quiet:
            print("  Counting rank-1 nodes from methods CSV files...")

        if methods_df_cache is None:
            methods_df_cache = {}

        # Load methods files once per graph, then build a full lookup table vectorially
        unique_graphs = df['graph_name'].unique()
        rank1_frames = []
        for graph in tqdm(unique_graphs, desc="Counting rank-1 nodes", disable=quiet):
            if graph not in methods_df_cache:
                methods_file = find_data_file(methods_dir, f"{graph}_methods")
                methods_df_cache[graph] = read_data_file(methods_file) if methods_file else None
            df_m = methods_df_cache[graph]
            if df_m is not None:
                counts = (
                    df_m[df_m['rank'] == 1]
                    .groupby(['ranking', 'method'])
                    .size()
                    .reset_index(name='num_rank1_nodes')
                )
                counts['graph_name'] = graph
                rank1_frames.append(counts)

        if rank1_frames:
            rank1_lookup = pd.concat(rank1_frames, ignore_index=True)
            df = df.merge(rank1_lookup, on=['graph_name', 'ranking', 'method'], how='left')
            # Fill 0 for graph+ranking+method combos that exist but have no rank-1 nodes
            found_graphs = {g for g in unique_graphs if methods_df_cache.get(g) is not None}
            mask = df['graph_name'].isin(found_graphs)
            df.loc[mask, 'num_rank1_nodes'] = df.loc[mask, 'num_rank1_nodes'].fillna(0)
        else:
            df['num_rank1_nodes'] = None
    else:
        df['num_rank1_nodes'] = None

    return df


def parse_graph_name(graph_name: str):
    """Extract features from graph name"""
    features = {
        'graph_name': graph_name,
        'graph_type': None,
        'size_distribution': None,
        'group_assignment': None,
        'base_config': None
    }

    # Graph type
    if graph_name.startswith('ER_'):
        features['graph_type'] = 'ER'
        features['group_assignment'] = 'random'
    elif graph_name.startswith('SBM_'):
        features['graph_type'] = 'SBM'
        if 'aligned' in graph_name:
            features['group_assignment'] = 'aligned'
        elif 'random' in graph_name:
            features['group_assignment'] = 'random'

    # Size distribution
    if 'uniform' in graph_name:
        features['size_distribution'] = 'uniform'
    elif 'imbalanced' in graph_name:
        features['size_distribution'] = 'imbalanced'

    # Base config (remove _repX suffix)
    if '_rep' in graph_name:
        features['base_config'] = graph_name.rsplit('_rep', 1)[0]
    else:
        features['base_config'] = graph_name

    # Detailed type
    if features['graph_type'] == 'SBM':
        features['detailed_type'] = f"SBM_{features['group_assignment']}"
    elif features['graph_type'] == 'ER':
        features['detailed_type'] = 'ER'
    else:
        features['detailed_type'] = 'real'

    return features


def load_all_comparisons(comparison_dir: Path, quiet=False):
    """Load all comparison files (parquet or CSV) with metadata"""
    # Try parquet first, then CSV for backward compatibility
    comparison_files = list(comparison_dir.glob('*_comparison.parquet'))
    if not comparison_files:
        comparison_files = list(comparison_dir.glob('*_comparison.csv'))

    if not comparison_files:
        if not quiet:
            print(f"❌ No comparison files found in {comparison_dir}")
        return None

    # Filter to only files with ranking (new format with reach vectors)
    # Pattern: {graph}_r{ranking}_p{prob}_comparison.parquet (or .csv)
    ranking_files = [f for f in comparison_files if '_r' in f.stem and '_p' in f.stem]

    if not ranking_files:
        if not quiet:
            print(f"❌ No ranking-specific comparison files found in {comparison_dir}")
            print(f"   Please regenerate comparisons using the latest compare_results.py")
        return None

    if not quiet:
        print(f"Found {len(ranking_files)} ranking-specific comparison files")
        if len(comparison_files) > len(ranking_files):
            print(f"  (Skipped {len(comparison_files) - len(ranking_files)} old-format files without ranking)\n")
        else:
            print()

    all_results = []
    for data_file in tqdm(ranking_files, desc="Loading comparisons", unit="file", disable=quiet):
        df = read_data_file(data_file)

        filename = data_file.stem

        # Extract ranking if present
        # Use rsplit to split from the right to handle graph names with '_r' in them
        ranking = None
        ranking_str = None
        if '_r' in filename and '_p' in filename:
            # Split from right to get the last occurrence of _r (before ranking)
            parts = filename.rsplit('_r', 1)
            if len(parts) == 2:
                ranking_part = parts[1].split('_p')[0]
                ranking_str = ranking_part
                try:
                    ranking = [int(x) for x in ranking_part.split('_')]
                except ValueError:
                    # If parsing fails, skip this file
                    ranking = None
                    ranking_str = None

        # Extract infection prob
        if '_p' in filename:
            p_part = filename.rsplit('_p', 1)[-1].split('_')[0]
            infection_prob = float(p_part)
            if '_r' in filename and ranking_str:
                # Use rsplit to get graph name (everything before last _r)
                graph_name = filename.rsplit('_r', 1)[0]
            else:
                graph_name = filename.rsplit('_p', 1)[0]
        else:
            graph_name = filename.replace('_comparison', '')
            infection_prob = None

        features = parse_graph_name(graph_name)

        for key, value in features.items():
            df[key] = value
        df['ranking'] = ranking_str
        df['ranking_list'] = str(ranking) if ranking else None
        df['infection_prob'] = infection_prob
        df['source_file'] = data_file.name

        all_results.append(df)

    combined_df = pd.concat(all_results, ignore_index=True)

    if not quiet:
        print(f"Loaded data:")
        print(f"  Total rows: {len(combined_df)}")
        print(f"  Unique graphs: {combined_df['graph_name'].nunique()}")
        print(f"  Unique base configs: {combined_df['base_config'].nunique()}")
        print(f"  Unique rankings: {combined_df['ranking'].nunique() if 'ranking' in combined_df.columns else 0}")
        if 'ranking' in combined_df.columns and combined_df['ranking'].notna().any():
            print(f"  Rankings: {sorted(combined_df['ranking'].dropna().unique())}")
        print(f"  Methods: {sorted(combined_df['method'].unique())}")

    return combined_df


def aggregate_level(df: pd.DataFrame, group_by):
    """Generic aggregation function"""
    agg_dict = {}

    # Legacy metrics
    if 'kendall_correlation' in df.columns:
        agg_dict['kendall_correlation'] = ['mean', 'std']

    if 'precision_at_10' in df.columns:
        agg_dict['precision_at_10'] = ['mean', 'std']

    if 'num_method_top1' in df.columns and df['num_method_top1'].notna().any():
        agg_dict['num_method_top1'] = ['mean', 'std', 'min', 'max']

    if 'num_rank1_nodes' in df.columns and df['num_rank1_nodes'].notna().any():
        agg_dict['num_rank1_nodes'] = ['mean', 'std', 'min', 'max']

    # SNR-based metrics
    for col in ['weighted_kendall_tau', 'mrr']:
        if col in df.columns and df[col].notna().any():
            agg_dict[col] = ['mean', 'std']

    # arhr@k columns (absolute and percentage-based)
    for col in [c for c in df.columns if c.startswith('arhr_at_')]:
        if df[col].notna().any():
            agg_dict[col] = ['mean', 'std']

    # precision_snr@k columns
    for col in [c for c in df.columns if c.startswith('precision_snr_at_')]:
        if df[col].notna().any():
            agg_dict[col] = ['mean', 'std']

    summary = df.groupby(group_by).agg(agg_dict).round(4)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary['count'] = df.groupby(group_by).size()

    # Ensure integer columns have consistent types
    int_columns = ['count']
    for col in int_columns:
        if col in summary.columns:
            summary[col] = summary[col].astype(int)

    # Sort by weighted_kendall_tau if available, otherwise kendall
    sort_col = 'weighted_kendall_tau_mean' if 'weighted_kendall_tau_mean' in summary.columns else 'kendall_correlation_mean'
    if sort_col in summary.columns:
        if isinstance(group_by, list):
            summary = summary.sort_values(group_by + [sort_col], ascending=[True]*len(group_by) + [False])
        else:
            summary = summary.sort_values(sort_col, ascending=False)

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Method selection logic
# ──────────────────────────────────────────────────────────────────────────────

# Patterns that identify composite methods (lexipeeling combined with another method)
_COMPOSITE_PATTERNS = [
    'lexipeeling_coreness', 'lexipeeling_degree', 'lexipeeling_degree_lexicographic',
    'degree_lexicographic_lexipeeling', 'restricted_degree_lexicographic',
]

# Always-included methods in selected outputs
_ALWAYS_INCLUDED = {'lexipeeling', 'lexipeeling_extended'}

# Base ranking-dependent method names (not composite, not restricted, not betas)
_BASE_RANKING_DEPENDENT = {
    'lexipeeling', 'lexipeeling_extended', 'degree_lexicographic', 'random',
    'pagerank_uniform_jump', 'pagerank_exponential_jump',
    'fairgd', 'adaptgd', 'fspr', 'lfpr_u', 'lfpr_n',
}


def is_composite_method(method_name: str) -> bool:
    """Return True if this method is a composite (lexipeeling + tiebreaker)."""
    return any(pattern in method_name for pattern in _COMPOSITE_PATTERNS)


def is_ranking_independent(method_name: str) -> bool:
    """Return True if this method does not require a group ranking to run."""
    return (
        method_name not in _BASE_RANKING_DEPENDENT
        and not method_name.startswith('betas_')
        and not method_name.startswith('restricted_')
        and not is_composite_method(method_name)
    )


def load_method_groups_yaml(yaml_path: Path) -> List[Dict]:
    """
    Load method groups from YAML file.

    Expected format:
        method_groups:
          - name: "PageRank variants"
            methods: [pagerank, pagerank_uniform_jump, ...]
          - name: "Coreness variants"
            methods: [coreness, h1_index, ...]

    Returns list of dicts with 'name' and 'methods' keys, or [] on error.
    """
    import yaml
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('method_groups', [])
    except Exception as e:
        print(f"  Warning: Could not load method groups from {yaml_path}: {e}")
        return []


def select_methods_for_metric(
    aggregated_df: pd.DataFrame,
    metric_col: str,
    method_col: str = 'method',
    groups_yaml: Path = None,
    top_n_baselines: int = 3,
    top_n_composites: int = 3,
    extra_always_include: set = None,
) -> Dict:
    """
    Select methods for a given metric column.

    Selection logic:
    1. From baselines (non-composite, non-always-included): top_n_baselines by metric_col
    2. From composites: top_n_composites by metric_col
    3. Always include lexipeeling and lexipeeling_extended
    4. From each YAML group: top 1 by metric_col if not already in top_n_baselines

    Args:
        aggregated_df: DataFrame with method × metric (one row per method, metric already averaged)
        metric_col: Column name of the metric (e.g., 'mrr_mean', 'weighted_kendall_tau_mean')
        method_col: Column name of the method
        groups_yaml: Path to YAML file with method groups
        top_n_baselines: Number of top baselines to include
        top_n_composites: Number of top composites to include

    Returns:
        dict with 'full_selected' and 'basic_selected' method lists
    """
    if metric_col not in aggregated_df.columns:
        return {'full_selected': [], 'basic_selected': []}

    # Build a clean per-method DataFrame
    method_scores = aggregated_df[[method_col, metric_col]].dropna().copy()
    method_scores = method_scores.groupby(method_col)[metric_col].mean().reset_index()
    method_scores = method_scores.sort_values(metric_col, ascending=False)

    available_methods = set(method_scores[method_col].tolist())

    # Separate into categories
    baselines = [m for m in method_scores[method_col]
                 if m not in _ALWAYS_INCLUDED and not is_composite_method(m)]
    composites = [m for m in method_scores[method_col] if is_composite_method(m)]

    # Top baselines and composites
    top_baselines = baselines[:top_n_baselines]
    top_composites = composites[:top_n_composites]

    # YAML group top-1
    group_top1 = []
    if groups_yaml and groups_yaml.exists():
        groups = load_method_groups_yaml(groups_yaml)
        for group in groups:
            group_methods = [m for m in group.get('methods', [])
                             if m in available_methods and m not in _ALWAYS_INCLUDED
                             and not is_composite_method(m)]
            if not group_methods:
                continue
            # Sort by metric and pick top 1 if not already in top_baselines
            group_scores = method_scores[method_scores[method_col].isin(group_methods)]
            if group_scores.empty:
                continue
            top1 = group_scores.iloc[0][method_col]
            if top1 not in top_baselines:
                group_top1.append(top1)

    # Assemble selected sets
    always_set = _ALWAYS_INCLUDED | (extra_always_include or set())
    always = list(always_set & available_methods)
    full_selected = list(dict.fromkeys(top_baselines + top_composites + always + group_top1))
    basic_selected = list(dict.fromkeys(top_baselines + always + group_top1))

    return {'full_selected': full_selected, 'basic_selected': basic_selected}


def save_per_metric_selections(
    detail_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    output_dir: Path,
    primary_metric_cols: List[str],
    groups_yaml: Path = None,
    quiet: bool = False,
):
    """
    For each metric, compute per-metric method selection and save filtered DataFrames.

    Saves:
        {output_dir}/selections/{metric}_selected_full.parquet
        {output_dir}/selections/{metric}_selected_basic.parquet
        {output_dir}/selections/{metric}_all_methods.parquet

    Args:
        detail_df: Full detail DataFrame (one row per method × graph × ranking × probability)
        aggregated_df: Aggregated DataFrame (one row per method, averaged)
        output_dir: Base output directory
        primary_metric_cols: List of metric column names to use for selection
        groups_yaml: Optional path to method groups YAML
        quiet: Suppress output
    """
    sel_dir = output_dir / 'selections'
    sel_dir.mkdir(parents=True, exist_ok=True)

    all_selections: Dict[str, Dict] = {}

    for metric_col in primary_metric_cols:
        agg_col = f'{metric_col}_mean' if not metric_col.endswith('_mean') else metric_col
        if agg_col not in aggregated_df.columns:
            continue

        selection = select_methods_for_metric(
            aggregated_df.reset_index() if aggregated_df.index.names != [None] else aggregated_df,
            metric_col=agg_col,
            groups_yaml=groups_yaml,
        )

        all_selections[metric_col] = selection

        metric_slug = metric_col.replace('_mean', '').replace('@', '_at_')

        # Save per-metric selections
        for sel_name, sel_methods in [('full', selection['full_selected']),
                                       ('basic', selection['basic_selected'])]:
            if sel_methods:
                sel_df = detail_df[detail_df['method'].isin(sel_methods)]
                write_data_file(sel_df, sel_dir / f'{metric_slug}_selected_{sel_name}.csv')

        # Save all methods (full detail, no filtering)
        write_data_file(detail_df, sel_dir / f'{metric_slug}_all_methods.csv')

    return all_selections


def format_table(df):
    """Format table with compact display of SNR-based metrics."""
    display = pd.DataFrame(index=df.index)

    # Determine which optional columns are present
    has_wkendall = 'weighted_kendall_tau_mean' in df.columns
    has_mrr = 'mrr_mean' in df.columns
    has_arhr10 = 'arhr_at_10_mean' in df.columns
    has_p10_snr = 'precision_snr_at_10_mean' in df.columns
    has_p10_legacy = 'precision_at_10_mean' in df.columns
    rank1_col = 'num_method_top1_mean' if 'num_method_top1_mean' in df.columns else 'num_rank1_nodes_mean'
    has_rank1 = rank1_col in df.columns

    def fmt(val, std, decimals=3):
        if pd.isna(val):
            return "N/A"
        fmt_str = f"{{:.{decimals}f}}"
        return f"{fmt_str.format(val)} ± {fmt_str.format(std if pd.notna(std) else 0)}"

    cols = {
        'WKendall': [], 'Kendall': [], 'MRR': [],
        'ARHR@10': [], 'P@10': [], '#Rank1': [],
    }

    for _, row in df.iterrows():
        if has_wkendall:
            cols['WKendall'].append(fmt(row.get('weighted_kendall_tau_mean'), row.get('weighted_kendall_tau_std')))
        cols['Kendall'].append(fmt(row.get('kendall_correlation_mean'), row.get('kendall_correlation_std')))
        if has_mrr:
            cols['MRR'].append(fmt(row.get('mrr_mean'), row.get('mrr_std')))
        if has_arhr10:
            cols['ARHR@10'].append(fmt(row.get('arhr_at_10_mean'), row.get('arhr_at_10_std')))
        if has_p10_snr:
            cols['P@10'].append(fmt(row.get('precision_snr_at_10_mean'), row.get('precision_snr_at_10_std')))
        elif has_p10_legacy:
            cols['P@10'].append(fmt(row.get('precision_at_10_mean'), row.get('precision_at_10_std')))
        if has_rank1:
            prefix = rank1_col.replace('_mean', '')
            mv = row.get(f'{prefix}_mean', 0)
            sv = row.get(f'{prefix}_std', 0)
            lo = row.get(f'{prefix}_min', 0)
            hi = row.get(f'{prefix}_max', 0)
            if pd.notna(mv):
                cols['#Rank1'].append(f"{mv:.1f} ± {sv:.1f} [{int(lo)}-{int(hi)}]")
            else:
                cols['#Rank1'].append("N/A")

    # Build display DataFrame
    col_order = ['WKendall', 'Kendall', 'MRR', 'ARHR@10', 'P@10', '#Rank1']
    present = {
        'WKendall': has_wkendall,
        'Kendall': True,
        'MRR': has_mrr,
        'ARHR@10': has_arhr10,
        'P@10': has_p10_snr or has_p10_legacy,
        '#Rank1': has_rank1,
    }
    for col in col_order:
        if present[col]:
            display[col] = cols[col]

    # Mark best value per numeric column with '*'
    numeric_cols = [c for c in col_order if c != '#Rank1' and present[c]]
    for col in numeric_cols:
        # Extract mean values for comparison (first number before ' ±')
        means = []
        for v in display[col]:
            try:
                means.append(float(v.split(' ±')[0]))
            except (ValueError, AttributeError):
                means.append(float('-inf'))
        if means:
            best_val = max(means)
            if best_val != float('-inf'):
                display[col] = [
                    f"*{v}" if means[i] == best_val else v
                    for i, v in enumerate(display[col])
                ]

    return display


def lexicographic_compare_vectors(vec1, vec2):
    """
    Compare two vectors lexicographically.
    Returns: -1 if vec1 < vec2, 0 if equal, 1 if vec1 > vec2

    For SIR vectors with std, compare by means first, then by std if means are equal
    """
    # If vectors are tuples (mean, std) pairs
    if isinstance(vec1[0], tuple):
        for (m1, s1), (m2, s2) in zip(vec1, vec2):
            if m1 > m2:
                return 1
            elif m1 < m2:
                return -1
            # If means equal, prefer smaller std (more consistent)
            elif s1 < s2:
                return 1
            elif s1 > s2:
                return -1
        return 0
    else:
        # Simple numeric vectors
        for v1, v2 in zip(vec1, vec2):
            if v1 > v2:
                return 1
            elif v1 < v2:
                return -1
        return 0


def compute_median_vector(vectors_with_std):
    """
    Compute median vector from list of (mean_vector, std_vector) tuples.

    Sorts vectors lexicographically (descending) and returns the median.
    If even number, averages the two middle vectors.

    Args:
        vectors_with_std: List of (mean_vec, std_vec) tuples

    Returns:
        (median_mean_vec, median_std_vec) or None
    """
    if not vectors_with_std:
        return None

    # Combine mean and std into tuples for each position
    combined = []
    for mean_vec, std_vec in vectors_with_std:
        combined.append(list(zip(mean_vec, std_vec)))

    # Sort lexicographically (descending)
    from functools import cmp_to_key
    sorted_vecs = sorted(combined, key=cmp_to_key(lambda a, b: -lexicographic_compare_vectors(a, b)))

    n = len(sorted_vecs)
    if n == 0:
        return None

    if n % 2 == 1:
        # Odd: take middle element
        median_combined = sorted_vecs[n // 2]
        median_mean = [m for m, s in median_combined]
        median_std = [s for m, s in median_combined]
    else:
        # Even: average two middle elements
        mid1 = sorted_vecs[n // 2 - 1]
        mid2 = sorted_vecs[n // 2]

        median_mean = [(m1 + m2) / 2 for (m1, s1), (m2, s2) in zip(mid1, mid2)]
        # For std: use average of the two stds
        median_std = [(s1 + s2) / 2 for (m1, s1), (m2, s2) in zip(mid1, mid2)]

    return median_mean, median_std


def analyze_top1_nodes_simple(df: pd.DataFrame):
    """
    Compute mean and median SIR reach for top-1 nodes chosen by each method.

    For each combination i, get the top-1 nodes (y_i nodes), compute mean among those y_i.
    Then compute the mean of these means across all combinations.
    Also computes median by sorting vectors lexicographically.
    """
    if 'method_top_1_node' not in df.columns:
        return None

    # Get ALL methods from dataframe
    methods = sorted(df['method'].unique())

    if not methods:
        return None

    results = {}

    for method in methods:
        method_df = df[df['method'] == method]

        # Get all top-1 reach values for this method
        reach_values = method_df['method_top_1_reach_total'].dropna().tolist()

        if len(reach_values) == 0:
            continue

        # Mean of means (each combination contributes its mean)
        mean_reach = np.mean(reach_values)
        std_reach = np.std(reach_values)

        # Parse reach vectors if available
        group_vectors = []
        if 'method_top_1_reach_vector' in method_df.columns:
            for vec_str in method_df['method_top_1_reach_vector'].dropna():
                try:
                    # Handle numpy dtype strings like "[np.float64(0.424), np.float64(0.12)]"
                    vec_str_clean = str(vec_str).replace('np.float64(', '').replace(')', '')
                    vec = ast.literal_eval(vec_str_clean)
                    if isinstance(vec, list):
                        group_vectors.append(vec)
                except:
                    pass

        # Compute mean and std for each group position
        group_means = []
        group_stds = []
        if group_vectors:
            num_groups = len(group_vectors[0])
            for g_idx in range(num_groups):
                g_values = [vec[g_idx] for vec in group_vectors if len(vec) > g_idx]
                if g_values:
                    group_means.append(np.mean(g_values))
                    group_stds.append(np.std(g_values))

        # Compute median vector (lexicographically sorted)
        # For median, we sort mean vectors and take the middle one(s)
        median_means = None
        if group_vectors and len(group_vectors) > 0:
            # Sort vectors lexicographically (descending)
            from functools import cmp_to_key
            sorted_vectors = sorted(group_vectors, key=cmp_to_key(lambda a, b: -lexicographic_compare_vectors(a, b)))

            n = len(sorted_vectors)
            if n % 2 == 1:
                # Odd: take middle
                median_means = sorted_vectors[n // 2]
            else:
                # Even: average two middle
                mid1 = sorted_vectors[n // 2 - 1]
                mid2 = sorted_vectors[n // 2]
                median_means = [(m1 + m2) / 2 for m1, m2 in zip(mid1, mid2)]

        results[method] = {
            'mean': mean_reach,
            'std': std_reach,
            'n': len(reach_values),
            'group_means': group_means if group_means else None,
            'group_stds': group_stds if group_stds else None,
            'median_means': median_means if median_means else None,
            'median_stds': None  # Not computed for median
        }

    # Add SIR optimal if available
    if 'sir_top_1_reach_total' in df.columns:
        sir_reach_values = df['sir_top_1_reach_total'].dropna().tolist()
        if len(sir_reach_values) > 0:
            # Parse SIR reach vectors
            group_vectors = []
            if 'sir_top_1_reach_vector' in df.columns:
                for vec_str in df['sir_top_1_reach_vector'].dropna():
                    try:
                        # Handle numpy dtype strings like "[np.float64(0.424), np.float64(0.12)]"
                        vec_str_clean = str(vec_str).replace('np.float64(', '').replace(')', '')
                        vec = ast.literal_eval(vec_str_clean)
                        if isinstance(vec, list):
                            group_vectors.append(vec)
                    except:
                        pass

            # Compute mean and std for each group position
            group_means = []
            group_stds = []
            if group_vectors:
                num_groups = len(group_vectors[0])
                for g_idx in range(num_groups):
                    g_values = [vec[g_idx] for vec in group_vectors if len(vec) > g_idx]
                    if g_values:
                        group_means.append(np.mean(g_values))
                        group_stds.append(np.std(g_values))

            # Compute median vector for SIR_OPTIMAL
            median_means = None
            if group_vectors and len(group_vectors) > 0:
                from functools import cmp_to_key
                sorted_vectors = sorted(group_vectors, key=cmp_to_key(lambda a, b: -lexicographic_compare_vectors(a, b)))

                n = len(sorted_vectors)
                if n % 2 == 1:
                    median_means = sorted_vectors[n // 2]
                else:
                    mid1 = sorted_vectors[n // 2 - 1]
                    mid2 = sorted_vectors[n // 2]
                    median_means = [(m1 + m2) / 2 for m1, m2 in zip(mid1, mid2)]

            results['sir_optimal'] = {
                'mean': np.mean(sir_reach_values),
                'std': np.std(sir_reach_values),
                'n': len(sir_reach_values),
                'group_means': group_means if group_means else None,
                'group_stds': group_stds if group_stds else None,
                'median_means': median_means if median_means else None,
                'median_stds': None  # Not computed for median
            }

    return results if results else None


def print_top1_simple(results, show_vectors=False):
    """
    Print simple top-1 node analysis

    Args:
        results: Results dict from analyze_top1_nodes_simple
        show_vectors: If True, show group-wise vectors and median (for ranking-specific analysis)
                     If False, only show total reach (for cross-ranking aggregation)
    """
    if not results:
        return

    for method, data in results.items():
        # Always show total mean reach
        sir_str = f"{data['mean']:.2f} ± {data['std']:.2f}"

        # Add group breakdown (mean) ONLY if show_vectors=True
        if show_vectors and data.get('group_means') is not None and data.get('group_stds') is not None:
            group_strs = []
            for g_mean, g_std in zip(data['group_means'], data['group_stds']):
                group_strs.append(f"{g_mean:.2f} ± {g_std:.2f}")
            sir_str += f", [{', '.join(group_strs)}]"

        print(f"Method {method.upper()}: mean SIR reach = {sir_str}")

        # Median line ONLY if show_vectors=True
        if show_vectors and data.get('median_means') is not None:
            median_str = f"median SIR vector = [{', '.join([f'{m:.2f}' for m in data['median_means']])}]"
            print(f"{'':>{len('Method ')}}  {median_str}")


def load_methods_data(methods_dir: Path, graph_name: str, ranking: str):
    """Load method results for a specific graph and ranking"""
    if not methods_dir:
        return None

    # Try to find methods file (parquet first, then CSV)
    methods_file = find_data_file(methods_dir, f"{graph_name}_methods")
    if not methods_file:
        return None

    try:
        df = read_data_file(methods_file)
        # Filter to this ranking
        df = df[df['ranking'] == ranking]
        return df
    except Exception as e:
        return None


def get_method_value_for_node(methods_df: pd.DataFrame, method: str, node_id: int, debug=False):
    """Get the value/vector that a method assigned to a node"""
    if methods_df is None:
        return None

    node_data = methods_df[(methods_df['method'] == method) & (methods_df['node_id'] == node_id)]
    if node_data.empty:
        return None

    if 'value' not in node_data.columns:
        return None

    value = node_data['value'].values[0]

    if debug:
        print(f"    DEBUG get_method_value: method={method}, node={node_id}, value={value}, type={type(value)}")

    # For lexipeeling and degree_lexicographic, the value might be a string with underscores like "7_0"
    if method in ['lexipeeling', 'degree_lexicographic']:
        if isinstance(value, str):
            # Try underscore-separated FIRST (before ast.literal_eval, which treats "14_0" as numeric literal 140)
            if '_' in value:
                try:
                    parsed = [int(x) for x in value.split('_')]
                    if debug:
                        print(f"      Parsed underscore-separated: {parsed}")
                    return parsed
                except Exception as e:
                    if debug:
                        print(f"      Failed to parse underscore-separated: {e}")

            # Try parsing as list notation "[7, 0]"
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    if debug:
                        print(f"      Parsed as list notation: {parsed}")
                    return parsed
            except:
                pass

            # Return as-is if nothing worked
            if debug:
                print(f"      Returning as-is: {value}")
            return value
        else:
            if debug:
                print(f"      Value is not string, type={type(value)}, returning as-is")
            return value
    else:
        # For scalar methods like coreness and degree
        return value


def load_sir_data(comparison_dir: Path, graph_name: str, ranking: str, infection_prob: float, sir_dir: Path = None):
    """Load SIR results for a specific graph and infection probability"""
    # SIR files are named: {graph}_sir_p{prob}.parquet (or .csv for backward compatibility)

    sir_file = None

    # First, try user-specified SIR directory
    if sir_dir:
        sir_file = find_data_file(sir_dir, f"{graph_name}_sir_p{infection_prob:.4f}")
        if sir_file:
            try:
                sir_df = read_data_file(sir_file)
                return sir_df
            except:
                return None

    # Fallback to auto-detection
    # comparison_dir is like: results/comparison/email-eu-core_2026-01-12_20-00
    # SIR dir should be:      results/sir/email-eu-core_2026-01-12_20-00

    experiment_name = comparison_dir.name  # e.g., "email-eu-core_2026-01-12_20-00"

    # Try different locations
    search_dirs = [
        comparison_dir.parent.parent / 'sir' / experiment_name,
        comparison_dir.parent.parent / 'sir',
        comparison_dir.parent,
    ]

    for search_dir in search_dirs:
        sir_file = find_data_file(search_dir, f"{graph_name}_sir_p{infection_prob:.4f}")
        if sir_file:
            try:
                sir_df = read_data_file(sir_file)
                return sir_df
            except:
                continue

    return None


def analyze_top1_nodes_detailed(df: pd.DataFrame, methods_dir: Path, comparison_dir: Path = None, sir_dir: Path = None):
    """
    Detailed top-1 node analysis per ranking/graph.

    For each graph, for each method, show top-1 nodes with:
    - The method's own value/vector for that node
    - SIR mean ± std for that node on that graph
    - Values from ALL other methods for that node
    - Aggregate statistics: mean and median SIR vectors across all rank-1 nodes
    """
    if 'method_top_1_node' not in df.columns:
        return None

    methods = ['lexipeeling', 'coreness', 'degree_lexicographic']
    methods = [m for m in methods if m in df['method'].unique()]

    results = {}

    for graph_name in df['graph_name'].unique():
        graph_df = df[df['graph_name'] == graph_name]

        # Get ranking and infection_prob for this graph
        ranking = graph_df['ranking'].iloc[0] if 'ranking' in graph_df.columns else None
        infection_prob = graph_df['infection_prob'].iloc[0] if 'infection_prob' in graph_df.columns else None

        # Load methods data for cross-referencing
        methods_df = load_methods_data(methods_dir, graph_name, ranking) if methods_dir else None

        # Load SIR data directly
        sir_df = load_sir_data(comparison_dir, graph_name, ranking, infection_prob, sir_dir) if comparison_dir and infection_prob else None

        graph_results = {}
        for method in methods:
            node_info = []

            # Get ALL rank-1 nodes for this method from the methods CSV (not comparison CSV)
            if methods_df is not None:
                method_rows = methods_df[methods_df['method'] == method]
                rank_1_nodes = method_rows[method_rows['rank'] == 1]['node_id'].unique()

                for node_id in rank_1_nodes:
                    info = {
                        'node_id': int(node_id),
                    }

                    # Try to get SIR reach for this node from SIR data
                    if sir_df is not None:
                        node_sir_data = sir_df[sir_df['node_id'] == node_id]
                        if not node_sir_data.empty:
                            info['sir_mean'] = node_sir_data['total_mean'].values[0]
                            info['sir_std'] = node_sir_data['total_std'].values[0]

                            # Extract group-specific reach values according to ranking
                            if ranking:
                                # Parse ranking if it's a string like "4_14"
                                if isinstance(ranking, str):
                                    ranking_list = [int(x) for x in ranking.split('_')]
                                else:
                                    ranking_list = ranking

                                group_reaches = []
                                for group_id in ranking_list:
                                    group_mean_col = f'group_{group_id}_mean'
                                    group_std_col = f'group_{group_id}_std'
                                    if group_mean_col in node_sir_data.columns:
                                        g_mean = node_sir_data[group_mean_col].values[0]
                                        g_std = node_sir_data[group_std_col].values[0]
                                        group_reaches.append((g_mean, g_std))
                                if group_reaches:
                                    info['sir_group_reaches'] = group_reaches
                        else:
                            # Try comparison data as fallback
                            node_comparison_data = graph_df[graph_df['method_top_1_node'] == node_id]
                            reaches = node_comparison_data['method_top_1_reach_total'].dropna()
                            if len(reaches) > 0:
                                info['sir_mean'] = np.mean(reaches)
                                info['sir_std'] = np.std(reaches)
                            else:
                                info['sir_mean'] = None
                                info['sir_std'] = None
                    else:
                        # Fallback to comparison data
                        node_comparison_data = graph_df[graph_df['method_top_1_node'] == node_id]
                        reaches = node_comparison_data['method_top_1_reach_total'].dropna()
                        if len(reaches) > 0:
                            info['sir_mean'] = np.mean(reaches)
                            info['sir_std'] = np.std(reaches)
                        else:
                            info['sir_mean'] = None
                            info['sir_std'] = None

                    # Get this method's own value for the node
                    own_value = get_method_value_for_node(methods_df, method, node_id)
                    if own_value is not None:
                        info['own_value'] = own_value

                    # Get values from ALL other methods
                    other_values = {}
                    for other_method in methods:
                        if other_method != method:
                            val = get_method_value_for_node(methods_df, other_method, node_id)
                            if val is not None:
                                other_values[other_method] = val
                    if other_values:
                        info['other_methods'] = other_values

                    node_info.append(info)
            else:
                # Fallback to old behavior if methods_df not available
                method_df = graph_df[graph_df['method'] == method]
                top_nodes = method_df['method_top_1_node'].dropna().unique()

                for node in top_nodes:
                    node_id = int(node)
                    node_data = method_df[method_df['method_top_1_node'] == node]
                    reaches = node_data['method_top_1_reach_total'].dropna()

                    if len(reaches) > 0:
                        info = {
                            'node_id': node_id,
                            'sir_mean': np.mean(reaches),
                            'sir_std': np.std(reaches)
                        }
                        node_info.append(info)

            # Compute aggregate statistics across all rank-1 nodes
            aggregate_stats = {}

            # Mean SIR reach (total)
            total_means = [n['sir_mean'] for n in node_info if n.get('sir_mean') is not None]
            if total_means:
                aggregate_stats['mean_sir_total'] = np.mean(total_means)
                aggregate_stats['std_sir_total'] = np.std(total_means)

            # Mean and median SIR vectors (group-wise)
            if node_info and 'sir_group_reaches' in node_info[0]:
                # Extract all group reach vectors
                num_groups = len(node_info[0]['sir_group_reaches'])

                # Compute mean for each group position
                group_means = []
                group_stds = []
                for g_idx in range(num_groups):
                    g_values = [n['sir_group_reaches'][g_idx][0] for n in node_info
                               if 'sir_group_reaches' in n and len(n['sir_group_reaches']) > g_idx]
                    if g_values:
                        group_means.append(np.mean(g_values))
                        group_stds.append(np.std(g_values))

                if group_means:
                    aggregate_stats['mean_sir_vector'] = group_means
                    aggregate_stats['std_sir_vector'] = group_stds

                # Compute median vector (lexicographically sorted)
                # Extract just the mean values for each node's vector
                vectors = []
                for n in node_info:
                    if 'sir_group_reaches' in n:
                        vec = [g_mean for g_mean, g_std in n['sir_group_reaches']]
                        vectors.append(vec)

                if vectors:
                    # Sort lexicographically (descending)
                    from functools import cmp_to_key
                    sorted_vectors = sorted(vectors, key=cmp_to_key(lambda a, b: -lexicographic_compare_vectors(a, b)))

                    n = len(sorted_vectors)
                    if n % 2 == 1:
                        # Odd: take middle
                        aggregate_stats['median_sir_vector'] = sorted_vectors[n // 2]
                    else:
                        # Even: average two middle
                        mid1 = sorted_vectors[n // 2 - 1]
                        mid2 = sorted_vectors[n // 2]
                        aggregate_stats['median_sir_vector'] = [(m1 + m2) / 2 for m1, m2 in zip(mid1, mid2)]

            # Store both node info and aggregate stats
            graph_results[method] = {
                'nodes': node_info,
                'aggregate': aggregate_stats
            }

        results[graph_name] = graph_results

    return results if results else None


def get_sir_optimal_info(df: pd.DataFrame):
    """
    Get SIR_OPTIMAL node info from comparison data.

    Returns dict with:
        - node_id: SIR optimal node ID
        - reach_total: Total reach mean
        - reach_vector: Group-wise reach vector
    """
    if 'sir_top_1_node' not in df.columns or 'sir_top_1_reach_total' not in df.columns:
        return None

    # Get the first non-null entry (should be the same across all methods for this graph/ranking)
    sir_row = df[df['sir_top_1_node'].notna()].iloc[0] if len(df[df['sir_top_1_node'].notna()]) > 0 else None

    if sir_row is None:
        return None

    result = {
        'node_id': int(sir_row['sir_top_1_node']),
        'reach_total': sir_row['sir_top_1_reach_total']
    }

    # Parse reach vector if available
    if 'sir_top_1_reach_vector' in sir_row and pd.notna(sir_row['sir_top_1_reach_vector']):
        try:
            # Handle numpy dtype strings like "[np.float64(0.424), np.float64(0.12)]"
            vec_str = str(sir_row['sir_top_1_reach_vector'])
            vec_str_clean = vec_str.replace('np.float64(', '').replace(')', '')
            vec = ast.literal_eval(vec_str_clean)
            if isinstance(vec, list):
                result['reach_vector'] = vec
        except:
            pass

    return result


def print_sir_optimal_info(sir_info):
    """Print SIR_OPTIMAL information"""
    if not sir_info:
        return

    print(f"  SIR_OPTIMAL (ground truth): Node {sir_info['node_id']}")

    sir_str = f"{sir_info['reach_total']:.2f}"
    if 'reach_vector' in sir_info:
        vec_str = ', '.join([f"{v:.2f}" for v in sir_info['reach_vector']])
        sir_str += f", [{vec_str}]"

    print(f"    Total reach: {sir_str}")


def print_top1_detailed(results):
    """Print detailed top-1 node analysis"""
    if not results:
        return

    for graph_name, graph_data in results.items():
        print(f"\nGraph: {graph_name}")

        for method, method_data in graph_data.items():
            # Extract nodes and aggregate stats from new structure
            if isinstance(method_data, dict) and 'nodes' in method_data:
                nodes = method_data['nodes']
                aggregate = method_data.get('aggregate', {})
            else:
                # Fallback for old structure (just a list of nodes)
                nodes = method_data
                aggregate = {}

            if len(nodes) == 0:
                continue

            print(f"  {method.upper()}:")

            # ALWAYS show aggregate statistics first (if available)
            if aggregate:
                if 'mean_sir_total' in aggregate:
                    agg_str = f"{aggregate['mean_sir_total']:.2f} ± {aggregate['std_sir_total']:.2f}"

                    # Add group breakdown (mean)
                    if 'mean_sir_vector' in aggregate:
                        group_strs = []
                        for g_mean, g_std in zip(aggregate['mean_sir_vector'], aggregate['std_sir_vector']):
                            group_strs.append(f"{g_mean:.2f} ± {g_std:.2f}")
                        agg_str += f", [{', '.join(group_strs)}]"

                    print(f"    Mean SIR reach: {agg_str}")

                # Show median vector
                if 'median_sir_vector' in aggregate:
                    median_str = ', '.join([f"{m:.2f}" for m in aggregate['median_sir_vector']])
                    print(f"    Median SIR vector: [{median_str}]")

                print()  # Blank line before individual nodes

            # Show individual node details (if ≤8 nodes)
            if len(nodes) <= 8:
                for node_info in nodes:
                    # First line: Node ID + method's own value
                    if 'own_value' in node_info:
                        value = node_info['own_value']
                        if isinstance(value, list):
                            value_str = str(value)
                        elif isinstance(value, (int, float)):
                            value_str = f"{value:.3f}"
                        else:
                            value_str = str(value)
                        print(f"    Node {node_info['node_id']}: {value_str}")
                    else:
                        print(f"    Node {node_info['node_id']}:")

                    # Second line: SIR reach (total and group breakdown)
                    if node_info['sir_mean'] is not None:
                        sir_str = f"{node_info['sir_mean']:.2f} ± {node_info['sir_std']:.2f}"

                        # Add group breakdown if available
                        if 'sir_group_reaches' in node_info:
                            group_strs = []
                            for g_mean, g_std in node_info['sir_group_reaches']:
                                group_strs.append(f"{g_mean:.2f} ± {g_std:.2f}")
                            sir_str += f", [{', '.join(group_strs)}]"

                        print(f"      SIR reach: {sir_str}")
                    else:
                        print(f"      SIR reach: N/A (not simulated)")

                    # Following lines: values from other methods
                    if 'other_methods' in node_info:
                        for other_method, value in node_info['other_methods'].items():
                            if isinstance(value, list):
                                value_str = str(value)
                            elif isinstance(value, (int, float)):
                                value_str = f"{value:.3f}"
                            else:
                                value_str = str(value)
                            print(f"      {other_method}: {value_str}")
            else:
                # More than 8 nodes: print count, IDs, and one example value
                print(f"    {len(nodes)} rank-1 nodes (showing IDs and one example value):")
                node_ids = [n['node_id'] for n in nodes]
                print(f"    Node IDs: {node_ids}")

                # Show the value from one random (or first) node
                example_node = nodes[0]
                if 'own_value' in example_node:
                    value = example_node['own_value']
                    if isinstance(value, list):
                        value_str = str(value)
                    elif isinstance(value, (int, float)):
                        value_str = f"{value:.3f}"
                    else:
                        value_str = str(value)
                    print(f"    Example value (node {example_node['node_id']}): {value_str}")


def get_lexicographic_degeneracy(methods_dir: Path, graph_name: str, ranking: str, debug=False):
    """
    Get the lexicographic degeneracy vector for a graph.
    This is the lexicoreness vector of the top-1 node found by lexipeeling.
    """
    methods_df = load_methods_data(methods_dir, graph_name, ranking)
    if methods_df is None:
        if debug:
            print(f"  DEBUG: Could not load methods data for {graph_name}, ranking {ranking}")
        return None

    # Get lexipeeling top-1 node
    lexi_df = methods_df[methods_df['method'] == 'lexipeeling']
    if lexi_df.empty:
        if debug:
            print(f"  DEBUG: No lexipeeling rows found for {graph_name}, ranking {ranking}")
        return None

    top_nodes = lexi_df[lexi_df['rank'] == 1]
    if top_nodes.empty:
        if debug:
            print(f"  DEBUG: No rank-1 nodes found for {graph_name}, ranking {ranking}")
        return None

    # Get the vector for the first top-1 node
    if 'value' in top_nodes.columns:
        value = top_nodes['value'].values[0]
        # Value might be a string representation of a list or already a list
        if isinstance(value, str):
            # Try underscore-separated FIRST (before ast.literal_eval, which treats "14_0" as numeric literal 140)
            if '_' in value:
                try:
                    vec = [int(x) for x in value.split('_')]
                    if debug:
                        print(f"  DEBUG: Found degeneracy {vec} for {graph_name}, ranking {ranking}")
                    return vec
                except Exception as e:
                    if debug:
                        print(f"  DEBUG: Failed to parse underscore-separated '{value}': {e}")

            # Try parsing as list notation "[7, 0]"
            try:
                vec = ast.literal_eval(value)
                if isinstance(vec, list):
                    if debug:
                        print(f"  DEBUG: Found degeneracy {vec} for {graph_name}, ranking {ranking}")
                    return vec
            except:
                pass

            if debug:
                print(f"  DEBUG: Could not parse value '{value}'")
            return None
        elif isinstance(value, list):
            if debug:
                print(f"  DEBUG: Found degeneracy {value} for {graph_name}, ranking {ranking}")
            return value
        else:
            if debug:
                print(f"  DEBUG: Value is not a vector: {value} (type: {type(value)})")
            return None
    else:
        if debug:
            print(f"  DEBUG: No 'value' column in methods data")
            print(f"  Available columns: {list(top_nodes.columns)}")
        return None


def compute_tie_statistics(ranks: List) -> Dict:
    """
    Compute statistics about ties in a ranking.

    Args:
        ranks: List of rank values for nodes

    Returns:
        Dictionary with tie statistics
    """
    from collections import Counter

    if not ranks or len(ranks) == 0:
        return None

    rank_counts = Counter(ranks)  # How many nodes at each rank value
    tie_sizes = list(rank_counts.values())

    return {
        'num_nodes': len(ranks),
        'num_unique_ranks': len(rank_counts),
        'max_tie_size': max(tie_sizes),
        'mean_tie_size': np.mean(tie_sizes),
        'pct_nodes_in_ties': sum(c for c in tie_sizes if c > 1) / len(ranks) * 100
    }


def analyze_best_spreader_group_reach(
    df: pd.DataFrame,
    methods_dir: Path,
    sir_dir: Path,
    graphs_dir: Path,
    comparison_dir: Path,
    betas_group_label: str = 'none',
    betas_info: Dict = None,
    quiet: bool = False
) -> Optional[pd.DataFrame]:
    """
    For each (graph, ranking, infection_prob, method), identify the best spreader
    among rank-1 nodes (by lexicographic SIR reach), then report per-group percentage
    of influenced nodes (mean + std).

    For betas groups, reach vectors are blended before lexicographic comparison.
    """
    if not methods_dir or not sir_dir or not graphs_dir:
        return None

    from collections import Counter
    from lexicographic_utils import sort_by_ranking

    methods_files = list(methods_dir.glob("*_methods.parquet"))
    if not methods_files:
        methods_files = list(methods_dir.glob("*_methods.csv"))
    if not methods_files:
        return None

    # Caches
    group_sizes_cache: Dict[str, Dict[int, int]] = {}
    sir_cache: Dict[Tuple[str, float], pd.DataFrame] = {}

    rows = []

    for methods_file in tqdm(methods_files, desc="Best spreaders", unit="graph", disable=quiet):
        try:
            df_methods = read_data_file(methods_file)
            graph_name = methods_file.stem.replace('_methods', '')

            if 'ranking' not in df_methods.columns or 'method' not in df_methods.columns or 'rank' not in df_methods.columns:
                continue

            # Load group sizes (cached)
            if graph_name not in group_sizes_cache:
                group_file = graphs_dir / f"{graph_name}_groups.txt"
                if not group_file.exists():
                    found = list(graphs_dir.rglob(f"{graph_name}_groups.txt"))
                    group_file = found[0] if found else None
                if not group_file or not group_file.exists():
                    if not quiet:
                        print(f"    Warning: Group file not found for {graph_name}")
                    continue
                group_assignments = {}
                with open(group_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            group_assignments[int(parts[0])] = int(parts[1])
                group_sizes_cache[graph_name] = dict(Counter(group_assignments.values()))

            group_sizes = group_sizes_cache[graph_name]

            # Find all SIR files for this graph
            sir_files = list(sir_dir.glob(f"{graph_name}_sir_*.parquet"))
            if not sir_files:
                sir_files = list(sir_dir.glob(f"{graph_name}_sir_*.csv"))
            if not sir_files:
                continue

            rankings = df_methods['ranking'].dropna().unique()

            for sir_file in sir_files:
                # Parse infection probability
                filename = sir_file.stem
                try:
                    prob_str = filename.split('_p')[-1]
                    infection_prob = float(prob_str)
                except (ValueError, IndexError):
                    continue

                # Load SIR data (cached)
                cache_key = (graph_name, infection_prob)
                if cache_key not in sir_cache:
                    sir_cache[cache_key] = read_data_file(sir_file)
                df_sir = sir_cache[cache_key]

                if 'node_id' not in df_sir.columns:
                    continue

                sir_indexed = df_sir.set_index('node_id')

                for ranking_str in rankings:
                    # Parse ranking
                    try:
                        ranking_list = [int(x) for x in ranking_str.split('_')]
                    except (ValueError, AttributeError):
                        continue

                    # Check that SIR has group columns for this ranking
                    if not all(f'group_{gid}_mean' in df_sir.columns for gid in ranking_list):
                        continue

                    # Get betas vector for this ranking if applicable
                    betas_vector = None
                    if betas_group_label and betas_group_label != 'none' and betas_info:
                        if ranking_str in betas_info:
                            betas_vector = betas_info[ranking_str].get('betas')

                    # Filter methods for this ranking
                    ranking_methods = df_methods[df_methods['ranking'] == ranking_str]
                    # Only process methods present in the df (this betas group)
                    group_methods = set(df['method'].unique()) if df is not None else None
                    methods_in_ranking = ranking_methods['method'].unique()

                    for method in methods_in_ranking:
                        if group_methods and method not in group_methods:
                            continue

                        method_data = ranking_methods[ranking_methods['method'] == method]
                        rank1_nodes = method_data[method_data['rank'] == 1]['node_id'].unique()
                        if len(rank1_nodes) == 0:
                            continue

                        # Build reach vectors for rank-1 nodes
                        group_data = {}
                        node_raw_data = {}  # Store raw (unblended) mean and std per node
                        for node in rank1_nodes:
                            if node not in sir_indexed.index:
                                continue
                            row_data = sir_indexed.loc[node]

                            # Raw per-group means and stds
                            raw_means = {gid: row_data[f'group_{gid}_mean'] for gid in ranking_list}
                            raw_stds = {gid: row_data.get(f'group_{gid}_std', 0.0) for gid in ranking_list}
                            node_raw_data[node] = {'means': raw_means, 'stds': raw_stds}

                            # Build vector for lexicographic comparison (blended if betas)
                            if betas_vector and len(betas_vector) == len(ranking_list) - 1:
                                s = len(ranking_list)
                                raw = [raw_means[gid] for gid in ranking_list]
                                blended = {}
                                for i in range(s - 1):
                                    blended[ranking_list[i]] = raw[i] + betas_vector[i] * raw[i + 1]
                                blended[ranking_list[s - 1]] = raw[s - 1]
                                group_data[node] = blended
                            else:
                                group_data[node] = raw_means

                        if not group_data:
                            continue

                        # Sort lexicographically (best first)
                        sorted_nodes = sort_by_ranking(group_data, ranking_list, reverse=True)

                        # Identify best spreaders (all tied for first place)
                        best_vector = sorted_nodes[0][1]
                        best_spreaders = [node for node, vec in sorted_nodes if tuple(vec) == tuple(best_vector)]

                        # Compute per-group percentages for each best spreader, then average
                        pct_means = {i: [] for i in range(len(ranking_list))}
                        pct_stds = {i: [] for i in range(len(ranking_list))}

                        for node in best_spreaders:
                            raw = node_raw_data[node]
                            for i, gid in enumerate(ranking_list):
                                gs = group_sizes.get(gid, 1)
                                pct_means[i].append(raw['means'][gid] / gs * 100)
                                pct_stds[i].append(raw['stds'][gid] / gs * 100)

                        row = {
                            'graph_name': graph_name,
                            'ranking': ranking_str,
                            'infection_prob': infection_prob,
                            'method': method,
                            'num_rank1_nodes': len(rank1_nodes),
                            'num_best_spreaders': len(best_spreaders),
                            'best_spreader_nodes': str(best_spreaders),
                            'ranking_groups': str(ranking_list),
                        }

                        for i in range(len(ranking_list)):
                            row[f'group_{i}_pct_mean'] = np.mean(pct_means[i])
                            if len(pct_stds[i]) == 1:
                                row[f'group_{i}_pct_std'] = pct_stds[i][0]
                            else:
                                # Combined std: sqrt(mean(std^2) + var(means))
                                row[f'group_{i}_pct_std'] = np.sqrt(
                                    np.mean(np.array(pct_stds[i])**2) + np.var(pct_means[i])
                                )

                        rows.append(row)

        except Exception as e:
            if not quiet:
                print(f"    Warning: Could not process {methods_file} for best spreader analysis: {e}")

    if not rows:
        return None

    return pd.DataFrame(rows)


def aggregate_best_spreader_across_probs(detail_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Aggregate best spreader group reach across infection probabilities.

    Groups by (graph_name, ranking, method) and computes:
    - mean of pct_means, mean of pct_stds, std of pct_means across probs
    """
    if detail_df is None or detail_df.empty:
        return None

    # Find all group_*_pct_mean and group_*_pct_std columns
    pct_mean_cols = [c for c in detail_df.columns if c.startswith('group_') and c.endswith('_pct_mean')]
    pct_std_cols = [c for c in detail_df.columns if c.startswith('group_') and c.endswith('_pct_std')]

    group_cols = ['graph_name', 'ranking', 'method']
    agg_dict = {}
    for col in pct_mean_cols:
        agg_dict[col] = ['mean', 'std']
    for col in pct_std_cols:
        agg_dict[col] = ['mean']
    agg_dict['num_rank1_nodes'] = 'first'
    agg_dict['num_best_spreaders'] = 'mean'
    agg_dict['ranking_groups'] = 'first'

    grouped = detail_df.groupby(group_cols).agg(agg_dict)

    # Flatten multi-level columns
    flat_cols = []
    for col, agg_type in grouped.columns:
        if agg_type == 'first':
            flat_cols.append(col)
        elif agg_type == 'mean' and col.endswith('_pct_mean'):
            flat_cols.append(col + '_across_probs_mean')
        elif agg_type == 'std' and col.endswith('_pct_mean'):
            flat_cols.append(col + '_across_probs_std')
        elif agg_type == 'mean' and col.endswith('_pct_std'):
            flat_cols.append(col + '_across_probs_mean')
        elif agg_type == 'mean':
            flat_cols.append(col + '_mean')
        else:
            flat_cols.append(f'{col}_{agg_type}')

    grouped.columns = flat_cols
    grouped = grouped.reset_index()

    return grouped


def analyze_ties_from_methods(methods_dir: Path, sir_dir: Path = None, quiet: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze tie statistics from methods results and optionally SIR results.

    Args:
        methods_dir: Directory containing methods parquet files
        sir_dir: Optional directory containing SIR parquet files
        quiet: Suppress progress output

    Returns:
        Tuple of (methods_ties_df, sir_ties_df) DataFrames
    """
    methods_ties_data = []
    sir_ties_data = []

    # Find all methods files
    methods_files = list(methods_dir.glob("*_methods.parquet"))
    if not methods_files:
        methods_files = list(methods_dir.glob("*_methods.csv"))

    for methods_file in tqdm(methods_files, desc="Ties (methods)", unit="file", disable=quiet):
        try:
            df_methods = read_data_file(methods_file)
            graph_name = methods_file.stem.replace('_methods', '')

            # Get unique rankings
            rankings = df_methods['ranking'].unique()

            for ranking in rankings:
                df_ranking = df_methods[df_methods['ranking'] == ranking]

                # Get unique methods
                methods = df_ranking['method'].unique()

                for method in methods:
                    df_method = df_ranking[df_ranking['method'] == method]
                    ranks = df_method['rank'].tolist()

                    tie_stats = compute_tie_statistics(ranks)
                    if tie_stats:
                        methods_ties_data.append({
                            'graph_name': graph_name,
                            'ranking': ranking,
                            'method': method,
                            **tie_stats
                        })
        except Exception as e:
            if not quiet:
                print(f"    Warning: Could not process {methods_file}: {e}")

    # Analyze SIR files if provided
    if sir_dir:
        sir_files = list(sir_dir.glob("*_sir_*.parquet"))

        for sir_file in tqdm(sir_files, desc="Ties (SIR)", unit="file", disable=quiet):
            try:
                df_sir = read_data_file(sir_file)
                # SIR files have total_mean column - rank by this
                if 'total_mean' in df_sir.columns:
                    df_sir = df_sir.sort_values('total_mean', ascending=False)
                    df_sir['rank'] = df_sir['total_mean'].rank(method='min', ascending=False).astype(int)
                    ranks = df_sir['rank'].tolist()

                    tie_stats = compute_tie_statistics(ranks)
                    if tie_stats:
                        # Extract graph name and infection prob from filename
                        parts = sir_file.stem.split('_sir_p')
                        graph_name = parts[0]
                        infection_prob = parts[1] if len(parts) > 1 else 'unknown'

                        sir_ties_data.append({
                            'graph_name': graph_name,
                            'infection_prob': infection_prob,
                            'method': 'SIR_ground_truth',
                            **tie_stats
                        })
            except Exception as e:
                if not quiet:
                    print(f"    Warning: Could not process {sir_file}: {e}")

    # Create DataFrames
    methods_ties_df = pd.DataFrame(methods_ties_data) if methods_ties_data else None
    sir_ties_df = pd.DataFrame(sir_ties_data) if sir_ties_data else None

    return methods_ties_df, sir_ties_df


def aggregate_tie_statistics(ties_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate tie statistics per method (across all rankings).

    Args:
        ties_df: DataFrame with tie statistics per (graph, ranking, method)

    Returns:
        Aggregated DataFrame per method
    """
    if ties_df is None or len(ties_df) == 0:
        return None

    agg_dict = {
        'num_unique_ranks': ['mean', 'std'],
        'max_tie_size': ['mean', 'max'],
        'mean_tie_size': ['mean', 'std'],
        'pct_nodes_in_ties': ['mean', 'std']
    }

    summary = ties_df.groupby('method').agg(agg_dict).round(2)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary['count'] = ties_df.groupby('method').size()

    return summary.sort_values('pct_nodes_in_ties_mean', ascending=True)


def extract_node_ranks(methods_dir: Path, sir_dir: Path, comparison_dir: Path = None,
                       quiet: bool = False,
                       output_path: Path = None,
                       group_methods=None,
                       sem_percentile: float = 50.0) -> Optional[pd.DataFrame]:
    """
    Extract node-level ranks from methods results and SIR ground truth.

    Creates a dataset where each row contains a node's rank according to a method
    and its ground truth rank (from SIR).

    Now processes ALL SIR files (different infection probabilities) to capture
    how ground truth rankings vary with infection probability.

    Args:
        methods_dir: Directory containing methods parquet files
        sir_dir: Directory containing SIR parquet files
        comparison_dir: Optional directory containing comparison files (for ranking info)
        quiet: Suppress progress output
        output_path: If provided, write chunks directly to this CSV path (avoids OOM
                     for large datasets). Returns None when used.
        group_methods: If provided, filter to only these methods before writing/accumulating.

    Returns:
        DataFrame with columns: graph_name, ranking, infection_prob, method, node_id,
                               method_rank, ground_truth_rank, n_nodes
        (or None if output_path was used and data was written directly to disk)
    """
    if not methods_dir or not sir_dir:
        return None

    from compare_results import rank_sir_results_snr as _rank_snr

    chunks = []
    parquet_writer = None

    methods_files = list(methods_dir.glob("*_methods.parquet"))
    if not methods_files:
        methods_files = list(methods_dir.glob("*_methods.csv"))

    for methods_file in tqdm(methods_files, desc="Node ranks", unit="graph", disable=quiet):
        try:
            df_methods = read_data_file(methods_file)
            graph_name = methods_file.stem.replace('_methods', '')

            if 'ranking' not in df_methods.columns or 'method' not in df_methods.columns:
                continue

            # Get unique rankings
            rankings = df_methods['ranking'].dropna().unique()

            # Find ALL SIR files for this graph (different infection probabilities)
            sir_files = list(sir_dir.glob(f"{graph_name}_sir_*.parquet"))
            if not sir_files:
                continue

            # Process each SIR file (each infection probability)
            for sir_file in sir_files:
                # Extract infection probability from filename
                # Format: {graph_name}_sir_p{prob}.parquet
                filename = sir_file.stem
                try:
                    prob_str = filename.split('_p')[-1]
                    infection_prob = float(prob_str)
                except (ValueError, IndexError):
                    if not quiet:
                        print(f"    Warning: Could not parse infection_prob from {filename}")
                    continue

                df_sir = read_data_file(sir_file)

                if 'total_mean' not in df_sir.columns or 'node_id' not in df_sir.columns:
                    continue

                # Compute ground truth rank from SIR
                df_sir = df_sir.sort_values('total_mean', ascending=False)
                df_sir['ground_truth_rank'] = df_sir['total_mean'].rank(method='min', ascending=False).astype(int)
                gt_rank_map = df_sir.set_index('node_id')['ground_truth_rank']
                n_nodes = len(df_sir)

                for ranking in rankings:
                    ranking_data = df_methods[df_methods['ranking'] == ranking]

                    if 'rank' not in ranking_data.columns or 'node_id' not in ranking_data.columns:
                        continue

                    if group_methods is not None:
                        ranking_data = ranking_data[ranking_data['method'].isin(group_methods)]

                    if len(ranking_data) == 0:
                        continue

                    # Compute SNR-based ground truth rank for this ranking
                    try:
                        ranking_list = [int(x) for x in ranking.split('_')] if isinstance(ranking, str) else list(ranking)
                        sir_df_snr, _ = _rank_snr(df_sir.copy(), ranking_list, verbose=False, sem_percentile=sem_percentile)
                        snr_rank_map = sir_df_snr.set_index('node_id')['sir_rank_snr']
                    except Exception:
                        snr_rank_map = None

                    # Build chunk with vectorized ops (no iterrows)
                    chunk = ranking_data[['node_id', 'method', 'rank']].copy()
                    chunk['ground_truth_rank'] = chunk['node_id'].map(gt_rank_map)
                    chunk = chunk.dropna(subset=['ground_truth_rank'])
                    if len(chunk) == 0:
                        continue
                    chunk['ground_truth_rank'] = chunk['ground_truth_rank'].astype(int)
                    chunk['method_rank'] = chunk['rank'].astype(int)
                    if snr_rank_map is not None:
                        chunk['sir_rank_snr'] = chunk['node_id'].map(snr_rank_map)
                    chunk['graph_name'] = graph_name
                    chunk['ranking'] = ranking
                    chunk['infection_prob'] = infection_prob
                    chunk['n_nodes'] = n_nodes
                    chunk = chunk.drop(columns=['rank'])

                    if output_path is not None:
                        # Write directly to disk to avoid accumulating in RAM
                        import pyarrow as pa
                        import pyarrow.parquet as pq
                        table = pa.Table.from_pandas(chunk, preserve_index=False)
                        if parquet_writer is None:
                            parquet_writer = pq.ParquetWriter(output_path, table.schema)
                        parquet_writer.write_table(table)
                    else:
                        chunks.append(chunk)

        except Exception as e:
            if not quiet:
                print(f"    Warning: Could not process {methods_file}: {e}")

    if output_path is not None:
        if parquet_writer is not None:
            parquet_writer.close()
        return None

    if not chunks:
        return None

    return pd.concat(chunks, ignore_index=True)


def analyze_by_lexicographic_degeneracy(df: pd.DataFrame, methods_dir: Path, debug=False):
    """
    Group graphs by their lexicographic degeneracy vector and compute performance.

    Note: Each (graph, ranking) combination can have a different degeneracy vector
    since lexipeeling's top-1 node depends on the ranking.
    """
    if not methods_dir or 'ranking' not in df.columns:
        return None

    # Build mapping: (graph_name, ranking) -> lexicographic degeneracy vector
    graph_ranking_to_degeneracy = {}

    # Get unique (graph_name, ranking) combinations
    unique_combos = df[['graph_name', 'ranking']].drop_duplicates()

    if debug:
        print(f"DEBUG: Processing {len(unique_combos)} (graph, ranking) combinations")

    for _, row in unique_combos.iterrows():
        graph_name = row['graph_name']
        ranking = row['ranking']

        if pd.notna(ranking):
            if debug:
                print(f"DEBUG: Extracting degeneracy for {graph_name}, ranking {ranking}")
            degeneracy = get_lexicographic_degeneracy(methods_dir, graph_name, ranking, debug=debug)
            if degeneracy:
                graph_ranking_to_degeneracy[(graph_name, ranking)] = tuple(degeneracy)

    if not graph_ranking_to_degeneracy:
        return None

    # Add degeneracy to dataframe
    df_copy = df.copy()
    df_copy['lexicographic_degeneracy'] = df_copy.apply(
        lambda row: str(list(graph_ranking_to_degeneracy.get((row['graph_name'], row['ranking']))))
        if (row['graph_name'], row['ranking']) in graph_ranking_to_degeneracy else None,
        axis=1
    )

    # Filter to only rows with degeneracy
    df_with_deg = df_copy[df_copy['lexicographic_degeneracy'].notna()]

    if df_with_deg.empty:
        return None

    # Aggregate by degeneracy and method
    by_degeneracy = aggregate_level(df_with_deg, ['lexicographic_degeneracy', 'method'])

    # Sort by degeneracy vector length, then lexicographically
    degeneracy_order = sorted(
        df_with_deg['lexicographic_degeneracy'].unique(),
        key=lambda x: (len(ast.literal_eval(x)), ast.literal_eval(x))
    )

    # Reorder
    try:
        by_degeneracy = by_degeneracy.reindex(
            [(deg, method) for deg in degeneracy_order for method in by_degeneracy.index.get_level_values(1).unique()],
            level=[0, 1]
        )
    except:
        # If reindex fails, just return as is
        pass

    return by_degeneracy


def load_betas_metadata_for_aggregation(methods_dir: Path) -> Dict:
    """
    Load betas metadata from all *_betas_metadata.json files in methods_dir.

    Returns a dict: {graph_name: {ranking_str: {label: metadata_dict}}}
    """
    result = {}
    if not methods_dir:
        return result

    metadata_files = list(methods_dir.glob('*_betas_metadata.json'))
    for meta_file in metadata_files:
        # Extract graph name: {graph_name}_betas_metadata.json
        graph_name = meta_file.stem.replace('_betas_metadata', '')
        try:
            with open(meta_file, 'r') as f:
                data = json.load(f)
            rankings_data = data.get('rankings', {})
            result[graph_name] = rankings_data
        except Exception:
            pass

    return result


def extract_betas_info_for_group(betas_metadata: Dict, group_label: str) -> Optional[Dict]:
    """
    Extract betas info for a specific group label from metadata.

    Returns a dict: {ranking_str: {betas, beta_value}}
    or None if not available.
    """
    if not betas_metadata or group_label == 'none':
        return None

    info = {}
    for graph_name, rankings_data in betas_metadata.items():
        for ranking_str, labels_data in rankings_data.items():
            if group_label in labels_data:
                meta = labels_data[group_label]
                info[ranking_str] = {
                    'betas': meta.get('betas', []),
                    'beta_value': meta.get('beta_value'),
                }
        # Only need one graph's metadata (values are the same structure)
        if info:
            break

    return info if info else None


def _select_for_display(aggregated_df: pd.DataFrame, args, betas_group_label: str = None) -> pd.DataFrame:
    """Filter an aggregated (method-indexed) DataFrame to only the curated display subset.

    Selection: top-3 baselines + top-3 composites + always-included + top-1 per YAML group.
    For a betas group, betas_lexipeeling_{label} and betas_lexipeeling_extended_{label} are
    always included (analogous to lexipeeling/lexipeeling_extended in the standard group).
    Falls back to full df if no metric column is present.
    Result is sorted by the primary metric descending.
    """
    primary_col = (
        'weighted_kendall_tau_mean' if 'weighted_kendall_tau_mean' in aggregated_df.columns
        else 'kendall_correlation_mean'
    )
    if primary_col not in aggregated_df.columns:
        return aggregated_df

    # For betas groups always include betas_lexipeeling_{label} (betas_lexipeeling_extended
    # has no betas variant; lexipeeling_extended itself is included via _ALWAYS_INCLUDED)
    extra_always: set = set()
    if betas_group_label and betas_group_label != 'none':
        extra_always.add(f'betas_lexipeeling_{betas_group_label}')

    groups_yaml = getattr(args, 'method_groups_yaml', None)
    selection = select_methods_for_metric(
        aggregated_df.reset_index(),
        metric_col=primary_col,
        groups_yaml=groups_yaml,
        extra_always_include=extra_always if extra_always else None,
    )
    selected = selection.get('full_selected', [])
    if not selected:
        return aggregated_df

    # Filter to selected methods preserving original MultiIndex structure
    idx = aggregated_df.index
    if isinstance(idx, pd.MultiIndex):
        method_level = idx.names.index('method')
        mask = aggregated_df.index.get_level_values('method').isin(selected)
    else:
        mask = idx.isin(selected)

    filtered = aggregated_df[mask]
    if primary_col in filtered.columns:
        # Sort by the innermost (method) level after filtering
        filtered = filtered.sort_values(primary_col, ascending=False)
    return filtered


# ─── Terminal output helpers ──────────────────────────────────────────────────

_DISPLAY_METRICS = [
    ('weighted_kendall_tau', 'WKendall'),
    ('kendall_correlation', 'Kendall'),
    ('precision_snr_at_3', 'P@3'),
    ('precision_snr_at_5', 'P@5'),
    ('precision_snr_at_10', 'P@10'),
    ('precision_snr_at_1pct', 'P@1%'),
    ('precision_snr_at_3pct', 'P@3%'),
    ('precision_snr_at_5pct', 'P@5%'),
    ('precision_snr_at_10pct', 'P@10%'),
    ('mrr', 'MRR'),
    ('arhr_at_3', 'ARHR@3'),
    ('arhr_at_5', 'ARHR@5'),
    ('arhr_at_10', 'ARHR@10'),
    ('arhr_at_1pct', 'ARHR@1%'),
    ('arhr_at_3pct', 'ARHR@3%'),
    ('arhr_at_5pct', 'ARHR@5%'),
    ('arhr_at_10pct', 'ARHR@10%'),
]

_ANSI_UNDERLINE = '\033[4m'
_ANSI_RESET = '\033[0m'


def _print_top3_per_metric(overall):
    """Print top-3 methods for each display metric present in overall."""
    present = [(col, label) for col, label in _DISPLAY_METRICS
               if f'{col}_mean' in overall.columns]
    if not present:
        return
    max_label = max(len(label) for _, label in present)
    for col, label in present:
        sorted_m = overall[f'{col}_mean'].dropna().sort_values(ascending=False)
        top3 = sorted_m.head(3)
        entries = ', '.join(f'{m} ({v:.4f})' for m, v in top3.items())
        print(f"  {label:<{max_label}} top-3:  {entries}")


def _print_method_table(overall, row_methods):
    """Print comparison table with * (best) and underline (second-best) per column."""
    present = [(col, label) for col, label in _DISPLAY_METRICS
               if f'{col}_mean' in overall.columns]
    row_methods = [m for m in row_methods if m in overall.index]
    if not row_methods or not present:
        return

    has_rank1 = 'num_rank1_nodes_mean' in overall.columns

    # Collect raw values and display strings per metric column
    col_raw: Dict[str, Dict[str, Optional[float]]] = {}
    col_str: Dict[str, Dict[str, str]] = {}
    for col, _ in present:
        col_mean, col_std = f'{col}_mean', f'{col}_std'
        raw, disp = {}, {}
        for m in row_methods:
            v = overall.loc[m, col_mean] if m in overall.index else None
            s = (overall.loc[m, col_std]
                 if col_std in overall.columns and m in overall.index else None)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                raw[m], disp[m] = None, 'N/A'
            else:
                raw[m] = float(v)
                disp[m] = (f'{v:.3f}\u00b1{s:.3f}'
                           if s is not None and not pd.isna(s) else f'{v:.3f}')
        col_raw[col], col_str[col] = raw, disp

    # #Rank1 column strings
    rank1_str: Dict[str, str] = {}
    if has_rank1:
        for m in row_methods:
            if m not in overall.index:
                rank1_str[m] = 'N/A'
                continue
            mn = overall.loc[m, 'num_rank1_nodes_mean']
            if pd.isna(mn):
                rank1_str[m] = 'N/A'
                continue
            sd = (overall.loc[m, 'num_rank1_nodes_std']
                  if 'num_rank1_nodes_std' in overall.columns else None)
            mi = (overall.loc[m, 'num_rank1_nodes_min']
                  if 'num_rank1_nodes_min' in overall.columns else None)
            ma = (overall.loc[m, 'num_rank1_nodes_max']
                  if 'num_rank1_nodes_max' in overall.columns else None)
            s = f'{mn:.1f}'
            if sd is not None and not pd.isna(sd):
                s += f'\u00b1{sd:.1f}'
            if mi is not None and ma is not None and not pd.isna(mi) and not pd.isna(ma):
                s += f' [{int(mi)}\u2013{int(ma)}]'
            rank1_str[m] = s

    # Determine best/second-best rank per column
    def _rank(raw_vals):
        clean = {m: v for m, v in raw_vals.items() if v is not None}
        if not clean:
            return {}
        max_v = max(clean.values())
        rest = [v for v in clean.values() if v < max_v]
        sec_v = max(rest) if rest else None
        return {m: ('best' if v == max_v
                    else 'second' if sec_v is not None and v == sec_v
                    else None)
                for m, v in clean.items()}

    markers = {col: _rank(col_raw[col]) for col, _ in present}

    # Column widths (based on raw string lengths)
    method_w = max(len('Method'), max(len(m) for m in row_methods))
    metric_ws = [max(len(label),
                     max(len(col_str[col].get(m, 'N/A')) + 2 for m in row_methods))
                 for col, label in present]
    rank1_w = (max(len('#Rank1'), max(len(rank1_str.get(m, 'N/A')) for m in row_methods))
               if has_rank1 else 0)

    # Header line
    header_parts = [f"  {'Method':<{method_w}}"]
    for i, (_, label) in enumerate(present):
        header_parts.append(f" {label:^{metric_ws[i]}}")
    if has_rank1:
        header_parts.append(f" {'#Rank1':^{rank1_w}}")
    print('|'.join(header_parts))

    # Separator line
    seps = ['  ' + '-' * method_w] + ['-' * (w + 1) for w in metric_ws]
    if has_rank1:
        seps.append('-' * (rank1_w + 1))
    print('+'.join(seps))

    # Data rows
    for m in row_methods:
        parts = [f"  {m:<{method_w}}"]
        for i, (col, _) in enumerate(present):
            raw_s = col_str[col].get(m, 'N/A')
            rank = markers[col].get(m)
            w = metric_ws[i]
            if rank == 'best':
                parts.append(' ' + f'{(raw_s + " *"):<{w}}')
            elif rank == 'second':
                parts.append(' ' + _ANSI_UNDERLINE + raw_s + _ANSI_RESET + ' ' * (w - len(raw_s)))
            else:
                parts.append(f' {raw_s:<{w}}')
        if has_rank1:
            parts.append(f' {rank1_str.get(m, "N/A"):<{rank1_w}}')
        print('|'.join(parts))


def _parse_node_ids(raw_nodes) -> List[int]:
    """Extract integer node IDs from a stored string (handles np.int64 repr etc.)."""
    import re as _re
    try:
        parsed = ast.literal_eval(str(raw_nodes))
        if isinstance(parsed, (set, list)):
            return sorted(int(n) for n in parsed)
        else:
            return [int(parsed)]
    except Exception:
        ids = _re.findall(r'\d+', str(raw_nodes))
        return [int(x) for x in ids] if ids else []


def _print_ranking_boxplot(overall, best_spreader_df, row_methods, df=None,
                           betas_group_label=None):
    """Print per-ranking numeric reach table (mean \u00b1 std per group per method)."""
    if best_spreader_df is None or len(best_spreader_df) == 0:
        return

    BOXPLOT_METRICS = ['mrr', 'arhr_at_3', 'arhr_at_5', 'arhr_at_10']
    non_comp = [m for m in overall.index if not is_composite_method(m)]
    second_best: List[str] = []
    for metric in BOXPLOT_METRICS:
        col_mean = f'{metric}_mean'
        if col_mean not in overall.columns:
            continue
        pool = overall.loc[[m for m in non_comp if m in overall.index], col_mean].dropna()
        if len(pool) >= 2:
            second_best.append(pool.sort_values(ascending=False).index[1])
    second_best = list(dict.fromkeys(second_best))

    always = ['lexipeeling', 'lexipeeling_extended']
    if betas_group_label and betas_group_label != 'none':
        always = list(dict.fromkeys(always + [
            f'betas_lexipeeling_{betas_group_label}',
            f'betas_lexipeeling_extended_{betas_group_label}',
        ]))
    composites_in_table = [m for m in row_methods if is_composite_method(m)]
    candidates = list(dict.fromkeys(always + composites_in_table + second_best))

    for ranking in sorted(best_spreader_df['ranking'].unique()):
        df_r = best_spreader_df[best_spreader_df['ranking'] == ranking]
        try:
            group_ids = ast.literal_eval(df_r['ranking_groups'].iloc[0])
        except Exception:
            group_ids = list(range(
                sum(1 for c in df_r.columns if c.startswith('group_') and c.endswith('_pct_mean'))))
        n_groups = len(group_ids)

        avail = set(df_r['method'].unique())
        show = [m for m in candidates if m in avail]
        if not show:
            continue

        # Aggregate across all (graph_name, infection_prob) for this ranking
        agg: Dict[str, Tuple[list, list]] = {}
        for m in show:
            df_m = df_r[df_r['method'] == m]
            means, stds = [], []
            for i in range(n_groups):
                mc, sc = f'group_{i}_pct_mean', f'group_{i}_pct_std'
                means.append(df_m[mc].mean() if mc in df_m.columns else None)
                stds.append(df_m[sc].mean() if sc in df_m.columns else 0.0)
            agg[m] = (means, stds)

        ranking_display = '[' + ', '.join(ranking.split('_')) + ']'
        group_labels = [f'Group {g}' for g in group_ids]

        # Collect strategies for this ranking from the enriched df
        strategy_str = ''
        if df is not None and 'ranking_strategy' in df.columns:
            strategies = (df[df['ranking'] == ranking]['ranking_strategy']
                          .dropna().unique().tolist())
            if strategies:
                strategy_str = '  (' + ', '.join(sorted(strategies)) + ')'

        print(f"\n  Ranking {ranking_display}{strategy_str}  \u2014  avg reach % from best-spreader among rank-1 nodes")

        # Best achievable per group: max avg spread across ALL methods in df_r
        all_method_means: Dict[str, list] = {}
        for m_all in df_r['method'].unique():
            df_m_all = df_r[df_r['method'] == m_all]
            all_method_means[m_all] = [
                df_m_all[f'group_{i}_pct_mean'].mean()
                if f'group_{i}_pct_mean' in df_m_all.columns else None
                for i in range(n_groups)
            ]

        best_info = []  # (best_val_str, node_str) per group
        for i in range(n_groups):
            best_val, best_m = None, None
            for m_all, means_all in all_method_means.items():
                v = means_all[i]
                if v is not None and not pd.isna(v) and (best_val is None or v > best_val):
                    best_val, best_m = v, m_all
            if best_val is None or best_m is None:
                best_info.append(('?', '?'))
                continue
            # Node from the single raw row with the highest group_i_pct_mean for best_m
            mc = f'group_{i}_pct_mean'
            df_bm = df_r[df_r['method'] == best_m]
            node_disp = '?'
            if mc in df_bm.columns and len(df_bm) > 0:
                best_row = df_bm.loc[df_bm[mc].idxmax()]
                raw_nodes = best_row.get('best_spreader_nodes', None)
                if raw_nodes is not None:
                    ids = _parse_node_ids(raw_nodes)
                    if ids:
                        node_disp = (str(ids[0]) if len(ids) == 1
                                     else '/'.join(str(n) for n in ids[:3]))
            best_info.append((f'{best_val:.1f}%', node_disp))

        # Column widths: accommodate both the group label and the best-info line
        best_labels = [f'{v} (n={nd})' for v, nd in best_info]
        method_w = max(len('Method'), max(len(m) for m in show))
        group_w = max(
            max(len(g) for g in group_labels),
            max(len(b) for b in best_labels),
            14,
        )

        # Two-row header: best info row + group name row
        best_header = f"  {'':>{method_w}}"
        for bl in best_labels:
            best_header += f" | {bl:^{group_w}}"
        print(best_header)

        header = f"  {'Method':<{method_w}}"
        for g in group_labels:
            header += f" | {g:^{group_w}}"
        print(header)
        print('  ' + '-' * method_w + (' | ' + '-' * group_w) * n_groups)

        for m in show:
            means, stds = agg[m]
            row = f"  {m:<{method_w}}"
            for i in range(n_groups):
                cell = (f'{means[i]:.1f} \u00b1 {stds[i]:.1f}'
                        if means[i] is not None else 'N/A')
                row += f" | {cell:^{group_w}}"
            print(row)


def _print_betas_block(overall, df, best_spreader_df, selections_by_metric, betas_group_label):
    """Print the full terminal summary for one betas group."""
    label = betas_group_label if betas_group_label else 'none'
    header = f"BETAS: {label}"
    w = len(header) + 4
    print(f"\n{'═' * w}")
    print(f"  {header}")
    print(f"{'═' * w}")

    # Section 1: Top-3 per metric
    print("\n  TOP-3 METHODS PER METRIC")
    _print_top3_per_metric(overall)

    # Union of full_selected across all metrics, sorted by primary metric descending
    row_methods: List[str] = []
    for sel in selections_by_metric.values():
        for m in sel.get('full_selected', []):
            if m not in row_methods:
                row_methods.append(m)
    primary_col = ('weighted_kendall_tau_mean' if 'weighted_kendall_tau_mean' in overall.columns
                   else 'kendall_correlation_mean')
    if primary_col in overall.columns:
        row_methods.sort(
            key=lambda m: -float(overall.loc[m, primary_col])
            if m in overall.index and not pd.isna(overall.loc[m, primary_col]) else 0.0
        )

    # Section 2: Method comparison table
    print("\n  METHOD COMPARISON TABLE")
    _print_method_table(overall, row_methods)

    # Section 3: Per-ranking group reach
    if best_spreader_df is not None and len(best_spreader_df) > 0:
        print("\n  PER-RANKING GROUP REACH")
        _print_ranking_boxplot(overall, best_spreader_df, row_methods, df,
                               betas_group_label=betas_group_label)


def run_aggregation_pipeline(df, output_dir, args, betas_group_label=None, betas_info=None):
    """
    Run the full aggregation pipeline on a subset of data.

    Args:
        df: DataFrame subset for this betas group
        output_dir: Where to save output files (subdirectory per group)
        args: CLI args
        betas_group_label: e.g. "none", "equal_0.5"
        betas_info: dict with per-ranking betas values
    """
    quiet = args.quiet
    verbose = args.verbose

    # Level 1: Overall
    overall = aggregate_level(df, 'method')
    overall.to_csv(output_dir / 'level1_overall.csv', index=True)

    # Levels 2-5: optional extended views
    if args.extended_levels:
        # Level 2: By type
        by_type = aggregate_level(df, ['detailed_type', 'method'])
        by_type.to_csv(output_dir / 'level2_by_type.csv', index=True)

        by_dist = aggregate_level(df, ['size_distribution', 'method'])
        by_dist.to_csv(output_dir / 'level3_by_distribution.csv', index=True)

        by_config = aggregate_level(df, ['base_config', 'method'])
        by_config.to_csv(output_dir / 'level4_by_config.csv', index=True)

        combined = aggregate_level(df, ['detailed_type', 'size_distribution', 'method'])
        combined.to_csv(output_dir / 'level5_combined.csv', index=True)

    # Ranking Analysis
    if 'ranking' in df.columns and df['ranking'].notna().any():
        by_ranking = aggregate_level(df[df['ranking'].notna()], ['ranking', 'method'])
        by_ranking.to_csv(output_dir / 'by_ranking.csv', index=True)

    # By Ranking Strategy
    if 'ranking_strategy' in df.columns and df['ranking_strategy'].notna().any():
        df_with_strategy = df[df['ranking_strategy'].notna()]
        by_strategy = aggregate_level(df_with_strategy, ['ranking_strategy', 'method'])
        by_strategy.to_csv(output_dir / 'by_ranking_strategy.csv', index=True)

    # By Ranking Length
    if 'ranking_length' in df.columns and df['ranking_length'].notna().any():
        df_with_length = df[df['ranking_length'].notna()]
        by_length = aggregate_level(df_with_length, ['ranking_length', 'method'])
        by_length.to_csv(output_dir / 'by_ranking_length.csv', index=True)

    # By Ranking Strategy AND Length
    if ('ranking_strategy' in df.columns and df['ranking_strategy'].notna().any() and
        'ranking_length' in df.columns and df['ranking_length'].notna().any()):
        df_with_both = df[df['ranking_strategy'].notna() & df['ranking_length'].notna()]
        by_strategy_length = aggregate_level(df_with_both, ['ranking_strategy', 'ranking_length', 'method'])
        by_strategy_length.to_csv(output_dir / 'by_ranking_strategy_and_length.csv', index=True)

    # By SIR Infection Probability
    if 'infection_prob' in df.columns and df['infection_prob'].notna().any():
        df_with_prob = df[df['infection_prob'].notna()].copy()
        df_with_prob['infection_prob_rounded'] = df_with_prob['infection_prob'].round(4)
        by_prob = aggregate_level(df_with_prob, ['infection_prob_rounded', 'method'])
        by_prob.to_csv(output_dir / 'by_infection_prob.csv', index=True)

    # By Ranking Strategy AND SIR Probability
    if ('ranking_strategy' in df.columns and df['ranking_strategy'].notna().any() and
        'infection_prob' in df.columns and df['infection_prob'].notna().any()):
        df_with_both = df[df['ranking_strategy'].notna() & df['infection_prob'].notna()].copy()
        df_with_both['infection_prob_rounded'] = df_with_both['infection_prob'].round(4)
        by_strategy_prob = aggregate_level(df_with_both, ['ranking_strategy', 'infection_prob_rounded', 'method'])
        by_strategy_prob.to_csv(output_dir / 'by_ranking_strategy_and_prob.csv', index=True)

    # By Lexicographic Degeneracy
    by_degeneracy = None
    if args.methods_dir:
        by_degeneracy = analyze_by_lexicographic_degeneracy(df, args.methods_dir, debug=False)
        if by_degeneracy is not None:
            by_degeneracy.to_csv(output_dir / 'by_lexicographic_degeneracy.csv', index=True)

    # (BY LEXICOGRAPHIC DEGENERACY table suppressed from terminal; data saved to file above)

    # Tie Statistics
    # Filter to only methods present in this betas group
    group_methods = set(df['method'].unique())

    if args.methods_dir:
        methods_ties_df, sir_ties_df = analyze_ties_from_methods(
            args.methods_dir, args.sir_dir, quiet=quiet
        )

        if methods_ties_df is not None and len(methods_ties_df) > 0:
            methods_ties_df = methods_ties_df[methods_ties_df['method'].isin(group_methods)]

        if methods_ties_df is not None and len(methods_ties_df) > 0:
            methods_ties_df.to_csv(output_dir / 'ties_detailed_methods.csv', index=False)

            methods_ties_agg = aggregate_tie_statistics(methods_ties_df)
            if methods_ties_agg is not None:
                methods_ties_agg.to_csv(output_dir / 'ties_by_method.csv', index=True)

            if 'ranking' in methods_ties_df.columns:
                ties_by_method_ranking = methods_ties_df.groupby(['method', 'ranking']).agg({
                    'num_unique_ranks': 'mean',
                    'max_tie_size': 'max',
                    'mean_tie_size': 'mean',
                    'pct_nodes_in_ties': 'mean'
                }).round(2)
                ties_by_method_ranking.to_csv(output_dir / 'ties_by_method_ranking.csv', index=True)

        if sir_ties_df is not None and len(sir_ties_df) > 0:
            sir_ties_df.to_csv(output_dir / 'ties_sir_ground_truth.csv', index=False)

    # Best Spreader Group Reach Analysis
    best_spreader_df = None
    if args.methods_dir and args.sir_dir and args.graphs_dir:
        best_spreader_df = analyze_best_spreader_group_reach(
            df=df,
            methods_dir=args.methods_dir,
            sir_dir=args.sir_dir,
            graphs_dir=args.graphs_dir,
            comparison_dir=args.comparison_dir,
            betas_group_label=betas_group_label,
            betas_info=betas_info,
            quiet=quiet
        )

        if best_spreader_df is not None and len(best_spreader_df) > 0:
            best_spreader_df.to_csv(output_dir / 'best_spreader_group_reach.csv', index=False)
            best_spreader_agg = aggregate_best_spreader_across_probs(best_spreader_df)
            if best_spreader_agg is not None and len(best_spreader_agg) > 0:
                best_spreader_agg.to_csv(output_dir / 'best_spreader_group_reach_agg_by_prob.csv', index=False)
        else:
            best_spreader_df = None

    # Node-level ranks — write directly to disk to avoid OOM on large datasets
    if args.methods_dir and args.sir_dir:
        node_ranks_output = output_dir / 'node_ranks.parquet'
        if node_ranks_output.exists():
            node_ranks_output.unlink()
        extract_node_ranks(
            methods_dir=args.methods_dir,
            sir_dir=args.sir_dir,
            comparison_dir=args.comparison_dir,
            quiet=quiet,
            output_path=node_ranks_output,
            group_methods=list(group_methods) if group_methods is not None else None,
            sem_percentile=getattr(args, 'sem_percentile', 50.0),
        )

    # Per-metric method selection
    groups_yaml = getattr(args, 'method_groups_yaml', None)
    primary_metrics = [
        'weighted_kendall_tau', 'mrr',
        'arhr_at_1', 'arhr_at_3', 'arhr_at_5', 'arhr_at_10',
        'arhr_at_1pct', 'arhr_at_3pct', 'arhr_at_5pct', 'arhr_at_10pct',
        'precision_snr_at_1', 'precision_snr_at_3', 'precision_snr_at_5', 'precision_snr_at_10',
        'precision_snr_at_1pct', 'precision_snr_at_3pct', 'precision_snr_at_5pct', 'precision_snr_at_10pct',
    ]
    primary_metrics = [m for m in primary_metrics if f'{m}_mean' in overall.columns]
    selections_by_metric: Dict[str, Dict] = {}
    if primary_metrics:
        selections_by_metric = save_per_metric_selections(
            detail_df=df,
            aggregated_df=overall.reset_index(),
            output_dir=output_dir,
            primary_metric_cols=primary_metrics,
            groups_yaml=groups_yaml,
            quiet=True,
        )

    if not quiet:
        _print_betas_block(overall, df, best_spreader_df, selections_by_metric, betas_group_label)
        print(f"\nOutput saved to: {output_dir}")



def main():
    parser = argparse.ArgumentParser(
        description='Multi-level aggregation of comparison results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--comparison-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--graphs-dir', type=Path, help='Directory with graph files (needed for ranking strategy mapping)')
    parser.add_argument('--methods-dir', type=Path, help='Directory with method results (needed for detailed analysis and rank-1 counts)')
    parser.add_argument('--sir-dir', type=Path, help='Directory with SIR results (if different from auto-detected path)')
    parser.add_argument('--verbose', type=int, default=0, choices=[0, 1, 2],
                        help='Verbosity level: 0=minimal, 1=include strategy+length/prob tables, 2=include all levels and ranking analysis')
    parser.add_argument('--quiet', action='store_true', help='Suppress all output (only save files)')
    parser.add_argument('--extended-levels', action='store_true',
                        help='Also produce level2_by_type, level3_by_distribution, level4_by_config, level5_combined')
    parser.add_argument('--method-groups-yaml', type=Path,
                        default=Path('methods-files/to_plot/methods_groups.yaml'),
                        help='Optional YAML file defining method groups for per-metric selection. '
                             'Format: method_groups: [{name: "...", methods: [...]}]')
    parser.add_argument('--sem-percentile', type=float, default=50.0,
                        help='Percentile of SEM distribution for SNR discretization '
                             '(default: 50). Lower → fewer ties, higher → more ties.')
    parser.add_argument('--config', type=Path, default=None,
                        help='Path to experiment config YAML. If sem_percentile is a list, '
                             'aggregates each sem_{p} subdirectory of --comparison-dir.')

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

    base_comparison_dir = args.comparison_dir
    base_output_dir = args.output_dir

    # Pre-load shared caches (persist across sem_percentile iterations)
    _strategies_cache = {}
    _methods_df_cache = {}
    _betas_metadata = load_betas_metadata_for_aggregation(args.methods_dir) if args.methods_dir else None

    for sem_p in sem_percentile_values:
        comparison_dir = base_comparison_dir / f'sem_{int(sem_p)}' if use_subdirs else base_comparison_dir
        output_dir = base_output_dir / f'sem_{int(sem_p)}' if use_subdirs else base_output_dir

        if use_subdirs and not args.quiet:
            print(f"\n{'='*70}")
            print(f"AGGREGATION — SEM percentile: {int(sem_p)}")
            print(f"{'='*70}")

        # Temporarily override args for this iteration
        args.sem_percentile = sem_p
        args.comparison_dir = comparison_dir
        args.output_dir = output_dir

        # Load data
        df = load_all_comparisons(comparison_dir, quiet=args.quiet)
        if df is None:
            if use_subdirs:
                print(f"  Skipping sem_{int(sem_p)}: no comparison files found in {comparison_dir}")
                continue
            return 1

        # Enrich dataframe with ranking info (length, strategy, rank-1 counts)
        df = enrich_dataframe_with_ranking_info(
            df,
            graphs_dir=args.graphs_dir,
            methods_dir=args.methods_dir,
            quiet=args.quiet,
            strategies_cache=_strategies_cache,
            methods_df_cache=_methods_df_cache,
        )

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed data (all groups combined) at top level
        detailed_file = output_dir / 'detailed_all_comparisons.csv'
        df.to_csv(detailed_file, index=False)

        # Print graph statistics (nodes, edges)
        if not args.quiet and args.graphs_dir:
            print(f"\n{'='*70}")
            print(f"GRAPH STATISTICS")
            print(f"{'='*70}")
            unique_graphs = df['graph_name'].unique()
            for graph_name in sorted(unique_graphs):
                edge_file = args.graphs_dir / f"{graph_name}_edges.txt"
                if edge_file.exists():
                    n_edges = sum(1 for _ in open(edge_file))
                    nodes = set()
                    with open(edge_file) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                nodes.add(parts[0])
                                nodes.add(parts[1])
                    n_nodes = len(nodes)
                    print(f"  {graph_name}: {n_nodes} nodes, {n_edges} edges")
                else:
                    found = list(args.graphs_dir.rglob(f"{graph_name}_edges.txt"))
                    if found:
                        n_edges = sum(1 for _ in open(found[0]))
                        nodes = set()
                        with open(found[0]) as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 2:
                                    nodes.add(parts[0])
                                    nodes.add(parts[1])
                        n_nodes = len(nodes)
                        print(f"  {graph_name}: {n_nodes} nodes, {n_edges} edges")
                    else:
                        print(f"  {graph_name}: edge file not found")
            print(f"{'='*70}")

        # Determine betas groups
        def _betas_sort_key(label: str) -> float:
            if label == 'none':
                return -1.0
            try:
                return float(label.split('_')[-1])
            except ValueError:
                return 0.0

        if 'betas_group' in df.columns:
            df['betas_group'] = df['betas_group'].fillna('none')
            betas_groups = sorted(df['betas_group'].unique(), key=_betas_sort_key)
        else:
            betas_groups = ['none']

        for group in betas_groups:
            if 'betas_group' in df.columns:
                df_group = df[df['betas_group'] == group]
            else:
                df_group = df

            if len(betas_groups) > 1:
                group_output_dir = output_dir / f'betas_{group}'
            else:
                group_output_dir = output_dir

            group_output_dir.mkdir(parents=True, exist_ok=True)

            betas_info = extract_betas_info_for_group(_betas_metadata, group)

            run_aggregation_pipeline(df_group, group_output_dir, args, group, betas_info)

    return 0


if __name__ == '__main__':
    exit(main())
