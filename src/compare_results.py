#!/usr/bin/env python3
"""
Compare Results

Comparison and evaluation of method results vs SIR ground truth.

Workflow:
1. Load method results 
2. Load SIR results 
3. Rank SIR results using lexicographic comparison
4. Compare method predictions vs SIR ground truth
5. Compute comparison metrics (rank correlation, Precision@K, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import argparse
from typing import List, Dict, Tuple
from scipy.stats import kendalltau, weightedtau
import warnings
import time
import re

sys.path.insert(0, str(Path(__file__).parent))
from lexicographic_utils import sort_by_ranking


def rank_sir_results(
    sir_df: pd.DataFrame,
    ranking: List[int],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Rank SIR results by lexicographic comparison of reach vectors

    Args:
        sir_df: DataFrame with SIR results (raw data)
        ranking: Group ranking
        verbose: Print progress messages

    Returns:
        DataFrame with added 'sir_rank' column
    """
    if verbose:
        print("Ranking SIR results by reach vectors...")

    cols = [f'group_{group_id}_mean' for group_id in ranking]
    sir_df = sir_df.copy()
    sorted_df = sir_df.sort_values(cols, ascending=False).reset_index(drop=True)

    # Mark rows where the vector differs from the previous row
    different = (sorted_df[cols] != sorted_df[cols].shift(1)).any(axis=1)
    different.iloc[0] = True
    group_num = different.cumsum() - 1

    # Rank = 1-indexed position of the first row in each group
    first_row_idx = different[different].index
    rank_for_group = pd.Series(first_row_idx + 1, index=range(len(first_row_idx)))
    sorted_df['sir_rank'] = group_num.map(rank_for_group).astype(int)

    sir_df = sir_df.merge(sorted_df[['node_id', 'sir_rank']], on='node_id')

    top_node = sorted_df.iloc[0]['node_id']
    if verbose:
        print(f"Ranked {len(sir_df)} nodes")
        print(f"Top SIR spreader: Node {top_node} (rank 1)")

    return sir_df


def _discretize_group(sir_df, group_id, sem_percentile, sqrt_n):
    """Compute SNR integer bins for one group. Returns int64 array."""
    mean_vals = sir_df[f'group_{group_id}_mean'].values.astype(float)
    if sem_percentile == 0:
        _, inverse = np.unique(mean_vals, return_inverse=True)
        return inverse.astype(np.int64)
    std_vals = sir_df[f'group_{group_id}_std'].values.astype(float)
    sem_vals = std_vals / sqrt_n
    positive_sem = sem_vals[sem_vals > 0]
    epsilon = np.percentile(positive_sem, sem_percentile) if len(positive_sem) > 0 else 1.0 / sqrt_n
    return np.floor(mean_vals / epsilon).astype(np.int64)


def rank_sir_results_betas(
    sir_df: pd.DataFrame,
    ranking: List[int],
    betas: List[float],
    verbose: bool = True,
    sem_percentile: float = 0.0,
) -> pd.DataFrame:
    """
    Rank SIR results by lexicographic comparison of blended reach vectors.

    When sem_percentile > 0, reach values are first discretized into SNR bins
    (same logic as rank_sir_results_snr), then blended with betas.
    When sem_percentile == 0, raw means are blended directly.

    For each node with values [v0, v1, ..., v_{s-1}]:
        blended[i] = v[i] + betas[i] * v[i+1]   for i < s-1
        blended[s-1] = v[s-1]

    Then rank blended vectors lexicographically.

    Args:
        sir_df: DataFrame with SIR results (raw data)
        ranking: Group ranking
        betas: Betas vector of length len(ranking) - 1
        sem_percentile: SEM percentile for discretization (0 = no discretization)

    Returns:
        DataFrame with added 'sir_rank' column
    """
    s = len(ranking)
    assert len(betas) == s - 1, f"betas length {len(betas)} != ranking length - 1 ({s - 1})"

    if verbose:
        print(f"Ranking SIR results by blended reach vectors (betas={betas}, sem_percentile={sem_percentile})...")

    sir_df = sir_df.copy()

    if sem_percentile > 0:
        # Discretize first, then blend
        n_sim = float(sir_df['num_simulations'].iloc[0]) if 'num_simulations' in sir_df.columns else 1000.0
        sqrt_n = np.sqrt(n_sim)
        value_cols = []
        for group_id in ranking:
            col = f'_snr_{group_id}'
            sir_df[col] = _discretize_group(sir_df, group_id, sem_percentile, sqrt_n)
            value_cols.append(col)
    else:
        # Blend raw means directly
        value_cols = [f'group_{group_id}_mean' for group_id in ranking]

    # Compute blended values vectorized
    blend_cols = []
    for i in range(s - 1):
        col = f'_blend_{i}'
        sir_df[col] = sir_df[value_cols[i]] + betas[i] * sir_df[value_cols[i + 1]]
        blend_cols.append(col)
    last_col = f'_blend_{s - 1}'
    sir_df[last_col] = sir_df[value_cols[s - 1]]
    blend_cols.append(last_col)

    sorted_df = sir_df.sort_values(blend_cols, ascending=False).reset_index(drop=True)

    # Mark rows where the blended vector differs from the previous row
    different = (sorted_df[blend_cols] != sorted_df[blend_cols].shift(1)).any(axis=1)
    different.iloc[0] = True
    group_num = different.cumsum() - 1

    # Rank = 1-indexed position of the first row in each group
    first_row_idx = different[different].index
    rank_for_group = pd.Series(first_row_idx + 1, index=range(len(first_row_idx)))
    sorted_df['sir_rank'] = group_num.map(rank_for_group).astype(int)

    sir_df = sir_df.merge(sorted_df[['node_id', 'sir_rank']], on='node_id')
    drop_cols = blend_cols[:]
    if sem_percentile > 0:
        drop_cols += [c for c in sir_df.columns if c.startswith('_snr_')]
    sir_df = sir_df.drop(columns=drop_cols)

    top_node = sorted_df.iloc[0]['node_id']
    if verbose:
        print(f"Ranked {len(sir_df)} nodes (blended)")
        print(f"Top SIR spreader (blended): Node {top_node} (rank 1)")

    return sir_df


def rank_sir_results_snr(
    sir_df: pd.DataFrame,
    ranking: List[int],
    verbose: bool = True,
    sem_percentile: float = 50.0,
) -> Tuple[pd.DataFrame, int]:
    """
    Rank SIR results using global-epsilon discretization.

    For each group j in ranking, compute a global epsilon:
        ε_j = percentile(std_j(node) / sqrt(n_sim), sem_percentile)
              over nodes with std_j > 0

    Then for each node:
        snr_j = floor(mean_j / ε_j)

    This discretizes continuous reach values into bins of width ε_j.
    Lower percentile → smaller epsilon → more bins → fewer ties.
    Higher percentile → larger epsilon → fewer bins → more ties.

    Adds columns to sir_df:
        snr_{group_id}   (int64)  — SNR component for each group in ranking
        sir_rank_snr     (int)    — rank based on SNR vector (primary ground truth)

    Args:
        sir_df: DataFrame with SIR results (raw data)
        ranking: Group ranking
        verbose: Print progress messages

    Returns:
        (sir_df_with_snr_cols, U) where U = max(all snr values) + 1
    """
    if verbose:
        print("Ranking SIR results by SNR-discretized reach vectors...")

    sir_df = sir_df.copy()
    s = len(ranking)

    if 'num_simulations' in sir_df.columns:
        n_sim = float(sir_df['num_simulations'].iloc[0])
    else:
        n_sim = 1000.0
        if verbose:
            print("  Warning: 'num_simulations' column not found, assuming n_sim=1000")
    sqrt_n = np.sqrt(n_sim)

    snr_cols = []
    for group_id in ranking:
        snr_col = f'snr_{group_id}'
        sir_df[snr_col] = _discretize_group(sir_df, group_id, sem_percentile, sqrt_n)
        snr_cols.append(snr_col)

    U = int(max(sir_df[col].max() for col in snr_cols)) + 1

    sorted_df = sir_df.sort_values(snr_cols, ascending=False).reset_index(drop=True)

    different = (sorted_df[snr_cols] != sorted_df[snr_cols].shift(1)).any(axis=1)
    different.iloc[0] = True
    group_num = different.cumsum() - 1

    first_row_idx = different[different].index
    rank_for_group = pd.Series(first_row_idx + 1, index=range(len(first_row_idx)))
    sorted_df['sir_rank_snr'] = group_num.map(rank_for_group).astype(int)

    sir_df = sir_df.merge(sorted_df[['node_id', 'sir_rank_snr']], on='node_id')

    if verbose:
        top_node = sorted_df.iloc[0]['node_id']
        num_unique = int((sorted_df['sir_rank_snr'] != sorted_df['sir_rank_snr'].shift(1)).sum())
        print(f"  SNR-ranked {len(sir_df)} nodes, U={U}")
        print(f"  Unique SNR ranks: {num_unique}/{len(sir_df)} ({num_unique/len(sir_df):.1%})")
        print(f"  Top SNR spreader: Node {top_node} (rank 1)")

    return sir_df, U


def compute_tie_statistics(
    sir_df: pd.DataFrame,
    ranking: List[int],
    infection_prob: float = None,
) -> List[Dict]:
    """
    Count ties at each component position for raw-mean and SNR variants.

    For each variant (raw_mean, snr) and for each prefix length j (1..s):
    - Count nodes that share identical values for the first j components with at least one other node

    Args:
        sir_df: DataFrame with SIR results; snr_{id} columns must exist for SNR variant
        ranking: Group ranking
        infection_prob: Infection probability (stored in output, not used for computation)

    Returns:
        List of dicts: variant, component_index, group_id, infection_prob,
                       num_nodes_tied, num_unique_vectors, num_nodes, fraction_tied
    """
    n = len(sir_df)
    rows = []

    variants = [('raw_mean', [f'group_{g}_mean' for g in ranking])]
    snr_cols = [f'snr_{g}' for g in ranking]
    if all(c in sir_df.columns for c in snr_cols):
        variants.append(('snr', snr_cols))

    for variant_name, col_list in variants:
        for j in range(len(ranking)):
            prefix_cols = col_list[:j + 1]
            group_sizes = sir_df.groupby(prefix_cols, sort=False).size()
            tied_groups = group_sizes[group_sizes >= 2]
            num_nodes_tied = int(tied_groups.sum())
            num_unique_vectors = int(len(group_sizes))
            rows.append({
                'variant': variant_name,
                'component_index': j,
                'group_id': ranking[j],
                'infection_prob': infection_prob,
                'num_nodes_tied': num_nodes_tied,
                'num_unique_vectors': num_unique_vectors,
                'num_nodes': n,
                'fraction_tied': num_nodes_tied / n if n > 0 else 0.0,
            })

    return rows


def print_tie_statistics(tie_stats: List[Dict], ranking: List[int],
                         infection_prob: float = None, verbose: bool = True):
    """Print tie statistics — full table (verbose=True) or one summary line (verbose=False)."""
    if not tie_stats:
        return
    prob_str = f"  p={infection_prob:.4f}" if infection_prob is not None else ""
    ranking_str = '_'.join(map(str, ranking))

    if not verbose:
        # One compact line per variant showing fraction tied at each component
        by_variant: Dict[str, List] = {}
        for stat in tie_stats:
            by_variant.setdefault(stat['variant'], []).append(stat)
        parts = []
        for variant, rows in by_variant.items():
            rows_sorted = sorted(rows, key=lambda r: r['component_index'])
            comp_parts = '  '.join(
                f"comp{r['component_index']}={r['fraction_tied']:.1%}" for r in rows_sorted
            )
            parts.append(f"{variant}: {comp_parts}")
        print(f"  Ties  r={ranking_str}{prob_str}  |  " + '  |  '.join(parts))
        return

    print(f"\n  Tie statistics  ranking={ranking_str}{prob_str}:")
    print(f"  {'Variant':>10}  {'Comp':>5}  {'Group':>5}  {'Tied nodes':>12}  {'Unique vecs':>12}  {'Frac tied':>10}")
    print(f"  {'-'*10}  {'-'*5}  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*10}")
    for stat in tie_stats:
        print(f"  {stat['variant']:>10}  {stat['component_index']:>5}  {stat['group_id']:>5}  "
              f"{stat['num_nodes_tied']:>12,}  {stat['num_unique_vectors']:>12,}  "
              f"{stat['fraction_tied']:>10.3%}")


def save_tie_statistics(
    tie_stats: List[Dict],
    output_path: Path,
    ranking: List[int],
):
    """Save tie statistics to CSV file."""
    if not tie_stats:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(tie_stats)
    df['ranking'] = '_'.join(map(str, ranking))
    df.to_csv(output_path, index=False)


def compute_precision_at_k(
    method_ranks: pd.Series,
    sir_ranks: pd.Series,
    k: int
) -> Tuple[float, int]:
    """
    Compute Precision@K: fraction of method's top-K that are in SIR's top-K

    Args:
        method_ranks: Series mapping node_id -> method rank
        sir_ranks: Series mapping node_id -> SIR rank
        k: K value

    Returns:
        Tuple of (precision@k score, effective_k)
        effective_k is min(k, num_evaluated_nodes)
    """
    num_evaluated = len(method_ranks)
    effective_k = min(k, num_evaluated)

    method_top_k = set(method_ranks[method_ranks <= effective_k].index)
    sir_top_k = set(sir_ranks[sir_ranks <= effective_k].index)

    if len(method_top_k) == 0:
        return 0.0, effective_k

    intersection = method_top_k & sir_top_k
    return min(len(intersection), effective_k) / effective_k, effective_k



def compute_weighted_kendall(method_ranks: pd.Series, sir_ranks: pd.Series) -> Tuple[float, float]:
    """Compute weighted Kendall's tau (hyperbolic weights) between two rank series."""
    common = method_ranks.index.intersection(sir_ranks.index)
    if len(common) < 2:
        return None, None
    m = method_ranks.loc[common].values.astype(float)
    s = sir_ranks.loc[common].values.astype(float)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            result = weightedtau(m, s)
        tau = float(result.statistic) if hasattr(result, 'statistic') else float(result[0])
        pval = float(result.pvalue) if hasattr(result, 'pvalue') else None
        if np.isnan(tau):
            tau = 0.0
        return tau, pval
    except Exception:
        return None, None


def compute_mrr(method_ranks: pd.Series, sir_ranks: pd.Series) -> float:
    """
    Compute MRR (ground-truth-centric): mean(1/method_rank) over all SIR top-1 nodes.
    """
    top_sir = sir_ranks[sir_ranks == 1].index
    rrs = [1.0 / float(method_ranks[n]) for n in top_sir if n in method_ranks.index]
    return float(np.mean(rrs)) if rrs else None


def compute_arhr_at_k(method_ranks: pd.Series, sir_ranks: pd.Series, k: int) -> float:
    """
    Compute arhr@k: mean(1/method_rank) over all nodes in SIR's top-k.
    """
    top_sir = sir_ranks[sir_ranks <= k].index
    rrs = [1.0 / float(method_ranks[n]) for n in top_sir if n in method_ranks.index]
    return float(np.mean(rrs)) if rrs else None



def compute_method_fractional_ranks(method_ranks: pd.Series) -> pd.Series:
    """
    Convert integer ranks to fractional ranks (ties get mean of their rank range).
    Used for hexbin correlation plots.
    """
    rank_counts = method_ranks.value_counts().sort_index()
    frac_ranks = {}
    pos = 1
    for rank_val, count in rank_counts.items():
        mean_pos = pos + (count - 1) / 2.0
        for node in method_ranks[method_ranks == rank_val].index:
            frac_ranks[node] = mean_pos
        pos += count
    return pd.Series(frac_ranks)


def compare_method_vs_sir(
    methods_df: pd.DataFrame,
    sir_df: pd.DataFrame,
    method_name: str,
    ranking: List[int],
    k_values: List[int] = [1, 5, 10, 20],
    k_precision_snr: List[int] = [1, 3, 5, 10],
    k_arhr_abs: List[int] = [1, 3, 5, 10],
    k_arhr_pct: List[float] = [0.01, 0.03, 0.05, 0.10],
    verbose: bool = True,
    method_ranks_cache=None,
) -> Dict:
    """
    Compare a single method against SIR ground truth.

    When sir_df contains sir_rank_snr column (added by rank_sir_results_snr),
    also computes: weighted Kendall's tau, MRR, arhr@k, precision_snr@k.

    Args:
        methods_df: Methods results DataFrame
        sir_df: SIR results DataFrame (with sir_rank; optionally sir_rank_snr)
        method_name: Name of method to compare
        ranking: Group ranking
        k_values: K values for legacy Precision@K (vs raw-mean sir_rank)
        k_precision_snr: K values for SNR-based Precision@K
        k_arhr_abs: Absolute k values for arhr@k
        k_arhr_pct: Fractional k values for arhr@k (e.g. 0.01 = top 1%)
        verbose: Print detailed output

    Returns:
        Dict with comparison metrics
    """
    ranking_str = '_'.join(map(str, ranking))

    # Get method ranks (use cache if available, else filter from full df)
    if method_ranks_cache is not None and method_name in method_ranks_cache:
        method_ranks_full = method_ranks_cache[method_name]
    else:
        method_df = methods_df[
            (methods_df['method'] == method_name) &
            (methods_df['ranking'] == ranking_str)
        ]
        method_ranks_full = method_df.set_index('node_id')['rank']

    # Get ranks for nodes evaluated by both method and SIR
    common_nodes = set(method_ranks_full.index) & set(sir_df['node_id'])

    if len(common_nodes) == 0:
        return {
            'method': method_name,
            'num_common_nodes': 0,
            'error': 'No common nodes between method and SIR results'
        }

    sir_ranks = sir_df.set_index('node_id')['sir_rank']

    # Filter to common nodes
    method_ranks = method_ranks_full[method_ranks_full.index.isin(common_nodes)]
    sir_ranks = sir_ranks[sir_ranks.index.isin(common_nodes)]

    # Compute legacy Precision@K
    precision_at_k = {}
    effective_ks = {}
    for k in k_values:
        prec, eff_k = compute_precision_at_k(method_ranks, sir_ranks, k)
        precision_at_k[k] = prec
        effective_ks[k] = eff_k

    # Top-1 nodes (used for reach vector diagnostics and num_method_top1)
    method_top = method_ranks[method_ranks == 1].index.tolist()
    sir_top = sir_ranks[sir_ranks == 1].index.tolist()
    num_method_top1 = len(method_top)

    # === SNR-BASED METRICS ===

    n_common = len(common_nodes)

    # Determine which ground truth column to use for new metrics:
    # - sir_rank_snr if available (standard comparison with SNR variant)
    # - sir_rank otherwise (betas comparison, uses blended means)
    has_snr = 'sir_rank_snr' in sir_df.columns
    sir_snr_idx = sir_df.set_index('node_id')
    if has_snr:
        sir_gt_ranks = sir_snr_idx['sir_rank_snr'][sir_snr_idx.index.isin(common_nodes)]
    else:
        sir_gt_ranks = sir_ranks[sir_ranks.index.isin(common_nodes)]

    method_ranks_common = method_ranks[method_ranks.index.isin(common_nodes)]
    sir_gt_common = sir_gt_ranks

    # Kendall's tau (vs discretized ground truth)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        kendall_corr, kendall_pval = kendalltau(method_ranks_common, sir_gt_common)
    if np.isnan(kendall_corr):
        if verbose:
            print(f"Warning: Constant input for {method_name} (all ranks identical)")
        kendall_corr = 0.0
        kendall_pval = 1.0

    # Weighted Kendall's tau
    wtau, wtau_pval = compute_weighted_kendall(method_ranks_common, sir_gt_common)

    # MRR (ground-truth-centric, SNR-based if available)
    mrr = compute_mrr(method_ranks_common, sir_gt_common)

    # MRR top-spreaders info (SNR-based)
    snr_top = sir_gt_common[sir_gt_common == 1].index.tolist()
    num_snr_top1 = len(snr_top)

    # arhr@k — absolute k values
    arhr_abs = {}
    for k in k_arhr_abs:
        arhr_abs[k] = compute_arhr_at_k(method_ranks_common, sir_gt_common, k)

    # arhr@k — percentage-based k values
    arhr_pct = {}
    for pct in k_arhr_pct:
        k_eff = max(1, int(pct * n_common))
        arhr_pct[pct] = compute_arhr_at_k(method_ranks_common, sir_gt_common, k_eff)

    # Precision@k — percentage-based (SNR ground truth)
    _pct_prec_vals = [0.01, 0.03, 0.05, 0.10]
    _pct_prec_label = {0.01: '1pct', 0.03: '3pct', 0.05: '5pct', 0.10: '10pct'}
    precision_snr_pct = {}
    for pct in _pct_prec_vals:
        k_eff = max(1, int(pct * n_common))
        top_method = method_ranks_common[method_ranks_common <= k_eff].index
        top_sir = sir_gt_common[sir_gt_common <= k_eff].index
        precision_snr_pct[pct] = min(len(set(top_method) & set(top_sir)), k_eff) / k_eff

    # Precision@k (SNR-based, absolute k)
    precision_snr = {}
    effective_k_snr = {}
    for k in k_precision_snr:
        prec, eff_k = compute_precision_at_k(method_ranks_common, sir_gt_common, k)
        precision_snr[k] = prec
        effective_k_snr[k] = eff_k

    # === END SNR-BASED METRICS ===

    # Reach vectors for diagnostic info
    method_top_1_reach_vectors = {}
    if len(method_top) > 0:
        top_node = method_top[0]
        sir_row = sir_df[sir_df['node_id'] == top_node]
        if not sir_row.empty:
            reach_vec = [sir_row[f'group_{g}_mean'].values[0] for g in ranking]
            method_top_1_reach_vectors = {
                'node_id': top_node,
                'reach_vector': reach_vec,
                'reach_total': sir_row['total_mean'].values[0],
            }

    sir_top_1_reach_vectors = {}
    if len(sir_top) > 0:
        top_node = sir_top[0]
        sir_row = sir_df[sir_df['node_id'] == top_node]
        if not sir_row.empty:
            reach_vec = [sir_row[f'group_{g}_mean'].values[0] for g in ranking]
            sir_top_1_reach_vectors = {
                'node_id': top_node,
                'reach_vector': reach_vec,
                'reach_total': sir_row['total_mean'].values[0],
            }

    pct_label = {0.01: '1pct', 0.03: '3pct', 0.05: '5pct', 0.10: '10pct', 0.20: '20pct'}

    result = {
        'method': method_name,
        'ranking': ranking_str,
        'num_common_nodes': n_common,
        'kendall_correlation': kendall_corr,
        'kendall_pvalue': kendall_pval,
        'method_top_1_node': method_top_1_reach_vectors.get('node_id'),
        'method_top_1_reach_vector': str(method_top_1_reach_vectors.get('reach_vector')),
        'method_top_1_reach_total': method_top_1_reach_vectors.get('reach_total'),
        'sir_top_1_node': sir_top_1_reach_vectors.get('node_id'),
        'sir_top_1_reach_vector': str(sir_top_1_reach_vectors.get('reach_vector')),
        'sir_top_1_reach_total': sir_top_1_reach_vectors.get('reach_total'),
        'num_method_top1': num_method_top1,
        **{f'precision_at_{k}': v for k, v in precision_at_k.items()},
        **{f'effective_k_{k}': eff_k for k, eff_k in effective_ks.items()},
        # SNR-based metrics
        'weighted_kendall_tau': wtau,
        'weighted_kendall_pvalue': wtau_pval,
        'mrr': mrr,
        'num_snr_top1': num_snr_top1,
        **{f'arhr_at_{k}': v for k, v in arhr_abs.items()},
        **{f'arhr_at_{pct_label.get(pct, f"{int(pct*100)}pct")}': v for pct, v in arhr_pct.items()},
        **{f'precision_snr_at_{_pct_prec_label[p]}': v for p, v in precision_snr_pct.items()},
        **{f'precision_snr_at_{k}': v for k, v in precision_snr.items()},
        **{f'effective_k_snr_{k}': eff_k for k, eff_k in effective_k_snr.items()},
    }

    return result


def extract_top_k_ground_truth(
    methods_df: pd.DataFrame,
    sir_df: pd.DataFrame,
    ranking: List[int],
    top_k: int = 3,
    method_ranks_cache=None,
) -> List[Dict]:
    """
    For each method, extract its top-k ranked nodes and their ground truth scores.

    sir_df must already have 'sir_rank' computed (standard or betas-blended).

    Args:
        methods_df: Methods results DataFrame (with 'method', 'ranking', 'rank', 'node_id')
        sir_df: SIR results DataFrame (with 'sir_rank', 'node_id', group columns)
        ranking: Group ranking as list of ints
        top_k: Number of top method ranks to include

    Returns:
        List of dicts, one per (method, node) for nodes ranked 1..top_k by the method.
    """
    ranking_str = '_'.join(map(str, ranking))
    sir_lookup = sir_df.set_index('node_id')

    rows = []
    names = list(method_ranks_cache.keys()) if method_ranks_cache is not None else methods_df['method'].unique()
    for method_name in names:
        # Get top_k nodes for this method
        if method_ranks_cache is not None:
            method_ranks_series = method_ranks_cache[method_name]
            top_nodes_iter = method_ranks_series[method_ranks_series <= top_k].sort_values().items()
        else:
            method_df = methods_df[
                (methods_df['method'] == method_name) &
                (methods_df['ranking'] == ranking_str)
            ]
            if method_df.empty:
                continue
            top_df = method_df[method_df['rank'] <= top_k].sort_values('rank')
            top_nodes_iter = ((row['node_id'], int(row['rank'])) for _, row in top_df.iterrows())

        for node_id, method_rank in top_nodes_iter:
            row = {
                'method': method_name,
                'ranking': ranking_str,
                'node_id': node_id,
                'method_rank': int(method_rank),
            }

            if node_id in sir_lookup.index:
                sir_row = sir_lookup.loc[node_id]
                row['ground_truth_rank'] = int(sir_row['sir_rank'])
                # Per-group reach (raw counts)
                for i, gid in enumerate(ranking):
                    col_mean = f'group_{gid}_mean'
                    col_std = f'group_{gid}_std'
                    if col_mean in sir_row.index:
                        row[f'group_{i}_reach_mean'] = sir_row[col_mean]
                    if col_std in sir_row.index:
                        row[f'group_{i}_reach_std'] = sir_row[col_std]
                if 'total_mean' in sir_row.index:
                    row['total_reach_mean'] = sir_row['total_mean']
            else:
                row['ground_truth_rank'] = None

            rows.append(row)

    return rows


def compare_all_methods_betas(
    methods_file: Path,
    sir_file: Path,
    ranking: List[int],
    betas_label: str,
    betas_vector: List[float],
    output_file: Path,
    k_values: List[int] = [1, 5, 10, 20],
    k_precision_snr: List[int] = [1, 3, 5, 10],
    k_arhr_abs: List[int] = [1, 3, 5, 10],
    k_arhr_pct: List[float] = [0.01, 0.03, 0.05, 0.10],
    top_k: int = 3,
    verbose: bool = True,
    methods_df: pd.DataFrame = None,
    sir_df: pd.DataFrame = None,
    method_ranks_cache=None,
    sem_percentile: float = 0.0,
):
    """
    Compare betas-specific methods against blended SIR ground truth.

    Uses blended reach vectors as ground truth. When sem_percentile > 0,
    reach values are discretized before blending.

    Args:
        methods_file: Path to methods results (parquet or CSV)
        sir_file: Path to SIR results (parquet or CSV)
        ranking: Group ranking
        betas_label: Betas config label (e.g., "equal_0.5")
        betas_vector: Betas vector for blending
        output_file: Path to save comparison results
        k_values: K values for legacy Precision@K metrics
        k_precision_snr: K values for Precision@K (vs blended ground truth)
        k_arhr_abs: Absolute k values for arhr@k (vs blended ground truth)
        k_arhr_pct: Fractional k values for arhr@k
        top_k: Save ground truth info for each method's top-k ranked nodes (0 to disable)
        verbose: Print detailed output
        sem_percentile: SEM percentile for discretization before blending (0 = no discretization)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"COMPARING METHODS VS BLENDED SIR (betas_group={betas_label})")
        print(f"{'='*80}")
        print(f"Betas vector: {betas_vector}")

    # Load data
    if methods_df is None:
        if str(methods_file).endswith('.parquet'):
            methods_df = pd.read_parquet(methods_file, engine='pyarrow')
        else:
            methods_df = pd.read_csv(methods_file)

    if sir_df is None:
        if str(sir_file).endswith('.parquet'):
            sir_df = pd.read_parquet(sir_file, engine='pyarrow')
        else:
            sir_df = pd.read_csv(sir_file)

    # Compute blended ground truth (sir_df must be raw, without sir_rank)
    sir_df = rank_sir_results_betas(sir_df, ranking, betas_vector, verbose=verbose, sem_percentile=sem_percentile)

    # Compare ALL methods against the betas-blended ground truth
    all_methods = methods_df['method'].unique()
    selected_methods = list(all_methods)

    if verbose:
        print(f"Comparing {len(selected_methods)} methods against betas_group={betas_label}")

    comparisons = []
    for method_name in selected_methods:
        if verbose:
            print(f"\n  {method_name}:")

        comparison = compare_method_vs_sir(
            methods_df, sir_df, method_name, ranking,
            k_values=k_values,
            k_precision_snr=k_precision_snr,
            k_arhr_abs=k_arhr_abs,
            k_arhr_pct=k_arhr_pct,
            verbose=verbose,
            method_ranks_cache=method_ranks_cache,
        )

        if 'error' in comparison:
            if verbose:
                print(f"Error: {comparison['error']}")
        else:
            if verbose:
                wtau = comparison.get('weighted_kendall_tau')
                mrr_v = comparison.get('mrr')
                print(f"  Weighted Kendall: {wtau:.4f}" if wtau is not None else "  Weighted Kendall: N/A")
                print(f"  MRR:              {mrr_v:.4f}" if mrr_v is not None else "  MRR: N/A")

        comparisons.append(comparison)

    comparison_df = pd.DataFrame(comparisons)
    comparison_df['betas_group'] = betas_label

    # Save results
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if str(output_file).endswith('.parquet'):
            comparison_df.to_parquet(output_file, index=False, engine='pyarrow')
        else:
            comparison_df.to_csv(output_file, index=False)
        if verbose:
            print(f"\n  Saved betas comparison results: {output_file}")

    # Extract top-k ground truth detail
    if top_k > 0 and output_file:
        top_k_rows = extract_top_k_ground_truth(methods_df, sir_df, ranking, top_k,
                                                method_ranks_cache=method_ranks_cache)
        if top_k_rows:
            top_k_df = pd.DataFrame(top_k_rows)
            top_k_df['betas_group'] = betas_label
            top_k_file = Path(str(output_file).replace('_comparison.parquet', '_top_k_detail.parquet')
                              .replace('_comparison.csv', '_top_k_detail.csv'))
            if str(top_k_file).endswith('.parquet'):
                top_k_df.to_parquet(top_k_file, index=False, engine='pyarrow')
            else:
                top_k_df.to_csv(top_k_file, index=False)
            if verbose:
                print(f"Saved top-{top_k} ground truth detail: {top_k_file} ({len(top_k_df)} rows)")

    return comparison_df


def compare_all_methods(
    methods_file: Path,
    sir_file: Path,
    ranking: List[int],
    output_file: Path = None,
    k_values: List[int] = [1, 5, 10, 20],
    k_precision_snr: List[int] = [1, 3, 5, 10],
    k_arhr_abs: List[int] = [1, 3, 5, 10],
    k_arhr_pct: List[float] = [0.01, 0.03, 0.05, 0.10],
    top_k: int = 3,
    tie_stats_dir: Path = None,
    verbose: bool = True,
    verbose_ties: bool = False,
    methods_df: pd.DataFrame = None,
    betas_meta: dict = None,
    method_ranks_cache=None,
    sem_percentile: float = 50.0,
    sir_df_raw: pd.DataFrame = None,
    sir_df_ranked: pd.DataFrame = None,
):
    """
    Compare all methods against SIR ground truth.

    Computes both legacy metrics (vs raw-mean sir_rank) and new SNR-based metrics
    (weighted Kendall, MRR, arhr@k, precision_snr@k).

    Args:
        methods_file: Path to methods results (parquet or CSV)
        sir_file: Path to SIR results (parquet or CSV)
        ranking: Group ranking
        output_file: Path to save comparison results
        k_values: K values for legacy Precision@K
        k_precision_snr: K values for SNR-based Precision@K
        k_arhr_abs: Absolute k values for arhr@k
        k_arhr_pct: Fractional k values for arhr@k
        top_k: Save ground truth info for each method's top-k ranked nodes (0 to disable)
        tie_stats_dir: Directory to save tie statistics CSV (None to skip)
        verbose: Print detailed output
        sir_df_raw: Pre-loaded raw SIR DataFrame (skips file read if provided)
        sir_df_ranked: Pre-ranked SIR DataFrame with raw-mean ranking (skips both
                       file read and raw-mean ranking if provided; sir_df_raw must
                       also be provided for betas reuse)
    """
    if verbose:
        print(f"\n{'='*80}")
        print("COMPARING METHODS VS SIR")
        print(f"{'='*80}")
        print(f"Methods: {methods_file}")
        print(f"SIR: {sir_file}")
        print(f"Ranking: {ranking}")
        print(f"K values: {k_values}")

    # Load data (support both parquet and CSV)
    if verbose:
        print("\nLoading data...")

    if methods_df is None:
        t0 = time.time()
        if str(methods_file).endswith('.parquet'):
            methods_df = pd.read_parquet(methods_file, engine='pyarrow')
        else:
            methods_df = pd.read_csv(methods_file)
        if verbose:
            print(f"  methods loaded: {len(methods_df)} rows  [{time.time()-t0:.1f}s]")

    # Load SIR data — reuse pre-loaded/pre-ranked DataFrames if provided
    if sir_df_ranked is not None:
        sir_df = sir_df_ranked.copy()
        if verbose:
            print(f"  SIR reused (pre-ranked): {len(sir_df)} rows")
    elif sir_df_raw is not None:
        if verbose:
            print(f"  SIR reused (pre-loaded): {len(sir_df_raw)} rows")
        t0 = time.time()
        sir_df = rank_sir_results(sir_df_raw, ranking, verbose=verbose)
        if verbose:
            print(f"  SIR ranked (raw mean):  [{time.time()-t0:.1f}s]")
    else:
        t0 = time.time()
        if str(sir_file).endswith('.parquet'):
            sir_df_raw = pd.read_parquet(sir_file, engine='pyarrow')
        else:
            sir_df_raw = pd.read_csv(sir_file)
        if verbose:
            print(f"  SIR loaded:    {len(sir_df_raw)} rows  [{time.time()-t0:.1f}s]")
        t0 = time.time()
        sir_df = rank_sir_results(sir_df_raw, ranking, verbose=verbose)
        if verbose:
            print(f"  SIR ranked (raw mean):  [{time.time()-t0:.1f}s]")

    # Extract infection probability from SIR filename for tie stats
    _prob_match = re.search(r'_p([0-9.]+)\.(parquet|csv)$', str(sir_file))
    infection_prob = float(_prob_match.group(1)) if _prob_match else None

    # Rank SIR results (SNR variant) — adds sir_rank_snr column
    t0 = time.time()
    sir_df, snr_U = rank_sir_results_snr(sir_df, ranking, verbose=verbose, sem_percentile=sem_percentile)
    if verbose:
        print(f"  SIR ranked (SNR):       [{time.time()-t0:.1f}s]")

    # Compute and save tie statistics (once per graph+probability+ranking)
    tie_stats = []
    if tie_stats_dir is not None or verbose:
        tie_stats = compute_tie_statistics(sir_df, ranking, infection_prob=infection_prob)
        if verbose:
            print_tie_statistics(tie_stats, ranking, infection_prob, verbose=verbose_ties)
        if tie_stats_dir is not None:
            ranking_str_ts = '_'.join(map(str, ranking))
            prob_str_ts = f"p{infection_prob:.4f}" if infection_prob is not None else "pNA"
            ts_file = Path(tie_stats_dir) / f"ties_r{ranking_str_ts}_{prob_str_ts}.csv"
            save_tie_statistics(tie_stats, ts_file, ranking)

    # Get unique methods, excluding betas methods from standard comparison
    all_methods = methods_df['method'].unique()
    methods = [m for m in all_methods if 'betas_' not in m]
    if verbose:
        excluded = len(all_methods) - len(methods)
        if excluded > 0:
            print(f"\n  Excluding {excluded} betas methods from standard comparison")
        print(f"\n  Methods to compare: {list(methods)}")
        print("\nComparing methods...")

    comparisons = []

    for method_name in methods:
        if verbose:
            print(f"\n  {method_name}:")

        comparison = compare_method_vs_sir(
            methods_df, sir_df, method_name, ranking,
            k_values=k_values,
            k_precision_snr=k_precision_snr,
            k_arhr_abs=k_arhr_abs,
            k_arhr_pct=k_arhr_pct,
            verbose=verbose,
            method_ranks_cache=method_ranks_cache,
        )

        if 'error' in comparison:
            if verbose:
                print(f"Error: {comparison['error']}")
        else:
            if verbose:
                print(f"Kendall tau: {comparison['kendall_correlation']:.4f}")
                if comparison.get('method_top_1_node'):
                    print(f"Method's top-1 node: {comparison['method_top_1_node']}")
                    print(f"-> Reach vector: {comparison['method_top_1_reach_vector']}")
                    print(f"-> Total reach: {comparison['method_top_1_reach_total']:.2f}")
                for k in k_values:
                    eff_k = comparison[f'effective_k_{k}']
                    prec = comparison[f'precision_at_{k}']
                    k_note = f"(evaluated on {eff_k} nodes)" if eff_k < k else ""
                    print(f"Precision@{k}: {prec:.4f}{k_note}")

        comparisons.append(comparison)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparisons)
    comparison_df['betas_group'] = 'none'

    # Save results
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if str(output_file).endswith('.parquet'):
            comparison_df.to_parquet(output_file, index=False, engine='pyarrow')
        else:
            comparison_df.to_csv(output_file, index=False)
        if verbose:
            print(f"\n  Saved comparison results: {output_file}")

    # Extract top-k ground truth detail
    if top_k > 0 and output_file:
        top_k_rows = extract_top_k_ground_truth(methods_df, sir_df, ranking, top_k,
                                                method_ranks_cache=method_ranks_cache)
        if top_k_rows:
            top_k_df = pd.DataFrame(top_k_rows)
            top_k_df['betas_group'] = 'none'
            top_k_file = Path(str(output_file).replace('_comparison.parquet', '_top_k_detail.parquet')
                              .replace('_comparison.csv', '_top_k_detail.csv'))
            if str(top_k_file).endswith('.parquet'):
                top_k_df.to_parquet(top_k_file, index=False, engine='pyarrow')
            else:
                top_k_df.to_csv(top_k_file, index=False)
            if verbose:
                print(f"Saved top-{top_k} ground truth detail: {top_k_file} ({len(top_k_df)} rows)")

    # Print summary
    if verbose:
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")

        comparison_df_sorted = comparison_df.sort_values('weighted_kendall_tau', ascending=False)
        print("\nMethod Performance (sorted by weighted Kendall's tau):")
        for _, row in comparison_df_sorted.iterrows():
            if 'error' not in row or pd.isna(row.get('error')):
                wtau = row.get('weighted_kendall_tau')
                mrr_val = row.get('mrr')
                p10 = row.get('precision_snr_at_10', row.get('precision_at_10'))
                print(f"\n  {row['method']}:")
                if wtau is not None:
                    print(f"  Weighted Kendall: {wtau:.4f}")
                if mrr_val is not None:
                    print(f"  MRR:              {mrr_val:.4f}")
                if p10 is not None:
                    print(f"  Precision@10:     {p10:.4f}")

    # Auto-detect betas metadata and run betas comparisons
    if output_file:
        methods_file = Path(methods_file)
        _betas_meta = betas_meta  # use pre-loaded if available
        if _betas_meta is None:
            betas_meta_path = methods_file.parent / methods_file.name.replace(
                '_methods.parquet', '_betas_metadata.json'
            ).replace('_methods.csv', '_betas_metadata.json')

            if betas_meta_path.exists():
                if verbose:
                    print(f"\n  Found betas metadata: {betas_meta_path}")
                with open(betas_meta_path, 'r') as f:
                    _betas_meta = json.load(f)

        if _betas_meta is not None:
            ranking_str = '_'.join(map(str, ranking))
            ranking_configs = _betas_meta.get('rankings', {}).get(ranking_str, {})

            if ranking_configs:
                if verbose:
                    print(f"Found {len(ranking_configs)} betas configs for ranking {ranking_str}")

                for label, meta in ranking_configs.items():
                    betas_vector = meta['betas']
                    output_stem = str(output_file).replace('_comparison.parquet', '').replace('_comparison.csv', '')
                    ext = output_file.suffix
                    betas_output = Path(f"{output_stem}_betas_{label}_comparison{ext}")

                    compare_all_methods_betas(
                        methods_file=methods_file,
                        sir_file=sir_file,
                        ranking=ranking,
                        betas_label=label,
                        betas_vector=betas_vector,
                        output_file=betas_output,
                        k_values=k_values,
                        k_precision_snr=k_precision_snr,
                        k_arhr_abs=k_arhr_abs,
                        k_arhr_pct=k_arhr_pct,
                        top_k=top_k,
                        verbose=verbose,
                        methods_df=methods_df,
                        sir_df=sir_df_raw,  # reuse already-loaded raw SIR df
                        method_ranks_cache=method_ranks_cache,
                        sem_percentile=sem_percentile,
                    )
            else:
                if verbose:
                    print(f"No betas configs for ranking {ranking_str} in metadata")

    return comparison_df, tie_stats


def main():
    parser = argparse.ArgumentParser(
        description="Compare method results vs SIR ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script performs the comparison workflow:
  1. Loads method results
  2. Loads SIR results
  3. Ranks SIR results using lexicographic SNR comparison
  4. Compares each method against SIR ground truth
  5. Computes metrics: rank correlation, Precision@K

SNR discretization:
  --sem-percentile controls how SIR means are discretized into integer bins.
  Higher values → more ties (coarser bins), lower → fewer ties.
  Use 0 for no discretization (raw mean ordering).

Example:
  python compare_results.py \\
    --methods results/methods/ER_2groups_rep0_methods.parquet \\
    --sir results/sir/ER_2groups_rep0_sir_p0.0500.parquet \\
    --ranking 0 1 \\
    --sem-percentile 25 \\
    --output results/comparison/ER_2groups_rep0_comparison.parquet
        """
    )
    parser.add_argument(
        "--methods",
        type=Path,
        required=True,
        help="Path to methods results file (parquet or CSV)"
    )
    parser.add_argument(
        "--sir",
        nargs='+',
        type=Path,
        required=True,
        help="Path(s) to SIR results file(s) (parquet or CSV); accepts multiple files"
    )
    parser.add_argument(
        "--ranking",
        nargs='+',
        type=int,
        default=[0, 1],
        help="Group ranking (default: 0 1)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (parquet or CSV, optional); for single --sir use"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; filenames are derived from SIR filenames (use with multiple --sir)"
    )
    parser.add_argument(
        "--k-values",
        nargs='+',
        type=int,
        default=[1, 5, 10, 20],
        help="K values for legacy Precision@K (default: 1 5 10 20)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Save ground truth detail for each method's top-k ranked nodes (default: 3, 0 to disable)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output (only show final results)"
    )
    parser.add_argument(
        "--sem-percentile",
        type=float,
        default=50.0,
        help="Percentile of SEM distribution for SNR discretization "
             "(default: 50). 0 = no discretization (raw means)."
    )

    args = parser.parse_args()

    if len(args.sir) > 1 and not args.output_dir:
        parser.error("--output-dir is required when passing multiple --sir files")

    # Load methods_df once (avoid reloading for each SIR file)
    methods_file = Path(args.methods)
    if str(methods_file).endswith('.parquet'):
        methods_df = pd.read_parquet(methods_file, engine='pyarrow')
    else:
        methods_df = pd.read_csv(methods_file)

    # Load betas metadata once
    betas_meta_path = methods_file.parent / methods_file.name.replace(
        '_methods.parquet', '_betas_metadata.json'
    ).replace('_methods.csv', '_betas_metadata.json')
    betas_meta = None
    if betas_meta_path.exists():
        if not args.quiet:
            print(f"Found betas metadata: {betas_meta_path}")
        with open(betas_meta_path, 'r') as f:
            betas_meta = json.load(f)

    graph_name = methods_file.stem.replace('_methods', '')
    ranking_str = '_'.join(map(str, args.ranking))
    prob_pattern = re.compile(r'_p([0-9.]+)\.(parquet|csv)$')

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for sir_file in args.sir:
        sir_file = Path(sir_file)
        t0 = time.time()

        # Determine output file path
        if args.output_dir:
            match = prob_pattern.search(sir_file.name)
            if not match:
                print(f"WARNING: Could not extract infection_prob from {sir_file.name}, skipping")
                continue
            infection_prob = float(match.group(1))
            output_file = args.output_dir / f"{graph_name}_r{ranking_str}_p{infection_prob:.4f}_comparison.parquet"
        else:
            output_file = args.output  # may be None (single-file / no-save mode)

        compare_all_methods(
            methods_file=args.methods,
            sir_file=sir_file,
            ranking=args.ranking,
            output_file=output_file,
            k_values=args.k_values,
            top_k=args.top_k,
            verbose=not args.quiet,
            methods_df=methods_df,
            betas_meta=betas_meta,
            sem_percentile=args.sem_percentile,
        )

        if not args.quiet:
            elapsed = time.time() - t0
            m = prob_pattern.search(sir_file.name)
            prob_str = f"p={float(m.group(1)):.4f}" if m else sir_file.name
            print(f"  {prob_str} — ranked + compared in {elapsed:.1f}s")

    if not args.quiet:
        print(f"\n{'='*80}")
        print("COMPARISON COMPLETE")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
