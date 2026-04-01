#!/usr/bin/env python3
"""Generate cross-dataset strategy comparison tables for the appendix.

Usage: python3 src/generate_strategy_tables.py [--output-dir tables/cross_dataset/appendix]
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────

DATASETS = {
    'blogs_gcc': 'sem_10',
    'books': 'sem_10',
    'email-eu-core_gcc': 'sem_10',
    'mind': 'sem_10',
    'brexit': None,
    'iphone_samsung': None,
    'lastfm_asia': None,
}
DS_ORDER = ['blogs_gcc', 'books', 'email-eu-core_gcc', 'mind',
            'brexit', 'iphone_samsung', 'lastfm_asia']
DISPLAY_NAMES = {
    'blogs_gcc': 'PolBlogs (LCC)', 'books': 'PolBooks',
    'email-eu-core_gcc': 'Email-EU-Core (LCC)', 'mind': 'MIND',
    'brexit': 'Brexit', 'iphone_samsung': 'iPhone-Samsung',
    'lastfm_asia': 'LastFM-Asia',
}

AGG = Path('results/aggregated')

# Display names for methods in LaTeX output
METHOD_DISPLAY = {
    'lexipeeling':                      'lexipeeling',
    'lexipeeling_extended':             'lexipeeling-ext.',
    'coreness':                         'coreness',
    'degree':                           'degree',
    'degree_lexicographic':             'lexicographic-degree',
    'coreness_lexipeeling':             'coreness-lexipeeling',
    'lexipeeling_coreness':             'lexipeeling-coreness',
    'degree_lexipeeling':               'degree-lexipeeling',
    'lexipeeling_degree':               'lexipeeling-degree',
    'degree_lexicographic_lexipeeling': 'lexicographic-degree-lexipeeling',
    'lexipeeling_degree_lexicographic': 'lexipeeling-lexicographic-degree',
    'extended_gravity_centrality':      'ext. gravity',
}


def _method_tex(name: str) -> str:
    """Return LaTeX-safe display name for a method."""
    if name in METHOD_DISPLAY:
        display = METHOD_DISPLAY[name]
    else:
        display = name.replace('degree_lexicographic', 'lexicographic_degree')
        display = display.replace('restricted_first_', 'top-restricted-')
        display = display.replace('_', '-')
    return display.replace('%', r'\%')


def _strategy_display(s: str) -> str:
    """Convert a strategy name to display form (no LaTeX escaping needed after this)."""
    s = s.replace('restricted_first_', 'top-restricted-')
    s = s.replace('restricted_first', 'top-restricted')
    s = s.replace('_', '-')
    return s.replace('%', r'\%')


# (file_stem, metric_col, method_a, method_b, display_metric, caption_extra)
TABLE_SPECS = [
    ('strat_mrr_lexi_vs_core',
     'mrr_mean', 'lexipeeling', 'coreness',
     'MRR', 'lexipeeling vs.\\ coreness'),
    ('strat_wkendall_lexi_vs_core',
     'weighted_kendall_tau_mean', 'lexipeeling', 'coreness',
     r'Wtd.\ Kendall $\tau$', 'lexipeeling vs.\\ coreness'),
    ('strat_kendall_lexi_vs_core',
     'kendall_correlation_mean', 'lexipeeling', 'coreness',
     r'Kendall $\tau$', 'lexipeeling vs.\\ coreness'),
    ('strat_wkendall_lexi_vs_extgrav',
     'weighted_kendall_tau_mean', 'lexipeeling', 'extended_gravity_centrality',
     r'Wtd.\ Kendall $\tau$', 'lexipeeling vs.\\ ext. gravity'),
    ('strat_kendall_lexi_vs_extgrav',
     'kendall_correlation_mean', 'lexipeeling', 'extended_gravity_centrality',
     r'Kendall $\tau$', 'lexipeeling vs.\\ ext. gravity'),
    ('strat_wkendall_lexiext_vs_extgrav',
     'weighted_kendall_tau_mean', 'lexipeeling_extended', 'extended_gravity_centrality',
     r'Wtd.\ Kendall $\tau$', 'lexipeeling-ext. vs.\\ ext. gravity'),
    ('strat_kendall_lexiext_vs_extgrav',
     'kendall_correlation_mean', 'lexipeeling_extended', 'extended_gravity_centrality',
     r'Kendall $\tau$', 'lexipeeling-ext. vs.\\ ext. gravity'),
]


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_strategy_data() -> Dict[str, pd.DataFrame]:
    """Load by_ranking_strategy.csv for each dataset."""
    data = {}
    for ds in DS_ORDER:
        sem = DATASETS[ds]
        if sem:
            p = AGG / ds / sem / 'betas_none' / 'by_ranking_strategy.csv'
        else:
            p = AGG / ds / 'betas_none' / 'by_ranking_strategy.csv'
        if not p.exists():
            print(f'  Warning: {p} not found, skipping {ds}')
            continue
        data[ds] = pd.read_csv(p)
    return data


def get_common_strategies(data: Dict[str, pd.DataFrame], min_datasets: int = 5) -> List[str]:
    """Return strategies present in >= min_datasets, sorted by frequency then name."""
    counts: Dict[str, int] = {}
    for ds, df in data.items():
        for s in df['ranking_strategy'].unique():
            counts[s] = counts.get(s, 0) + 1
    common = [(s, n) for s, n in counts.items() if n >= min_datasets]
    common.sort(key=lambda x: (-x[1], x[0]))
    return [s for s, _ in common]


# ─── Table Generation ─────────────────────────────────────────────────────────

def _escape_latex(s: str) -> str:
    return s.replace('_', r'\_').replace('%', r'\%')


def generate_table(
    data: Dict[str, pd.DataFrame],
    strategies: List[str],
    metric_col: str,
    method_a: str,
    method_b: str,
    file_stem: str,
    display_metric: str,
    caption_extra: str,
    output_dir: Path,
) -> None:
    """Generate one strategy comparison table (CSV + LaTeX)."""
    print(f'  {file_stem}...')

    # Build data: cells[strategy][ds] = (val_a, val_b)
    cells: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for strat in strategies:
        cells[strat] = {}
        for ds in DS_ORDER:
            if ds not in data:
                continue
            df = data[ds]
            row_a = df[(df['ranking_strategy'] == strat) & (df['method'] == method_a)]
            row_b = df[(df['ranking_strategy'] == strat) & (df['method'] == method_b)]
            if row_a.empty or metric_col not in df.columns:
                continue
            va = float(row_a[metric_col].iloc[0]) if not row_a.empty else np.nan
            vb = float(row_b[metric_col].iloc[0]) if not row_b.empty else np.nan
            if not np.isnan(va):
                cells[strat][ds] = (va, vb if not np.isnan(vb) else np.nan)

    # Summary row: wins/losses/ties per strategy
    summary: Dict[str, Tuple[int, int, int]] = {}
    for strat in strategies:
        a_wins = b_wins = ties = 0
        for ds, (va, vb) in cells.get(strat, {}).items():
            if np.isnan(vb):
                continue
            if va > vb + 1e-6:
                a_wins += 1
            elif vb > va + 1e-6:
                b_wins += 1
            else:
                ties += 1
        summary[strat] = (a_wins, b_wins, ties)

    # CSV output
    csv_rows = []
    for strat in strategies:
        row = {'strategy': strat}
        for ds in DS_ORDER:
            if ds in cells.get(strat, {}):
                va, vb = cells[strat][ds]
                if np.isnan(vb):
                    row[DISPLAY_NAMES[ds]] = f'{va:.3f}'
                else:
                    row[DISPLAY_NAMES[ds]] = f'{va:.3f} ({vb:.3f})'
            else:
                row[DISPLAY_NAMES[ds]] = '---'
        aw, bw, t = summary[strat]
        row['A_wins'] = aw
        row['B_wins'] = bw
        row['ties'] = t
        csv_rows.append(row)
    csv_df = pd.DataFrame(csv_rows)
    csv_path = output_dir / f'{file_stem}.csv'
    csv_df.to_csv(csv_path, index=False)

    # LaTeX output
    n_ds = len(DS_ORDER)
    ds_headers = [_escape_latex(DISPLAY_NAMES[ds]) for ds in DS_ORDER]
    col_spec = 'l' + 'r' * n_ds + 'r'  # strategy + datasets + summary

    method_a_short = _method_tex(method_a)
    method_b_short = _method_tex(method_b)

    lines = [
        r'\begin{longtable}{' + col_spec + '}',
        r'\caption{' + display_metric + ' by ranking strategy: ' + caption_extra +
        r'. Each cell shows ' + method_a_short + r'(' + method_b_short +
        r'), \\textbf{bold} = higher. Last column: A wins--B wins--ties.}',
        r'\label{tab:' + file_stem + '}' + r' \\',
        r'\toprule',
        'Strategy & ' + ' & '.join(ds_headers) + r' & W--L--T \\',
        r'\midrule',
        r'\endfirsthead',
        r'\toprule',
        'Strategy & ' + ' & '.join(ds_headers) + r' & W--L--T \\',
        r'\midrule',
        r'\endhead',
    ]

    for strat in strategies:
        parts = [_strategy_display(strat)]
        for ds in DS_ORDER:
            if ds in cells.get(strat, {}):
                va, vb = cells[strat][ds]
                if np.isnan(vb):
                    parts.append(f'{va:.2f}')
                else:
                    va_str = f'{va:.2f}'
                    vb_str = f'{vb:.2f}'
                    if va > vb + 1e-6:
                        parts.append(r'\textbf{' + va_str + '}(' + vb_str + ')')
                    elif vb > va + 1e-6:
                        parts.append(va_str + r'(\textbf{' + vb_str + '})')
                    else:
                        parts.append(va_str + '(' + vb_str + ')')
            else:
                parts.append('---')
        aw, bw, t = summary[strat]
        parts.append(f'{aw}--{bw}--{t}')
        lines.append(' & '.join(parts) + r' \\')

    # Total summary row
    total_aw = sum(s[0] for s in summary.values())
    total_bw = sum(s[1] for s in summary.values())
    total_t = sum(s[2] for s in summary.values())
    lines.append(r'\midrule')
    total_parts = [r'\textit{Total}'] + [''] * n_ds + [f'{total_aw}--{total_bw}--{total_t}']
    lines.append(' & '.join(total_parts) + r' \\')

    lines += [r'\bottomrule', r'\end{longtable}']

    tex_path = output_dir / f'{file_stem}.tex'
    tex_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'    -> {csv_path.name}, {tex_path.name}')


# ─── Cumulative@1% per strategy ────────────────────────────────────────────────

def _get_ranking_strategy_map(ds: str) -> Dict[str, str]:
    """Extract ranking → strategy mapping from detailed_all_comparisons.csv (memory-efficient)."""
    sem = DATASETS[ds]
    if sem:
        p = AGG / ds / sem / 'detailed_all_comparisons.csv'
    else:
        p = AGG / ds / 'detailed_all_comparisons.csv'
    if not p.exists():
        return {}
    # Read in chunks to save memory — only need unique pairs
    mapping = {}
    for chunk in pd.read_csv(p, usecols=['ranking', 'ranking_strategy'], chunksize=100000):
        for _, row in chunk.drop_duplicates().iterrows():
            mapping[row['ranking']] = row['ranking_strategy']
    return mapping


def _compute_cumul_per_strategy(ds: str, methods: List[str], threshold_pct: float = 1.0
                                 ) -> Dict[str, Dict[str, float]]:
    """Compute Cumulative@threshold_pct% per strategy per method from node_ranks."""
    sem = DATASETS[ds]
    if sem:
        nr_path = AGG / ds / sem / 'betas_none' / 'node_ranks.parquet'
    else:
        nr_path = AGG / ds / 'betas_none' / 'node_ranks.parquet'

    if not nr_path.exists():
        nr_path = nr_path.with_suffix('.csv')
    if not nr_path.exists():
        return {}

    ranking_map = _get_ranking_strategy_map(ds)
    if not ranking_map:
        return {}

    # Read node_ranks — filter to needed methods and rank-1 only
    cols = ['method', 'method_rank', 'sir_rank_snr', 'ranking', 'n_nodes', 'infection_prob']
    if nr_path.suffix == '.parquet':
        import pyarrow.parquet as pq
        df = pd.read_parquet(nr_path, columns=cols,
                             filters=[('method', 'in', methods),
                                      ('method_rank', '==', 1)])
    else:
        chunks = []
        for chunk in pd.read_csv(nr_path, usecols=cols, chunksize=500000):
            mask = chunk['method'].isin(methods) & (chunk['method_rank'] == 1)
            chunks.append(chunk[mask])
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    if df.empty:
        return {}

    df['ranking_strategy'] = df['ranking'].map(ranking_map)
    df = df.dropna(subset=['ranking_strategy'])

    results: Dict[str, Dict[str, float]] = {}
    for (method, strat), strat_group in df.groupby(['method', 'ranking_strategy']):
        n_nodes = strat_group['n_nodes'].iloc[0]
        top_threshold = max(1, int(np.ceil(n_nodes * threshold_pct / 100)))
        # Per (ranking, infection_prob) normalization: each instance contributes equally
        per_instance_scores = []
        for _, rgroup in strat_group.groupby(['ranking', 'infection_prob']):
            in_top = (rgroup['sir_rank_snr'] <= top_threshold).sum()
            total = len(rgroup)
            per_instance_scores.append(in_top / total if total > 0 else 0.0)
        pct = float(np.mean(per_instance_scores) * 100)
        if method not in results:
            results[method] = {}
        results[method][strat] = pct

    return results


def generate_cumul_strategy_table(strategies: List[str], output_dir: Path,
                                   method_b: str = 'coreness',
                                   file_stem: str = 'strat_cumul1_lexi_vs_core'):
    """Generate Cumulative@1% strategy table: lexipeeling vs method_b."""
    method_a = 'lexipeeling'

    cells: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for strat in strategies:
        cells[strat] = {}

    for ds in DS_ORDER:
        print(f'    Computing cumulative for {ds}...')
        cumul = _compute_cumul_per_strategy(ds, [method_a, method_b])
        if not cumul:
            print(f'      Skipped (no data)')
            continue
        for strat in strategies:
            va = cumul.get(method_a, {}).get(strat, np.nan)
            vb = cumul.get(method_b, {}).get(strat, np.nan)
            if not np.isnan(va):
                cells[strat][ds] = (va, vb if not np.isnan(vb) else np.nan)

    # Summary
    summary: Dict[str, Tuple[int, int, int]] = {}
    for strat in strategies:
        aw = bw = t = 0
        for ds, (va, vb) in cells.get(strat, {}).items():
            if np.isnan(vb):
                continue
            if va > vb + 0.5:
                aw += 1
            elif vb > va + 0.5:
                bw += 1
            else:
                t += 1
        summary[strat] = (aw, bw, t)

    # CSV
    csv_rows = []
    for strat in strategies:
        row = {'strategy': strat}
        for ds in DS_ORDER:
            if ds in cells.get(strat, {}):
                va, vb = cells[strat][ds]
                if np.isnan(vb):
                    row[DISPLAY_NAMES[ds]] = f'{va:.1f}'
                else:
                    row[DISPLAY_NAMES[ds]] = f'{va:.1f} ({vb:.1f})'
            else:
                row[DISPLAY_NAMES[ds]] = '---'
        aw, bw, t2 = summary[strat]
        row['A_wins'] = aw
        row['B_wins'] = bw
        row['ties'] = t2
        csv_rows.append(row)
    csv_path = output_dir / f'{file_stem}.csv'
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    # LaTeX
    n_ds = len(DS_ORDER)
    ds_headers = [_escape_latex(DISPLAY_NAMES[ds]) for ds in DS_ORDER]
    col_spec = 'l' + 'r' * n_ds + 'r'
    ma_short = _method_tex(method_a)
    mb_short = _method_tex(method_b)

    lines = [
        r'\begin{longtable}{' + col_spec + '}',
        r'\caption{Cumulative@1\% by ranking strategy: ' + ma_short +
        r' vs.\ ' + mb_short +
        r'. Each cell shows ' + ma_short + r'(' + mb_short +
        r'), \textbf{bold} = higher.}',
        r'\label{tab:' + file_stem + r'} \\',
        r'\toprule',
        'Strategy & ' + ' & '.join(ds_headers) + r' & W--L--T \\',
        r'\midrule',
        r'\endfirsthead',
        r'\toprule',
        'Strategy & ' + ' & '.join(ds_headers) + r' & W--L--T \\',
        r'\midrule',
        r'\endhead',
    ]

    for strat in strategies:
        parts = [_strategy_display(strat)]
        for ds in DS_ORDER:
            if ds in cells.get(strat, {}):
                va, vb = cells[strat][ds]
                if np.isnan(vb):
                    parts.append(f'{va:.1f}')
                else:
                    va_s = f'{va:.1f}'
                    vb_s = f'{vb:.1f}'
                    if va > vb + 0.5:
                        parts.append(r'\textbf{' + va_s + '}(' + vb_s + ')')
                    elif vb > va + 0.5:
                        parts.append(va_s + r'(\textbf{' + vb_s + '})')
                    else:
                        parts.append(va_s + '(' + vb_s + ')')
            else:
                parts.append('---')
        aw, bw, t2 = summary[strat]
        parts.append(f'{aw}--{bw}--{t2}')
        lines.append(' & '.join(parts) + r' \\')

    total_aw = sum(s[0] for s in summary.values())
    total_bw = sum(s[1] for s in summary.values())
    total_t = sum(s[2] for s in summary.values())
    lines.append(r'\midrule')
    total_parts = [r'\textit{Total}'] + [''] * n_ds + [f'{total_aw}--{total_bw}--{total_t}']
    lines.append(' & '.join(total_parts) + r' \\')
    lines += [r'\bottomrule', r'\end{longtable}']

    tex_path = output_dir / f'{file_stem}.tex'
    tex_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'    -> {csv_path.name}, {tex_path.name}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate strategy comparison tables')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('tables/cross_dataset/appendix'),
                        help='Output directory for tables')
    parser.add_argument('--min-datasets', type=int, default=5,
                        help='Minimum number of datasets a strategy must appear in')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading strategy data...')
    data = load_strategy_data()
    print(f'  Loaded {len(data)} datasets')

    strategies = get_common_strategies(data, min_datasets=args.min_datasets)
    print(f'  {len(strategies)} strategies with >= {args.min_datasets} datasets: {strategies}')

    print('\nGenerating tables...')
    for spec in TABLE_SPECS:
        stem, metric_col, method_a, method_b, display_metric, caption_extra = spec
        generate_table(data, strategies, metric_col, method_a, method_b,
                       stem, display_metric, caption_extra, args.output_dir)

    # Tables 8-9: Cumulative@1% from node_ranks
    for method_b, stem in [('coreness', 'strat_cumul1_lexi_vs_core'),
                            ('degree', 'strat_cumul1_lexi_vs_degree')]:
        print(f'\n  {stem}...')
        generate_cumul_strategy_table(strategies, args.output_dir,
                                      method_b=method_b, file_stem=stem)

    print(f'\nDone. {len(TABLE_SPECS)} tables written to {args.output_dir}')


if __name__ == '__main__':
    main()
    # Force clean exit to avoid crash during cleanup of large DataFrames
    import gc
    gc.collect()
    import os
    os._exit(0)
