"""
Paper-quality plots for the LexiCoreness paper.

Usage:
    python3 src/paper_plots.py \
        --aggregated-dir results/aggregated/books/ \
        --output-dir plots/books/paper/ \
        --sir-dir results/sir/books/ \
        --graph-name books \
        --method-groups-yaml methods-files/to_plot/methods_groups.yaml \
        [--graph-file data/graphs/books/books.txt]

Plots generated (one PNG per file unless noted as facet):
  P1   : SIR reach violins (one violin per infection probability)
  P2_pct : ARHR@k% facet vs infection probability  [4-panel facet]
  P2_abs : ARHR@k  facet vs infection probability  [4-panel facet]
  P4_pct_by_{dim}  : MRR/ARHR@k%  4-panel facet by ranking strategy
  P4_abs_by_{dim}  : MRR/ARHR@k   4-panel facet by ranking strategy
  P6_pct_by_prob   : MRR/ARHR@k%  4-panel line facet by infection prob
  P6_abs_by_prob   : MRR/ARHR@k   4-panel line facet by infection prob
  P7_pct_by_degeneracy : ARHR@k%  4-panel barplot by lex degeneracy
  P7_abs_by_degeneracy : ARHR@k   4-panel barplot by lex degeneracy
  P8_pct_by_length : ARHR@k%  4-panel barplot by ranking length
  P8_abs_by_length : ARHR@k   4-panel barplot by ranking length
  P9_prec_pct_by_{dim} : Precision@k% 4-panel facet × 4 dims  (4 files)
  P9_prec_abs_by_{dim} : Precision@k  4-panel facet × 4 dims  (4 files)
  P11  : Weighted Kendall boxplot by ranking strategy
  P12  : Standard Kendall boxplot by ranking strategy
  P13_wkendall : Weighted Kendall boxplot by lex degeneracy
  P13_kendall  : Standard Kendall boxplot by lex degeneracy
  P14  : Weighted Kendall vs infection probability (line plot)
  P15  : Standard Kendall vs infection probability (line plot)
  P16  : Cumulative distribution of rank-1 node GT quality
  P17  : Weighted Kendall heatmap by ranking strategy
  P18  : Standard Kendall heatmap by ranking strategy
  P19_pareto_prec_{k}_vs_mrr : Pareto scatter Precision@k vs MRR  (8 files)
  P20  : Pareto scatter Weighted Kendall vs MRR
  P21  : Pareto scatter Standard Kendall vs MRR
  P22_beta_{b}_length_{L}: Beta group reach facet (per beta × ranking length)
  P23  : MRR vs beta line plot
  P24_pareto_cumul_k{k}_t{t}_vs_mrr : Pareto scatter Cumulative@(k,t) vs MRR  (2 files)
  P25  : Worst-case cumulative score drop bar chart (cross-dataset robustness)
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau
try:
    from scipy.stats import weightedtau as _scipy_weightedtau
except ImportError:
    _scipy_weightedtau = None
import yaml

# ─── Constants ────────────────────────────────────────────────────────────────

_MARKERS = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', '<', '>']

_EXTRA_PALETTE = [
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#E69F00',  # orange
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7',  # reddish purple
    '#F0E442',  # yellow
    '#000000',  # black
]

_ARHR_PCT_COLS   = ['arhr_at_1pct',  'arhr_at_3pct',  'arhr_at_5pct',  'arhr_at_10pct']
_ARHR_PCT_LABELS = ['ARHR@1%', 'ARHR@3%', 'ARHR@5%', 'ARHR@10%']
_ARHR_PCT_SLUGS  = ['mrr', '3pct', '5pct', '10pct']

_ARHR_ABS_COLS   = ['arhr_at_1',  'arhr_at_3',  'arhr_at_5',  'arhr_at_10']
_ARHR_ABS_LABELS = ['MRR', 'ARHR@3', 'ARHR@5', 'ARHR@10']
_ARHR_ABS_SLUGS  = ['mrr', '3', '5', '10']

_PRECISION_PCT_COLS   = ['precision_snr_at_1pct',  'precision_snr_at_3pct',
                          'precision_snr_at_5pct',  'precision_snr_at_10pct']
_PRECISION_PCT_LABELS = ['P@1%', 'P@3%', 'P@5%', 'P@10%']
_PRECISION_PCT_SLUGS  = ['1pct', '3pct', '5pct', '10pct']

_PRECISION_ABS_COLS   = ['precision_snr_at_1',  'precision_snr_at_3',
                          'precision_snr_at_5',  'precision_snr_at_10']
_PRECISION_ABS_LABELS = ['P@1', 'P@3', 'P@5', 'P@10']
_PRECISION_ABS_SLUGS  = ['1', '3', '5', '10']

_PARETO_PRECISION_KEYS = [
    ('precision_snr_at_1',    'P@1'),
    ('precision_snr_at_3',    'P@3'),
    ('precision_snr_at_5',    'P@5'),
    ('precision_snr_at_10',   'P@10'),
    ('precision_snr_at_1pct', 'P@1%'),
    ('precision_snr_at_3pct', 'P@3%'),
    ('precision_snr_at_5pct', 'P@5%'),
    ('precision_snr_at_10pct','P@10%'),
]

_BETAS_SORT_KEY = lambda label: -1.0 if label == 'none' else float(label)

# Dimension descriptors: (dim_col, dim_label, data_key_in_load_betas_subdir)
_DIMENSIONS = [
    ('ranking_strategy',         'Ranking Strategy',      'by_strategy'),
    ('infection_prob_rounded',   'Infection Probability', 'by_prob'),
    ('lexicographic_degeneracy', 'Lex. Degeneracy',       'by_degeneracy'),
    ('ranking_length',           'Ranking Length',        'by_length'),
]
_DIM_SLUGS = {
    'ranking_strategy':         'strategy',
    'infection_prob_rounded':   'prob',
    'lexicographic_degeneracy': 'degeneracy',
    'ranking_length':           'length',
}


# ─── Color / style utilities ──────────────────────────────────────────────────

# Fixed preferred colors (Okabe-Ito + distinct accents). A method NOT in this
# dict gets a color assigned sequentially at plot time by assign_method_style.
_METHOD_COLORS: Dict[str, str] = {
    'lexipeeling':                         '#D55E00',  # vermillion
    'degree_lexicographic':                '#56B4E9',  # sky blue
    'degree':                              '#0072B2',  # blue
    'coreness':                            '#009E73',  # bluish green
    'betweenness':                         '#E69F00',  # orange
    'pagerank':                            '#CC79A7',  # reddish purple
    # H-index — teal/purple family (distinct from each other)
    'h1_index':                            '#44AA99',
    'h2_index':                            '#882255',
    'h3_index':                            '#332288',
    'h4_index':                            '#117733',
    'h5_index':                            '#AA4499',
    # Restricted h-index
    'restricted_first_h3_index':           '#6699CC',
    'restricted_first_h4_index':           '#CC6677',
    # Composite lexi methods
    'lexipeeling_extended':                '#2CA02C',  # green
    'lexipeeling_coreness':                '#A0522D',
    'lexipeeling_degree':                  '#C97A50',
    'lexipeeling_degree_lexicographic':    '#8B3A0F',
    'degree_lexicographic_lexipeeling':    '#7B9DA8',
    # Restricted baselines
    'restricted_degree_lexicographic':     '#4477AA',
    'restricted_first_coreness':           '#228833',
    'restricted_coreness':                 '#66CCEE',
    'restricted_first_degree':             '#BBCC33',
    # PageRank variants
    'pagerank_uniform_jump':               '#EE6677',
    'pagerank_exponential_jump':           '#AA3377',
    # Other
    'fairgd':  '#DDCC77',  # muted yellow (Paul Tol "Muted" palette)
    'random':  '#BBBBBB',
}

# Sequential color pool used when a method has no fixed color; also used by
# assign_method_style to avoid repeating colors across methods in the same plot.
_COLOR_POOL = [
    '#D55E00', '#56B4E9', '#009E73', '#E69F00', '#0072B2', '#CC79A7',
    '#44AA99', '#882255', '#332288', '#117733', '#AA4499', '#6699CC',
    '#CC6677', '#F0A040', '#4477AA', '#228833', '#66CCEE', '#BBCC33',
    '#EE6677', '#AA3377', '#DDCC77', '#999933',
]


def _get_method_color(method: str) -> str:
    """Return a fixed color for known methods, else a deterministic hash-based color."""
    if method in _METHOD_COLORS:
        return _METHOD_COLORS[method]
    return _COLOR_POOL[hash(method) % len(_COLOR_POOL)]


def assign_method_style(methods: List[str]) -> Dict[str, dict]:
    """
    Assign color + marker to each method ensuring NO TWO methods share the same color.

    Priority: use fixed color from _METHOD_COLORS if not already taken by another
    method in this call; otherwise fall back to the next unused color in _COLOR_POOL.
    """
    styles: Dict[str, dict] = {}
    used_colors: set = set()
    pool_idx = 0

    for idx, m in enumerate(methods):
        preferred = _METHOD_COLORS.get(m)
        if preferred and preferred not in used_colors:
            color = preferred
        else:
            # Walk the pool until we find an unused color
            while _COLOR_POOL[pool_idx % len(_COLOR_POOL)] in used_colors:
                pool_idx += 1
            color = _COLOR_POOL[pool_idx % len(_COLOR_POOL)]
            pool_idx += 1
        used_colors.add(color)
        styles[m] = {'color': color, 'marker': _MARKERS[idx % len(_MARKERS)]}

    return styles


def _short(method: str, max_len: int = 22) -> str:
    """Shorten a method name for display in plots.

    Always applies naming conventions:
      - degree_lexicographic → lexicographic_degree (swap order)
      - restricted_first → top-restricted
      - _ → -
    Then abbreviates only if the result still exceeds max_len.
    """
    m = method
    # Swap order: degree_lexicographic → lexicographic_degree
    m = m.replace('degree_lexicographic', 'lexicographic_degree')
    # Rename: restricted_first → top_restricted
    m = m.replace('restricted_first_', 'top_restricted_')
    # Remove common suffixes (before underscore→hyphen)
    m = m.replace('_centrality', '')
    m = m.replace('_correlation', '')
    # Replace underscores with hyphens
    m = m.replace('_', '-')
    if len(m) <= max_len:
        return m
    # Abbreviate for tight spaces
    m = m.replace('top-restricted-', 't.r.-')
    m = m.replace('restricted-', 'r.-')
    m = m.replace('lexipeeling', 'lexi')
    m = m.replace('lexicographic-degree', 'lex.-degree')
    m = m.replace('lexicographic', 'lex.')
    m = m.replace('gravity', 'grav')
    if len(m) > max_len:
        m = m[:max_len - 2] + '..'
    return m


# ─── Data loading helpers ─────────────────────────────────────────────────────

def _find_file(base: Path, stem: str) -> Optional[Path]:
    for ext in ['.parquet', '.csv']:
        p = base / f'{stem}{ext}'
        if p.exists():
            return p
    return None


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _read_node_ranks(path: Path, max_method_rank: int = 10) -> pd.DataFrame:
    """Read node_ranks file, keeping only method_rank <= max_method_rank rows (avoids OOM)."""
    if path.suffix == '.parquet':
        return pd.read_parquet(path, filters=[('method_rank', '<=', max_method_rank)])
    chunks = []
    for chunk in pd.read_csv(path, chunksize=500_000):
        chunks.append(chunk[chunk['method_rank'] <= max_method_rank])
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def load_betas_subdir(subdir: Path, load_node_ranks: bool = True) -> Dict[str, pd.DataFrame]:
    """Load key DataFrames from one betas_* subdir."""
    data = {}
    for stem, key in [
        ('level1_overall',            'overall'),
        ('by_ranking_strategy',       'by_strategy'),
        ('by_ranking_length',         'by_length'),
        ('by_infection_prob',         'by_prob'),
        ('by_lexicographic_degeneracy', 'by_degeneracy'),
        ('node_ranks',                'node_ranks'),
        ('best_spreader_group_reach', 'reach'),
    ]:
        if key == 'node_ranks' and not load_node_ranks:
            continue
        f = _find_file(subdir, stem)
        if f is not None:
            if key == 'node_ranks':
                data[key] = _read_node_ranks(f)
            else:
                data[key] = _read(f)
    return data


def load_all_betas(aggregated_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load data for all betas_* subdirs. Returns {label: {key: df}}."""
    result = {}
    for subdir in sorted(aggregated_dir.glob('betas_*/'),
                         key=lambda d: _BETAS_SORT_KEY(d.name.replace('betas_', ''))):
        label = subdir.name.replace('betas_', '')
        result[label] = load_betas_subdir(subdir, load_node_ranks=(label == 'none'))
    return result


def load_sir_data(sir_dir: Path, graph_name: str) -> Dict[float, pd.DataFrame]:
    """Load SIR parquet files. Returns {infection_prob: df}."""
    result = {}
    for f in sorted(sir_dir.glob(f'{graph_name}_sir_p*.parquet')):
        m = re.search(r'_sir_p([\d.]+)\.parquet$', f.name)
        if m:
            prob = float(m.group(1))
            result[prob] = pd.read_parquet(f)
    return result


def compute_avg_degree(graph_file: Path) -> Optional[float]:
    """Compute average degree from an edge list file (one edge per line: u v)."""
    try:
        degrees: Dict[int, int] = {}
        with open(graph_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    degrees[u] = degrees.get(u, 0) + 1
                    degrees[v] = degrees.get(v, 0) + 1
        if degrees:
            return sum(degrees.values()) / len(degrees)
    except Exception:
        pass
    return None


# ─── Method selection ─────────────────────────────────────────────────────────

def _is_baseline(method: str) -> bool:
    """Return True if method is a non-lexipeeling baseline (eligible for selected_methods)."""
    return 'lexipeeling' not in method


def select_top3_for_metric(df_overall: pd.DataFrame, metric_mean_col: str) -> List[str]:
    """
    Return ['lexipeeling'] + top-3 baselines ranked by metric_mean_col.

    Baselines = methods with 'lexipeeling' not in name.
    """
    df = df_overall.reset_index() if df_overall.index.name == 'method' else df_overall.copy()
    if 'method' not in df.columns:
        return []

    available = set(df['method'].tolist())
    result: List[str] = []
    if 'lexipeeling' in available:
        result.append('lexipeeling')

    if metric_mean_col not in df.columns:
        return result

    baselines = [m for m in available if _is_baseline(m)]
    top3 = (df[df['method'].isin(baselines)]
            .sort_values(metric_mean_col, ascending=False)['method']
            .head(3).tolist())
    for m in top3:
        if m not in result:
            result.append(m)
    return result


def select_union_for_metrics(df_overall: pd.DataFrame, metric_cols: List[str]) -> List[str]:
    """
    Return the union of select_top3_for_metric across multiple metric base columns.

    metric_cols: list of base column names (without _mean suffix), e.g.
                 ['arhr_at_1pct', 'arhr_at_3pct', 'arhr_at_5pct', 'arhr_at_10pct']
    """
    df = df_overall.reset_index() if df_overall.index.name == 'method' else df_overall.copy()
    available = set(df['method'].tolist())
    result: List[str] = []
    if 'lexipeeling' in available:
        result.append('lexipeeling')
    for col in metric_cols:
        mean_col = f'{col}_mean'
        for m in select_top3_for_metric(df, mean_col):
            if m not in result:
                result.append(m)
    return result


def _is_composite(method: str) -> bool:
    """Return True if the method is a composite lexipeeling variant (not the base)."""
    return 'lexipeeling' in method and method != 'lexipeeling'


def _find_best_non_lexi(df_overall: pd.DataFrame, metric: str = 'mrr_mean') -> Optional[str]:
    df = df_overall.reset_index() if df_overall.index.name == 'method' else df_overall.copy()
    if metric not in df.columns:
        return None
    candidates = df[~df['method'].str.contains('lexipeeling', na=False)]
    if candidates.empty:
        return None
    return candidates.sort_values(metric, ascending=False)['method'].iloc[0]


# ─── Pareto front ─────────────────────────────────────────────────────────────

def compute_pareto_front(df: pd.DataFrame, x_col: str, y_col: str,
                         methods: List[str]) -> List[str]:
    """Maximization Pareto front among given methods. Returns list sorted by x descending."""
    sub = df[df['method'].isin(methods)][[x_col, y_col, 'method']].dropna()
    if sub.empty:
        return []
    pts = list(sub.itertuples(index=False))
    pareto = []
    for row in pts:
        dominated = False
        for other in pts:
            if other.method == row.method:
                continue
            ox, oy = getattr(other, x_col), getattr(other, y_col)
            rx, ry = getattr(row, x_col), getattr(row, y_col)
            if ox >= rx and oy >= ry and (ox > rx or oy > ry):
                dominated = True
                break
        if not dominated:
            pareto.append(row.method)
    pareto.sort(key=lambda m: sub.loc[sub['method'] == m, x_col].values[0], reverse=True)
    return pareto


# ─── Generic helpers ──────────────────────────────────────────────────────────

def _plot_metric_line(df: pd.DataFrame, mean_col: str, x_col: str, x_label: str,
                      y_label: str, methods: List[str], styles: Dict[str, dict],
                      output_path: Path, title: str):
    """Line plot: X=x_col, Y=mean_col, one line per method. Uses aggregated data."""
    if mean_col not in df.columns:
        print(f'  Skip {output_path.name} — {mean_col} not in data')
        return
    if x_col not in df.columns:
        print(f'  Skip {output_path.name} — {x_col} not in data')
        return
    df_m = df[df['method'].isin(methods)].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        sub = df_m[df_m['method'] == method].sort_values(x_col)
        if sub.empty:
            continue
        st = styles.get(method, {'color': '#333333', 'marker': 'o'})
        ax.plot(sub[x_col], sub[mean_col], color=st['color'], marker=st['marker'],
                linewidth=2, markersize=7, label=_short(method))
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {output_path.name}')


def _plot_metric_barplot(df: pd.DataFrame, mean_col: str, dim_col: str, dim_label: str,
                          methods: List[str], styles: Dict[str, dict],
                          output_path: Path, title: str):
    """Grouped barplot: X=dim_col values, grouped bars per method, Y=mean_col."""
    if mean_col not in df.columns:
        print(f'  Skip {output_path.name} — {mean_col} not in data')
        return
    if dim_col not in df.columns:
        print(f'  Skip {output_path.name} — {dim_col} not in data')
        return
    df_m = df[df['method'].isin(methods)].copy()
    dim_vals = sorted(df_m[dim_col].dropna().unique())
    present = [m for m in methods if m in df_m['method'].unique()]
    n_methods = len(present)
    n_dims = len(dim_vals)
    if n_dims == 0 or n_methods == 0:
        return
    width = 0.8 / max(n_methods, 1)
    x = np.arange(n_dims)
    fig, ax = plt.subplots(figsize=(max(6, n_dims * n_methods * 0.5 + 2), 5))
    for i, method in enumerate(present):
        st = styles.get(method, {'color': '#333333', 'marker': 'o'})
        vals = []
        for dv in dim_vals:
            v = df_m[(df_m['method'] == method) & (df_m[dim_col] == dv)][mean_col]
            vals.append(float(v.mean()) if not v.empty else np.nan)
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.85,
               color=st['color'], alpha=0.75, label=_short(method))
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in dim_vals], fontsize=9, rotation=30, ha='right')
    ax.set_xlabel(dim_label, fontsize=11)
    ax.set_ylabel(title.split('—')[0].strip(), fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    for gi in range(n_dims - 1):
        ax.axvline((x[gi] + x[gi + 1]) / 2.0, color='gray', alpha=0.35,
                    linewidth=1.0, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {output_path.name}')


def _plot_metric_facet_barplot(df: pd.DataFrame,
                                metric_cols: List[str], metric_labels: List[str],
                                dim_col: str, dim_label: str,
                                methods: List[str], styles: Dict[str, dict],
                                output_path: Path, suptitle: str,
                                group_spacing: float = 1.4):
    """4-panel facet barplot: one panel per k value. Same method set across all panels.
    Draws a soft dashed vertical separator between each pair of adjacent dim values."""
    df_m = df[df['method'].isin(methods)].copy()
    dim_vals = sorted(df_m[dim_col].dropna().unique())
    n_dims = len(dim_vals)
    n_methods = len(methods)
    if n_dims == 0 or n_methods == 0:
        print(f'  Skip {output_path.name} — no data')
        return
    width = 0.6 / max(n_methods, 1)
    x = np.arange(n_dims) * group_spacing

    n_panels = len(metric_cols)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, col, label in zip(axes, metric_cols, metric_labels):
        mean_col = f'{col}_mean'
        if mean_col not in df.columns:
            ax.set_visible(False)
            continue
        for i, method in enumerate(methods):
            st = styles.get(method, {'color': '#333333'})
            vals = []
            for dv in dim_vals:
                v = df_m[(df_m['method'] == method) & (df_m[dim_col] == dv)][mean_col]
                vals.append(float(v.mean()) if not v.empty else np.nan)
            offset = (i - (n_methods - 1) / 2) * width
            ax.bar(x + offset, vals, width=width * 0.85,
                   color=st['color'], alpha=0.8, label=_short(method))
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in dim_vals], fontsize=8, rotation=30, ha='right')
        ax.set_xlabel(dim_label, fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        # Soft dashed separator between dim-value groups
        for gi in range(n_dims - 1):
            sep = (x[gi] + x[gi + 1]) / 2.0
            ax.axvline(sep, color='gray', alpha=0.35, linewidth=1.0, linestyle='--')

    handles = [mpatches.Patch(color=styles.get(m, {'color': '#333333'})['color'],
                               label=_short(m)) for m in methods]
    fig.legend(handles=handles, bbox_to_anchor=(1.01, 0.5), loc='center left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {output_path.name}')


def _plot_metric_facet_heatmap(df: pd.DataFrame,
                                 metric_cols: List[str], metric_labels: List[str],
                                 dim_col: str, dim_label: str,
                                 methods: List[str],
                                 output_path: Path, suptitle: str):
    """4-panel facet heatmap: one panel per k value. Rows=methods, cols=dim values."""
    df_m = df[df['method'].isin(methods)].copy()
    dim_vals = sorted(df_m[dim_col].dropna().unique())
    if len(dim_vals) == 0 or len(methods) == 0:
        print(f'  Skip {output_path.name} — no data')
        return

    n_panels = len(metric_cols)
    fig, axes = plt.subplots(1, n_panels,
                              figsize=(max(4, len(dim_vals) * 1.5) * n_panels,
                                       max(3, len(methods) * 0.65 + 1.5)))
    if n_panels == 1:
        axes = [axes]

    for ax, col, label in zip(axes, metric_cols, metric_labels):
        mean_col = f'{col}_mean'
        if mean_col not in df.columns:
            ax.set_visible(False)
            continue
        matrix = pd.DataFrame(index=methods,
                               columns=[str(v) for v in dim_vals],
                               dtype=float)
        for m in methods:
            for dv in dim_vals:
                v = df_m[(df_m['method'] == m) & (df_m[dim_col] == dv)][mean_col]
                matrix.loc[m, str(dv)] = float(v.mean()) if not v.empty else np.nan
        sns.heatmap(matrix.astype(float), ax=ax, annot=True, fmt='.3f',
                    cmap='RdYlGn', cbar=True, linewidths=0.5,
                    yticklabels=[_short(m) for m in methods])
        ax.set_xlabel(dim_label, fontsize=9)
        ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {output_path.name}')


def _plot_metric_facet_line(df: pd.DataFrame,
                              metric_cols: List[str], metric_labels: List[str],
                              x_col: str, x_label: str,
                              methods: List[str], styles: Dict[str, dict],
                              output_path: Path, suptitle: str):
    """4-panel facet line plot: one panel per k value. X=x_col, lines=methods."""
    df_m = df[df['method'].isin(methods)].copy()

    n_panels = len(metric_cols)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, col, label in zip(axes, metric_cols, metric_labels):
        mean_col = f'{col}_mean'
        if mean_col not in df.columns:
            ax.set_visible(False)
            continue
        for method in methods:
            sub = df_m[df_m['method'] == method].sort_values(x_col)
            if sub.empty or mean_col not in sub.columns:
                continue
            st = styles.get(method, {'color': '#333333', 'marker': 'o'})
            ax.plot(sub[x_col], sub[mean_col], color=st['color'],
                    marker=st['marker'], linewidth=2, markersize=7)
        ax.set_xlabel(x_label, fontsize=9)
        ax.grid(True, alpha=0.3)

    handles = [Line2D([0], [0], color=styles.get(m, {'color': '#333333'})['color'],
                       marker=styles.get(m, {'marker': 'o'})['marker'],
                       linewidth=2, markersize=7, label=_short(m))
               for m in methods if m in df_m['method'].values]
    fig.legend(handles=handles, bbox_to_anchor=(1.01, 0.5), loc='center left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {output_path.name}')


def _plot_metric_facet_vs_prob(df_by_prob: pd.DataFrame,
                                metric_cols: List[str], metric_labels: List[str],
                                methods: List[str], styles: Dict[str, dict],
                                output_path: Path, y_label: str = 'Score'):
    """
    4-panel facet (one panel per metric), X=infection_prob, Y=metric, lines=methods.
    All panels show the same method set.
    """
    x_col = 'infection_prob_rounded'
    if x_col not in df_by_prob.columns:
        for alt in ['infection_prob', 'prob']:
            if alt in df_by_prob.columns:
                x_col = alt
                break

    cols_present = [(c, l) for c, l in zip(metric_cols, metric_labels)
                    if f'{c}_mean' in df_by_prob.columns]
    if not cols_present:
        print(f'  Skip {output_path.name} — no metric columns found')
        return

    n = len(cols_present)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    df_m = df_by_prob[df_by_prob['method'].isin(methods)].copy()

    for ax, (col, label) in zip(axes, cols_present):
        mean_col = f'{col}_mean'
        for method in methods:
            sub = df_m[df_m['method'] == method].sort_values(x_col)
            if sub.empty or mean_col not in sub.columns:
                continue
            st = styles.get(method, {'color': '#333333', 'marker': 'o'})
            ax.plot(sub[x_col], sub[mean_col], color=st['color'],
                    marker=st['marker'], linewidth=2, markersize=7, label=method)
        ax.set_xlabel('Infection probability', fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

    handles = [mpatches.Patch(color=styles.get(m, {'color': '#333'})['color'], label=m)
               for m in methods if m in df_m['method'].values]
    fig.legend(handles=handles, bbox_to_anchor=(1.01, 0.5), loc='center left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {output_path.name}')


# ─── P1: SIR violin ───────────────────────────────────────────────────────────

def plot_sir_violins(sir_data: Dict[float, pd.DataFrame], output_dir: Path,
                    graph_name: str, avg_degree: Optional[float] = None):
    probs = sorted(sir_data.keys())
    if not probs:
        print('  P1: no SIR data found, skipping')
        return

    fig, ax = plt.subplots(figsize=(max(8, len(probs) * 1.2), 5))

    data_list = [sir_data[p]['total_mean'].dropna().values for p in probs]
    positions = list(range(len(probs)))

    parts = ax.violinplot(data_list, positions=positions, showmedians=True, showextrema=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#4c72b0')
        pc.set_alpha(0.7)

    if avg_degree is not None:
        ax.axhline(avg_degree, color='red', linestyle=':', linewidth=2,
                   label=f'Avg degree = {avg_degree:.2f}')
        ax.legend(fontsize=9)

    ax.set_xticks(positions)
    ax.set_xticklabels([f'{p:.4f}' for p in probs], rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Infection probability', fontsize=11)
    ax.set_ylabel('Mean infected nodes (total)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    out = output_dir / 'P1_sir_violins.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ P1: {out.name}')


# ─── P2: ARHR facet vs infection probability ──────────────────────────────────

def plot_arhr_pct_facet_vs_prob(df_by_prob: pd.DataFrame, methods: List[str],
                                 styles: Dict[str, dict], output_dir: Path):
    """Plot 2: ARHR@k% facet vs infection probability."""
    _plot_metric_facet_vs_prob(df_by_prob, _ARHR_PCT_COLS, _ARHR_PCT_LABELS,
                                methods, styles,
                                output_dir / 'P2_arhr_pct_vs_prob.png',
                                y_label='ARHR')


def plot_arhr_abs_facet_vs_prob(df_by_prob: pd.DataFrame, methods: List[str],
                                 styles: Dict[str, dict], output_dir: Path):
    """Plot 3: ARHR@k (absolute) facet vs infection probability."""
    _plot_metric_facet_vs_prob(df_by_prob, _ARHR_ABS_COLS, _ARHR_ABS_LABELS,
                                methods, styles,
                                output_dir / 'P2_arhr_abs_vs_prob.png',
                                y_label='ARHR')


# ─── P4–P8: ARHR heatmaps by dimension ───────────────────────────────────────

def plot_arhr_heatmaps(df_overall: pd.DataFrame,
                       data_by_dim: Dict[str, pd.DataFrame],
                       output_dir: Path):
    """
    Plots 4–8: ARHR by dimension. One facet file per (metric_type × dimension).
    - P4 (by_strategy): 4-panel heatmap facet
    - P6 (by_prob):     4-panel line facet
    - P7 (by_degeneracy): 4-panel barplot facet (only 2 unique values)
    - P8 (by_length):   4-panel barplot facet
    """
    dim_info = [
        ('by_strategy',   'ranking_strategy',         'strategy',   'P4',  'barplot'),
        ('by_prob',       'infection_prob_rounded',   'prob',       'P6',  'line'),
        ('by_degeneracy', 'lexicographic_degeneracy', 'degeneracy', 'P7',  'barplot'),
        ('by_length',     'ranking_length',           'length',     'P8',  'barplot'),
    ]

    for metric_cols, metric_labels, ktype in [
        (_ARHR_PCT_COLS, _ARHR_PCT_LABELS, 'pct'),
        (_ARHR_ABS_COLS, _ARHR_ABS_LABELS, 'abs'),
    ]:
        for data_key, dim_col, dim_slug, plot_prefix, plot_type in dim_info:
            df_dim = data_by_dim.get(data_key)
            if df_dim is None:
                continue
            methods = select_union_for_metrics(df_overall, metric_cols)
            if not methods:
                continue
            styles = assign_method_style(methods)
            fname = f'{plot_prefix}_arhr_{ktype}_by_{dim_slug}.png'
            suptitle = f'ARHR — by {dim_slug}'
            path = output_dir / fname
            dim_label = dim_col.replace('_', ' ').title()
            if plot_type == 'heatmap':
                _plot_metric_facet_heatmap(df_dim, metric_cols, metric_labels,
                                            dim_col, dim_label, methods, path, suptitle)
            elif plot_type == 'line':
                _plot_metric_facet_line(df_dim, metric_cols, metric_labels,
                                         dim_col, dim_label, methods, styles, path, suptitle)
            elif plot_type == 'barplot':
                spacing = 1.4
                _plot_metric_facet_barplot(df_dim, metric_cols, metric_labels,
                                            dim_col, dim_label, methods, styles,
                                            path, suptitle, group_spacing=spacing)


# ─── P9: Precision heatmaps by dimension ─────────────────────────────────────

def plot_precision_heatmaps(df_overall: pd.DataFrame,
                             data_by_dim: Dict[str, pd.DataFrame],
                             output_dir: Path):
    """Plot 9: Precision@k — one facet file per (metric_type × dimension)."""
    dim_info = [
        ('by_strategy',   'ranking_strategy',         'strategy',   'barplot'),
        ('by_prob',       'infection_prob_rounded',   'prob',       'line'),
        ('by_degeneracy', 'lexicographic_degeneracy', 'degeneracy', 'barplot'),
        ('by_length',     'ranking_length',           'length',     'barplot'),
    ]

    for metric_cols, metric_labels, ktype in [
        (_PRECISION_PCT_COLS, _PRECISION_PCT_LABELS, 'pct'),
        (_PRECISION_ABS_COLS, _PRECISION_ABS_LABELS, 'abs'),
    ]:
        for data_key, dim_col, dim_slug, plot_type in dim_info:
            df_dim = data_by_dim.get(data_key)
            if df_dim is None:
                continue
            methods = select_union_for_metrics(df_overall, metric_cols)
            if not methods:
                continue
            styles = assign_method_style(methods)
            fname = f'P9_prec_{ktype}_by_{dim_slug}.png'
            suptitle = f'Precision — by {dim_slug}'
            path = output_dir / fname
            dim_label = dim_col.replace('_', ' ').title()
            if plot_type == 'heatmap':
                _plot_metric_facet_heatmap(df_dim, metric_cols, metric_labels,
                                            dim_col, dim_label, methods, path, suptitle)
            elif plot_type == 'line':
                _plot_metric_facet_line(df_dim, metric_cols, metric_labels,
                                         dim_col, dim_label, methods, styles, path, suptitle)
            elif plot_type == 'barplot':
                spacing = 1.4
                _plot_metric_facet_barplot(df_dim, metric_cols, metric_labels,
                                            dim_col, dim_label, methods, styles,
                                            path, suptitle, group_spacing=spacing)


# ─── P11/P12/P13: Kendall barplots ───────────────────────────────────────────

def _plot_kendall_barplot(df_agg: pd.DataFrame, metric_mean_col: str, metric_label: str,
                           dim_col: str, dim_label: str,
                           methods: List[str], styles: Dict[str, dict],
                           output_path: Path):
    """Barplot: X=dimension values, grouped bars per method, Y=mean metric."""
    if metric_mean_col not in df_agg.columns:
        print(f'  Skip {output_path.name} — {metric_mean_col} not in data')
        return
    if dim_col not in df_agg.columns:
        print(f'  Skip {output_path.name} — {dim_col} not in data')
        return
    df_m = df_agg[df_agg['method'].isin(methods)].copy()
    if df_m.empty:
        print(f'  Skip {output_path.name} — no data')
        return
    dim_vals = sorted(df_m[dim_col].dropna().unique())
    present = [m for m in methods if m in df_m['method'].unique()]
    n_methods, n_dims = len(present), len(dim_vals)
    if n_methods == 0 or n_dims == 0:
        return
    width = 0.8 / max(n_methods, 1)
    x = np.arange(n_dims)
    fig, ax = plt.subplots(figsize=(max(6, n_dims * n_methods * 0.5 + 2), 5))
    for i, method in enumerate(present):
        st = styles.get(method, {'color': '#333333', 'marker': 'o'})
        vals = []
        for dv in dim_vals:
            v = df_m[(df_m['method'] == method) & (df_m[dim_col] == dv)][metric_mean_col]
            vals.append(float(v.mean()) if not v.empty else np.nan)
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.85,
               color=st['color'], alpha=0.75, label=_short(method))
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in dim_vals], rotation=30, ha='right', fontsize=9)
    ax.set_xlabel(dim_label, fontsize=10)
    ax.set_ylabel(metric_label, fontsize=10)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    for gi in range(n_dims - 1):
        ax.axvline((x[gi] + x[gi + 1]) / 2.0, color='gray', alpha=0.35,
                    linewidth=1.0, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {output_path.name}')


def plot_kendall_barplots_by_strategy(df_by_strategy: pd.DataFrame,
                                       df_overall: pd.DataFrame,
                                       output_dir: Path):
    """Plots 11 and 12: weighted/standard kendall barplot by ranking strategy."""
    for metric_col, metric_label, slug, pnum in [
        ('weighted_kendall_tau',  'Weighted Kendall τ', 'wkendall', 'P11'),
        ('kendall_correlation',   'Kendall τ',          'kendall',  'P12'),
    ]:
        mean_col = f'{metric_col}_mean'
        avail_col = mean_col if mean_col in df_overall.columns else 'weighted_kendall_tau_mean'
        methods = select_top3_for_metric(df_overall, avail_col)
        styles = assign_method_style(methods)
        _plot_kendall_barplot(
            df_by_strategy, mean_col, metric_label,
            'ranking_strategy', 'Ranking Strategy',
            methods, styles,
            output_dir / f'{pnum}_{slug}_barplot_by_strategy.png',
        )


def plot_kendall_barplots_by_degeneracy(df_by_degeneracy: pd.DataFrame,
                                         df_overall: pd.DataFrame,
                                         output_dir: Path):
    """Plot 13: weighted/standard kendall barplot by lexicographic degeneracy."""
    for metric_col, metric_label, slug, pnum in [
        ('weighted_kendall_tau', 'Weighted Kendall τ', 'wkendall', 'P13a'),
        ('kendall_correlation',  'Kendall τ',          'kendall',  'P13b'),
    ]:
        mean_col = f'{metric_col}_mean'
        avail_col = mean_col if mean_col in df_overall.columns else 'weighted_kendall_tau_mean'
        methods = select_top3_for_metric(df_overall, avail_col)
        styles = assign_method_style(methods)
        _plot_kendall_barplot(
            df_by_degeneracy, mean_col, metric_label,
            'lexicographic_degeneracy', 'Lex. Degeneracy',
            methods, styles,
            output_dir / f'{pnum}_{slug}_barplot_by_degeneracy.png',
        )


# ─── P14/P15: Kendall vs infection probability ────────────────────────────────

def _plot_metric_vs_prob(df_by_prob: pd.DataFrame, metric_mean_col: str,
                          metric_label: str, methods: List[str],
                          styles: Dict[str, dict], output_path: Path):
    """Line plot: X=infection_prob, Y=metric, one line per method."""
    x_col = 'infection_prob_rounded'
    if x_col not in df_by_prob.columns:
        for alt in ['infection_prob', 'prob']:
            if alt in df_by_prob.columns:
                x_col = alt
                break

    if metric_mean_col not in df_by_prob.columns:
        print(f'  Skip {output_path.name} — {metric_mean_col} not found')
        return

    df_m = df_by_prob[df_by_prob['method'].isin(methods)].copy()
    fig, ax = plt.subplots(figsize=(8, 5))

    for method in methods:
        sub = df_m[df_m['method'] == method].sort_values(x_col)
        if sub.empty:
            continue
        st = styles.get(method, {'color': '#333333', 'marker': 'o'})
        ax.plot(sub[x_col], sub[metric_mean_col],
                color=st['color'], marker=st['marker'],
                linewidth=2, markersize=7, label=_short(method))

    ax.set_xlabel('Infection probability', fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {output_path.name}')


def plot_kendall_vs_prob(df_by_prob: pd.DataFrame, df_overall: pd.DataFrame,
                          output_dir: Path):
    """Plots 14 and 15: wkendall/kendall vs infection probability."""
    for metric_mean_col, metric_label, pnum in [
        ('weighted_kendall_tau_mean', 'Weighted Kendall τ', 'P14'),
        ('kendall_correlation_mean',  'Kendall τ',          'P15'),
    ]:
        methods = select_top3_for_metric(df_overall, metric_mean_col)
        styles = assign_method_style(methods)
        metric_raw = metric_mean_col.replace('_mean', '')
        _plot_metric_vs_prob(df_by_prob, metric_mean_col, metric_label,
                              methods, styles,
                              output_dir / f'{pnum}_{metric_raw}_vs_prob.png')


# ─── P16: Cumulative distribution ─────────────────────────────────────────────

def plot_cumulative_dist(df_node_ranks: pd.DataFrame, df_overall: pd.DataFrame,
                          output_dir: Path, k_values: List[int] = None):
    """Plot 16: cumulative distribution of rank-k node GT quality.

    Generates one plot per k value, using absolute GT rank thresholds on the x-axis.
    """
    if k_values is None:
        k_values = [1, 5]
    required = ['method', 'method_rank', 'sir_rank_snr', 'n_nodes']
    if not all(c in df_node_ranks.columns for c in required):
        print('  P16: missing columns in node_ranks, skipping')
        return

    methods = select_top3_for_metric(df_overall, 'mrr_mean')
    styles = assign_method_style(methods)
    gb_cols = ['ranking', 'infection_prob'] if 'infection_prob' in df_node_ranks.columns else ['ranking']
    t_values = np.array([1, 2, 3, 5, 10, 20, 50])

    for k in k_values:
        fig, ax = plt.subplots(figsize=(10, 7))

        for method in methods:
            sub = df_node_ranks[(df_node_ranks['method'] == method) &
                                (df_node_ranks['method_rank'] <= k)]
            if sub.empty:
                continue
            per_instance = []
            for _, rgroup in sub.groupby(gb_cols):
                gt_vals = rgroup['sir_rank_snr'].values
                per_instance.append([(gt_vals <= t).mean() for t in t_values])
            cumulative = (np.mean(per_instance, axis=0) * 100).tolist()
            st = styles.get(method, {'color': '#333', 'marker': 'o'})
            ax.plot(t_values, cumulative, color=st['color'], marker=st['marker'],
                    linewidth=2, markersize=8, label=method)

        ax.set_xlabel('Ground Truth Rank Threshold (t)', fontsize=11)
        ax.set_ylabel(f'% of method-rank-≤{k} nodes with GT rank ≤ t', fontsize=11)
        ax.set_xlim([0, t_values[-1] + 2])
        ax.set_ylim([0, 105])
        ax.set_xticks(t_values)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f'P16_cumulative_dist_k{k}.png'
        plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ {fname}')


def plot_cumulative_dist_with_lexi(df_node_ranks: pd.DataFrame, df_overall: pd.DataFrame,
                                    output_dir: Path, k_values: List[int] = None):
    """Plot 16b: cumulative distribution including all lexipeeling composites.

    Generates one plot per k value, using absolute GT rank thresholds.
    """
    if k_values is None:
        k_values = [1, 5]
    required = ['method', 'method_rank', 'sir_rank_snr', 'n_nodes']
    if not all(c in df_node_ranks.columns for c in required):
        print('  P16b: missing columns in node_ranks, skipping')
        return

    # Top-3 baselines + lexipeeling + all lexi composites present in node_ranks
    base_methods = select_top3_for_metric(df_overall, 'mrr_mean')
    all_present = df_node_ranks['method'].unique().tolist()
    lexi_composites = [m for m in all_present if _is_composite(m)]
    methods_16b = list(dict.fromkeys(base_methods + sorted(lexi_composites)))
    styles = assign_method_style(methods_16b)
    gb_cols = ['ranking', 'infection_prob'] if 'infection_prob' in df_node_ranks.columns else ['ranking']
    t_values = np.array([1, 2, 3, 5, 10, 20, 50])
    summary_thresholds = [1, 5, 10]

    for k in k_values:
        summary_rows = []
        fig, ax = plt.subplots(figsize=(10, 7))

        for method in methods_16b:
            sub = df_node_ranks[(df_node_ranks['method'] == method) &
                                (df_node_ranks['method_rank'] <= k)]
            if sub.empty:
                continue
            per_instance = []
            for _, rgroup in sub.groupby(gb_cols):
                gt_vals = rgroup['sir_rank_snr'].values
                per_instance.append([(gt_vals <= t).mean() for t in t_values])
            cumulative = (np.mean(per_instance, axis=0) * 100).tolist()
            st = styles.get(method, {'color': '#333', 'marker': 'o'})
            ls = '--' if _is_composite(method) else '-'
            ax.plot(t_values, cumulative, color=st['color'], marker=st['marker'],
                    linestyle=ls, linewidth=2, markersize=7, label=_short(method))
            # Collect summary at key thresholds
            row = {'method': method}
            for t in summary_thresholds:
                idx = list(t_values).index(t)
                row[f'cumul_at_{t}'] = cumulative[idx]
            summary_rows.append(row)

        ax.set_xlabel('Ground Truth Rank Threshold (t)', fontsize=11)
        ax.set_ylabel(f'% of method-rank-≤{k} nodes with GT rank ≤ t', fontsize=11)
        ax.set_xlim([0, t_values[-1] + 2])
        ax.set_ylim([0, 105])
        ax.set_xticks(t_values)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f'P16b_cumulative_dist_with_lexi_k{k}.png'
        plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ {fname}')

        # Save summary CSV
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            csv_path = output_dir / f'P16b_cumulative_summary_k{k}.csv'
            df_summary.to_csv(csv_path, index=False)
            print(f'  ✓ P16b_cumulative_summary_k{k}.csv')


# ─── Pcorr: removed (O(n²) pairwise Kendall computation) ─────────────────────

# ─── P17/P18: Kendall heatmaps by ranking strategy ───────────────────────────

# ─── P19–P21: Pareto scatters ─────────────────────────────────────────────────

def _plot_pareto_scatter(df_overall: pd.DataFrame, x_col: str, y_col: str,
                          x_label: str, y_label: str, output_path: Path):
    df = df_overall.reset_index() if df_overall.index.name == 'method' else df_overall.copy()
    if x_col not in df.columns or y_col not in df.columns:
        print(f'  Skip {output_path.name} — missing {x_col} or {y_col}')
        return

    df_clean = df[['method', x_col, y_col]].dropna().reset_index(drop=True)
    if df_clean.empty:
        return

    all_methods = df_clean['method'].tolist()

    # Identify method(s) with best X-axis value for bold-outline highlighting
    best_x_val = df_clean[x_col].max()
    best_x_methods = set(df_clean[df_clean[x_col] == best_x_val]['method'].tolist())

    # Single unified Pareto front across ALL methods
    pareto = compute_pareto_front(df_clean, x_col, y_col, all_methods)
    pareto_set = set(pareto)

    # Assign one distinct color per Pareto-front method (deterministic via palette index)
    pareto_colors = {m: _EXTRA_PALETTE[i % len(_EXTRA_PALETTE)]
                     for i, m in enumerate(pareto)}

    _LEXI_EXT_COLOR = '#2CA02C'  # green for lexipeeling_extended

    fig, ax = plt.subplots(figsize=(10, 7))

    # ── Layer 1: background dots (non-Pareto, non-special) ──────────────────────
    _special = {'lexipeeling', 'lexipeeling_extended'}
    for _, row in df_clean.iterrows():
        m = row['method']
        if m in pareto_set or m in _special:
            continue
        marker = '^' if _is_composite(m) else 'o'
        if m in best_x_methods:
            ax.scatter(row[x_col], row[y_col], c='#6baed6', marker=marker,
                       s=150, alpha=0.9, zorder=2,
                       edgecolors='#888888', linewidths=2.0)
        else:
            ax.scatter(row[x_col], row[y_col], c='#6baed6', marker=marker,
                       s=28, alpha=0.25, zorder=2)

    # ── Layer 2: Pareto-front staircase ─────────────────────────────────────────
    if len(pareto) >= 2:
        px = [df_clean.loc[df_clean['method'] == m, x_col].values[0] for m in pareto]
        py = [df_clean.loc[df_clean['method'] == m, y_col].values[0] for m in pareto]
        ax.step(px, py, where='post', color='black', linewidth=1.5, alpha=0.5, zorder=3)

    # ── Layer 3: Pareto-front methods (individual colors, larger) ───────────────
    for m in pareto:
        if m in _special:
            continue  # drawn in dedicated layers below
        row = df_clean[df_clean['method'] == m].iloc[0]
        color = pareto_colors[m]
        if m in best_x_methods:
            ax.scatter(row[x_col], row[y_col], c=color, marker='o',
                       s=120, alpha=0.92, zorder=4,
                       edgecolors='#888888', linewidths=2.0)
        else:
            ax.scatter(row[x_col], row[y_col], c=color, marker='o',
                       s=120, alpha=0.92, zorder=4,
                       edgecolors='#cccccc', linewidths=0.4)

    # ── Layer 4: lexipeeling_extended (green triangle) ──────────────────────────
    lexi_ext_row = df_clean[df_clean['method'] == 'lexipeeling_extended']
    lexi_ext_on_front = 'lexipeeling_extended' in pareto_set
    lexi_ext_is_best = 'lexipeeling_extended' in best_x_methods
    if not lexi_ext_row.empty:
        ext_alpha = 1.0 if lexi_ext_on_front else 0.45
        ext_s = 320 if lexi_ext_is_best else 180
        ext_ec = '#888888' if lexi_ext_is_best else ('#cccccc' if lexi_ext_on_front else 'none')
        ext_lw = 2.0 if lexi_ext_is_best else (0.4 if lexi_ext_on_front else 0)
        ax.scatter(lexi_ext_row[x_col], lexi_ext_row[y_col],
                   c=_LEXI_EXT_COLOR, marker='^', s=ext_s, alpha=ext_alpha, zorder=5,
                   edgecolors=ext_ec, linewidths=ext_lw)

    # ── Layer 5: lexipeeling (always topmost) ────────────────────────────────────
    lexi_row = df_clean[df_clean['method'] == 'lexipeeling']
    lexi_on_front = 'lexipeeling' in pareto_set
    lexi_is_best = 'lexipeeling' in best_x_methods
    if not lexi_row.empty:
        lexi_alpha = 1.0 if lexi_on_front else 0.45
        lexi_s = 480 if lexi_is_best else 280
        lexi_ec = '#888888' if lexi_is_best else ('#cccccc' if lexi_on_front else 'none')
        lexi_lw = 2.0 if lexi_is_best else (0.4 if lexi_on_front else 0)
        ax.scatter(lexi_row[x_col], lexi_row[y_col],
                   c='#D55E00', marker='*', s=lexi_s, alpha=lexi_alpha, zorder=6,
                   edgecolors=lexi_ec, linewidths=lexi_lw)

    # ── Legend ───────────────────────────────────────────────────────────────────
    # Only Pareto-front and best-X methods get named entries.
    # lexipeeling / lexipeeling_extended always appear.
    # Pareto-front labels are tracked for post-creation bolding.
    legend_handles = []
    pareto_legend_labels: set = set()
    _best_suffix = f' (best {x_label})'

    # lexipeeling entry (always)
    if not lexi_row.empty:
        _base = 'lexipeeling' if lexi_on_front else 'lexipeeling (★ = orange star)'
        lexi_label = _base + (f'{_best_suffix}' if lexi_is_best else '')
        lexi_alpha_leg = 1.0 if lexi_on_front else 0.45
        legend_handles.append(
            Line2D([0], [0], marker='*', color='#D55E00', markersize=14,
                   markeredgecolor='none', alpha=lexi_alpha_leg,
                   linestyle='None', label=lexi_label))
        if lexi_on_front:
            pareto_legend_labels.add(lexi_label)

    # lexipeeling_extended entry (always)
    if not lexi_ext_row.empty:
        _base = 'lexipeeling_extended' if lexi_ext_on_front else 'lexipeeling_extended (▲ = green triangle)'
        ext_label = _base + (f'{_best_suffix}' if lexi_ext_is_best else '')
        ext_alpha_leg = 1.0 if lexi_ext_on_front else 0.45
        legend_handles.append(
            Line2D([0], [0], marker='^', color=_LEXI_EXT_COLOR, markersize=11,
                   markeredgecolor='none', alpha=ext_alpha_leg,
                   linestyle='None', label=ext_label))
        if lexi_ext_on_front:
            pareto_legend_labels.add(ext_label)

    # One entry per Pareto-front method (not special)
    for m in pareto:
        if m in _special:
            continue
        color = pareto_colors[m]
        is_best = m in best_x_methods
        label_m = _short(m) + (_best_suffix if is_best else '')
        ec_leg = '#888888' if is_best else '#cccccc'
        lw_leg = 2.0 if is_best else 0.4
        legend_handles.append(
            Line2D([0], [0], marker='o', color=color, markersize=9,
                   markeredgecolor=ec_leg, markeredgewidth=lw_leg,
                   linestyle='None', label=label_m))
        pareto_legend_labels.add(label_m)

    # Background best-X methods (not on front, not special): named entry, no bold
    bg_best = best_x_methods - pareto_set - _special
    for m in sorted(bg_best):
        label_m = f'{_short(m)}{_best_suffix}'
        legend_handles.append(
            Line2D([0], [0], marker='o', color='#6baed6', markersize=9,
                   markeredgecolor='#888888', markeredgewidth=2.0,
                   linestyle='None', label=label_m))

    legend_obj = ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1),
                           loc='upper left', fontsize=8, framealpha=0.9)

    # Bold all Pareto-front entries (works natively without LaTeX)
    for text in legend_obj.get_texts():
        if text.get_text() in pareto_legend_labels:
            text.set_fontweight('bold')
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {output_path.name}')


def plot_pareto_precision_vs_mrr(df_overall: pd.DataFrame, output_dir: Path):
    """Plot 19: Pareto scatter Precision@k vs MRR (8 files)."""
    df = df_overall.reset_index() if df_overall.index.name == 'method' else df_overall.copy()
    y_col = 'mrr_mean'
    if y_col not in df.columns:
        print('  P19: mrr_mean not found, skipping')
        return
    for col, label in _PARETO_PRECISION_KEYS:
        mean_col = f'{col}_mean'
        slug = col.replace('precision_snr_', '')
        _plot_pareto_scatter(df, mean_col, y_col, label, 'MRR',
                              output_dir / f'P19_pareto_prec_{slug}_vs_mrr.png')


def plot_pareto_wkendall_vs_mrr(df_overall: pd.DataFrame, output_dir: Path):
    """Plot 20: Pareto scatter Weighted Kendall vs MRR."""
    df = df_overall.reset_index() if df_overall.index.name == 'method' else df_overall.copy()
    _plot_pareto_scatter(df, 'weighted_kendall_tau_mean', 'mrr_mean',
                          'Weighted Kendall τ', 'MRR',
                          output_dir / 'P20_pareto_wkendall_vs_mrr.png')


def plot_pareto_kendall_vs_mrr(df_overall: pd.DataFrame, output_dir: Path):
    """Plot 21: Pareto scatter Standard Kendall vs MRR."""
    df = df_overall.reset_index() if df_overall.index.name == 'method' else df_overall.copy()
    _plot_pareto_scatter(df, 'kendall_correlation_mean', 'mrr_mean',
                          'Standard Kendall τ', 'MRR',
                          output_dir / 'P21_pareto_kendall_vs_mrr.png')


def _compute_cumulative_for_pareto(df_node_ranks: pd.DataFrame,
                                    k: int, t: int) -> pd.Series:
    """Compute cumulative score per method from node_ranks DataFrame.

    Returns a Series indexed by method name with cumulative score values.
    """
    sub = df_node_ranks[df_node_ranks['method_rank'] <= k]
    if sub.empty:
        return pd.Series(dtype=float)

    gb_cols = ['ranking', 'infection_prob'] if 'infection_prob' in sub.columns else ['ranking']
    results = {}
    for method, mdf in sub.groupby('method'):
        per_instance = []
        for _, rgroup in mdf.groupby(gb_cols):
            gt_vals = rgroup['sir_rank_snr'].values
            per_instance.append((gt_vals <= t).mean())
        if per_instance:
            results[method] = float(np.mean(per_instance) * 100)
    return pd.Series(results)


def plot_pareto_cumulative_vs_mrr(df_node_ranks: pd.DataFrame,
                                   df_overall: pd.DataFrame,
                                   output_dir: Path):
    """Plot P24: Pareto scatter Cumulative@(k,t) vs MRR."""
    df = df_overall.reset_index() if df_overall.index.name == 'method' else df_overall.copy()
    if 'mrr_mean' not in df.columns:
        print('  P24: mrr_mean not found, skipping')
        return

    required = ['method', 'method_rank', 'sir_rank_snr', 'ranking']
    if not all(c in df_node_ranks.columns for c in required):
        print('  P24: missing columns in node_ranks, skipping')
        return

    for k, t in [(1, 1), (5, 5)]:
        col_name = f'cumul_k{k}_t{t}'
        cumul = _compute_cumulative_for_pareto(df_node_ranks, k, t)
        if cumul.empty:
            print(f'  P24: no cumulative data for k={k}, t={t}, skipping')
            continue
        # Merge cumulative into df
        df_merged = df.copy()
        df_merged[col_name] = df_merged['method'].map(cumul)
        _plot_pareto_scatter(df_merged, col_name, 'mrr_mean',
                              f'Cumulative@(k={k}, t={t})', 'MRR',
                              output_dir / f'P24_pareto_cumul_k{k}_t{t}_vs_mrr.png')


def plot_worst_case_drop(output_dir: Path, top_n: int = 15):
    """Plot P25: worst-case cumulative score drop bar chart (cross-dataset)."""
    tables_dir = Path(__file__).parent.parent / 'tables/cross_dataset/appendix'
    k_values, t_values = [1, 5, 10], [1, 5, 10]
    ds_list = ['blogs_gcc', 'books', 'email-eu-core_gcc', 'mind',
               'brexit', 'iphone_samsung', 'lastfm_asia']
    exclude_prefixes = ('betas_', 'random')

    # Collect per-dataset avg scores for each method
    from collections import defaultdict
    method_ds_scores: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for k in k_values:
        for t in t_values:
            path = tables_dir / f'cross_cumul_k{k}_t{t}.csv'
            if not path.exists():
                continue
            df = pd.read_csv(path, index_col=0)
            for ds in ds_list:
                if ds not in df.columns:
                    continue
                col = df[ds].dropna()
                col = col[~col.index.str.startswith(exclude_prefixes)]
                for m, v in col.items():
                    method_ds_scores[m][ds].append(v)

    if not method_ds_scores:
        print('  P25: no cumulative data found, skipping')
        return

    # Average across (k,t) per dataset
    method_ds_avg = {m: {ds: np.mean(s) for ds, s in ds_scores.items()}
                     for m, ds_scores in method_ds_scores.items()}

    # Top score per dataset
    top_per_ds = {ds: max(method_ds_avg[m].get(ds, -np.inf)
                          for m in method_ds_avg) for ds in ds_list}

    # Max drop per method (only methods present on all datasets)
    stats = {}
    for m, ds_avgs in method_ds_avg.items():
        if len(ds_avgs) < len(ds_list):
            continue
        drops = {ds: top_per_ds[ds] - ds_avgs.get(ds, 0) for ds in ds_list}
        worst_ds = max(drops, key=drops.get)
        stats[m] = (drops[worst_ds], worst_ds)

    ranked = sorted(stats.items(), key=lambda x: x[1][0])[:top_n]

    # Plot
    methods = [m for m, _ in ranked]
    drops = [d for _, (d, _) in ranked]
    worst_ds = [w for _, (_, w) in ranked]
    labels = [_short(m, 30) for m in methods]

    ds_short = {'blogs_gcc': 'Blogs', 'books': 'Books',
                'email-eu-core_gcc': 'Email', 'mind': 'MIND',
                'brexit': 'Brexit', 'iphone_samsung': 'iPhone',
                'lastfm_asia': 'LastFM'}

    fig_height = max(3, 1.0 + top_n * 0.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    colors = ['#D55E00' if m == 'lexipeeling_degree' else '#6baed6'
              for m in methods]
    bars = ax.barh(range(len(methods)), drops, color=colors, edgecolor='#333333',
                   linewidth=0.5, height=0.7)

    # Add worst-dataset labels
    for i, (d, w) in enumerate(zip(drops, worst_ds)):
        ax.text(d + 0.3, i, f'{ds_short.get(w, w)}', va='center', fontsize=8,
                color='#555555')

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Worst-case gap (points below dataset winner)', fontsize=11)
    ax.set_title('Robustness: smallest worst-case cumulative score drop', fontsize=12)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    out = output_dir / 'P25_worst_case_drop.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out.name}')


# ─── P22: Beta group reach boxplot ────────────────────────────────────────────

def _parse_ranking_groups(raw) -> List[int]:
    try:
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, (list, tuple)):
            return [int(x) for x in parsed]
    except Exception:
        ids = re.findall(r'\d+', str(raw))
        return [int(x) for x in ids]
    return []


def plot_beta_group_reach_boxplot(reach_df: pd.DataFrame, ranking: str,
                                   beta_label: str, best_none_method: Optional[str],
                                   output_dir: Path,
                                   ax=None):
    """
    Plot 22: For each group in the ranking, show 3 side-by-side boxplots
    (lexipeeling, betas_lexipeeling_{b}, best_non_lexi) of group_pct_mean
    across infection probabilities, normalized to the max coverage of each group.

    If ax is provided, draws into it (for facet use). Otherwise creates its own
    figure and saves to output_dir.
    """
    df_r = reach_df[reach_df['ranking'] == ranking].copy()
    if df_r.empty:
        return

    sample = df_r['ranking_groups'].dropna().iloc[0] if 'ranking_groups' in df_r.columns else None
    group_ids = _parse_ranking_groups(sample) if sample is not None else []
    if not group_ids:
        group_ids = sorted(int(m.group(1))
                           for m in (re.match(r'group_(\d+)_pct_mean', c)
                                     for c in df_r.columns) if m)
    if not group_ids:
        print(f'  P22: cannot determine groups for ranking {ranking}, skipping')
        return

    lexi_method = 'lexipeeling'
    beta_method = f'betas_lexipeeling_{beta_label}'
    bar_methods = [m for m in [lexi_method, beta_method, best_none_method] if m is not None]
    present_methods = [m for m in bar_methods if m in df_r['method'].unique()]

    if not present_methods:
        return

    # Normalization: max avg spread per group across ALL methods and probs
    norm: Dict[int, float] = {}
    for pos, gid in enumerate(group_ids):
        col = f'group_{pos}_pct_mean'
        norm[gid] = float(df_r[col].max()) if col in df_r.columns and df_r[col].max() > 0 else 1.0

    # Collect data positionally: group_ids[i] → column f'group_{i}_pct_mean'
    data: Dict[str, Dict[int, np.ndarray]] = {}
    for method in present_methods:
        df_m = df_r[df_r['method'] == method]
        data[method] = {}
        for pos, gid in enumerate(group_ids):
            col = f'group_{pos}_pct_mean'  # positional, not by actual gid
            if col in df_m.columns:
                vals = df_m[col].dropna().values
                data[method][gid] = vals / norm[gid] * 100 if norm[gid] > 0 else vals
            else:
                data[method][gid] = np.array([])

    method_colors = {
        lexi_method: '#D55E00',
        beta_method: '#E69F00',
        best_none_method: '#0072B2',
    }

    n_groups = len(group_ids)
    n_methods = len(present_methods)
    width = 0.7 / max(n_methods, 1)
    x = np.arange(n_groups)

    created_own_fig = ax is None
    if created_own_fig:
        fig, ax = plt.subplots(figsize=(max(6, n_groups * 1.8 + 2), 5))

    for i, method in enumerate(present_methods):
        offset = (i - (n_methods - 1) / 2) * width
        bp_data = [data[method].get(gid, np.array([])) for gid in group_ids]
        bp_data = [d if len(d) > 0 else np.array([np.nan]) for d in bp_data]
        positions = x + offset
        color = method_colors.get(method, '#333333')
        ax.boxplot(bp_data,
                   positions=positions,
                   widths=width * 0.85,
                   patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.6),
                   medianprops=dict(color='black', linewidth=2),
                   whiskerprops=dict(color=color),
                   capprops=dict(color=color),
                   flierprops=dict(marker='o', color=color, markersize=4, alpha=0.5),
                   showfliers=True)
        ax.plot([], [], color=color, linewidth=4, alpha=0.7, label=_short(method))

    ax.set_xticks(x)
    ax.set_xticklabels([f'Group {gid}' for gid in group_ids], fontsize=10)
    ax.set_ylabel('Avg reach % (norm.)', fontsize=9)
    ax.set_ylim([0, 115])
    ax.grid(True, alpha=0.3, axis='y')

    # Vertical separator lines between group clusters
    for gi in range(n_groups - 1):
        sep_x = (x[gi] + x[gi + 1]) / 2.0
        ax.axvline(sep_x, color='gray', alpha=0.4, linewidth=1.2, linestyle='--')

    if created_own_fig:
        ax.legend(fontsize=8)
        plt.tight_layout()
        safe_ranking = ranking.replace('/', '_').replace(' ', '_')
        out = output_dir / f'P22_beta_{beta_label}_reach_{safe_ranking}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ {out.name}')


def plot_beta_group_reach_by_length(reach_df: pd.DataFrame, beta_label: str,
                                     best_none_method: Optional[str],
                                     output_dir: Path):
    """Plot 22: One facet figure per ranking length; panels = rankings of that length."""
    if 'ranking' not in reach_df.columns:
        return
    rankings = sorted(reach_df['ranking'].unique())

    by_length: Dict[int, List[str]] = {}
    for r in rankings:
        length = len(r.split('_'))
        by_length.setdefault(length, []).append(r)

    for length, rnk_list in sorted(by_length.items()):
        rnk_sorted = sorted(rnk_list)
        ncols = 3
        nrows = (len(rnk_sorted) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(ncols * 6, nrows * 4.5),
                                  squeeze=False)
        axes_flat = axes.flatten()

        for i, ranking in enumerate(rnk_sorted):
            plot_beta_group_reach_boxplot(reach_df, ranking, beta_label,
                                          best_none_method, output_dir,
                                          ax=axes_flat[i])

        for i in range(len(rnk_sorted), len(axes_flat)):
            axes_flat[i].set_visible(False)

        # Shared legend from first visible axes
        handles, labels = axes_flat[0].get_legend_handles_labels()
        for axi in axes_flat[:len(rnk_sorted)]:
            leg = axi.get_legend()
            if leg:
                leg.remove()
        if handles:
            fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.5),
                       loc='center left', fontsize=9)

        plt.tight_layout()
        out = output_dir / f'P22_beta_{beta_label}_length_{length}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ {out.name}')


# ─── P23: MRR vs beta ─────────────────────────────────────────────────────────

def plot_mrr_vs_beta(betas_data: Dict[str, Dict[str, pd.DataFrame]],
                     output_dir: Path):
    """
    Plot 23: MRR vs beta (X axis = GT beta value).

    Each betas_lexipeeling_X method is plotted as a continuous line across ALL
    GT beta values: at each GT beta b, we read betas_lexipeeling_X from
    betas_{b}/level1_overall.csv — i.e. method X evaluated against GT at b.
    This is cross-evaluation: method_beta != gt_beta in general.

    Also shown as dashed baselines: lexipeeling + top-3 non-lexi methods
    (evaluated against standard SNR GT, same at every x).
    """
    metric = 'mrr_mean'
    x_numeric = {label: (0.0 if label == 'none' else float(label))
                  for label in betas_data}
    x_sorted = sorted(x_numeric.items(), key=lambda kv: kv[1])
    x_nonzero = [(label, xv) for label, xv in x_sorted if label != 'none']
    beta_labels = [label for label, _ in x_nonzero]  # ['0.01', '0.1', '0.5', '0.99']

    # ── discover pure betas_lexipeeling_{number} methods (no composite variants) ─
    # Matches betas_lexipeeling_0.1 but NOT betas_lexipeeling_coreness_0.1 etc.
    _pure_betas_re = re.compile(r'^betas_lexipeeling_[\d.]+$')
    betas_lexi_methods: List[str] = []
    for label, _ in x_nonzero:
        df_ov = betas_data[label].get('overall')
        if df_ov is None:
            continue
        df = df_ov.reset_index() if df_ov.index.name == 'method' else df_ov.copy()
        for m in df['method'].tolist():
            if _pure_betas_re.match(m) and m not in betas_lexi_methods:
                betas_lexi_methods.append(m)
    betas_lexi_methods.sort()

    # ── cross-evaluation: for each method_X, read its MRR at every GT beta ──────
    # cross_data[method_X][gt_beta_xval] = mrr
    cross_data: Dict[str, Dict[float, float]] = {m: {} for m in betas_lexi_methods}
    for gt_label, gt_xv in x_nonzero:
        df_ov = betas_data[gt_label].get('overall')
        if df_ov is None:
            continue
        df = df_ov.reset_index() if df_ov.index.name == 'method' else df_ov.copy()
        if metric not in df.columns:
            continue
        for m in betas_lexi_methods:
            row = df[df['method'] == m]
            if not row.empty:
                cross_data[m][gt_xv] = float(row[metric].values[0])

    # ── standard baselines: lexipeeling + top-1 non-lexi by MRR ─────────────────
    none_overall = betas_data.get('none', {}).get('overall')
    std_methods: List[str] = []
    if none_overall is not None:
        df0 = none_overall.reset_index() if none_overall.index.name == 'method' else none_overall.copy()
        if 'lexipeeling' in df0['method'].values:
            std_methods.append('lexipeeling')
        baselines = df0[~df0['method'].str.contains('lexipeeling', na=False)]
        if metric in df0.columns and not baselines.empty:
            top1 = baselines.sort_values(metric, ascending=False)['method'].iloc[0]
            std_methods.append(top1)

    std_mrr: Dict[str, Dict[float, float]] = {m: {} for m in std_methods}
    for label, xv in x_sorted:
        df_ov = betas_data[label].get('overall')
        if df_ov is None:
            continue
        df = df_ov.reset_index() if df_ov.index.name == 'method' else df_ov.copy()
        if metric not in df.columns:
            continue
        for m in std_methods:
            row = df[df['method'] == m]
            if not row.empty:
                std_mrr[m][xv] = float(row[metric].values[0])

    # ── plot ────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    marker_idx = 0

    all_methods_p23 = std_methods + betas_lexi_methods
    styles_p23 = assign_method_style(all_methods_p23)

    # Standard baselines (solid)
    for m in std_methods:
        pts = sorted(std_mrr[m].items())
        if not pts:
            continue
        xs, ys = zip(*pts)
        st = styles_p23[m]
        lw = 2.5 if m == 'lexipeeling' else 1.8
        ax.plot(xs, ys, color=st['color'], marker=st['marker'],
                linewidth=lw, markersize=7, linestyle='-', label=_short(m))

    # Cross-evaluation curves for each betas_lexipeeling_X (solid lines)
    for m in betas_lexi_methods:
        pts = sorted(cross_data[m].items())
        if len(pts) < 2:
            continue
        xs, ys = zip(*pts)
        st = styles_p23[m]
        ax.plot(xs, ys, color=st['color'], marker=st['marker'],
                linewidth=1.8, markersize=7, linestyle='-',
                label=_short(m.replace('betas_lexipeeling_', 'betas_lexi_')))

    ax.set_xlabel('Beta value', fontsize=11)
    ax.set_ylabel('MRR (mean)', fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'P23_mrr_vs_beta.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ P23_mrr_vs_beta.png')


# ─── Robustness Grid ─────────────────────────────────────────────────────────

def _select_robustness_methods(df_overall: pd.DataFrame, metric_mean_col: str) -> List[str]:
    """Select methods for the robustness plot.

    Rules:
    1. lexipeeling is always included.
    2. Among baselines (non-lexipeeling, non-lexipeeling-variants):
       - If >3 baselines tie at the maximum baseline value → include ALL of them.
       - Otherwise → include the top 3 baselines (random among ties).
    Result is sorted by method name for deterministic legend order.
    """
    df = df_overall.reset_index() if df_overall.index.name == 'method' else df_overall.copy()
    if 'method' not in df.columns or metric_mean_col not in df.columns:
        return []

    result: List[str] = []
    if 'lexipeeling' in df['method'].values:
        result.append('lexipeeling')

    # Baselines = everything that doesn't contain 'lexipeeling' in the name
    baselines_df = df[~df['method'].str.contains('lexipeeling', na=False)].copy()
    if baselines_df.empty:
        return sorted(result)

    max_baseline_val = baselines_df[metric_mean_col].max()
    at_max = baselines_df.loc[
        baselines_df[metric_mean_col] == max_baseline_val, 'method'].tolist()

    if len(at_max) > 3:
        # Many ties — show all of them
        result.extend(at_max)
    else:
        # Top 3 baselines (head(3) picks randomly among ties due to sort stability)
        top3 = (baselines_df
                .sort_values(metric_mean_col, ascending=False)['method']
                .head(3).tolist())
        result.extend(top3)

    return sorted(set(result))


def plot_robustness_grid(data_dir: Path, df_overall: pd.DataFrame,
                         output_dir: Path, dataset_name: str):
    """2x2 grid: MRR by infection_prob, ranking_strategy, ranking_length,
    lexicographic_degeneracy."""
    methods = _select_robustness_methods(
        df_overall if 'method' not in df_overall.columns
        else df_overall.set_index('method') if df_overall.index.name != 'method'
        else df_overall,
        'mrr_mean')
    styles = assign_method_style(methods)
    mrr_col = 'mrr_mean'

    # Load dimensional data
    dim_specs = [
        ('by_infection_prob',         'infection_prob_rounded', 'Infection Probability',    'line'),
        ('by_ranking_strategy',       'ranking_strategy',      'Ranking Strategy',         'bar'),
        ('by_ranking_length',         'ranking_length',        'Ranking Length',            'bar'),
        ('by_lexicographic_degeneracy', 'lexicographic_degeneracy', 'Lex. Degeneracy',     'bar'),
    ]

    dfs_dim = {}
    for stem, _, _, _ in dim_specs:
        for ext in ['.csv', '.parquet']:
            p = data_dir / f'{stem}{ext}'
            if p.exists():
                dfs_dim[stem] = pd.read_parquet(p) if ext == '.parquet' else pd.read_csv(p)
                break

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for idx, (stem, dim_col, dim_label, plot_type) in enumerate(dim_specs):
        ax = axes_flat[idx]
        df_dim = dfs_dim.get(stem)
        if df_dim is None:
            ax.text(0.5, 0.5, f'{stem}\nnot found', transform=ax.transAxes,
                    ha='center', va='center')
            continue

        df_m = df_dim[df_dim['method'].isin(methods)].copy()
        dim_vals = sorted(df_m[dim_col].dropna().unique())

        if plot_type == 'line':
            for method in methods:
                sub = df_m[df_m['method'] == method].sort_values(dim_col)
                if sub.empty or mrr_col not in sub.columns:
                    continue
                st = styles.get(method, {'color': '#333', 'marker': 'o'})
                ax.plot(sub[dim_col], sub[mrr_col], color=st['color'],
                        marker=st['marker'], linewidth=2, markersize=7)
        else:  # barplot
            n_dims = len(dim_vals)
            n_methods = len(methods)
            width = 0.6 / max(n_methods, 1)
            x = np.arange(n_dims) * 1.4

            for i, method in enumerate(methods):
                st = styles.get(method, {'color': '#333'})
                vals = []
                for dv in dim_vals:
                    v = df_m[(df_m['method'] == method) & (df_m[dim_col] == dv)][mrr_col]
                    vals.append(float(v.mean()) if not v.empty else np.nan)
                offset = (i - (n_methods - 1) / 2) * width
                ax.bar(x + offset, vals, width=width * 0.85,
                       color=st['color'], alpha=0.8)

            ax.set_xticks(x)
            labels = [str(v) for v in dim_vals]
            ax.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')

        ax.set_xlabel(dim_label, fontsize=10)
        ax.set_ylabel('MRR', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title(f'MRR by {dim_label}', fontsize=11)

    # Shared legend at bottom
    handles = [mpatches.Patch(color=styles.get(m, {'color': '#333'})['color'],
                               label=_short(m)) for m in methods]
    fig.legend(handles=handles, loc='lower center', ncol=len(methods),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = output_dir / f'P_robustness_mrr_{dataset_name}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out.name}')


# ─── Tie Plot ────────────────────────────────────────────────────────────────

def _read_node_ranks_for_ties(path: Path, methods: List[str]) -> pd.DataFrame:
    """Memory-efficient loader for ALL ranks (not just rank-1) for specific methods.

    Tie sizes (groupby method_rank → count) are identical across infection_probs,
    so we read only a single infection_prob to keep memory bounded.
    """
    cols = ['method', 'method_rank', 'n_nodes', 'ranking', 'infection_prob']
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        # Sample one infection_prob from the file metadata
        pf = pq.ParquetFile(path)
        sample = pf.read_row_group(0, columns=['infection_prob'])
        sample_prob = sample.column('infection_prob')[0].as_py()
        df = pd.read_parquet(path, columns=cols,
                             filters=[('method', 'in', methods),
                                      ('infection_prob', '==', sample_prob)])
    else:
        # For CSV: read first chunk to get a sample infection_prob, then filter
        sample_prob = None
        for chunk in pd.read_csv(path, usecols=['infection_prob'], chunksize=1):
            sample_prob = chunk['infection_prob'].iloc[0]
            break
        chunks = []
        for chunk in pd.read_csv(path, usecols=cols, chunksize=500_000):
            mask = (chunk['method'].isin(methods) &
                    (chunk['infection_prob'] == sample_prob))
            chunks.append(chunk[mask])
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return df


def plot_tie_sizes(node_ranks_path: Path, output_dir: Path, dataset_name: str):
    """Tie plot: two-panel (full + zoom), shared Y axis, top-center legend."""
    tie_methods = ['lexipeeling', 'lexipeeling_degree', 'coreness', 'degree_lexicographic', 'degree']
    tie_styles = {
        'coreness':             {'color': '#888888', 'ls': '--', 'lw': 2.0},
        'lexipeeling':          {'color': '#0072B2', 'ls': '-',  'lw': 2.5},
        'degree_lexicographic': {'color': '#009E73', 'ls': '--', 'lw': 1.8},
        'degree':               {'color': '#D55E00', 'ls': '-',  'lw': 2.0},
        'lexipeeling_degree':   {'color': '#E69F00', 'ls': '-',  'lw': 2.5},
    }

    print(f'  Loading node_ranks for tie plot from {node_ranks_path.name}...')
    df = _read_node_ranks_for_ties(node_ranks_path, tie_methods)
    if df.empty:
        print('  Tie plot: no data found, skipping')
        return

    x_grid = np.linspace(0, 100, 200)

    # Pre-compute profiles once
    profiles_by_method = {}
    for method in tie_methods:
        if method not in df['method'].values:
            continue
        df_m = df[df['method'] == method]
        n_nodes = int(df_m['n_nodes'].iloc[0])
        all_profiles = []
        for ranking in df_m['ranking'].unique():
            sub = df_m[df_m['ranking'] == ranking]
            tie_counts = sub.groupby('method_rank').size().sort_index()
            cumsum = tie_counts.cumsum()
            x_pct = (cumsum / n_nodes) * 100
            y_sizes = tie_counts.values.astype(float)
            profile = np.interp(x_grid, x_pct.values, y_sizes,
                                left=y_sizes[0], right=y_sizes[-1])
            all_profiles.append(profile)
        arr = np.array(all_profiles)
        profiles_by_method[method] = {
            'median': np.median(arr, axis=0),
            'q25': np.percentile(arr, 25, axis=0),
            'q75': np.percentile(arr, 75, axis=0),
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    for ax, xlim, panel_label in [(ax1, [0, 100], '(a) Full ranking'),
                                   (ax2, [0, 20],  '(b) Top-ranked 20%')]:
        for method in tie_methods:
            if method not in profiles_by_method:
                continue
            p = profiles_by_method[method]
            st = tie_styles[method]
            ax.plot(x_grid, p['median'], color=st['color'], linestyle=st['ls'],
                    linewidth=st['lw'], label=_short(method))
            ax.fill_between(x_grid, p['q25'], p['q75'], color=st['color'], alpha=0.2)
        ax.set_yscale('log')
        ax.set_xlabel('Rank Position (% of nodes)', fontsize=11)
        ax.set_xlim(xlim)
        ax.grid(True, alpha=0.3)
        ax.set_title(panel_label, loc='left', fontsize=11)

    ax1.set_ylabel('Tie Size (# nodes at same rank)', fontsize=11)

    # Shared horizontal legend above both panels
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=4, fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.subplots_adjust(wspace=0.05)

    out = output_dir / f'P_tie_sizes_{dataset_name}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out.name}')


# ─── Distribution Plots ───────────────────────────────────────────────────────

def plot_distributions(graph_file: Path, output_dir: Path, dataset_name: str):
    """Degree and core decomposition distributions (1×2 panels)."""
    import networkx as nx

    # Build graph
    G = nx.Graph()
    try:
        with open(graph_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    G.add_edge(parts[0], parts[1])
    except Exception as e:
        print(f'  Distribution plot: failed to read graph: {e}')
        return

    if G.number_of_nodes() < 2:
        print('  Distribution plot: graph too small, skipping')
        return

    degrees = np.array([d for _, d in G.degree()])
    core_numbers = np.array(list(nx.core_number(G).values()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Degree distribution
    ax = axes[0]
    ax.hist(degrees, bins=50, color='#4C72B0', edgecolor='white', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('Degree', fontsize=10)
    ax.set_ylabel('# Nodes', fontsize=10)
    ax.set_title('(a) Degree distribution', loc='left', fontsize=11)
    ax.grid(True, alpha=0.3)

    # (b) Core number distribution
    ax = axes[1]
    core_vals, core_counts = np.unique(core_numbers, return_counts=True)
    ax.bar(core_vals, core_counts, color='#55A868', edgecolor='white', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('Core number', fontsize=10)
    ax.set_ylabel('# Nodes', fontsize=10)
    ax.set_title('(b) Core decomposition', loc='left', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / f'P_distributions_{dataset_name}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out.name}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description='Paper-quality plots for LexiCoreness')
    parser.add_argument('--aggregated-dir', type=Path, required=True,
                        help='Root aggregated results directory (contains betas_*/ subdirs)')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for plots')
    parser.add_argument('--sir-dir', type=Path, default=None,
                        help='Directory containing SIR parquet files (for P1)')
    parser.add_argument('--graph-name', type=str, required=True,
                        help='Graph name (e.g. books)')
    parser.add_argument('--method-groups-yaml', type=Path,
                        default=Path('methods-files/to_plot/methods_groups.yaml'),
                        help='Path to methods_groups.yaml (unused, kept for CLI compat)')
    parser.add_argument('--graph-file', type=Path, default=None,
                        help='Graph edge list file (for avg degree in P1)')
    parser.add_argument('--config', type=Path, default=None,
                        help='Path to experiment config YAML. If sem_percentile is a list, '
                             'generates plots for each sem_{p}/ subdirectory.')
    args = parser.parse_args()

    # Resolve sem_percentile list from config
    sem_percentile_values = [None]  # None = no subdirs (default)
    if args.config and args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        sp = config.get('sem_percentile')
        if sp is not None:
            sem_percentile_values = sp if isinstance(sp, list) else [sp]
    use_subdirs = len(sem_percentile_values) > 1

    base_aggregated_dir = args.aggregated_dir
    base_output_dir = args.output_dir

    for sem_p in sem_percentile_values:
        if use_subdirs and sem_p is not None:
            args.aggregated_dir = base_aggregated_dir / f'sem_{int(sem_p)}'
            args.output_dir = base_output_dir / f'sem_{int(sem_p)}'
            print(f"\n{'='*70}")
            print(f"PLOTS — SEM percentile: {int(sem_p)}")
            print(f"{'='*70}")
        else:
            args.aggregated_dir = base_aggregated_dir
            args.output_dir = base_output_dir

        if not args.aggregated_dir.exists():
            print(f"  Skipping: {args.aggregated_dir} does not exist")
            continue

        _run_plots(args)


def _run_plots(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print('Loading data...')
    betas_data = load_all_betas(args.aggregated_dir)
    if not betas_data:
        print('ERROR: no betas_* subdirectories found in', args.aggregated_dir)
        sys.exit(1)

    none_data = betas_data.get('none', {})
    df_overall_raw = none_data.get('overall')
    if df_overall_raw is None:
        print('ERROR: betas_none/level1_overall.csv not found')
        sys.exit(1)

    df_overall = df_overall_raw.copy()
    if df_overall.index.name != 'method' and 'method' in df_overall.columns:
        df_overall = df_overall.set_index('method')
    df_overall_reset = df_overall.reset_index()

    data_by_dim = {
        'by_strategy':   none_data.get('by_strategy'),
        'by_prob':       none_data.get('by_prob'),
        'by_degeneracy': none_data.get('by_degeneracy'),
        'by_length':     none_data.get('by_length'),
    }

    # ── P1: SIR violin ────────────────────────────────────────────────────────
    print('\nP1: SIR violin plots...')
    # Auto-detect graph file if not provided
    if args.graph_file is None:
        for candidate in [
            Path('data/graphs') / args.graph_name / f'{args.graph_name}.txt',
            Path('data/graphs') / args.graph_name / f'{args.graph_name}.edges',
            Path('data/graphs') / args.graph_name / f'{args.graph_name}_edges.txt',
            Path('data/graphs') / args.graph_name / f'{args.graph_name}.edgelist',
        ]:
            if candidate.exists():
                args.graph_file = candidate
                print(f'  Auto-detected graph file: {args.graph_file}')
                break
    if args.sir_dir and args.sir_dir.exists():
        sir_data = load_sir_data(args.sir_dir, args.graph_name)
        avg_degree = compute_avg_degree(args.graph_file) if args.graph_file else None
        if avg_degree:
            print(f'  Average degree: {avg_degree:.2f}')
        plot_sir_violins(sir_data, args.output_dir, args.graph_name, avg_degree)
    else:
        print('  Skipping (no --sir-dir)')

    # ── P2: ARHR facets vs infection probability ───────────────────────────────
    print('\nP2: ARHR facets vs infection probability...')
    df_by_prob = data_by_dim.get('by_prob')
    if df_by_prob is not None:
        union_pct = select_union_for_metrics(df_overall_reset, _ARHR_PCT_COLS)
        union_abs = select_union_for_metrics(df_overall_reset, _ARHR_ABS_COLS)
        styles_pct = assign_method_style(union_pct)
        styles_abs = assign_method_style(union_abs)
        print(f'  ARHR@% methods ({len(union_pct)}): {union_pct}')
        print(f'  ARHR@abs methods ({len(union_abs)}): {union_abs}')
        plot_arhr_pct_facet_vs_prob(df_by_prob, union_pct, styles_pct, args.output_dir)
        plot_arhr_abs_facet_vs_prob(df_by_prob, union_abs, styles_abs, args.output_dir)
    else:
        print('  Skipping (by_infection_prob not found)')

    # ── P4–P8: ARHR heatmaps by dimension ─────────────────────────────────────
    print('\nP4–P8: ARHR heatmaps by dimension...')
    plot_arhr_heatmaps(df_overall_reset, data_by_dim, args.output_dir)

    # ── P9: Precision heatmaps by dimension ───────────────────────────────────
    print('\nP9: Precision heatmaps by dimension...')
    plot_precision_heatmaps(df_overall_reset, data_by_dim, args.output_dir)

    # ── P11/P12: Kendall barplots by ranking strategy ─────────────────────────
    print('\nP11/P12: Kendall barplots by ranking strategy...')
    df_by_strategy = data_by_dim.get('by_strategy')
    if df_by_strategy is not None:
        plot_kendall_barplots_by_strategy(df_by_strategy, df_overall_reset, args.output_dir)
    else:
        print('  Skipping (by_ranking_strategy not found)')

    # ── P13: Kendall barplots by lexicographic degeneracy ─────────────────────
    print('\nP13: Kendall barplots by lexicographic degeneracy...')
    df_by_degeneracy = data_by_dim.get('by_degeneracy')
    if df_by_degeneracy is not None:
        plot_kendall_barplots_by_degeneracy(df_by_degeneracy, df_overall_reset, args.output_dir)
    else:
        print('  Skipping (by_lexicographic_degeneracy not found)')

    # ── P14/P15: Kendall vs infection probability ──────────────────────────────
    print('\nP14/P15: Kendall vs infection probability...')
    if df_by_prob is not None:
        plot_kendall_vs_prob(df_by_prob, df_overall_reset, args.output_dir)
    else:
        print('  Skipping (by_infection_prob not found)')

    # ── P16: Cumulative distribution ───────────────────────────────────────────
    print('\nP16: Cumulative distribution...')
    df_node_ranks = none_data.get('node_ranks')
    if df_node_ranks is not None:
        plot_cumulative_dist(df_node_ranks, df_overall_reset, args.output_dir)
        plot_cumulative_dist_with_lexi(df_node_ranks, df_overall_reset, args.output_dir)
    else:
        print('  Skipping (node_ranks not found)')

    # ── Robustness grid (2×2 MRR) ────────────────────────────────────────────
    print('\nRobustness grid (2×2 MRR)...')
    betas_none_dir = args.aggregated_dir / 'betas_none'
    if betas_none_dir.is_dir():
        plot_robustness_grid(betas_none_dir, df_overall_reset, args.output_dir, args.graph_name)
    else:
        print('  Skipping (betas_none dir not found)')

    # ── Tie plot ──────────────────────────────────────────────────────────────
    print('\nTie plot...')
    nr_path = _find_file(args.aggregated_dir / 'betas_none', 'node_ranks')
    if nr_path is not None:
        plot_tie_sizes(nr_path, args.output_dir, args.graph_name)
    else:
        print('  Skipping (node_ranks not found)')

    # ── Distribution plots ─────────────────────────────────────────────────────
    print('\nDistribution plots...')
    if args.graph_file and args.graph_file.exists():
        plot_distributions(args.graph_file, args.output_dir, args.graph_name)
    else:
        print('  Skipping (no --graph-file)')

    # ── P19–P25: Pareto scatters + robustness ─────────────────────────────────
    print('\nP19: Pareto scatter Precision@k vs MRR...')
    plot_pareto_precision_vs_mrr(df_overall, args.output_dir)

    print('\nP20: Pareto scatter Weighted Kendall vs MRR...')
    plot_pareto_wkendall_vs_mrr(df_overall, args.output_dir)

    print('\nP21: Pareto scatter Standard Kendall vs MRR...')
    plot_pareto_kendall_vs_mrr(df_overall, args.output_dir)

    print('\nP24: Pareto scatter Cumulative vs MRR...')
    if df_node_ranks is not None:
        plot_pareto_cumulative_vs_mrr(df_node_ranks, df_overall_reset, args.output_dir)
    else:
        print('  Skipping (node_ranks not found)')

    # ── P22: Beta group reach boxplots ─────────────────────────────────────────
    print('\nP22: Beta group reach boxplots...')
    best_none = _find_best_non_lexi(df_overall_reset, 'mrr_mean')
    print(f'  Best non-lexi method at beta=0: {best_none}')
    for beta_label, beta_d in betas_data.items():
        if beta_label == 'none':
            continue
        reach_df = beta_d.get('reach')
        if reach_df is None:
            print(f'  beta={beta_label}: no reach data, skipping')
            continue
        plot_beta_group_reach_by_length(reach_df, beta_label, best_none, args.output_dir)

    # ── P23: MRR vs beta ───────────────────────────────────────────────────────
    print('\nP23: MRR vs beta...')
    betas_overall = {label: d for label, d in betas_data.items() if 'overall' in d}
    if len(betas_overall) > 1:
        plot_mrr_vs_beta(betas_overall, args.output_dir)
    else:
        print('  Need at least 2 beta groups, skipping')

    print(f'\nDone. Plots saved to {args.output_dir}')


if __name__ == '__main__':
    main()
