"""
Generate paper-quality LaTeX tables for the LexiCoreness paper.

Usage:
    python3 src/paper_tables.py \
        --aggregated-dir results/aggregated/books/ results/aggregated/highschool12/ \
        --graphs-dir data/graphs/books/ data/graphs/highschool-marseilles-2012/ \
        --output-dir plots/paper/tables/ \
        --methods-groups-yaml methods-files/to_plot/methods_groups.yaml

Output structure:
  {output_dir}/
    T1_graph_stats.tex/.csv
    main/
      T2_mrr_rank1.tex/.csv
      T3_kendall.tex/.csv
      T4_prec5pct.tex/.csv
    appendix/
      beta_none/
        TA_mrr_rank1.tex/.csv  TB_kendall.tex/.csv  ...
      beta_0.01/
        TA_mrr_rank1.tex/.csv  ...
      beta_0.1/  beta_0.5/  beta_0.99/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# ─── Utilities copied verbatim from paper_plots.py ───────────────────────────
# (Not imported to avoid matplotlib side effects)

def _is_composite(method: str) -> bool:
    """True if method is a composite lexipeeling variant (not base lexipeeling)."""
    return 'lexipeeling' in method and method != 'lexipeeling'


def _is_baseline(method: str) -> bool:
    """True if method contains no 'lexipeeling'."""
    return 'lexipeeling' not in method


def _short(method: str, max_len: int = 22) -> str:
    """Shorten a method name for display.

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


def _find_file(base: Path, stem: str) -> Optional[Path]:
    for ext in ['.parquet', '.csv']:
        p = base / f'{stem}{ext}'
        if p.exists():
            return p
    return None


def _read_df(path: Path) -> pd.DataFrame:
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ─── Constants ────────────────────────────────────────────────────────────────

BETA_LABELS_ORDERED = ['none', '0.01', '0.1', '0.5', '0.99']

# Matches betas_lexipeeling_0.01, betas_lexipeeling_0.1, etc. (not composites like betas_lexipeeling_degree_*)
_BETAS_LEXI_RE = re.compile(r'^betas_lexipeeling_[\d.]+$')

# Display names for datasets (internal key → paper-friendly name)
DATASET_DISPLAY_NAMES: Dict[str, str] = {
    'blogs_gcc':         'PolBlogs (LCC)',
    'books':             'PolBooks',
    'email-eu-core_gcc': 'Email-EU-Core (LCC)',
    'mind':              'MIND',
    'brexit':            'Brexit',
    'iphone_samsung':    'iPhone-Samsung',
    'lastfm_asia':       'LastFM-Asia',
}

# (display_label, mean_col, is_compound, extra_cols)
# extra_cols is a dict for compound columns (rank1): {key: col_name}
METRIC_DEFS: Dict[str, Tuple] = {
    'mrr':       ('MRR',                  'mrr_mean',                        False, None),
    'rank1':     (r'Rank-1 \#',           'num_rank1_nodes_mean',            True,
                  {'min': 'num_rank1_nodes_min', 'max': 'num_rank1_nodes_max'}),
    'wkendall':  (r'Wtd.\ Ken.',          'weighted_kendall_tau_mean',       False, None),
    'kendall':   (r'Kendall $\tau$',      'kendall_correlation_mean',        False, None),
    # ARHR absolute (arhr@1 = MRR, shown in TA via 'mrr' key)
    'arhr3':     ('ARHR@3',               'arhr_at_3_mean',                  False, None),
    'arhr5':     ('ARHR@5',               'arhr_at_5_mean',                  False, None),
    'arhr10':    ('ARHR@10',              'arhr_at_10_mean',                 False, None),
    # ARHR pct (arhr@1% = MRR, shown separately)
    'arhr3pct':  (r'ARHR@3\%',           'arhr_at_3pct_mean',               False, None),
    'arhr5pct':  (r'ARHR@5\%',           'arhr_at_5pct_mean',               False, None),
    'arhr10pct': (r'ARHR@10\%',          'arhr_at_10pct_mean',              False, None),
    # Precision absolute
    'prec1':     ('P@1',                  'precision_snr_at_1_mean',         False, None),
    'prec3':     ('P@3',                  'precision_snr_at_3_mean',         False, None),
    'prec5':     ('P@5',                  'precision_snr_at_5_mean',         False, None),
    'prec10':    ('P@10',                 'precision_snr_at_10_mean',        False, None),
    # Precision pct
    'prec1pct':  (r'P@1\%',              'precision_snr_at_1pct_mean',      False, None),
    'prec3pct':  (r'P@3\%',              'precision_snr_at_3pct_mean',      False, None),
    'prec5pct':  (r'P@5\%',              'precision_snr_at_5pct_mean',      False, None),
    'prec10pct': (r'P@10\%',             'precision_snr_at_10pct_mean',     False, None),
}

# (table_id, beta_scope, row_scope, metric_keys, file_stem)
TABLE_SPECS = [
    ('T2',     'none_only', 'selected', ['mrr', 'rank1'],
     'T2_mrr_rank1'),
    ('T3',     'none_only', 'selected', ['wkendall', 'kendall'],
     'T3_kendall'),
    ('T4',     'none_only', 'selected', ['prec5pct'],
     'T4_prec5pct'),
    ('TA',     'all_betas', 'all',      ['mrr', 'rank1'],
     'TA_mrr_rank1'),
    ('TB',     'all_betas', 'all',      ['wkendall', 'kendall'],
     'TB_kendall'),
    ('TC-abs', 'all_betas', 'all',      ['prec1', 'prec3', 'prec5', 'prec10'],
     'TC_prec_abs'),
    ('TC-pct', 'all_betas', 'all',      ['prec1pct', 'prec3pct', 'prec5pct', 'prec10pct'],
     'TC_prec_pct'),
    ('TE-abs', 'all_betas', 'all',      ['arhr3', 'arhr5', 'arhr10'],
     'TE_arhr_abs'),
    ('TE-pct', 'all_betas', 'all',      ['arhr3pct', 'arhr5pct', 'arhr10pct'],
     'TE_arhr_pct'),
]


# ─── Data Loading ─────────────────────────────────────────────────────────────


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure method is the index."""
    if df.index.name != 'method' and 'method' in df.columns:
        df = df.set_index('method')
    return df


def load_dataset_overall(agg_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Returns {beta_label: df(index=method)} for all available betas.
    Tries betas_*/level1_overall first (new structure).
    Falls back to level1_overall at root (old flat structure) → {'none': df}.
    """
    result: Dict[str, pd.DataFrame] = {}

    # Try new betas structure
    betas_dirs = sorted(agg_dir.glob('betas_*/'),
                        key=lambda d: (-1.0 if d.name == 'betas_none'
                                       else float(d.name.replace('betas_', ''))))
    for bdir in betas_dirs:
        label = bdir.name.replace('betas_', '')
        f = _find_file(bdir, 'level1_overall')
        if f is not None:
            df = _normalize_index(_read_df(f))
            result[label] = df

    if result:
        return result

    # Flat fallback
    f = _find_file(agg_dir, 'level1_overall')
    if f is not None:
        df = _normalize_index(_read_df(f))
        result['none'] = df

    return result


# ─── Method Selection ─────────────────────────────────────────────────────────

def _top_k_with_ties(series: pd.Series, k: int = 3) -> List[str]:
    """Return index labels of top-k values, expanding ties at the k-th position."""
    s = series.sort_values(ascending=False)
    if s.empty:
        return []
    threshold = s.iloc[min(k - 1, len(s) - 1)]
    return s[s >= threshold].index.tolist()


def select_methods_for_table(df: pd.DataFrame, metric_keys: List[str]) -> List[str]:
    """
    Returns ['lexipeeling'] + union of top-3 (with tie expansion) per metric_key.
    Candidates: methods where not _is_composite(m) (this also excludes lexipeeling_extended).
    lexipeeling is always first (excluded from the top-3 candidate pool).
    """
    if df.empty:
        return []

    result: List[str] = []
    if 'lexipeeling' in df.index:
        result.append('lexipeeling')

    candidates = [m for m in df.index if not _is_composite(m) and m != 'lexipeeling']

    for mk in metric_keys:
        if mk == 'rank1':
            # Use mrr as proxy for rank1 selection
            mean_col = METRIC_DEFS['mrr'][1]
        else:
            mean_col = METRIC_DEFS[mk][1]
        if mean_col not in df.columns:
            continue
        sub = df.loc[[m for m in candidates if m in df.index], mean_col].dropna()
        for m in _top_k_with_ties(sub):
            if m not in result:
                result.append(m)

    return result


def get_all_methods_ordered(
    methods: List[str],
    methods_groups: Optional[List[dict]],
) -> Tuple[List[str], List[Tuple[int, str]]]:
    """
    Returns (ordered_methods, group_boundaries).
    Order: lexi variants (alpha) → groups from YAML (alpha within) → ungrouped (alpha).
    group_boundaries = [(row_index, group_name)] for longtable separators.
    """
    method_set = set(methods)

    # Group 0: lexi variants
    lexi = sorted([m for m in method_set if 'lexipeeling' in m])
    ordered: List[str] = list(lexi)
    placed = set(lexi)
    boundaries: List[Tuple[int, str]] = [(0, 'LexiPeeling variants')]

    if methods_groups:
        for group in methods_groups:
            gname = group.get('name', 'Other')
            gms = sorted([m for m in group.get('methods', []) if m in method_set and m not in placed])
            if gms:
                boundaries.append((len(ordered), gname))
                ordered.extend(gms)
                placed.update(gms)

    # Ungrouped
    leftover = sorted([m for m in method_set if m not in placed])
    if leftover:
        boundaries.append((len(ordered), 'Other'))
        ordered.extend(leftover)

    return ordered, boundaries


# ─── Cell Formatting ──────────────────────────────────────────────────────────

def format_cell(df: pd.DataFrame, method: str, metric_key: str) -> str:
    """Return a formatted string for one table cell. '---' if missing or NaN."""
    label, mean_col, is_compound, extra = METRIC_DEFS[metric_key]

    if df.empty or method not in df.index or mean_col not in df.columns:
        return '---'

    val = df.at[method, mean_col]
    if pd.isna(val):
        return '---'

    if is_compound:
        # rank1: "mean [min–max]"
        min_col = extra['min']
        max_col = extra['max']
        min_v = df.at[method, min_col] if min_col in df.columns else float('nan')
        max_v = df.at[method, max_col] if max_col in df.columns else float('nan')
        s = f'{val:.1f}'
        if not pd.isna(min_v) and not pd.isna(max_v):
            s += f' [{int(min_v)}--{int(max_v)}]'
        return s

    # Scalar: 3dp for kendall, 4dp for everything else
    if metric_key in ('kendall', 'wkendall'):
        return f'{val:.3f}'
    return f'{val:.4f}'


def apply_bold_best(
    cells: Dict[Tuple, str],
    methods: List[str],
    ds_key: str,
    metric_key: str,
) -> None:
    """Wrap best value(s) with \\textbf{} and second-best with \\underline{}."""
    if metric_key == 'rank1':
        return  # compound string — skip

    vals: Dict[str, float] = {}
    for m in methods:
        raw = cells.get((m, ds_key, metric_key), '---')
        if raw == '---':
            continue
        try:
            vals[m] = float(raw)
        except ValueError:
            continue

    if not vals:
        return

    sorted_unique = sorted(set(vals.values()), reverse=True)
    best_val = sorted_unique[0]
    second_val = sorted_unique[1] if len(sorted_unique) > 1 else None

    for m, v in vals.items():
        old = cells[(m, ds_key, metric_key)]
        if v == best_val:
            cells[(m, ds_key, metric_key)] = r'\textbf{' + old + '}'
        elif second_val is not None and v == second_val:
            cells[(m, ds_key, metric_key)] = r'\underline{' + old + '}'


# ─── LaTeX Helpers ────────────────────────────────────────────────────────────

def _escape_latex(s: str) -> str:
    """Escape underscores and percent signs in text strings."""
    s = s.replace('_', r'\_')
    s = s.replace('%', r'\%')
    return s


def _method_display(method: str) -> str:
    return _escape_latex(_short(method, max_len=28))


def _col_spec(metric_keys: List[str], n_datasets: int) -> str:
    """Build LaTeX column spec: l followed by r (or l for rank1) per cell."""
    cols = ['l']
    for _ in range(n_datasets):
        for mk in metric_keys:
            cols.append('l' if mk == 'rank1' else 'r')
    return ''.join(cols)


def _header_rows(
    dataset_keys: List[str],
    dataset_display_names: Dict[str, str],
    metric_keys: List[str],
) -> str:
    """
    Build table header rows.
    - If multiple datasets: two rows (dataset multicolumns + metric sub-headers).
    - If single dataset: one row (metric sub-headers only, no dataset name).
    """
    n_m = len(metric_keys)
    n_ds = len(dataset_keys)

    if n_ds > 1:
        # Row 1: dataset multicolumns with cmidrules
        parts1 = ['']
        cmidrules = []
        col_start = 2
        for ds_key in dataset_keys:
            dname = dataset_display_names[ds_key]
            parts1.append(r'\multicolumn{' + str(n_m) + r'}{c}{' + dname + r'}')
            cmidrules.append(
                r'\cmidrule(lr){' + str(col_start) + '-' + str(col_start + n_m - 1) + r'}')
            col_start += n_m
        row1 = ' & '.join(parts1) + r' \\' + '\n' + ' '.join(cmidrules)
        # Row 2: metric sub-headers
        parts2 = ['Method']
        for _ in dataset_keys:
            for mk in metric_keys:
                parts2.append(METRIC_DEFS[mk][0])
        row2 = ' & '.join(parts2) + r' \\'
        return row1 + '\n' + row2
    else:
        # Single dataset: just metric column headers
        parts = ['Method'] + [METRIC_DEFS[mk][0] for mk in metric_keys]
        return ' & '.join(parts) + r' \\'


def render_latex_tabular(
    ordered_methods: List[str],
    dataset_keys: List[str],
    dataset_display_names: Dict[str, str],
    metric_keys: List[str],
    cells: Dict[Tuple, str],
) -> str:
    """Render a small (selected-methods) table using tabular + booktabs. No caption/label."""
    col_spec = _col_spec(metric_keys, len(dataset_keys))
    header = _header_rows(dataset_keys, dataset_display_names, metric_keys)

    lines = [
        r'\begin{tabular}{' + col_spec + '}',
        r'\toprule',
        header,
        r'\midrule',
    ]

    for method in ordered_methods:
        row_parts = [_method_display(method)]
        for ds_key in dataset_keys:
            for mk in metric_keys:
                row_parts.append(cells.get((method, ds_key, mk), '---'))
        lines.append(' & '.join(row_parts) + r' \\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
    ]
    return '\n'.join(lines) + '\n'


def render_latex_longtable(
    ordered_methods: List[str],
    dataset_keys: List[str],
    dataset_display_names: Dict[str, str],
    metric_keys: List[str],
    cells: Dict[Tuple, str],
    group_boundaries: List[Tuple[int, str]],
) -> str:
    """Render a large (all-methods) table using longtable + booktabs. No caption/label."""
    n_cols = 1 + len(dataset_keys) * len(metric_keys)
    col_spec = _col_spec(metric_keys, len(dataset_keys))
    header = _header_rows(dataset_keys, dataset_display_names, metric_keys)

    boundary_map = {idx: name for idx, name in group_boundaries}

    header_block = (
        r'\toprule' + '\n' +
        header + '\n' +
        r'\midrule'
    )

    lines = [
        r'\begin{longtable}{' + col_spec + '}',
        header_block,
        r'\endfirsthead',
        r'\multicolumn{' + str(n_cols) + r'}{l}{\textit{(continued from previous page)}} \\',
        header_block,
        r'\endhead',
        r'\midrule',
        r'\multicolumn{' + str(n_cols) + r'}{r}{\textit{continued\ldots}} \\',
        r'\endfoot',
        r'\bottomrule',
        r'\endlastfoot',
    ]

    for i, method in enumerate(ordered_methods):
        if i in boundary_map:
            if i > 0:
                lines.append(r'\midrule')
            gname = _escape_latex(boundary_map[i])
            lines.append(
                r'\multicolumn{' + str(n_cols) + r'}{l}{\textit{' + gname + r'}} \\'
            )

        row_parts = [_method_display(method)]
        for ds_key in dataset_keys:
            for mk in metric_keys:
                row_parts.append(cells.get((method, ds_key, mk), '---'))
        lines.append(' & '.join(row_parts) + r' \\')

    lines.append(r'\end{longtable}')
    return '\n'.join(lines) + '\n'


# ─── Graph Stats (T1) ─────────────────────────────────────────────────────────

def compute_graph_stats(graph_dir: Path) -> dict:
    """
    Build an nx.Graph from edge/group files.
    Returns {n_nodes, n_edges, avg_degree, max_degree, max_coreness, assortativity};
    None for missing fields.
    """
    import networkx as nx

    stats: dict = {k: None for k in [
        'n_nodes', 'n_edges', 'avg_degree', 'max_degree', 'max_coreness', 'n_groups', 'assortativity']}

    ds_name = graph_dir.name
    edge_files = list(graph_dir.glob(f'{ds_name}_edges.txt')) + list(graph_dir.glob(f'{ds_name}_edges.csv'))
    if not edge_files:
        # Fallback to wildcard if dataset-specific name not found
        edge_files = list(graph_dir.glob('*_edges.txt')) + list(graph_dir.glob('*_edges.csv'))
    if not edge_files:
        return stats

    # Build graph from edge file
    G = nx.Graph()
    try:
        with open(edge_files[0]) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    G.add_edge(parts[0], parts[1])
    except Exception:
        return stats

    if G.number_of_nodes() < 2:
        return stats

    stats['n_nodes'] = G.number_of_nodes()
    stats['n_edges'] = G.number_of_edges()
    stats['avg_degree'] = 2 * stats['n_edges'] / stats['n_nodes']
    stats['max_degree'] = max(d for _, d in G.degree())
    stats['max_coreness'] = max(nx.core_number(G).values())

    # Load group attributes for assortativity
    group_files = list(graph_dir.glob(f'{ds_name}_groups.txt')) + list(graph_dir.glob(f'{ds_name}_groups.csv'))
    if not group_files:
        group_files = list(graph_dir.glob('*_groups.txt')) + list(graph_dir.glob('*_groups.csv'))
    if group_files:
        try:
            node_groups = {}
            with open(group_files[0]) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) == 1:
                        # single column: group value, node = line index
                        # (not expected here, but handle gracefully)
                        pass
                    elif len(parts) >= 2:
                        node_groups[parts[0]] = parts[1]
            stats['n_groups'] = len(set(node_groups.values()))
            nx.set_node_attributes(G, node_groups, 'group')
            import math
            assort = nx.attribute_assortativity_coefficient(G, 'group')
            if not math.isnan(assort):
                stats['assortativity'] = assort
        except Exception:
            pass

    return stats


def generate_t1_graph_stats(datasets_config: dict, output_dir: Path) -> None:
    """Write T1_graph_stats.tex and T1_graph_stats.csv inside main/."""
    print('\nT1: Graph statistics...')

    rows = []
    for ds_key, cfg in datasets_config.items():
        st = compute_graph_stats(cfg['graph_dir'])

        # Count unique rankings from aggregated by_ranking.csv
        n_rankings = None
        agg_dir = cfg.get('aggregated_dir')
        if agg_dir:
            agg_dir = Path(agg_dir)
            # Try betas_none subdir first, then direct
            for candidate in [agg_dir / 'betas_none' / 'by_ranking.csv', agg_dir / 'by_ranking.csv']:
                if candidate.exists():
                    try:
                        br = pd.read_csv(candidate, usecols=['ranking'], nrows=100000)
                        n_rankings = br['ranking'].nunique()
                    except Exception:
                        pass
                    break

        rows.append({
            'Dataset': cfg['display_name'],
            '|V|': st['n_nodes'],
            '|E|': st['n_edges'],
            'Avg. degree': st['avg_degree'],
            'Max degree': st['max_degree'],
            'Max coreness': st['max_coreness'],
            '|Groups|': st['n_groups'],
            '|Rankings|': n_rankings,
            'Assortativity': st['assortativity'],
        })

    df = pd.DataFrame(rows).set_index('Dataset')

    sub_dir = output_dir / 'main'
    sub_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = sub_dir / 'T1_graph_stats.csv'
    df.to_csv(csv_path)
    print(f'  Written: main/{csv_path.name}')

    # LaTeX
    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{Graph Statistics}',
        r'\label{tab:graph_stats}',
        r'\begin{tabular}{lrrrrrrrr}',
        r'\toprule',
        r'Dataset & $|V|$ & $|E|$ & Avg.\ deg. & Max deg. & Max core. & $|G|$ & $|\mathcal{R}|$ & Assort. \\',
        r'\midrule',
    ]
    for _, row in df.iterrows():
        def _fmt(v, fmt):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return '---'
            return fmt.format(int(v) if fmt.endswith('d}') else v)

        lines.append(
            f'{row.name} & '
            f'{_fmt(row["|V|"], "{:,d}")} & '
            f'{_fmt(row["|E|"], "{:,d}")} & '
            f'{_fmt(row["Avg. degree"], "{:.2f}")} & '
            f'{_fmt(row["Max degree"], "{:,d}")} & '
            f'{_fmt(row["Max coreness"], "{:,d}")} & '
            f'{_fmt(row["|Groups|"], "{:,d}")} & '
            f'{_fmt(row["|Rankings|"], "{:,d}")} & '
            f'{_fmt(row["Assortativity"], "{:.3f}")} \\\\'
        )
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    tex_path = sub_dir / 'T1_graph_stats.tex'
    tex_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'  Written: main/{tex_path.name}')


# ─── Master Table Generator ───────────────────────────────────────────────────

def _build_csv(
    ordered_methods: List[str],
    dataset_keys: List[str],
    dataset_display_names: Dict[str, str],
    metric_keys: List[str],
    dfs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build a raw CSV DataFrame with float values (not LaTeX strings)."""
    single_ds = len(dataset_keys) == 1
    records = []
    for method in ordered_methods:
        row: dict = {'method': method}
        for ds_key in dataset_keys:
            dname = dataset_display_names[ds_key]
            df = dfs.get(ds_key, pd.DataFrame())
            for mk in metric_keys:
                label, mean_col, is_compound, extra = METRIC_DEFS[mk]
                col_header = label if single_ds else f'{dname}__{label}'
                if not df.empty and method in df.index and mean_col in df.columns:
                    row[col_header] = df.at[method, mean_col]
                    if is_compound and extra:
                        for subkey, subcol in extra.items():
                            if subcol in df.columns:
                                sub_col_header = f'{col_header}__{subkey}' if not single_ds else f'{label}__{subkey}'
                                row[sub_col_header] = df.at[method, subcol]
                else:
                    row[col_header] = float('nan')
        records.append(row)
    return pd.DataFrame(records)


def generate_table(
    table_id: str,
    beta_label: str,
    metric_keys: List[str],
    row_scope: str,
    all_data: Dict[str, Dict[str, pd.DataFrame]],
    dataset_keys: List[str],
    dataset_display_names: Dict[str, str],
    methods_groups: Optional[List[dict]],
    output_dir: Path,
    file_stem: str,
) -> None:
    print(f'  {table_id} β={beta_label} ...', end=' ', flush=True)

    # Collect per-dataset DFs for this beta
    dfs: Dict[str, pd.DataFrame] = {}
    for ds_key in dataset_keys:
        beta_data = all_data.get(ds_key, {})
        df = beta_data.get(beta_label, pd.DataFrame())
        dfs[ds_key] = df

    # For all-methods tables: augment each dataset's df with betas_lexipeeling_* rows
    # from other betas subdirs (they don't appear in betas_none data).
    if row_scope == 'all':
        for ds_key in dataset_keys:
            all_beta_data = all_data.get(ds_key, {})
            extra: Dict[str, pd.DataFrame] = {}
            for other_df in all_beta_data.values():
                if other_df.empty:
                    continue
                for m in other_df.index:
                    if _BETAS_LEXI_RE.match(m) and m not in dfs[ds_key].index and m not in extra:
                        extra[m] = other_df.loc[[m]]
            if extra:
                dfs[ds_key] = pd.concat([dfs[ds_key]] + list(extra.values()))

    # Union of all methods
    all_methods: List[str] = []
    seen: set = set()
    for df in dfs.values():
        if not df.empty:
            for m in df.index:
                if m not in seen:
                    all_methods.append(m)
                    seen.add(m)

    if not all_methods:
        print('(no data, skipping)')
        return

    # Determine row ordering
    if row_scope == 'selected':
        # Use first non-empty dataset's df for selection
        ref_df = next((df for df in dfs.values() if not df.empty), pd.DataFrame())
        ordered_methods = select_methods_for_table(ref_df, metric_keys)
        group_boundaries: List[Tuple[int, str]] = []
        # Sort by first dataset × first non-rank1 metric, descending
        sort_mk = next((mk for mk in metric_keys if mk != 'rank1'), None)
        if sort_mk:
            sort_col = METRIC_DEFS[sort_mk][1]
            first_df = dfs.get(dataset_keys[0], pd.DataFrame()) if dataset_keys else pd.DataFrame()
            if not first_df.empty and sort_col in first_df.columns:
                def _sort_key(m: str) -> float:
                    if m in first_df.index:
                        v = first_df.at[m, sort_col]
                        return float(v) if not pd.isna(v) else float('-inf')
                    return float('-inf')
                ordered_methods = sorted(ordered_methods, key=_sort_key, reverse=True)
    else:
        ordered_methods, group_boundaries = get_all_methods_ordered(all_methods, methods_groups)

    # Build cell dict
    cells: Dict[Tuple, str] = {}
    for method in ordered_methods:
        for ds_key in dataset_keys:
            for mk in metric_keys:
                cells[(method, ds_key, mk)] = format_cell(dfs[ds_key], method, mk)

    # Bold best per (ds_key, metric_key) column
    for ds_key in dataset_keys:
        for mk in metric_keys:
            apply_bold_best(cells, ordered_methods, ds_key, mk)

    # Determine output subdirectory
    if row_scope == 'selected':
        sub_dir = output_dir / 'main'
    else:
        sub_dir = output_dir / 'appendix' / f'beta_{beta_label}'
    sub_dir.mkdir(parents=True, exist_ok=True)

    # Render LaTeX
    if row_scope == 'selected':
        latex = render_latex_tabular(
            ordered_methods, dataset_keys, dataset_display_names,
            metric_keys, cells,
        )
    else:
        latex = render_latex_longtable(
            ordered_methods, dataset_keys, dataset_display_names,
            metric_keys, cells, group_boundaries,
        )

    tex_path = sub_dir / f'{file_stem}.tex'
    tex_path.write_text(latex, encoding='utf-8')

    # CSV with raw floats
    csv_df = _build_csv(ordered_methods, dataset_keys, dataset_display_names, metric_keys, dfs)
    csv_path = sub_dir / f'{file_stem}.csv'
    csv_df.to_csv(csv_path, index=False)

    print(f'{len(ordered_methods)} rows → {tex_path.relative_to(output_dir)}')


# ─── Cross-Dataset MRR Table ──────────────────────────────────────────────────

# Default dataset list and path resolution
CROSS_DATASET_DEFAULTS = [
    ('blogs_gcc',         'sem_10'),
    ('books',             'sem_10'),
    ('email-eu-core_gcc', 'sem_10'),
    ('mind',              'sem_10'),
    ('brexit',            None),
    ('iphone_samsung',    None),
    ('lastfm_asia',      None),
]


def _resolve_betas_none_dir(dataset_name: str, base_dir: Path,
                            sem_subdir: Optional[str] = None) -> Optional[Path]:
    """Find the betas_none directory for a dataset under base_dir."""
    if sem_subdir:
        candidate = base_dir / dataset_name / sem_subdir / 'betas_none'
        if candidate.exists():
            return candidate
    candidate = base_dir / dataset_name / 'betas_none'
    if candidate.exists():
        return candidate
    # Flat layout (no betas_none wrapper)
    candidate = base_dir / dataset_name
    if candidate.exists() and _find_file(candidate, 'level1_overall') is not None:
        return candidate
    return None


def _select_methods_cross_dataset(dfs: Dict[str, pd.DataFrame]) -> List[str]:
    """Union of top-3 (with tie expansion) non-composite non-betas non-lexipeeling
    methods by MRR across all datasets, with lexipeeling first."""
    result: List[str] = ['lexipeeling']
    for ds_name, df in dfs.items():
        candidates = [m for m in df.index
                      if m != 'lexipeeling'
                      and not _is_composite(m)
                      and not m.startswith('betas_')]
        if 'mrr_mean' not in df.columns:
            continue
        sub = df.loc[[m for m in candidates if m in df.index], 'mrr_mean'].dropna()
        for m in _top_k_with_ties(sub):
            if m not in result:
                result.append(m)
    return result


def generate_cross_dataset_mrr_table(base_dir: Path, output_dir: Path,
                                     datasets: Optional[List[tuple]] = None) -> None:
    """Generate a cross-dataset MRR table (rows=datasets, cols=methods)."""
    if datasets is None:
        datasets = CROSS_DATASET_DEFAULTS

    print('\nCross-Dataset MRR Table...')

    # Load data
    dfs: Dict[str, pd.DataFrame] = {}
    for ds_name, sem_sub in datasets:
        betas_dir = _resolve_betas_none_dir(ds_name, base_dir, sem_subdir=sem_sub)
        if betas_dir is None:
            print(f'  Warning: skipping {ds_name} (not found under {base_dir})')
            continue
        f = _find_file(betas_dir, 'level1_overall')
        if f is None:
            print(f'  Warning: skipping {ds_name} (no level1_overall in {betas_dir})')
            continue
        df = _normalize_index(_read_df(f))
        dfs[ds_name] = df
        print(f'  Loaded {ds_name}: {len(df)} methods')

    if not dfs:
        print('  No datasets loaded, skipping cross-dataset MRR table.')
        return

    # Select methods
    methods = _select_methods_cross_dataset(dfs)
    print(f'  Selected {len(methods)} methods: {methods}')

    # Build table data
    rows_data = []
    for ds_name in dfs:
        df = dfs[ds_name]
        row = {'Dataset': DATASET_DISPLAY_NAMES.get(ds_name, ds_name)}
        for m in methods:
            if m in df.index and 'mrr_mean' in df.columns:
                mrr = df.loc[m, 'mrr_mean']
                rank1_col = 'num_rank1_nodes_mean'
                rank1 = df.loc[m, rank1_col] if rank1_col in df.columns else np.nan
                row[m] = (mrr, rank1)
            else:
                row[m] = (np.nan, np.nan)
        rows_data.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV output — combined "MRR (rank1)" cells like the LaTeX
    csv_rows = []
    for row in rows_data:
        csv_row = {'Dataset': row['Dataset']}
        for m in methods:
            mrr, rank1 = row[m]
            if np.isnan(mrr):
                csv_row[m] = '---'
            else:
                rank1_str = f'{rank1:.1f}' if not np.isnan(rank1) else '?'
                csv_row[m] = f'{mrr:.4f} ({rank1_str})'
        csv_rows.append(csv_row)
    csv_df = pd.DataFrame(csv_rows)
    csv_path = output_dir / 'cross_dataset_mrr.csv'
    csv_df.to_csv(csv_path, index=False)
    print(f'  Written: {csv_path.name}')

    # Find best and second-best MRR per dataset
    best_per_row: Dict[str, float] = {}
    second_per_row: Dict[str, float] = {}
    for row in rows_data:
        ds = row['Dataset']
        vals = [row[m][0] for m in methods if not np.isnan(row[m][0])]
        if vals:
            sorted_unique = sorted(set(vals), reverse=True)
            best_per_row[ds] = sorted_unique[0]
            if len(sorted_unique) > 1:
                second_per_row[ds] = sorted_unique[1]

    # LaTeX output
    def _escape(s: str) -> str:
        return s.replace('_', r'\_').replace('%', r'\%')

    method_headers = [_escape(_short(m, max_len=18)) for m in methods]
    n_cols = len(methods)

    tex_lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{MRR across datasets (avg.\ \#rank-1 nodes in parentheses). Methods: lexipeeling plus the union of top-3 non-variant baselines by MRR per dataset (with tie expansion).}',
        r'\label{tab:cross_dataset_mrr}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{l' + 'r' * n_cols + '}',
        r'\toprule',
        'Dataset & ' + ' & '.join(method_headers) + r' \\',
        r'\midrule',
    ]

    for row in rows_data:
        ds = row['Dataset']
        cells = [_escape(ds)]
        for m in methods:
            mrr, rank1 = row[m]
            if np.isnan(mrr):
                cell = '---'
            else:
                rank1_str = f'{rank1:.1f}' if not np.isnan(rank1) else '?'
                mrr_str = f'{mrr:.4f}'
                if ds in best_per_row and mrr == best_per_row[ds]:
                    mrr_str = r'\textbf{' + mrr_str + '}'
                elif ds in second_per_row and mrr == second_per_row[ds]:
                    mrr_str = r'\underline{' + mrr_str + '}'
                cell = f'{mrr_str} ({rank1_str})'
            cells.append(cell)
        tex_lines.append(' & '.join(cells) + r' \\')

    tex_lines += [r'\bottomrule', r'\end{tabular}', '}', r'\end{table}']

    tex_path = output_dir / 'cross_dataset_mrr.tex'
    tex_path.write_text('\n'.join(tex_lines) + '\n', encoding='utf-8')
    print(f'  Written: {tex_path.name}')


# ─── Cross-Dataset Cumulative Summary ────────────────────────────────────────

def generate_cross_dataset_cumulative_table(plots_dir: Path, output_dir: Path,
                                             datasets: Optional[List[tuple]] = None) -> None:
    """Aggregate P16b_cumulative_summary.csv across datasets to verify which
    method is best at each cumulative threshold."""
    if datasets is None:
        datasets = CROSS_DATASET_DEFAULTS

    print('\nCross-Dataset Cumulative Summary...')

    all_rows = []
    for ds_name, sem_sub in datasets:
        if sem_sub:
            csv_path = plots_dir / ds_name / sem_sub / 'P16b_cumulative_summary.csv'
        else:
            csv_path = plots_dir / ds_name / 'P16b_cumulative_summary.csv'
        if not csv_path.exists():
            print(f'  Warning: skipping {ds_name} ({csv_path} not found)')
            continue
        df = pd.read_csv(csv_path)
        df['dataset'] = ds_name
        all_rows.append(df)
        print(f'  Loaded {ds_name}: {len(df)} methods')

    if not all_rows:
        print('  No datasets loaded, skipping.')
        return

    combined = pd.concat(all_rows, ignore_index=True)
    thresholds = [c for c in combined.columns if c.startswith('pct_at_')]

    # Build summary: for each dataset × threshold, which method is best?
    output_dir.mkdir(parents=True, exist_ok=True)

    # Table: rows = methods, columns = dataset/threshold combos
    # Simpler: one CSV per threshold with methods as rows, datasets as columns
    for thr in thresholds:
        pivot = combined.pivot_table(index='method', columns='dataset', values=thr)
        # Reorder columns to match dataset order
        ds_order = [ds for ds, _ in datasets if ds in pivot.columns]
        pivot = pivot[ds_order]

        csv_path = output_dir / f'cross_dataset_cumulative_{thr}.csv'
        pivot.to_csv(csv_path)
        print(f'  Written: {csv_path.name}')

        # Print best method per dataset
        print(f'  --- {thr} ---')
        for ds in ds_order:
            col = pivot[ds].dropna()
            if col.empty:
                continue
            best = col.idxmax()
            print(f'    {ds}: best = {best} ({col[best]:.1f}%)')

        # Overall: which method is best most often?
        wins = {}
        for ds in ds_order:
            col = pivot[ds].dropna()
            if not col.empty:
                best = col.idxmax()
                wins[best] = wins.get(best, 0) + 1
        if wins:
            sorted_wins = sorted(wins.items(), key=lambda x: -x[1])
            print(f'    Winner: {sorted_wins[0][0]} ({sorted_wins[0][1]}/{len(ds_order)} datasets)')

    # Also save a single combined CSV for easy inspection
    combined_path = output_dir / 'cross_dataset_cumulative_all.csv'
    combined.to_csv(combined_path, index=False)
    print(f'  Written: {combined_path.name}')


# ─── SEM Discretization Ties Table ───────────────────────────────────────────

# Datasets with multiple sem values and their sem_percentile lists
_SEM_DATASETS = [
    ('blogs_gcc',         [0, 1, 10, 50]),
    ('books',             [0, 1, 10, 50]),
    ('email-eu-core_gcc', [0, 1, 10, 50]),
    ('mind',              [0, 1, 10, 50]),
]


def _find_file_in(directory: Path, stem: str) -> Optional[Path]:
    """Find stem.parquet or stem.csv in directory."""
    for ext in ['.parquet', '.csv']:
        p = directory / f'{stem}{ext}'
        if p.exists():
            return p
    return None


def _count_gt_ties_for_sem(node_ranks_path: Path) -> dict:
    """Count ground truth tie statistics from sir_rank_snr in a node_ranks file.

    Returns dict with keys: unique_mean, unique_std, n_nodes, n_rankings,
    max_tie_mean, max_tie_std, avg_tie_mean, avg_tie_std.
    """
    # Read one infection_prob to avoid loading redundant data
    if node_ranks_path.suffix == '.parquet':
        import pyarrow.parquet as pq
        try:
            pf = pq.ParquetFile(node_ranks_path)
        except Exception:
            return {}
        sample = pf.read_row_group(0, columns=['infection_prob'])
        sp = sample.column('infection_prob')[0].as_py()
        cols = ['method', 'ranking', 'infection_prob', 'sir_rank_snr', 'n_nodes']
        try:
            df = pd.read_parquet(node_ranks_path, columns=cols,
                                 filters=[('method', '==', 'coreness'),
                                          ('infection_prob', '==', sp)])
        except Exception:
            return {}
    else:
        chunks = []
        sample_prob = None
        for chunk in pd.read_csv(node_ranks_path,
                                  usecols=['method', 'ranking', 'infection_prob',
                                           'sir_rank_snr', 'n_nodes'],
                                  chunksize=300_000):
            sub = chunk[chunk['method'] == 'coreness']
            if sub.empty:
                continue
            if sample_prob is None:
                sample_prob = sub['infection_prob'].iloc[0]
            sub = sub[sub['infection_prob'] == sample_prob]
            if not sub.empty:
                chunks.append(sub)
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    if df.empty or 'sir_rank_snr' not in df.columns:
        return {}

    stats = []
    for _, grp in df.groupby('ranking'):
        n = int(grp['n_nodes'].iloc[0])
        u = grp['sir_rank_snr'].nunique()
        ties = grp['sir_rank_snr'].value_counts()
        stats.append({'unique': u, 'n': n, 'max_tie': ties.max(),
                       'avg_tie': ties.mean()})

    sdf = pd.DataFrame(stats)
    import numpy as np
    return {
        'unique_mean': sdf['unique'].mean(),
        'unique_std': sdf['unique'].std(),
        'n_nodes': int(sdf['n'].iloc[0]),
        'n_rankings': len(sdf),
        'max_tie_mean': sdf['max_tie'].mean(),
        'max_tie_std': sdf['max_tie'].std(),
        'avg_tie_mean': sdf['avg_tie'].mean(),
        'avg_tie_std': sdf['avg_tie'].std(),
    }


def generate_sem_ties_table(base_dir: Path, output_dir: Path) -> None:
    """Generate a table showing ground truth ties at each SEM percentile.

    Rows = datasets, columns = sem values.
    Cells = "unique_ranks / n_nodes (pct%)" averaged across rankings.
    """
    print('\nSEM Discretization Ties Table...')

    rows = []
    for ds_name, sem_values in _SEM_DATASETS:
        row = {'Dataset': _escape_latex(DATASET_DISPLAY_NAMES.get(ds_name, ds_name))}
        for sem in sem_values:
            betas_dir = base_dir / ds_name / f'sem_{sem}' / 'betas_none'
            nr = _find_file_in(betas_dir, 'node_ranks')
            if nr is None:
                row[f'sem_{sem}'] = '---'
                continue

            st = _count_gt_ties_for_sem(nr)
            if not st:
                row[f'sem_{sem}'] = '---'
                continue

            pct = st['unique_mean'] / st['n_nodes'] * 100
            row[f'sem_{sem}'] = (f'{st["unique_mean"]:.0f}'
                                 f'$\\pm${st["unique_std"]:.0f}'
                                 f' / {st["n_nodes"]}'
                                 f' ({pct:.1f}\\%)')
        rows.append(row)

    if not rows:
        print('  No data found, skipping.')
        return

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV (plain text version)
    csv_rows = []
    for ds_name, sem_values in _SEM_DATASETS:
        csv_row = {'dataset': ds_name}
        for sem in sem_values:
            betas_dir = base_dir / ds_name / f'sem_{sem}' / 'betas_none'
            nr = _find_file_in(betas_dir, 'node_ranks')
            if nr is None:
                continue
            st = _count_gt_ties_for_sem(nr)
            if not st:
                continue
            pct = st['unique_mean'] / st['n_nodes'] * 100
            csv_row[f'sem_{sem}_unique'] = f'{st["unique_mean"]:.0f}±{st["unique_std"]:.0f}'
            csv_row[f'sem_{sem}_n_nodes'] = st['n_nodes']
            csv_row[f'sem_{sem}_pct'] = f'{pct:.1f}'
            csv_row[f'sem_{sem}_max_tie'] = f'{st["max_tie_mean"]:.0f}±{st["max_tie_std"]:.0f}'
            csv_row[f'sem_{sem}_avg_tie'] = f'{st["avg_tie_mean"]:.1f}±{st["avg_tie_std"]:.1f}'
        csv_rows.append(csv_row)

    pd.DataFrame(csv_rows).to_csv(output_dir / 'T_sem_ties.csv', index=False)
    print(f'  Written: T_sem_ties.csv')

    # LaTeX
    sem_cols = [f'sem_{s}' for s in _SEM_DATASETS[0][1]]
    header = ' & '.join(['Dataset'] + [f'SEM {s}\\%' for s in _SEM_DATASETS[0][1]])
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Unique ground truth ranks after SEM discretization '
        r'(mean$\pm$std across rankings). '
        r'Higher SEM percentile $\to$ coarser bins $\to$ more ties.}',
        r'\label{tab:sem_ties}',
        f'\\begin{{tabular}}{{l{"r" * len(sem_cols)}}}',
        r'\toprule',
        header + r' \\',
        r'\midrule',
    ]
    for _, r in df.iterrows():
        cells = [r['Dataset']] + [r.get(c, '---') for c in sem_cols]
        lines.append(' & '.join(str(x) for x in cells) + r' \\')
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    tex = '\n'.join(lines) + '\n'
    (output_dir / 'T_sem_ties.tex').write_text(tex, encoding='utf-8')
    print(f'  Written: T_sem_ties.tex')


# ─── Cross-Dataset Appendix Tables ────────────────────────────────────────────

# Each spec: (file_stem, column_name, display_label, format_string, higher_is_better)
_CROSS_APPENDIX_SIMPLE_SPECS = [
    ('cross_prec_at_1',     'precision_snr_at_1_mean',     'P@1',       '{:.4f}', True),
    ('cross_prec_at_3',     'precision_snr_at_3_mean',     'P@3',       '{:.4f}', True),
    ('cross_prec_at_5',     'precision_snr_at_5_mean',     'P@5',       '{:.4f}', True),
    ('cross_prec_at_10',    'precision_snr_at_10_mean',    'P@10',      '{:.4f}', True),
    ('cross_prec_at_1pct',  'precision_snr_at_1pct_mean',  r'P@1\%',   '{:.4f}', True),
    ('cross_prec_at_3pct',  'precision_snr_at_3pct_mean',  r'P@3\%',   '{:.4f}', True),
    ('cross_prec_at_5pct',  'precision_snr_at_5pct_mean',  r'P@5\%',   '{:.4f}', True),
    ('cross_prec_at_10pct', 'precision_snr_at_10pct_mean', r'P@10\%',  '{:.4f}', True),
    ('cross_kendall',       'kendall_correlation_mean',    r'Kendall $\tau$', '{:.3f}', True),
    ('cross_wkendall',      'weighted_kendall_tau_mean',   r'Wtd.\ Kendall',  '{:.3f}', True),
]


def _render_cross_appendix_latex(
    ordered_methods: List[str],
    ds_names: List[str],
    cells: Dict[Tuple[str, str], str],
    raw_vals: Dict[Tuple[str, str], float],
    group_boundaries: List[Tuple[int, str]],
    caption: str,
    label: str,
    higher_better: bool = True,
    highlight: bool = True,
    ds_display: Optional[List[str]] = None,
) -> str:
    """Render a cross-dataset appendix longtable (methods as rows, datasets as cols).

    When highlight=True, best values get \\textbf{} and second-best get \\underline{}.
    ds_display: optional display names for column headers (same order as ds_names).
    """
    n_cols = 1 + len(ds_names)
    col_spec = 'l' + 'r' * len(ds_names)

    headers = ds_display if ds_display else ds_names
    ds_headers = [_escape_latex(ds) for ds in headers]
    header_row = 'Method & ' + ' & '.join(ds_headers) + r' \\'

    boundary_map = {idx: name for idx, name in group_boundaries}

    # Compute best and second-best per dataset from raw_vals
    best_val_per_ds: Dict[str, float] = {}
    second_val_per_ds: Dict[str, float] = {}
    if highlight and raw_vals:
        for ds in ds_names:
            ds_vals = [raw_vals[(m, ds)] for m in ordered_methods if (m, ds) in raw_vals]
            if not ds_vals:
                continue
            sorted_unique = sorted(set(ds_vals), reverse=higher_better)
            best_val_per_ds[ds] = sorted_unique[0]
            if len(sorted_unique) > 1:
                second_val_per_ds[ds] = sorted_unique[1]

    header_block = r'\toprule' + '\n' + header_row + '\n' + r'\midrule'

    lines = [
        r'\begin{longtable}{' + col_spec + '}',
        r'\caption{' + caption + r'}',
        r'\label{' + label + '}' + r' \\',
        header_block,
        r'\endfirsthead',
        r'\multicolumn{' + str(n_cols) + r'}{l}{\textit{(continued from previous page)}} \\',
        header_block,
        r'\endhead',
        r'\midrule',
        r'\multicolumn{' + str(n_cols) + r'}{r}{\textit{continued\ldots}} \\',
        r'\endfoot',
        r'\bottomrule',
        r'\endlastfoot',
    ]

    for i, method in enumerate(ordered_methods):
        if i in boundary_map:
            if i > 0:
                lines.append(r'\midrule')
            gname = _escape_latex(boundary_map[i])
            lines.append(
                r'\multicolumn{' + str(n_cols) + r'}{l}{\textit{' + gname + r'}} \\'
            )

        row_parts = [_method_display(method)]
        for ds in ds_names:
            cell = cells.get((method, ds), '---')
            if highlight and (method, ds) in raw_vals:
                v = raw_vals[(method, ds)]
                if ds in best_val_per_ds and v == best_val_per_ds[ds]:
                    cell = r'\textbf{' + cell + '}'
                elif ds in second_val_per_ds and v == second_val_per_ds[ds]:
                    cell = r'\underline{' + cell + '}'
            row_parts.append(cell)
        lines.append(' & '.join(row_parts) + r' \\')

    lines.append(r'\end{longtable}')
    return '\n'.join(lines) + '\n'


def _load_all_datasets_with_betas(
    base_dir: Path,
    datasets: List[tuple],
) -> Dict[str, pd.DataFrame]:
    """Load level1_overall for each dataset, augmenting with betas_lexipeeling
    methods from sibling betas_X/ directories."""
    dfs: Dict[str, pd.DataFrame] = {}
    for ds_name, sem_sub in datasets:
        betas_dir = _resolve_betas_none_dir(ds_name, base_dir, sem_subdir=sem_sub)
        if betas_dir is None:
            print(f'  Warning: skipping {ds_name} (betas_none not found)')
            continue
        f = _find_file(betas_dir, 'level1_overall')
        if f is None:
            print(f'  Warning: skipping {ds_name} (no level1_overall)')
            continue
        df = _normalize_index(_read_df(f))

        # Augment with betas_lexipeeling from sibling dirs
        parent = betas_dir.parent
        for sibling in sorted(parent.iterdir()):
            if not sibling.is_dir() or sibling == betas_dir:
                continue
            sib_f = _find_file(sibling, 'level1_overall')
            if sib_f is None:
                continue
            sib_df = _normalize_index(_read_df(sib_f))
            for m in sib_df.index:
                if _BETAS_LEXI_RE.match(m) and m not in df.index:
                    df = pd.concat([df, sib_df.loc[[m]]])

        dfs[ds_name] = df
        print(f'  Loaded {ds_name}: {len(df)} methods')
    return dfs


def _compute_cumulative_scores(
    base_dir: Path,
    datasets: List[tuple],
    methods: List[str],
    k_values: List[int],
    t_values: List[int],
    t_is_absolute: bool = True,
) -> Dict[Tuple[str, str, int, int], float]:
    """Compute cumulative scores from node_ranks for each (method, dataset, k, t).

    Args:
        k_values: method_rank thresholds (e.g. [1, 5, 10])
        t_values: GT rank thresholds — absolute (sir_rank_snr <= t) if t_is_absolute,
                  else percentage ((sir_rank_snr / n_nodes) * 100 <= t)
        t_is_absolute: if True, t is an absolute GT rank; if False, t is a percentage

    Returns:
        Dict[(method, dataset, k, t)] = percentage score
    """
    max_k = max(k_values)
    scores: Dict[Tuple[str, str, int, int], float] = {}
    for ds_name, sem_sub in datasets:
        betas_dir = _resolve_betas_none_dir(ds_name, base_dir, sem_subdir=sem_sub)
        if betas_dir is None:
            continue
        nr_path = _find_file(betas_dir, 'node_ranks')
        if nr_path is None:
            print(f'  Cumulative: {ds_name} — no node_ranks, skipping')
            continue
        print(f'  Cumulative: loading {ds_name}...')
        cols = ['method', 'method_rank', 'sir_rank_snr', 'n_nodes', 'ranking', 'infection_prob']
        if nr_path.suffix == '.parquet':
            df = pd.read_parquet(nr_path, columns=cols,
                                 filters=[('method_rank', '<=', max_k)])
        else:
            chunks = []
            for chunk in pd.read_csv(nr_path, usecols=cols, chunksize=500_000):
                chunks.append(chunk[chunk['method_rank'] <= max_k])
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

        if df.empty:
            continue

        for method in methods:
            method_df = df[df['method'] == method]
            if method_df.empty:
                continue
            n_nodes = method_df['n_nodes'].median()
            if pd.isna(n_nodes) or n_nodes == 0:
                continue

            for k in k_values:
                sub = method_df[method_df['method_rank'] <= k]
                if sub.empty:
                    continue
                per_instance = {t: [] for t in t_values}
                for _, rgroup in sub.groupby(['ranking', 'infection_prob']):
                    if t_is_absolute:
                        gt_vals = rgroup['sir_rank_snr'].values
                        for t in t_values:
                            per_instance[t].append((gt_vals <= t).mean())
                    else:
                        pct_ranks = (rgroup['sir_rank_snr'].values / n_nodes) * 100
                        for t in t_values:
                            per_instance[t].append((pct_ranks <= t).mean())
                for t in t_values:
                    if per_instance[t]:
                        scores[(method, ds_name, k, t)] = float(np.mean(per_instance[t]) * 100)

    return scores


def generate_cross_dataset_appendix_tables(
    base_dir: Path,
    output_dir: Path,
    methods_groups: Optional[List[dict]] = None,
    datasets: Optional[List[tuple]] = None,
) -> None:
    """Generate all cross-dataset appendix tables (rows=ALL methods, cols=datasets)."""
    if datasets is None:
        datasets = CROSS_DATASET_DEFAULTS

    print('\n' + '=' * 60)
    print('Cross-Dataset Appendix Tables')
    print('=' * 60)

    # 1. Load data
    dfs = _load_all_datasets_with_betas(base_dir, datasets)
    if not dfs:
        print('  No datasets loaded, skipping.')
        return

    ds_names = [ds for ds, _ in datasets if ds in dfs]
    ds_display = [DATASET_DISPLAY_NAMES.get(ds, ds) for ds in ds_names]

    # 2. Union of all methods, ordered
    all_methods_set: List[str] = []
    seen: set = set()
    for df in dfs.values():
        for m in df.index:
            if m not in seen:
                all_methods_set.append(m)
                seen.add(m)

    ordered_methods, group_boundaries = get_all_methods_ordered(all_methods_set, methods_groups)

    appendix_dir = output_dir / 'appendix'
    appendix_dir.mkdir(parents=True, exist_ok=True)

    # 3. Simple metric tables (precision, kendall, wkendall)
    for file_stem, col_name, display_label, fmt, higher_better in _CROSS_APPENDIX_SIMPLE_SPECS:
        print(f'\n  {file_stem}...')
        cells: Dict[Tuple[str, str], str] = {}
        raw_vals: Dict[Tuple[str, str], float] = {}
        for ds in ds_names:
            df = dfs[ds]
            for method in ordered_methods:
                if method in df.index and col_name in df.columns:
                    val = df.at[method, col_name]
                    if not pd.isna(val):
                        cells[(method, ds)] = fmt.format(val)
                        raw_vals[(method, ds)] = float(val)

        # CSV
        csv_rows = []
        for method in ordered_methods:
            row = {'method': method}
            for ds in ds_names:
                v = raw_vals.get((method, ds))
                row[ds] = v if v is not None else ''
            csv_rows.append(row)
        pd.DataFrame(csv_rows).to_csv(appendix_dir / f'{file_stem}.csv', index=False)

        # LaTeX
        caption_label = display_label.replace(r'\ ', ' ')
        latex = _render_cross_appendix_latex(
            ordered_methods, ds_names, cells, raw_vals, group_boundaries,
            caption=f'{caption_label} across datasets',
            label=f'tab:{file_stem}',
            higher_better=higher_better,
            ds_display=ds_display,
        )
        (appendix_dir / f'{file_stem}.tex').write_text(latex, encoding='utf-8')
        print(f'    {file_stem}.csv + .tex')

    # 4. MRR table (compound: MRR + avg rank-1)
    # Bold/underline applied only to MRR score, not the rank-1 suffix
    print('\n  cross_mrr_all...')
    cells_mrr_score: Dict[Tuple[str, str], str] = {}
    cells_mrr_suffix: Dict[Tuple[str, str], str] = {}
    raw_mrr: Dict[Tuple[str, str], float] = {}
    for ds in ds_names:
        df = dfs[ds]
        for method in ordered_methods:
            if method not in df.index:
                continue
            mrr = df.at[method, 'mrr_mean'] if 'mrr_mean' in df.columns else float('nan')
            r1 = df.at[method, 'num_rank1_nodes_mean'] if 'num_rank1_nodes_mean' in df.columns else float('nan')
            if not pd.isna(mrr):
                r1_str = f'{r1:.1f}' if not pd.isna(r1) else '?'
                cells_mrr_score[(method, ds)] = f'{mrr:.4f}'
                cells_mrr_suffix[(method, ds)] = f' ({r1_str})'
                raw_mrr[(method, ds)] = float(mrr)

    csv_rows = []
    for method in ordered_methods:
        row = {'method': method}
        for ds in ds_names:
            score = cells_mrr_score.get((method, ds), '')
            suffix = cells_mrr_suffix.get((method, ds), '')
            row[ds] = f'{score}{suffix}' if score else ''
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(appendix_dir / 'cross_mrr_all.csv', index=False)

    # Build highlighted cells: bold/underline the score part, append suffix
    cells_mrr_display: Dict[Tuple[str, str], str] = {}
    for ds in ds_names:
        ds_vals = {m: raw_mrr[(m, ds)] for m in ordered_methods if (m, ds) in raw_mrr}
        if not ds_vals:
            continue
        sorted_unique = sorted(set(ds_vals.values()), reverse=True)
        best_v = sorted_unique[0]
        second_v = sorted_unique[1] if len(sorted_unique) > 1 else None
        for m, v in ds_vals.items():
            score = cells_mrr_score[(m, ds)]
            suffix = cells_mrr_suffix[(m, ds)]
            if v == best_v:
                cells_mrr_display[(m, ds)] = r'\textbf{' + score + '}' + suffix
            elif second_v is not None and v == second_v:
                cells_mrr_display[(m, ds)] = r'\underline{' + score + '}' + suffix
            else:
                cells_mrr_display[(m, ds)] = score + suffix

    latex = _render_cross_appendix_latex(
        ordered_methods, ds_names, cells_mrr_display, raw_mrr, group_boundaries,
        caption='MRR across datasets --- all methods (avg.\\ \\#rank-1 nodes in parentheses)',
        label='tab:cross_mrr_all',
        highlight=False,  # already highlighted above
        ds_display=ds_display,
    )
    (appendix_dir / 'cross_mrr_all.tex').write_text(latex, encoding='utf-8')
    print('    cross_mrr_all.csv + .tex')

    # 5. Rank-1 bounds table (min, max)
    print('\n  cross_rank1_bounds...')
    cells_r1: Dict[Tuple[str, str], str] = {}
    raw_r1_min: Dict[Tuple[str, str], float] = {}
    for ds in ds_names:
        df = dfs[ds]
        for method in ordered_methods:
            if method not in df.index:
                continue
            mn = df.at[method, 'num_rank1_nodes_min'] if 'num_rank1_nodes_min' in df.columns else float('nan')
            mx = df.at[method, 'num_rank1_nodes_max'] if 'num_rank1_nodes_max' in df.columns else float('nan')
            if not pd.isna(mn) and not pd.isna(mx):
                cells_r1[(method, ds)] = f'[{int(mn)}, {int(mx)}]'
                raw_r1_min[(method, ds)] = float(mn)

    csv_rows = []
    for method in ordered_methods:
        row = {'method': method}
        for ds in ds_names:
            row[ds] = cells_r1.get((method, ds), '')
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(appendix_dir / 'cross_rank1_bounds.csv', index=False)

    latex = _render_cross_appendix_latex(
        ordered_methods, ds_names, cells_r1, raw_r1_min, group_boundaries,
        caption='Rank-1 node count bounds [min, max] across datasets',
        label='tab:cross_rank1_bounds',
        highlight=False,  # descriptive, not a score
        ds_display=ds_display,
    )
    (appendix_dir / 'cross_rank1_bounds.tex').write_text(latex, encoding='utf-8')
    print('    cross_rank1_bounds.csv + .tex')

    # 6. Cumulative score tables (absolute thresholds)
    cumul_k_values = [1, 5, 10]
    cumul_t_values = [1, 5, 10]
    print('\n  Computing cumulative scores from node_ranks...')
    cumul_scores = _compute_cumulative_scores(
        base_dir, datasets, ordered_methods,
        k_values=cumul_k_values, t_values=cumul_t_values, t_is_absolute=True,
    )

    for k in cumul_k_values:
        for t in cumul_t_values:
            stem = f'cross_cumul_k{k}_t{t}'
            print(f'\n  {stem}...')
            cells_c: Dict[Tuple[str, str], str] = {}
            raw_c: Dict[Tuple[str, str], float] = {}
            for ds in ds_names:
                for method in ordered_methods:
                    v = cumul_scores.get((method, ds, k, t))
                    if v is not None:
                        cells_c[(method, ds)] = f'{v:.1f}'
                        raw_c[(method, ds)] = v

            csv_rows = []
            for method in ordered_methods:
                row = {'method': method}
                for ds in ds_names:
                    v = raw_c.get((method, ds))
                    row[ds] = v if v is not None else ''
                csv_rows.append(row)
            pd.DataFrame(csv_rows).to_csv(appendix_dir / f'{stem}.csv', index=False)

            latex = _render_cross_appendix_latex(
                ordered_methods, ds_names, cells_c, raw_c, group_boundaries,
                caption=f'Cumulative score (k={k}, t={t}): percentage of method-rank-$\\leq${k} nodes with GT rank $\\leq${t}',
                label=f'tab:{stem}',
                ds_display=ds_display,
            )
            (appendix_dir / f'{stem}.tex').write_text(latex, encoding='utf-8')
            print(f'    {stem}.csv + .tex')

    print(f'\n  All appendix tables written to {appendix_dir}')

    # 7. Summary table: cumulative (k=1, t=1) for lexipeeling variants (main paper)
    _generate_cumul_variants_summary(appendix_dir, ds_names, ds_display,
                                      cumul_k_values, cumul_t_values)


def _generate_cumul_variants_summary(
    appendix_dir: Path,
    ds_names: List[str],
    ds_display: List[str],
    k_values: List[int] = None,
    t_values: List[int] = None,
) -> None:
    """Generate summary tables of cumulative scores for lexipeeling variants."""
    if k_values is None:
        k_values = [1]
    if t_values is None:
        t_values = [1]

    target_methods = [
        'degree', 'coreness', 'degree_lexicographic',
        'lexipeeling', 'lexipeeling_extended',
        'degree_lexipeeling', 'lexipeeling_degree',
        'coreness_lexipeeling', 'lexipeeling_coreness',
        'degree_lexicographic_lexipeeling', 'lexipeeling_degree_lexicographic',
    ]
    ds_disp_map = dict(zip(ds_names, ds_display))
    main_dir = appendix_dir.parent / 'main'
    main_dir.mkdir(parents=True, exist_ok=True)

    for k in k_values:
        for t in t_values:
            cumul_path = appendix_dir / f'cross_cumul_k{k}_t{t}.csv'
            if not cumul_path.exists():
                continue

            print(f'\n  cumul_variants_summary (k={k}, t={t})...')
            full_df = pd.read_csv(cumul_path).set_index('method')

            available = [m for m in target_methods if m in full_df.index]
            available_ds = [ds for ds in ds_names if ds in full_df.columns]

            # CSV
            rows_csv = []
            for m in available:
                csv_row = {'method': m}
                for ds in available_ds:
                    val = full_df.at[m, ds]
                    csv_row[ds_disp_map.get(ds, ds)] = f'{val:.1f}' if not pd.isna(val) else ''
                rows_csv.append(csv_row)

            stem = f'T_cumul_variants_k{k}_t{t}'
            pd.DataFrame(rows_csv).to_csv(main_dir / f'{stem}.csv', index=False)

            # Compute best/second-best per dataset
            raw_vals: Dict[Tuple[str, str], float] = {}
            for m in available:
                for ds in available_ds:
                    val = full_df.at[m, ds]
                    if not pd.isna(val):
                        raw_vals[(m, ds)] = float(val)

            best_per_ds: Dict[str, float] = {}
            second_per_ds: Dict[str, float] = {}
            for ds in available_ds:
                ds_vals = [raw_vals[(m, ds)] for m in available if (m, ds) in raw_vals]
                if ds_vals:
                    sorted_unique = sorted(set(ds_vals), reverse=True)
                    best_per_ds[ds] = sorted_unique[0]
                    if len(sorted_unique) > 1:
                        second_per_ds[ds] = sorted_unique[1]

            # LaTeX
            n_ds = len(available_ds)
            ds_headers = [_escape_latex(ds_disp_map.get(ds, ds)) for ds in available_ds]
            header_row = 'Method & ' + ' & '.join(ds_headers) + r' \\'

            lines = [
                r'\begin{table}[t]',
                r'\centering',
                r'\caption{Cumulative score (k=' + str(k) + r', t=' + str(t)
                + r'): percentage of method-rank-$\leq$' + str(k)
                + r' nodes with GT rank $\leq$' + str(t)
                + r'. Methods: three baselines and all lexipeeling composite variants.}',
                r'\label{tab:' + stem + '}',
                r'\resizebox{\textwidth}{!}{%',
                r'\begin{tabular}{l' + 'r' * n_ds + '}',
                r'\toprule',
                header_row,
                r'\midrule',
            ]

            baseline_methods = {'degree', 'coreness', 'degree_lexicographic'}
            added_sep = False
            for m in available:
                if m not in baseline_methods and not added_sep:
                    lines.append(r'\midrule')
                    added_sep = True

                row_parts = [_escape_latex(m)]
                for ds in available_ds:
                    val = raw_vals.get((m, ds))
                    if val is None:
                        row_parts.append('---')
                    else:
                        cell = f'{val:.1f}'
                        if ds in best_per_ds and val == best_per_ds[ds]:
                            cell = r'\textbf{' + cell + '}'
                        elif ds in second_per_ds and val == second_per_ds[ds]:
                            cell = r'\underline{' + cell + '}'
                        row_parts.append(cell)
                lines.append(' & '.join(row_parts) + r' \\')

            lines += [r'\bottomrule', r'\end{tabular}', '}', r'\end{table}']

            tex_out = main_dir / f'{stem}.tex'
            tex_out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            print(f'    main/{stem}.csv + .tex')


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Generate LaTeX tables for the LexiCoreness paper.')
    p.add_argument('--aggregated-dir', type=Path, nargs='+', required=True,
                   metavar='DIR',
                   help='One or more aggregated results directories')
    p.add_argument('--graphs-dir', type=Path, nargs='+', required=True,
                   metavar='DIR',
                   help='One or more graph directories (same order as --aggregated-dir)')
    p.add_argument('--output-dir', type=Path, required=True,
                   help='Directory to write .tex and .csv files')
    p.add_argument('--betas', nargs='+', default=None,
                   metavar='BETA',
                   help='Beta labels to generate appendix tables for '
                        '(default: all of none 0.01 0.1 0.5 0.99)')
    p.add_argument('--methods-groups-yaml', type=Path,
                   default=Path('methods-files/to_plot/methods_groups.yaml'),
                   help='YAML with method groups for row ordering in large tables')
    p.add_argument('--tables', nargs='+', default=None,
                   metavar='TABLE_ID',
                   help='Generate only specified table IDs (e.g. T2 TA TB). Default: all.')
    p.add_argument('--graph-stats-only', action='store_true',
                   help='Only generate T1 graph statistics table')
    p.add_argument('--config', type=Path, default=None,
                   help='Path to experiment config YAML. If sem_percentile is a list, '
                        'generates tables for each sem_{p}/ subdirectory.')
    p.add_argument('--cross-dataset-mrr', action='store_true',
                   help='Generate cross-dataset MRR table. Uses --aggregated-dir as '
                        'base directory (parent of dataset dirs).')
    p.add_argument('--cross-dataset-cumulative', action='store_true',
                   help='Aggregate P16b cumulative summaries across datasets. '
                        'Uses --aggregated-dir as plots base directory.')
    p.add_argument('--cross-dataset-appendix', action='store_true',
                   help='Generate all cross-dataset appendix tables (precision, '
                        'kendall, MRR, rank-1, cumulative). Uses --aggregated-dir as '
                        'base directory (parent of dataset dirs).')
    p.add_argument('--sem-ties', action='store_true',
                   help='Generate table showing ground truth ties at each SEM '
                        'percentile threshold.')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Cross-dataset MRR table mode
    if args.cross_dataset_mrr:
        base_dir = args.aggregated_dir[0]  # Use first aggregated-dir as base
        generate_cross_dataset_mrr_table(base_dir, args.output_dir)
        return

    # Cross-dataset cumulative summary mode
    if args.cross_dataset_cumulative:
        plots_dir = args.aggregated_dir[0]  # Use first arg as plots base directory
        generate_cross_dataset_cumulative_table(plots_dir, args.output_dir)
        return

    # Cross-dataset appendix tables mode
    if args.cross_dataset_appendix:
        base_dir = args.aggregated_dir[0]
        methods_groups = None
        if args.methods_groups_yaml and args.methods_groups_yaml.exists():
            with open(args.methods_groups_yaml) as f:
                methods_groups = yaml.safe_load(f).get('method_groups', [])
        generate_cross_dataset_appendix_tables(base_dir, args.output_dir, methods_groups)
        return

    # SEM discretization ties table
    if args.sem_ties:
        base_dir = args.aggregated_dir[0]
        generate_sem_ties_table(base_dir, args.output_dir)
        return

    # Resolve sem_percentile list from config
    sem_percentile_values = [None]
    if args.config and args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        sp = config.get('sem_percentile')
        if sp is not None:
            sem_percentile_values = sp if isinstance(sp, list) else [sp]
    use_subdirs = len(sem_percentile_values) > 1

    base_agg_dirs: List[Path] = args.aggregated_dir
    base_output_dir: Path = args.output_dir

    for sem_p in sem_percentile_values:
        if use_subdirs and sem_p is not None:
            agg_dirs = [d / f'sem_{int(sem_p)}' for d in base_agg_dirs]
            output_dir = base_output_dir / f'sem_{int(sem_p)}'
            print(f"\n{'='*70}")
            print(f"TABLES — SEM percentile: {int(sem_p)}")
            print(f"{'='*70}")
        else:
            agg_dirs = base_agg_dirs
            output_dir = base_output_dir

        # Skip if any aggregated dir is missing
        missing = [d for d in agg_dirs if not d.exists()]
        if missing:
            for d in missing:
                print(f"  Skipping: {d} does not exist")
            continue

        _run_tables(args, agg_dirs, output_dir)


def _run_tables(args, agg_dirs: 'List[Path]', output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets_config from CLI args
    graph_dirs: 'List[Path]' = args.graphs_dir
    if len(agg_dirs) != len(graph_dirs):
        raise ValueError(
            f'--aggregated-dir ({len(agg_dirs)}) and --graphs-dir ({len(graph_dirs)}) '
            'must have the same number of arguments')

    datasets_config: Dict[str, dict] = {}
    for agg_dir, graph_dir in zip(agg_dirs, graph_dirs):
        ds_key = graph_dir.name  # use graph directory basename as key
        datasets_config[ds_key] = {
            'aggregated_dir': agg_dir,
            'graph_dir': graph_dir,
            'display_name': DATASET_DISPLAY_NAMES.get(ds_key, ds_key),
        }

    dataset_keys = list(datasets_config.keys())
    dataset_display_names = {k: v['display_name'] for k, v in datasets_config.items()}
    print(f'Datasets: {dataset_keys}')

    # T1: graph stats (always)
    generate_t1_graph_stats(datasets_config, output_dir)

    if args.graph_stats_only:
        print('\nDone (graph-stats-only).')
        return

    # Load all data upfront
    print('\nLoading aggregated data...')
    all_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for ds_key, cfg in datasets_config.items():
        all_data[ds_key] = load_dataset_overall(cfg['aggregated_dir'])
        betas_found = list(all_data[ds_key].keys())
        n_methods = next((len(df) for df in all_data[ds_key].values() if not df.empty), 0)
        print(f'  {ds_key}: betas={betas_found}, n_methods={n_methods}')

    # Load method groups
    methods_groups: Optional[List[dict]] = None
    if args.methods_groups_yaml and args.methods_groups_yaml.exists():
        with open(args.methods_groups_yaml) as f:
            methods_groups = yaml.safe_load(f).get('method_groups', [])
        print(f'Method groups loaded: {[g["name"] for g in methods_groups]}')

    # Determine requested betas for appendix tables
    requested_betas = args.betas if args.betas else BETA_LABELS_ORDERED

    # Filter table specs if --tables specified
    requested_ids = set(args.tables) if args.tables else None
    specs = [s for s in TABLE_SPECS
             if requested_ids is None or s[0] in requested_ids]

    print(f'\nGenerating {len(specs)} table spec(s)...')
    for spec in specs:
        table_id, beta_scope, row_scope, metric_keys, stem = spec
        betas = ['none'] if beta_scope == 'none_only' else requested_betas
        for beta in betas:
            # main tables: plain stem; appendix tables: plain stem (beta label is the folder)
            file_stem = stem
            generate_table(
                table_id=table_id,
                beta_label=beta,
                metric_keys=metric_keys,
                row_scope=row_scope,
                all_data=all_data,
                dataset_keys=dataset_keys,
                dataset_display_names=dataset_display_names,
                methods_groups=methods_groups,
                output_dir=output_dir,
                file_stem=file_stem,
            )

    print(f'\nDone. Tables saved to {output_dir}')


if __name__ == '__main__':
    main()
