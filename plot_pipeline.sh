#!/usr/bin/env bash
set -e

# ── Plots ──

# datasets with sem_percentile config (auto-loops over sem_0, sem_1, sem_10, sem_50)
for ds in blogs_gcc books email-eu-core_gcc mind; do
    sir_flag=""
    [ -d "results/sir/$ds" ] && sir_flag="--sir-dir results/sir/$ds"
    python3 src/paper_plots.py \
        --aggregated-dir results/aggregated/$ds \
        --output-dir plots/$ds \
        --graph-name $ds \
        --graph-file data/graphs/$ds/${ds}_edges.txt \
        --config experiments/$ds/experiment_config.yaml \
        $sir_flag
done

# flat datasets (no sem)
for ds in brexit iphone_samsung lastfm_asia; do
    sir_flag=""
    if [ -d "results/sir/$ds/$ds" ]; then
        sir_flag="--sir-dir results/sir/$ds/$ds"
    elif [ -d "results/sir/$ds" ]; then
        sir_flag="--sir-dir results/sir/$ds"
    fi
    python3 src/paper_plots.py \
        --aggregated-dir results/aggregated/$ds \
        --output-dir plots/$ds \
        --graph-name $ds \
        --graph-file data/graphs/$ds/${ds}_edges.txt \
        $sir_flag
done

# ── Per-dataset tables ──

# datasets with sem_percentile config (auto-loops over sem_0, sem_1, sem_10, sem_50)
for ds in blogs_gcc books email-eu-core_gcc mind; do
    python3 src/paper_tables.py \
        --aggregated-dir results/aggregated/$ds \
        --graphs-dir data/graphs/$ds \
        --output-dir tables/$ds \
        --config experiments/$ds/experiment_config.yaml
done

# flat datasets (no sem)
for ds in brexit iphone_samsung lastfm_asia; do
    python3 src/paper_tables.py \
        --aggregated-dir results/aggregated/$ds \
        --graphs-dir data/graphs/$ds \
        --output-dir tables/$ds
done

# ── T1: all 7 datasets ──

python3 src/paper_tables.py \
    --aggregated-dir \
        results/aggregated/blogs_gcc/sem_10 \
        results/aggregated/books/sem_10 \
        results/aggregated/email-eu-core_gcc/sem_10 \
        results/aggregated/mind/sem_10 \
        results/aggregated/brexit \
        results/aggregated/iphone_samsung \
        results/aggregated/lastfm_asia \
    --graphs-dir \
        data/graphs/blogs_gcc \
        data/graphs/books \
        data/graphs/email-eu-core_gcc \
        data/graphs/mind \
        data/graphs/brexit \
        data/graphs/iphone_samsung \
        data/graphs/lastfm_asia \
    --output-dir tables/cross_dataset \
    --graph-stats-only

# ── Cross-dataset MRR (main paper table) ──

python3 src/paper_tables.py \
    --aggregated-dir results/aggregated \
    --graphs-dir data/graphs \
    --output-dir tables/cross_dataset \
    --cross-dataset-mrr

# ── Cross-dataset appendix (16 tables) ──

python3 src/paper_tables.py \
    --aggregated-dir results/aggregated \
    --graphs-dir data/graphs \
    --output-dir tables/cross_dataset \
    --cross-dataset-appendix

# ── SEM discretization ties table ──

python3 src/paper_tables.py \
    --aggregated-dir results/aggregated \
    --graphs-dir data/graphs \
    --output-dir tables/cross_dataset \
    --sem-ties

# ── Strategy comparison tables (appendix) ──

python3 src/generate_strategy_tables.py \
    --output-dir tables/cross_dataset/appendix

# ── P25: Cross-dataset worst-case cumulative drop plot ──
# Must run after --cross-dataset-appendix (which generates cross_cumul_k*_t*.csv)

mkdir -p plots/cross_dataset
python3 - <<'EOF'
from pathlib import Path
from src.paper_plots import plot_worst_case_drop
plot_worst_case_drop(Path('plots/cross_dataset'), top_n=5)
EOF
