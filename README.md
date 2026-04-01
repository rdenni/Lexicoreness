# Targeted Spreader Identification via Lexicographic Core Decomposition

Code written by Riccardo Denni.


## Installation

```bash
# Clone the repository
git clone https://github.com/rdenni/LexiCoreness.git
cd LexiCoreness

# Create a virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the Experiments

The experimental pipeline consists of the following steps:

### 1. Generate rankings

```bash
python3 src/generate_rankings.py \
    --graphs-dir data/graphs/{dataset} \
    --output-dir results/methods/{dataset} \
    --config experiments/{dataset}/experiment_config.yaml
```

### 2. Run centrality methods

```bash
python3 src/run_methods.py \
    --graphs-dir data/graphs/{dataset} \
    --output-dir results/methods/{dataset} \
    --methods-file methods-files/to_run/all.txt \
    --config experiments/{dataset}/experiment_config.yaml
```

### 3. Run SIR simulations

```bash
python3 src/run_sir_batch.py \
    --graphs-dir data/graphs/{dataset} \
    --output-dir results/sir/{dataset} \
    --config experiments/{dataset}/experiment_config.yaml
```

### 4. Compare methods against ground truth

```bash
python3 src/compare_batch.py \
    --methods-dir results/methods/{dataset} \
    --sir-dir results/sir/{dataset} \
    --output-dir results/comparison/{dataset} \
    --no-spread-filter \
    --config experiments/{dataset}/experiment_config.yaml
```

### 5. Aggregate results

```bash
python3 src/aggregate_comparisons.py \
    --comparison-dir results/comparison/{dataset} \
    --output-dir results/aggregated/{dataset} \
    --graphs-dir data/graphs/{dataset} \
    --methods-dir results/methods/{dataset} \
    --sir-dir results/sir/{dataset} \
    --config experiments/{dataset}/experiment_config.yaml
```

### 6. Generate tables and plots

```bash
bash plot_pipeline.sh
```

This generates all paper tables in `tables/` and plots in `plots/`.

## Repository Structure

```
src/
  lexicographic_utils.py    # Lexicographic vector operations
  methods.py                # All centrality methods (lexipeeling, composites, baselines)
  sir_simulator.py          # SIR epidemic simulation engine
  sir_evaluation.py         # Bridge between SIR and lexicographic ranking
  ranking_strategies.py     # Ranking generation strategies
  graph_generator.py        # Synthetic graph generation
  run_methods.py            # Run centrality methods on graphs
  run_sir.py                # Run SIR simulations
  run_sir_batch.py          # Batch SIR runner
  compare_results.py        # Compare methods vs ground truth
  compare_batch.py          # Batch comparison
  aggregate_comparisons.py  # Multi-level result aggregation
  generate_graphs.py        # Graph generation from config
  generate_rankings.py      # Ranking generation
  compute_sir_probabilities.py  # Epidemic threshold computation
  select_nodes.py           # Node selection for SIR
  paper_plots.py            # Generate paper plots
  paper_tables.py           # Generate paper tables
  generate_strategy_tables.py   # Strategy comparison tables
  extract_gcc.py            # Extract largest connected component
  graph_info.py             # Graph statistics
  other_baselines/
    FairLaR/                # FairLaR baseline (MIT license)

experiments/                # Experiment configurations (one per dataset)
methods-files/              # Method specification files
plot_pipeline.sh            # Plot and table generation script
requirements.txt            # Python dependencies
```

## External Baselines

The FairGD and AdaptGD baselines require code from [Wang et al. (2026)](https://github.com/HongLWang/Fair-Pagerank-via-Edge-Rewighting). To use them:

```bash
cd src/other_baselines
git clone https://github.com/HongLWang/Fair-Pagerank-via-Edge-Rewighting.git
cd Fair-Pagerank-via-Edge-Rewighting
git checkout 7882018
```

## Citation

If you use this code, please cite: citation will be provided soon

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The FairLaR baseline (`src/other_baselines/FairLaR/`) is also MIT licensed (Copyright 2020, Sotiris Tsioutsiouliklis).
