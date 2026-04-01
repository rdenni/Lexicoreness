#!/usr/bin/env python3
"""
Generate all graphs specified in experiment configuration
"""

import yaml
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from graph_generator import generate_graph_from_config


def load_config(config_path: Path) -> dict:
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_all_graphs(config_path: Path, output_dir: Path, tier: str = "all"):
    """
    Generate all graphs from configuration
    
    Args:
        config_path: Path to YAML config
        output_dir: Output directory for graphs
        tier: Which tier to generate ("sir_optimal", "fast_only", or "all")
    """
    config = load_config(config_path)
    
    num_replicates = config['global']['replicates_per_config']
    
    # Determine which graphs to generate
    if tier == "all":
        graph_configs = config['graphs']['sir_optimal'] + config['graphs']['fast_only']
    elif tier == "sir_optimal":
        graph_configs = config['graphs']['sir_optimal']
    elif tier == "fast_only":
        graph_configs = config['graphs']['fast_only']
    else:
        raise ValueError(f"Unknown tier: {tier}")
    
    print(f"Generating {len(graph_configs)} graph types × {num_replicates} replicates = {len(graph_configs) * num_replicates} total graphs")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Generate graphs
    for graph_config in graph_configs:
        print(f"\nGenerating: {graph_config['name']}")
        print("-" * 80)
        
        # Add global seed to config
        graph_config['random_seed'] = config['global']['random_seed']
        
        for rep_id in range(num_replicates):
            print(f"\n  Replicate {rep_id}:")
            generate_graph_from_config(
                config=graph_config,
                replicate_id=rep_id,
                output_dir=output_dir
            )
    
    print("\n" + "=" * 80)
    print(f"✓ All graphs generated successfully!")
    print(f"  Total files: {len(graph_configs) * num_replicates * 3}")  # 3 files per graph
    print(f"  Location: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic graphs for spreading experiments")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiment_config.yaml"),
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/random_graphs"),
        help="Output directory for generated graphs"
    )
    parser.add_argument(
        "--tier",
        choices=["all", "sir_optimal", "fast_only"],
        default="all",
        help="Which tier of graphs to generate"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    config_path = project_root / args.config
    output_dir = project_root / args.output
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    generate_all_graphs(config_path, output_dir, args.tier)


if __name__ == "__main__":
    main()
