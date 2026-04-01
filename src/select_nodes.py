#!/usr/bin/env python3
"""
Node Selection Script (Component 2.5)

Reads a methods results CSV file, a ranking, and an optional list of methods,
and selects a subset of candidate nodes based on a specified strategy.

This script's output is a simple text file containing one node ID per line,
which can be used as input for the SIR simulator.
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import List, Set, Optional

def get_nodes_from_methods_csv(
    methods_csv: Path,
    ranking: List[int],
    methods_to_consider: Optional[List[str]] = None,
    top_k: int = None,
    rank_1_only: bool = False,
    max_rank: int = None
) -> Set[int]:
    """
    Extract candidate nodes from methods results CSV.

    Args:
        methods_csv: Path to methods results CSV.
        ranking: Ranking to filter by.
        methods_to_consider: Optional list of methods to filter by.
        top_k: Get top K nodes by rank.
        rank_1_only: Only rank-1 nodes.
        max_rank: All nodes with rank <= max_rank.

    Returns:
        Set of node IDs.
    """
    ranking_str = '_'.join(map(str, ranking))

    if Path(methods_csv).suffix == '.parquet':
        import pyarrow.parquet as pq
        df_ranking = pq.read_table(
            methods_csv,
            filters=[('ranking', '==', ranking_str)],
            columns=['ranking', 'method', 'rank', 'node_id'],
        ).to_pandas()
    else:
        df = pd.read_csv(methods_csv)
        df_ranking = df[df['ranking'] == ranking_str]
    if df_ranking.empty:
        print(f"⚠️ Warning: No results found for ranking '{ranking_str}' in {methods_csv}")
        return set()

    # If specific methods are requested, filter the DataFrame further
    if methods_to_consider:
        print(f"    Considering only methods: {methods_to_consider}")
        df_ranking = df_ranking[df_ranking['method'].isin(methods_to_consider)]
        if df_ranking.empty:
            print(f"⚠️ Warning: No results found for the specified methods.")
            return set()

    candidates = set()
    if rank_1_only:
        # All nodes ranked #1 by ANY of the considered methods
        rank_1_nodes = df_ranking[df_ranking['rank'] == 1]['node_id'].unique()
        candidates = set(rank_1_nodes)
        print(f"    Strategy: rank-1 only for ranking {ranking}")
        print(f"    Found {len(candidates)} nodes.")

    elif max_rank is not None:
        # All nodes with rank <= max_rank by ANY of the considered methods
        top_nodes = df_ranking[df_ranking['rank'] <= max_rank]['node_id'].unique()
        candidates = set(top_nodes)
        print(f"    Strategy: max rank <= {max_rank} for ranking {ranking}")
        print(f"    Found {len(candidates)} nodes.")

    elif top_k is not None:
        # Top K nodes by minimum rank across the considered methods
        node_min_rank = df_ranking.groupby('node_id')['rank'].min().sort_values()
        top_nodes = node_min_rank.head(top_k).index.tolist()
        candidates = set(top_nodes)
        print(f"    Strategy: top-{top_k} for ranking {ranking}")
        print(f"    Found {len(candidates)} nodes.")

    else:
        raise ValueError("A selection strategy (top-k, rank-1-only, or max-rank) must be provided.")

    return candidates

def main():
    parser = argparse.ArgumentParser(
        description="Select candidate nodes from methods results.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--methods-csv",
        type=Path,
        required=True,
        help="Path to methods results CSV file."
    )
    parser.add_argument(
        "--ranking",
        nargs='+',
        type=int,
        required=True,
        help="Group ranking to filter by (e.g., 0 1)."
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to save the output list of node IDs."
    )
    parser.add_argument(
        "--methods",
        nargs='*',
        default=None,
        help="Optional: Specific methods to consider (e.g., lexipeeling coreness). If not provided, uses all."
    )

    # Selection strategy (mutually exclusive)
    strategy_group = parser.add_mutually_exclusive_group(required=True)
    strategy_group.add_argument(
        "--top-k",
        type=int,
        help="Select top K nodes by minimum rank across specified methods."
    )
    strategy_group.add_argument(
        "--rank-1-only",
        action="store_true",
        help="Select all nodes with rank 1 from any specified method."
    )
    strategy_group.add_argument(
        "--max-rank",
        type=int,
        help="Select all nodes with rank <= max_rank from any specified method."
    )

    args = parser.parse_args()

    # Create parent directory for output file if it doesn't exist
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get the set of nodes
    nodes_to_select = get_nodes_from_methods_csv(
        methods_csv=args.methods_csv,
        ranking=args.ranking,
        methods_to_consider=args.methods,
        top_k=args.top_k,
        rank_1_only=args.rank_1_only,
        max_rank=args.max_rank
    )

    # Write nodes to output file
    with open(args.output_file, 'w') as f:
        for node_id in sorted(list(nodes_to_select)):
            f.write(f"{node_id}\n")

    print(f"\n Selected {len(nodes_to_select)} nodes and saved to: {args.output_file}")

if __name__ == "__main__":
    main()
