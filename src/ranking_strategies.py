#!/usr/bin/env python3
"""
Ranking Strategy Generation

Generate group rankings based on various strategies:
- Size-based (top/bottom)
- Connectivity-based (coreness, degree)
- Mixed strategies
- Random permutations
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
import random


def compute_group_metrics(G: nx.Graph, group_assignments: Dict[int, int], num_groups: int) -> Dict[int, Dict]:
    """
    Compute metrics for each group

    Args:
        G: NetworkX graph
        group_assignments: Mapping node_id -> group_id
        num_groups: Number of groups (used for validation, actual group IDs from assignments)

    Returns:
        Dict mapping group_id -> {size, avg_coreness, avg_degree}
    """
    # Compute coreness for all nodes
    coreness = nx.core_number(G)

    # Get actual group IDs from the data (handles non-0-indexed groups)
    actual_group_ids = sorted(set(group_assignments.values()))

    metrics = {}
    for group_id in actual_group_ids:
        group_nodes = [n for n in G.nodes() if group_assignments[n] == group_id]

        if len(group_nodes) == 0:
            metrics[group_id] = {
                'size': 0,
                'avg_coreness': 0.0,
                'avg_degree': 0.0
            }
            continue

        # Size
        size = len(group_nodes)

        # Average coreness
        avg_coreness = np.mean([coreness[n] for n in group_nodes])

        # Average degree
        avg_degree = np.mean([G.degree(n) for n in group_nodes])

        metrics[group_id] = {
            'size': size,
            'avg_coreness': avg_coreness,
            'avg_degree': avg_degree
        }

    return metrics


def generate_rankings(group_metrics: Dict[int, Dict], s: int, num_random: int = 5) -> Dict[str, List[int]]:
    """
    Generate all ranking strategies for length s

    Args:
        group_metrics: Dict from compute_group_metrics
        s: Ranking length
        num_random: Number of random rankings to generate

    Returns:
        Dict mapping ranking_name -> ranking (list of group_ids)
    """
    num_groups = len(group_metrics)

    if s > num_groups:
        raise ValueError(f"Cannot create ranking of length {s} with only {num_groups} groups")

    rankings = {}

    # Sort groups by different metrics
    groups_by_size = sorted(group_metrics.keys(), key=lambda g: group_metrics[g]['size'], reverse=True)
    groups_by_coreness = sorted(group_metrics.keys(), key=lambda g: group_metrics[g]['avg_coreness'], reverse=True)
    groups_by_degree = sorted(group_metrics.keys(), key=lambda g: group_metrics[g]['avg_degree'], reverse=True)

    # 1. Size-based rankings
    rankings['top_size'] = groups_by_size[:s]
    rankings['bottom_size'] = groups_by_size[-s:]

    # 2. Connectivity-based rankings (coreness)
    rankings['top_coreness'] = groups_by_coreness[:s]
    rankings['bottom_coreness'] = groups_by_coreness[-s:]

    # 3. Connectivity-based rankings (degree)
    rankings['top_degree'] = groups_by_degree[:s]
    rankings['bottom_degree'] = groups_by_degree[-s:]

    # 4. Alternating rankings
    if s >= 2:
        alt_size = []
        for i in range(s):
            if i % 2 == 0:
                alt_size.append(groups_by_size[i // 2])
            else:
                alt_size.append(groups_by_size[-(i // 2 + 1)])
        rankings['alt_size'] = alt_size

        alt_core = []
        for i in range(s):
            if i % 2 == 0:
                alt_core.append(groups_by_coreness[i // 2])
            else:
                alt_core.append(groups_by_coreness[-(i // 2 + 1)])
        rankings['alt_coreness'] = alt_core

        alt_degree = []
        for i in range(s):
            if i % 2 == 0:
                alt_degree.append(groups_by_degree[i // 2])
            else:
                alt_degree.append(groups_by_degree[-(i // 2 + 1)])
        rankings['alt_degree'] = alt_degree

    # 5. Mixed size×coreness rankings
    if s >= 2:
        # [High-size, High-core]
        rankings['high_size_high_core'] = [groups_by_size[0], groups_by_coreness[0]][:s]

        # [High-size, Low-core]
        rankings['high_size_low_core'] = [groups_by_size[0], groups_by_coreness[-1]][:s]

        # [Low-size, High-core]
        rankings['low_size_high_core'] = [groups_by_size[-1], groups_by_coreness[0]][:s]

        # [Low-size, Low-core]
        rankings['low_size_low_core'] = [groups_by_size[-1], groups_by_coreness[-1]][:s]

    # 6. Mixed size×degree rankings
    if s >= 2:
        # [High-size, High-degree]
        rankings['high_size_high_degree'] = [groups_by_size[0], groups_by_degree[0]][:s]

        # [High-size, Low-degree]
        rankings['high_size_low_degree'] = [groups_by_size[0], groups_by_degree[-1]][:s]

        # [Low-size, High-degree]
        rankings['low_size_high_degree'] = [groups_by_size[-1], groups_by_degree[0]][:s]

        # [Low-size, Low-degree]
        rankings['low_size_low_degree'] = [groups_by_size[-1], groups_by_degree[-1]][:s]

    # 7. Mixed coreness×degree rankings
    if s >= 2:
        # [High-core, High-degree]
        rankings['high_core_high_degree'] = [groups_by_coreness[0], groups_by_degree[0]][:s]

        # [High-core, Low-degree]
        rankings['high_core_low_degree'] = [groups_by_coreness[0], groups_by_degree[-1]][:s]

        # [Low-core, High-degree]
        rankings['low_core_high_degree'] = [groups_by_coreness[-1], groups_by_degree[0]][:s]

        # [Low-core, Low-degree]
        rankings['low_core_low_degree'] = [groups_by_coreness[-1], groups_by_degree[-1]][:s]

    # 8. Random rankings
    # Use actual group IDs from group_metrics (handles non-0-indexed groups like 1-14)
    all_groups = list(group_metrics.keys())
    for i in range(num_random):
        random_ranking = random.sample(all_groups, s)
        rankings[f'random_{i+1}'] = random_ranking

    return rankings


def generate_rankings_for_lengths(group_metrics: Dict[int, Dict],
                                  s_values: List[int],
                                  num_random: int = 5) -> Dict[int, Dict[str, List[int]]]:
    """
    Generate rankings for multiple lengths

    Args:
        group_metrics: Dict from compute_group_metrics
        s_values: List of ranking lengths to generate (e.g., [2, 3])
        num_random: Number of random rankings per length

    Returns:
        Dict mapping s -> {ranking_name -> ranking}
    """
    all_rankings = {}

    for s in s_values:
        all_rankings[s] = generate_rankings(group_metrics, s, num_random)

    return all_rankings


def format_ranking_for_filename(ranking: List[int]) -> str:
    """Convert ranking [0, 1, 2] to string '0_1_2' for filenames"""
    return '_'.join(map(str, ranking))


def print_ranking_summary(group_metrics: Dict[int, Dict], rankings: Dict[str, List[int]]):
    """Print a summary of generated rankings with group metrics"""
    print("\nGenerated Rankings:")
    print("=" * 80)

    for name, ranking in rankings.items():
        print(f"\n{name}: {ranking}")
        print("-" * 40)

        for i, group_id in enumerate(ranking):
            metrics = group_metrics[group_id]
            print(f"  Position {i}: Group {group_id}")
            print(f"    Size: {metrics['size']}")
            print(f"    Avg coreness: {metrics['avg_coreness']:.2f}")
            print(f"    Avg degree: {metrics['avg_degree']:.2f}")


# Example usage
if __name__ == "__main__":
    # Example: test with dummy data
    import networkx as nx

    # Create a simple test graph with 4 groups
    G = nx.karate_club_graph()

    # Assign groups (for testing)
    group_assignments = {i: i % 4 for i in G.nodes()}

    # Compute metrics
    metrics = compute_group_metrics(G, group_assignments, 4)

    # Generate rankings for s=2 and s=3
    rankings = generate_rankings_for_lengths(metrics, [2, 3], num_random=5)

    print("Group Metrics:")
    for gid, m in metrics.items():
        print(f"  Group {gid}: size={m['size']}, coreness={m['avg_coreness']:.2f}, degree={m['avg_degree']:.2f}")

    for s, s_rankings in rankings.items():
        print(f"\n\n=== Rankings of length {s} ===")
        print_ranking_summary(metrics, s_rankings)
