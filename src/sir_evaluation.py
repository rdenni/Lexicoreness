"""
SIR Evaluation Module

Orchestration layer between SIR simulations and lexicographic ranking.
This module knows about BOTH SIR simulation results AND lexicographic comparisons.

Design:
- sir_simulator.py: Pure simulation (returns raw data)
- lexicographic_utils.py: Pure vector operations (no domain knowledge)
- sir_evaluation.py: Orchestration (bridges simulation and ranking)
"""

from typing import Dict, List, Any, Tuple
import sys
from pathlib import Path

# Import simulation and ranking modules
sys.path.insert(0, str(Path(__file__).parent))
from lexicographic_utils import rank_vectors_lexicographically


def extract_reach_vector(
    simulation_results: Dict,
    ranking: List[int]
) -> List[float]:
    """
    Extract reach vector aligned with ranking from simulation results.
    
    This is SIR-specific: knows about 'group_reach_mean' structure.
    
    Args:
        simulation_results: Dict with 'group_reach_mean' key from SIR simulator
        ranking: Ranking of group indices
    
    Returns:
        List of reach values in ranking order
    
    Example:
        >>> results = {'group_reach_mean': {0: 10.5, 1: 5.2, 2: 8.1}}
        >>> extract_reach_vector(results, [1, 0, 2])
        [5.2, 10.5, 8.1]  # Reordered according to ranking
    """
    group_reach = simulation_results.get('group_reach_mean', {})
    return [group_reach.get(group_id, 0.0) for group_id in ranking]


def evaluate_seeds_with_ranking(
    simulator,
    seed_nodes: List[Any],
    ranking: List[int],
    infection_prob: float,
    num_simulations: int = 1000,
    random_seed: int = 42,
    verbose: bool = False
) -> Dict[Any, Dict]:
    """
    Evaluate multiple seed nodes with respect to a group ranking.
    
    ORCHESTRATION FUNCTION: Coordinates between SIR simulation and ranking.
    
    Workflow:
    1. For each seed node:
       - Call simulator.run_multiple_simulations() (SIR domain)
       - Extract ranking-specific reach vector (bridge)
       - Package results
    
    Args:
        simulator: SIRSimulator instance
        seed_nodes: List of seed nodes to evaluate
        ranking: List of group IDs in priority order
        infection_prob: Infection probability
        num_simulations: Number of simulations per seed
        random_seed: Base random seed
        verbose: Print progress
    
    Returns:
        Dict mapping seed_node -> evaluation results with reach vectors
        
    Example:
        >>> from sir_simulator import SIRSimulator
        >>> simulator = SIRSimulator(G, groups)
        >>> results = evaluate_seeds_with_ranking(
        ...     simulator=simulator,
        ...     seed_nodes=[0, 100, 200],
        ...     ranking=[0, 1],
        ...     infection_prob=0.05,
        ...     num_simulations=1000
        ... )
        >>> # results[0]['reach_vector_for_ranking'] = [mean_group0, mean_group1]
    """
    results = {}
    
    if verbose:
        print(f"Evaluating {len(seed_nodes)} seed nodes with ranking {ranking}...")
    
    for idx, seed in enumerate(seed_nodes):
        if verbose:
            print(f"  Seed {idx + 1}/{len(seed_nodes)}: Node {seed}")
        
        # Run SIR simulations (simulator's job)
        sim_results = simulator.run_multiple_simulations(
            seed_nodes=[seed],
            infection_prob=infection_prob,
            num_simulations=num_simulations,
            random_seed=random_seed + idx * 1000,
            verbose=False
        )
        
        # Extract ranking-specific reach vector (bridge between domains)
        reach_vector = extract_reach_vector(sim_results, ranking)
        
        # Package results
        results[seed] = {
            **sim_results,  # Include all simulation results
            'seed_node': seed,
            'ranking': ranking,
            'reach_vector_for_ranking': reach_vector,
            'infection_prob': infection_prob
        }
    
    return results


def rank_seeds_by_reach(
    seed_results: Dict[Any, Dict]
) -> List[Tuple[Any, List[float], int]]:
    """
    Rank seeds by their lexicographic reach vector.
    
    This is a convenience wrapper that:
    1. Extracts reach vectors from SIR results (SIR-specific)
    2. Sorts using generic function (delegates to lexicographic_utils)
    3. Assigns ranks
    
    Args:
        seed_results: Dict from evaluate_seeds_with_ranking
    
    Returns:
        List of (seed_node, reach_vector, rank) sorted by rank (rank 1 = best)
    
    Example:
        >>> results = evaluate_seeds_with_ranking(simulator, [0, 100], [0, 1], ...)
        >>> ranked = rank_seeds_by_reach(results)
        >>> print(f"Best seed: {ranked[0][0]}")
    """
    # Extract ranking from first result (all should have same ranking)
    first_result = next(iter(seed_results.values()))
    ranking = first_result['ranking']
    
    # Build group_data: {seed: {group_id: reach_mean}}
    group_data = {}
    for seed, results in seed_results.items():
        group_data[seed] = results['group_reach_mean']
    
    # Use generic sort_by_ranking function
    from lexicographic_utils import sort_by_ranking
    sorted_items = sort_by_ranking(group_data, ranking, reverse=True)
    
    # Assign ranks (handle ties)
    ranked_seeds = []
    current_rank = 1
    prev_vector = None
    
    for seed, vector in sorted_items:
        if prev_vector is not None and tuple(vector) != tuple(prev_vector):
            current_rank = len(ranked_seeds) + 1
        ranked_seeds.append((seed, vector, current_rank))
        prev_vector = vector
    
    return ranked_seeds


# Backward compatibility aliases
def rank_seeds_lexicographically(seed_results: Dict, ranking: List[int] = None):
    """
    DEPRECATED: Use rank_seeds_by_reach() instead.
    
    Kept for backward compatibility.
    """
    return rank_seeds_by_reach(seed_results)


if __name__ == "__main__":
    print("Testing SIR Evaluation Module...")
    
    # This module is pure orchestration - needs actual SIR simulator to test
    print("  Module loaded successfully")
    print("  Functions available:")
    print("    - extract_reach_vector()")
    print("    - evaluate_seeds_with_ranking()")
    print("    - rank_seeds_by_reach()")
    print("  See sir_simulator.py for integration tests")
