"""
SIR Epidemic Simulator

Implements the Susceptible-Infected-Recovered (SIR) model for simulating
influence spreading on networks.

SIR Model:
- All nodes start as Susceptible (S) except the seed node(s) which start as Infected (I)
- At each discrete time step:
  - Each infected node attempts to infect each susceptible neighbor with probability p
  - After attempting infections, infected nodes become Recovered (R)
- Recovered nodes no longer participate in spreading
- Process continues until no infected nodes remain
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import time


class SIRSimulator:
    """SIR epidemic model simulator"""
    
    def __init__(self, G: nx.Graph, group_assignments: Dict[int, int]):
        """
        Initialize simulator with graph and group assignments
        
        Args:
            G: NetworkX undirected graph
            group_assignments: Dict mapping node_id -> group_id
        """
        self.G = G
        self.group_assignments = group_assignments
        self.n = G.number_of_nodes()
        
        # Precompute neighbor lists for efficiency
        self.neighbors = {node: list(G.neighbors(node)) for node in G.nodes()}
        
        # Get unique groups
        self.groups = sorted(set(group_assignments.values()))
        self.num_groups = len(self.groups)
    
    def run_single_simulation(
        self,
        seed_nodes: List[int],
        infection_prob: float,
        random_state: np.random.Generator
    ) -> Tuple[int, Dict[int, int], List[Set[int]]]:
        """
        Run a single SIR simulation

        Args:
            seed_nodes: Initial infected nodes
            infection_prob: Probability of infection along each edge
            random_state: NumPy random generator for reproducibility
        
        Returns:
            (total_recovered, group_recovered_counts, infection_history)
            where:
            - total_recovered: Total number of recovered nodes
            - group_recovered_counts: Dict mapping group_id -> count of recovered in that group
            - infection_history: List of sets of infected nodes at each time step
        """
        # Initialize states
        # 0 = Susceptible, 1 = Infected, 2 = Recovered
        state = {node: 0 for node in self.G.nodes()}
        for seed in seed_nodes:
            state[seed] = 1
        
        infected_nodes = set(seed_nodes)
        infection_history = [infected_nodes.copy()]
        
        # Run simulation until no infected nodes remain
        while infected_nodes:
            newly_infected = set()
            
            # Each infected node tries to infect susceptible neighbors
            for infected_node in infected_nodes:
                for neighbor in self.neighbors[infected_node]:
                    # Only susceptible nodes can be infected
                    if state[neighbor] == 0:
                        # Attempt infection with probability p
                        if random_state.random() < infection_prob:
                            newly_infected.add(neighbor)
            
            # Update states: infected -> recovered, newly infected -> infected
            for node in infected_nodes:
                state[node] = 2  # Recovered
            
            for node in newly_infected:
                state[node] = 1  # Infected
            
            infected_nodes = newly_infected
            if infected_nodes:
                infection_history.append(infected_nodes.copy())
        
        # Count recovered nodes (all non-susceptible)
        recovered_nodes = [node for node, s in state.items() if s == 2]
        total_recovered = len(recovered_nodes)
        
        # Count recovered by group
        group_recovered = defaultdict(int)
        for node in recovered_nodes:
            group_recovered[self.group_assignments[node]] += 1
        
        return total_recovered, dict(group_recovered), infection_history
    
    def run_multiple_simulations(
        self,
        seed_nodes: List[int],
        infection_prob: float,
        num_simulations: int = 1000,
        random_seed: int = 42,
        verbose: bool = False
    ) -> Dict:
        """
        Run multiple SIR simulations and aggregate results
        
        Args:
            seed_nodes: Initial infected nodes
            infection_prob: Probability of infection along each edge
            num_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
            verbose: Print progress
        
        Returns:
            Dict with keys:
            - 'total_reach_mean': Mean total recovered nodes
            - 'total_reach_std': Std of total recovered nodes
            - 'total_reach_all': All total_recovered values (array)
            - 'group_reach_mean': Dict of mean recovered per group
            - 'group_reach_std': Dict of std recovered per group
            - 'group_reach_all': 2D array (num_simulations × num_groups)
            - 'group_reach_vector_mean': List ordered by groups [0, 1, 2, ...]
            - 'cascade_sizes': List of total_recovered for each simulation
        """
        random_state = np.random.default_rng(random_seed)
        
        # Storage for results
        total_recovered_all = []
        group_recovered_all = defaultdict(list)
        
        if verbose:
            print(f"Running {num_simulations} SIR simulations...")
            progress_interval = max(1, num_simulations // 10)
        
        for sim_idx in range(num_simulations):
            if verbose and (sim_idx + 1) % progress_interval == 0:
                print(f"  Progress: {sim_idx + 1}/{num_simulations}")
            
            total_rec, group_rec, _ = self.run_single_simulation(
                seed_nodes, infection_prob, random_state
            )
            
            total_recovered_all.append(total_rec)
            
            # Store group counts (0 if group not reached)
            for group_id in self.groups:
                group_recovered_all[group_id].append(group_rec.get(group_id, 0))
        
        # Compute statistics
        total_recovered_all = np.array(total_recovered_all)
        
        group_reach_mean = {
            group_id: np.mean(group_recovered_all[group_id])
            for group_id in self.groups
        }
        
        group_reach_std = {
            group_id: np.std(group_recovered_all[group_id])
            for group_id in self.groups
        }
        
        # Create 2D array: rows = simulations, cols = groups
        group_reach_array = np.zeros((num_simulations, self.num_groups))
        for i, group_id in enumerate(self.groups):
            group_reach_array[:, i] = group_recovered_all[group_id]
        
        # Create mean vector ordered by groups
        group_reach_vector_mean = [group_reach_mean[group_id] for group_id in self.groups]
        
        return {
            'total_reach_mean': float(np.mean(total_recovered_all)),
            'total_reach_std': float(np.std(total_recovered_all)),
            'total_reach_all': total_recovered_all,
            'group_reach_mean': group_reach_mean,
            'group_reach_std': group_reach_std,
            'group_reach_all': group_reach_array,
            'group_reach_vector_mean': group_reach_vector_mean,
            'cascade_sizes': total_recovered_all.tolist(),
        }


if __name__ == "__main__":
    from methods import load_graph
    import argparse
    import os
    import time

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run SIR simulation on a graph.")
    parser.add_argument("edge_file", type=str, help="Path to the edge file")
    parser.add_argument("group_file", type=str, help="Path to the group file")
    parser.add_argument("seed_node", type=int, help="Seed node ID for simulation")
    parser.add_argument("num_sims", type=int, help="Number of simulations to run")
    args = parser.parse_args()

    # Validate file paths
    if not os.path.exists(args.edge_file):
        raise FileNotFoundError(f"Edge file not found: {args.edge_file}")
    if not os.path.exists(args.group_file):
        raise FileNotFoundError(f"Group file not found: {args.group_file}")

    # Load graph and group assignments
    print("\nLoading graph...")
    G, group_assignments = load_graph(args.edge_file, args.group_file)

    # Print graph statistics
    print(f"Graph statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
    print(f"  Avg degree: {avg_degree:.2f}")
    # Print groups statistics
    print(f"  Group statistics:")
    print(f"  Number of groups: {len(set(group_assignments.values()))}")
    # Count nodes per group
    group_counts = defaultdict(int)
    for group_id in group_assignments.values():
        group_counts[group_id] += 1
    for group_id in sorted(group_counts.keys()):
        print(f"    Group {group_id}: {group_counts[group_id]} nodes")
    
    # Avg degree per group
    group_degree_sums = defaultdict(int)
    for node in G.nodes():
        group_id = group_assignments[node]
        group_degree_sums[group_id] += G.degree(node)
    print(f"  Average degree per group:")
    for group_id in sorted(group_counts.keys()):
        avg_deg = group_degree_sums[group_id] / group_counts[group_id]
        print(f"    Group {group_id}: {avg_deg:.2f}")


    # Initialize simulator
    simulator = SIRSimulator(G, group_assignments)

    # Simulation parameters
    seed_node = args.seed_node
    # calculate infection probability based on avg degree
    infection_prob = 0.8 / avg_degree  # Below epidemic threshold
    num_sims = args.num_sims  # Small number for quick test
    ranking = sorted(set(group_assignments.values()))  # Default ranking by group IDs

   
    print("\nRunning SIR simulation...")
    print("=" * 40)
    print(f"  Seed: Node {seed_node}")
    print(f"  Group of seed: {group_assignments[seed_node]}")
    print(f"  Infection prob: {infection_prob:.4f}")
    print(f"  Simulations: {num_sims}")
    print(f"  Ranking: {ranking}")
    print("=" * 40)
    # run evaluation and measure time
    start_time = time.time()
    results = simulator.run_multiple_simulations(
        seed_nodes=[seed_node],
        infection_prob=infection_prob,
        num_simulations=num_sims,
        random_seed=42,
        verbose=True
    )
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds.")

    # Print results
    print("\nSimulation results:")
    print(f"  Total reach: {results['total_reach_mean']:.2f} ± {results['total_reach_std']:.2f}")
    print(f"  Group reaches:")
    for group_id in sorted(results['group_reach_mean'].keys()):
        mean = results['group_reach_mean'][group_id]
        std = results['group_reach_std'][group_id]
        print(f"    Group {group_id}: {mean:.2f} ± {std:.2f}")
    print("\nSIR simulation completed successfully!")
