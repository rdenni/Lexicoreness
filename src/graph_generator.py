"""
Graph Generation Module for Spreading Experiments

Generates synthetic networks with controlled structure and group assignments.
Groups are not necessarily Communities: Groups are node attributes, communities are network structure.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import sys

# Directory LexiCoreness/
LEXICORENESS_DIR = Path(__file__).resolve().parent.parent

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from ranking_strategies import compute_group_metrics, generate_rankings_for_lengths


class GraphGenerator:
    """Generate synthetic graphs with group assignments"""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
    
    def generate_er_graph(self, n: int, avg_degree: float) -> nx.Graph:
        """
        Generate Erdős-Rényi random graph
        
        Args:
            n: Number of nodes
            avg_degree: Target average degree
        
        Returns:
            NetworkX graph
        """
        # For ER: expected avg_degree = p * (n-1)
        p = avg_degree / (n - 1)
        G = nx.erdos_renyi_graph(n, p, seed=self.random_seed)
        return G
    
    def generate_sbm_graph(
        self, 
        n: int,
        community_sizes: List[int],
        avg_degree: float,
        alpha: float = 0.2
    ) -> Tuple[nx.Graph, Dict[int, int]]:
        """
        Generate Stochastic Block Model graph with this calculation:
        
        Uses:
            p_between = \alpha * (d / (n - 1))
            p_within_i = (d - p_between * (n - c_i)) / (c_i - 1)
        
        This gives intuitive control: \alpha represents the fraction of ER baseline
        connectivity for inter-community edges.
        
        Args:
            n: Number of nodes
            community_sizes: List of community sizes (must sum to n)
            avg_degree: Target average degree (d)
            alpha: Inter-community connectivity factor (0 < \alpha < 1)
                  \alpha = 0.1: Very isolated communities (10% of ER connectivity)
                  \alpha = 0.2: Weak inter-community edges (recommended)
                  \alpha = 0.5: Moderate community structure
        
        Returns:
            (G, community_assignments) where community_assignments[node] = community_id
        """
        assert sum(community_sizes) == n, f"Community sizes must sum to n: {sum(community_sizes)} != {n}"
        assert 0 < alpha < 1, f"alpha must be in (0, 1): {alpha}"
        
        num_communities = len(community_sizes)
        
        # Calculate global p_between based on ER baseline
        p_between = alpha * (avg_degree / (n - 1))
        
        # Calculate per-community p_within to achieve target degree
        p_within_list = []
        for c_i in community_sizes:
            if c_i <= 1:
                # Single-node community, no internal edges
                p_within_i = 0.0
            else:
                numerator = avg_degree - p_between * (n - c_i)
                denominator = c_i - 1
                p_within_i = numerator / denominator
            
            p_within_list.append(p_within_i)
            
            # Verify constraint: p_within_i > p_between for community structure
            if p_within_i < p_between and c_i > 1:
                print(f"⚠️  Warning: Community {len(p_within_list)-1} has weak structure!")
                print(f"   Size: {c_i}, p_within: {p_within_i:.4f}, p_between: {p_between:.4f}")
                print(f"   Consider: (a) smaller \alpha, (b) larger communities, or (c) lower avg_degree")
            
            # Calculate actual modularity ratio
            if p_between > 0:
                actual_ratio = p_within_i / p_between
                print(f"   Community {len(p_within_list)-1}: size={c_i}, "
                      f"p_within={p_within_i:.4f}, ratio={actual_ratio:.2f}")
        
        # Ensure probabilities are valid [0, 1]
        p_within_list = [min(max(p, 0.0), 1.0) for p in p_within_list]
        p_between = min(max(p_between, 0.0), 1.0)
        
        # Build probability matrix
        prob_matrix = np.zeros((num_communities, num_communities))
        for i in range(num_communities):
            for j in range(num_communities):
                if i == j:
                    prob_matrix[i, j] = p_within_list[i]
                else:
                    prob_matrix[i, j] = p_between
        
        # Generate graph
        G = nx.stochastic_block_model(community_sizes, prob_matrix, seed=self.random_seed)
        
        # Extract community assignments
        community_assignments = {}
        node_counter = 0
        for comm_id, size in enumerate(community_sizes):
            for _ in range(size):
                community_assignments[node_counter] = comm_id
                node_counter += 1
        
        return G, community_assignments
    
    def assign_groups(
        self,
        n: int,
        num_groups: int,
        group_sizes: List[float],
        assignment_type: str = "random",
        community_assignments: Optional[Dict[int, int]] = None
    ) -> Dict[int, int]:
        """
        Assign nodes to groups
        
        Args:
            n: Number of nodes
            num_groups: Number of groups
            group_sizes: Fraction of nodes in each group (must sum to 1.0)
            assignment_type: "random", "aligned" (groups = communities)
            community_assignments: Required if assignment_type="aligned"
        
        Returns:
            Dictionary mapping node_id -> group_id
        """
        assert len(group_sizes) == num_groups, "group_sizes length must match num_groups"
        assert abs(sum(group_sizes) - 1.0) < 1e-6, f"group_sizes must sum to 1.0: {sum(group_sizes)}"
        
        group_assignments = {}
        
        if assignment_type == "random":
            # Randomly assign nodes to groups according to group_sizes
            nodes = list(range(n))
            self.rng.shuffle(nodes)
            
            current_idx = 0
            for group_id, fraction in enumerate(group_sizes):
                group_size = int(np.round(fraction * n))
                # Handle rounding for last group
                if group_id == num_groups - 1:
                    group_size = n - current_idx
                
                for i in range(current_idx, current_idx + group_size):
                    group_assignments[nodes[i]] = group_id
                
                current_idx += group_size
        
        elif assignment_type == "aligned":
            # Groups = Communities
            assert community_assignments is not None, "community_assignments required for aligned assignment"
            group_assignments = community_assignments.copy()
        
        else:
            raise ValueError(f"Unknown assignment_type: {assignment_type}")
        
        return group_assignments
    
    def save_graph(
        self,
        G: nx.Graph,
        group_assignments: Dict[int, int],
        output_dir: Path,
        graph_name: str,
        replicate_id: int,
        metadata: Dict
    ):
        """
        Save graph to files
        
        Creates three files:
        - {graph_name}_rep{replicate_id}_edges.txt: Edge list
        - {graph_name}_rep{replicate_id}_groups.txt: Node-group assignments
        - {graph_name}_rep{replicate_id}_metadata.json: Graph metadata
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = f"{graph_name}_rep{replicate_id}"
        
        # Save edges (0-indexed)
        edge_file = output_dir / f"{prefix}_edges.txt"
        with open(edge_file, 'w') as f:
            for u, v in G.edges():
                f.write(f"{u} {v}\n")
        
        # Save group assignments (0-indexed)
        group_file = output_dir / f"{prefix}_groups.txt"
        with open(group_file, 'w') as f:
            for node in sorted(group_assignments.keys()):
                f.write(f"{node} {group_assignments[node]}\n")
        
        # Compute actual statistics
        actual_avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
        
        # Save metadata
        metadata_enhanced = {
            **metadata,
            "replicate_id": replicate_id,
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "actual_avg_degree": actual_avg_degree,
            "num_groups": len(set(group_assignments.values())),
            "group_counts": {
                str(gid): sum(1 for g in group_assignments.values() if g == gid)
                for gid in set(group_assignments.values())
            },
            "edge_file": str(edge_file.name),
            "group_file": str(group_file.name),
        }
        
        metadata_file = output_dir / f"{prefix}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata_enhanced, f, indent=2)
        
        print(f"Saved graph: {prefix}")
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        print(f"  Target avg_degree: {metadata.get('avg_degree', 'N/A')}, Actual: {actual_avg_degree:.2f}")
        print(f"  Groups: {metadata_enhanced['group_counts']}")


def generate_graph_from_config(config: Dict, replicate_id: int, output_dir: Path) -> Dict:
    """
    Generate a single graph from configuration
    
    Args:
        config: Graph configuration dictionary
        replicate_id: Replicate identifier
        output_dir: Output directory
    
    Returns:
        Metadata dictionary
    """
    # Set replicate-specific seed
    seed = config.get('random_seed', 42) + replicate_id * 1000
    generator = GraphGenerator(random_seed=seed)
    
    graph_type = config['type']
    n = config['n']
    
    # Generate base graph
    if graph_type == 'ER':
        G = generator.generate_er_graph(n, config['avg_degree'])
        community_assignments = None
        
    elif graph_type == 'SBM':
        G, community_assignments = generator.generate_sbm_graph(
            n=n,
            community_sizes=config['community_sizes'],
            avg_degree=config['avg_degree'],
            alpha=config.get('alpha', 0.2)  # Default to 0.2 if not specified
        )
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    # Assign groups
    group_assignments = generator.assign_groups(
        n=n,
        num_groups=config['num_groups'],
        group_sizes=config['group_sizes'],
        assignment_type=config.get('group_assignment', 'random'),
        community_assignments=community_assignments
    )

    # # Handle automatic ranking generation
    # rankings = config.get('rankings', [])

    # # Check if rankings should be auto-generated
    # if rankings == "auto" or (isinstance(rankings, list) and len(rankings) > 0 and isinstance(rankings[0], str) and rankings[0].startswith("s=")):
    #     # Auto-generate rankings based on graph structure

    #     # Extract s values from config
    #     if rankings == "auto":
    #         # Default: generate for s=2 and s=3
    #         s_values = [2, 3]
    #     else:
    #         # Parse s values from strings like ["s=2", "s=3"]
    #         s_values = [int(r.split("=")[1]) for r in rankings if r.startswith("s=")]

    #     # Compute group metrics
    #     group_metrics = compute_group_metrics(G, group_assignments, config['num_groups'])

    #     # Generate rankings for each s value
    #     all_rankings_by_s = generate_rankings_for_lengths(
    #         group_metrics,
    #         s_values,
    #         num_random=5  # Generate 5 random rankings per s
    #     )

    #     # Flatten into single list of rankings
    #     rankings = []
    #     for s, s_rankings in all_rankings_by_s.items():
    #         for name, ranking in s_rankings.items():
    #             rankings.append(ranking)

    #     print(f"  Auto-generated {len(rankings)} rankings for s={s_values}")

    # Prepare metadata
    metadata = {
        "name": config['name'],
        "type": graph_type,
        "n": n,
        "avg_degree": config['avg_degree'],
        "num_groups": config['num_groups'],
        "group_sizes": config['group_sizes'],
         # "rankings": rankings,
        "random_seed": seed,
    }
    
    if graph_type == 'SBM':
        metadata.update({
            "num_communities": config['num_communities'],
            "community_sizes": config['community_sizes'],
            "alpha": config.get('alpha', 0.2),  # Inter-community connectivity factor
            "group_assignment": config.get('group_assignment', 'random'),
        })
    
    # Save graph
    generator.save_graph(
        G=G,
        group_assignments=group_assignments,
        output_dir=output_dir,
        graph_name=config['name'],
        replicate_id=replicate_id,
        metadata=metadata
    )
    
    return metadata


if __name__ == "__main__":
    # Test with a simple example
    # sandbox directory
    output_dir = LEXICORENESS_DIR / "sandbox"
    
    test_config_ER = {
        "name": "test_ER",
        "type": "ER",
        "n": 100,
        "avg_degree": 6,
        "num_groups": 2,
        "group_sizes": [0.6, 0.4],
        "rankings": [[0, 1]],
        "random_seed": 42
    }
    
    print("Testing graph generation ER...")
    generate_graph_from_config(test_config_ER, replicate_id=0, output_dir=output_dir)
    print("\nTest successful!")
    print("Now testing SBM...")

    test_config_SBM = {
        "name": "test_SBM",
        "type": "SBM",
        "n": 100,
        "avg_degree": 6,
        "num_communities": 2,
        "community_sizes": [50, 50],
        "alpha": 0.2,
        "num_groups": 2,
        "group_sizes": [0.5, 0.5],
        "group_assignment": "aligned",
        "rankings": [[0, 1]],
        "random_seed": 42
    }
    print("Testing graph generation SBM...")
    generate_graph_from_config(test_config_SBM, replicate_id=0, output_dir=output_dir)
    print("\nTest successful!")

    print("\n✓ All tests passed!")

