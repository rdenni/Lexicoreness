"""
Methods Module: Centrality Measures for Spreader Identification

"""

import sys
from pathlib import Path
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
#from collections import defaultdict
import time
import concurrent.futures
import threading

# Default timeout for methods (in seconds): 1 hour
DEFAULT_METHOD_TIMEOUT = 3600

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from lexicographic_utils import LexicographicHeap, lexicographic_max

# Try to import FairGD
FAIRGD_AVAILABLE = False
fairgd_utils = None
try:
    _fairgd_path = str(Path(__file__).parent / "other_baselines" / "Fair-Pagerank-via-Edge-Rewighting")
    if _fairgd_path not in sys.path:
        sys.path.insert(0, _fairgd_path)
    from fairGD import FairGD
    import utils as fairgd_utils
    FAIRGD_AVAILABLE = True
except (ImportError, Exception):
    # FairGD/AdaptGD/LFPR_U/LFPR_N will be skipped
    pass

# Try to import FairLaR dependencies 
FAIRLAR_AVAILABLE = False
try:
    import cvxopt
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False  # Suppress cvxopt output
    FAIRLAR_AVAILABLE = True
except (ImportError, Exception):
    # FSPR will be skipped
    pass



def compute_betas_configs(ranking: List[int],
                          betas_values: List[float]) -> List[Tuple[str, List[float], dict]]:
    """
    Generate beta configurations for a ranking.

    Each value in betas_values defines a strategy where all betas are equal to that value.

    Args:
        ranking: List of group IDs in priority order
        betas_values: List of beta values, e.g. [0.99, 0.5, 0.1]

    Returns:
        List of (label, betas_vector, metadata) tuples.
        betas_vector has length len(ranking) - 1.
    """
    s = len(ranking)
    if s < 2:
        return []

    configs = []
    for v in betas_values:
        v = float(v)
        label = str(v)
        betas_vector = [v] * (s - 1)
        metadata = {
            'strategy': label,
            'beta_value': v,
            'betas': betas_vector,
        }
        configs.append((label, betas_vector, metadata))

    return configs


class CentralityMethods:
    """Collection of centrality measures for identifying influential spreaders"""
    
    def __init__(self, G: nx.Graph, group_assignments: Dict[int, int]):
        """
        Initialize with graph and group assignments
        
        Args:
            G: NetworkX undirected graph
            group_assignments: Dict mapping node_id -> group_id
        """
        self.G = G
        self.group_assignments = group_assignments
        self.n = G.number_of_nodes()
        self.m = G.number_of_edges()
        
        # Validate that all nodes have group assignments
        assert len(group_assignments) == self.n, "All nodes must have group assignments"
        
        # Get unique groups
        self.groups = sorted(set(group_assignments.values()))
        self.num_groups = len(self.groups)
    
    def get_degree_vector(self, node: int, ranking: List[int],
                         induced_by: Optional[set] = None) -> List[int]:
        """
        Compute degree vector for a node restricted to ranking

        For influence/reach prediction, the node counts itself in its own group.
        This is because when a node is a seed in SIR, it always reaches at least itself.

        Args:
            node: Node ID
            ranking: List of group IDs in priority order
            induced_by: If provided, only count edges to nodes in this set

        Returns:
            Degree vector [deg_in_group_r[0], deg_in_group_r[1], ...]
            where deg_in_group includes +1 if the node belongs to that group
        """
        degree_vector = []
        node_group = self.group_assignments[node]

        # Check if node is in the active set (for induced subgraphs during peeling)
        node_is_active = (induced_by is None) or (node in induced_by)

        for group_id in ranking:
            deg = 0

            # Count neighbors in this group
            for neighbor in self.G.neighbors(node):
                # Check if neighbor is in the induced set (if provided)
                if induced_by is not None and neighbor not in induced_by:
                    continue
                # Count if neighbor belongs to the target group
                if self.group_assignments[neighbor] == group_id:
                    deg += 1

            # Add +1 if this is the node's own group and the node is active
            # (for SIR influence: a seed always reaches itself)
            if group_id == node_group and node_is_active:
                deg += 1

            degree_vector.append(deg)

        return degree_vector

    def lexipeeling(self, ranking: List[int]) -> Dict[int, Tuple[List[int], int]]:
        """
        Lexipeeling algorithm
        Computes lexicoreness vector for each node according to ranking.
        
        Args:
            ranking: List of group IDs in priority order
        
        Returns:
            Dict mapping node_id -> (lexicoreness_vector, rank)
            where rank is 1 for best (lexicographically maximum)
        """
        start_time = time.time()
        
        s = len(ranking)  # Length of ranking
        k_curr = [0] * s  # Current threshold vector
        k = {node: [0] * s for node in self.G.nodes()}  # Lexicoreness vectors
        
        # Initialize degree vectors for all nodes
        degree_vectors = {}
        for node in self.G.nodes():
            degree_vectors[node] = self.get_degree_vector(node, ranking, induced_by=None)
        
        # Create min-heap with all nodes
        heap = LexicographicHeap()
        for node in self.G.nodes():
            heap.push(node, degree_vectors[node])
        
        # Store extraction order for debugging
        extraction_order = []

        while heap:
            # Extract node with lexicographically minimum degree vector
            min_node, min_degree_vec = heap.pop()

            # Assign lexicoreness: max(k_curr, degree_vector)
            k[min_node] = lexicographic_max(k_curr, min_degree_vec)
            k_curr = k[min_node]

            # Record extraction order
            extraction_order.append(min_node)

            # Update neighbors' degree vectors (incremental update)
            min_node_group = self.group_assignments[min_node]

            # Only update if this node's group is in the ranking
            if min_node_group in ranking:
                group_idx = ranking.index(min_node_group)

                for neighbor in self.G.neighbors(min_node):
                    if neighbor in degree_vectors:  # Still in heap
                        # Decrement the component corresponding to min_node's group
                        degree_vectors[neighbor][group_idx] -= 1

                        # Update heap with new degree vector
                        heap.push(neighbor, degree_vectors[neighbor])

            # Remove from degree_vectors (node processed)
            del degree_vectors[min_node]
        
        # Assign ranks based on lexicographic ordering of vectors
        unique_vectors = sorted(set(tuple(v) for v in k.values()), reverse=True)
        vector_to_rank = {vec: rank + 1 for rank, vec in enumerate(unique_vectors)}
        
        # Create result with ranks
        result = {
            node: (k[node], vector_to_rank[tuple(k[node])])
            for node in self.G.nodes()
        }
        
        runtime = time.time() - start_time

        return result, runtime

    def lexipeeling_extended(self, ranking: List[int]) -> Optional[Tuple[Dict[int, Tuple[List[int], int]], float]]:
        """
        Extended lexipeeling variant that includes excluded groups.

        This variant only runs when the ranking contains fewer groups than the graph has.
        It treats all nodes from excluded groups as belonging to a single virtual group,
        which is appended at the end of the ranking.

        Example:
            Graph has groups: {g_1, g_2, g_3, g_4, g_5}
            Input ranking: [g_3, g_2, g_5]
            Excluded groups: {g_1, g_4} -> merged into virtual group g_x
            Extended ranking: [g_3, g_2, g_5, g_x]
            Lexipeeling runs with this 4-element ranking

        Args:
            ranking: List of group IDs in priority order

        Returns:
            If len(ranking) < num_groups: (Dict mapping node_id -> (lexicoreness_vector, rank), runtime)
            If len(ranking) >= num_groups: None (silently, no warning)
        """
        # Only run if ranking has fewer groups than the graph
        if len(ranking) >= self.num_groups:
            return None

        start_time = time.time()

        # Find excluded groups
        ranking_groups = set(ranking)
        excluded_groups = set(self.groups) - ranking_groups

        # Create virtual group ID (unique, not in existing groups)
        virtual_group_id = max(self.groups) + 1

        # Create modified group assignments: excluded groups -> virtual group
        modified_assignments = {}
        for node, group in self.group_assignments.items():
            if group in excluded_groups:
                modified_assignments[node] = virtual_group_id
            else:
                modified_assignments[node] = group

        # Build extended ranking: original ranking + virtual group at the end
        extended_ranking = list(ranking) + [virtual_group_id]

        # Temporarily swap group assignments to run lexipeeling with modified groups
        original_assignments = self.group_assignments
        self.group_assignments = modified_assignments

        try:
            # Run lexipeeling with extended ranking
            result, _ = self.lexipeeling(extended_ranking)
        finally:
            # Restore original group assignments
            self.group_assignments = original_assignments

        runtime = time.time() - start_time

        return result, runtime

    def coreness(self) -> Dict[int, Tuple[int, int]]:
        """
        Standard k-core decomposition (group-agnostic)
        
        Returns:
            Dict mapping node_id -> (coreness_value, rank)
        """
        start_time = time.time()
        
        # Use NetworkX's k-core implementation
        core_number = nx.core_number(self.G)
        
        # Assign ranks (higher coreness = better rank)
        unique_cores = sorted(set(core_number.values()), reverse=True)
        core_to_rank = {core: rank + 1 for rank, core in enumerate(unique_cores)}
        
        result = {
            node: (core_number[node], core_to_rank[core_number[node]])
            for node in self.G.nodes()
        }
        
        runtime = time.time() - start_time
        
        return result, runtime
    
    def degree_centrality(self) -> Dict[int, Tuple[int, int]]:
        """
        Standard degree centrality (group-agnostic)
        
        Returns:
            Dict mapping node_id -> (degree, rank)
        """
        start_time = time.time()
        
        degrees = dict(self.G.degree())
        
        # Assign ranks (higher degree = better rank)
        unique_degrees = sorted(set(degrees.values()), reverse=True)
        degree_to_rank = {deg: rank + 1 for rank, deg in enumerate(unique_degrees)}
        
        result = {
            node: (degrees[node], degree_to_rank[degrees[node]])
            for node in self.G.nodes()
        }
        
        runtime = time.time() - start_time
        
        return result, runtime
    
    def degree_centrality_lexicographic(self, ranking: List[int]) -> Dict[int, Tuple[List[int], int]]:
        """
        Group-aware degree centrality with lexicographic comparison
        
        Simply computes degree vector for each node and ranks lexicographically.
        No iterative peeling like Lexipeeling.
        
        Args:
            ranking: List of group IDs in priority order
        
        Returns:
            Dict mapping node_id -> (degree_vector, rank)
        """
        start_time = time.time()
        
        # Compute degree vectors for all nodes (no induced subset)
        degree_vectors = {}
        for node in self.G.nodes():
            degree_vectors[node] = self.get_degree_vector(node, ranking, induced_by=None)
        
        # Assign ranks based on lexicographic ordering
        unique_vectors = sorted(set(tuple(v) for v in degree_vectors.values()), reverse=True)
        vector_to_rank = {vec: rank + 1 for rank, vec in enumerate(unique_vectors)}
        
        result = {
            node: (degree_vectors[node], vector_to_rank[tuple(degree_vectors[node])])
            for node in self.G.nodes()
        }
        
        runtime = time.time() - start_time

        return result, runtime

    def betas_degree_lexicographic(self, ranking: List[int],
                                    betas: List[float]) -> Tuple[Dict[int, Tuple[List[float], int]], float]:
        """
        Beta-blended lexicographic degree centrality.

        For each node with raw degree vector [x_0, x_1, ..., x_{s-1}]:
            blended[i] = x_i + betas[i] * x_{i+1}   for i < s-1
            blended[s-1] = x_{s-1}

        Then ranks the blended vectors lexicographically.

        Args:
            ranking: List of group IDs in priority order
            betas: List of beta weights (length s-1)

        Returns:
            Tuple of (result_dict, runtime) where result_dict maps
            node_id -> (blended_vector, rank)
        """
        start_time = time.time()
        s = len(ranking)

        # Compute raw degree vectors and blend with betas
        blended = {}
        for node in self.G.nodes():
            raw = self.get_degree_vector(node, ranking, induced_by=None)
            vec = [float(raw[i]) + betas[i] * float(raw[i + 1]) for i in range(s - 1)]
            vec.append(float(raw[s - 1]))
            blended[node] = vec

        # Rank lexicographically
        unique_vectors = sorted(set(tuple(v) for v in blended.values()), reverse=True)
        vector_to_rank = {vec: rank + 1 for rank, vec in enumerate(unique_vectors)}

        result = {
            node: (blended[node], vector_to_rank[tuple(blended[node])])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def betas_lexipeeling(self, ranking: List[int],
                           betas: List[float]) -> Tuple[Dict[int, Tuple[List[float], int]], float]:
        """
        Beta-blended lexipeeling algorithm.

        Like standard lexipeeling but with blended degree vectors:
            Initial: blended[i] = raw[i] + betas[i] * raw[i+1]  for i < s-1
                     blended[s-1] = raw[s-1]

        Update rule when removing a node of group at position j:
            blended[j] -= 1              (direct effect)
            blended[j-1] -= betas[j-1]   (cascade, if j > 0)

        Args:
            ranking: List of group IDs in priority order
            betas: List of beta weights (length s-1)

        Returns:
            Tuple of (result_dict, runtime) where result_dict maps
            node_id -> (lexicoreness_vector, rank)
        """
        start_time = time.time()

        s = len(ranking)
        k_curr = [0.0] * s
        k = {node: [0.0] * s for node in self.G.nodes()}

        # Compute raw degree vectors and blend with betas
        blended_vectors = {}
        for node in self.G.nodes():
            raw = self.get_degree_vector(node, ranking, induced_by=None)
            vec = [float(raw[i]) + betas[i] * float(raw[i + 1]) for i in range(s - 1)]
            vec.append(float(raw[s - 1]))
            blended_vectors[node] = vec

        # Create min-heap
        heap = LexicographicHeap()
        for node in self.G.nodes():
            heap.push(node, blended_vectors[node])

        while heap:
            # Extract node with lexicographically minimum blended vector
            min_node, min_degree_vec = heap.pop()

            # Assign lexicoreness: max(k_curr, blended_vector)
            k[min_node] = lexicographic_max(k_curr, min_degree_vec)
            k_curr = k[min_node]

            # Update neighbors' blended vectors
            min_node_group = self.group_assignments[min_node]

            if min_node_group in ranking:
                j = ranking.index(min_node_group)  # position in ranking

                for neighbor in self.G.neighbors(min_node):
                    if neighbor in blended_vectors:  # still in heap
                        # Direct effect: component j decreases by 1
                        blended_vectors[neighbor][j] -= 1.0
                        # Cascade effect: component j-1 decreases by betas[j-1]
                        if j > 0:
                            blended_vectors[neighbor][j - 1] -= betas[j - 1]

                        heap.push(neighbor, blended_vectors[neighbor])

            # Remove from blended_vectors (node processed)
            del blended_vectors[min_node]

        # Assign ranks based on lexicographic ordering
        unique_vectors = sorted(set(tuple(v) for v in k.values()), reverse=True)
        vector_to_rank = {vec: rank + 1 for rank, vec in enumerate(unique_vectors)}

        result = {
            node: (k[node], vector_to_rank[tuple(k[node])])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def betas_pagerank_exponential_jump(self, ranking: List[int],
                                         beta_value: float) -> Tuple[Dict[int, Tuple[float, int]], float]:
        """
        PageRank with exponential decay jump distribution, with geometric base = 0.5 + beta_value.

        The personalization (jump) distribution is:
        - (0.5 + beta_value)^(rank(group) - 1) if the node's group is in the ranking
        - 0 if the node's group is not in the ranking

        Args:
            ranking: List of group IDs in priority order
            beta_value: Beta value to add to the base 0.5

        Returns:
            (Dict mapping node_id -> (pagerank, rank), runtime)
        """
        start_time = time.time()
        base = 0.5 + beta_value

        # Map group_id -> position (1-indexed)
        group_to_position = {group_id: pos + 1 for pos, group_id in enumerate(ranking)}

        # Build personalization dict: exponential decay based on group position
        personalization = {}
        for node in self.G.nodes():
            node_group = self.group_assignments[node]
            if node_group in group_to_position:
                position = group_to_position[node_group]
                personalization[node] = base ** (position - 1)
            else:
                personalization[node] = 0.0

        # Handle edge case: if no nodes are in ranking groups, fall back to standard PageRank
        if sum(personalization.values()) == 0:
            pr = nx.pagerank(self.G)
        else:
            pr = nx.pagerank(self.G, personalization=personalization)

        # Assign ranks (higher PageRank = better rank)
        unique_values = sorted(set(pr.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (pr[node], value_to_rank[pr[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def betas_fairgd(self, ranking: List[int], beta_value: float,
                     gamma: float = 0.15, lr: float = 1.0,
                     max_iter: int = 50, verbose: bool = False) -> Tuple[Dict[int, Tuple[float, int]], float]:
        """
        FairGD with geometric base = 0.5 + beta_value for the target distribution.

        Args:
            ranking: List of group IDs in priority order
            beta_value: Beta value to add to the base 0.5
            gamma: PageRank damping factor (default 0.15)
            lr: Learning rate for gradient descent (default 1.0)
            max_iter: Maximum iterations (default 50)
            verbose: Print progress during optimization

        Returns:
            (Dict mapping node_id -> (pagerank, rank), runtime)
        """
        if not FAIRGD_AVAILABLE:
            raise ImportError("FairGD not available.")

        start_time = time.time()
        base = 0.5 + beta_value

        G_labeled = self._prepare_graph_for_fairgd()
        fgd = FairGD(G_labeled, attr_name='label')

        target_dist = self._build_fairgd_target_distribution(ranking, base=base)

        targets = np.zeros(fgd.n_labels)
        for orig_label, proportion in target_dist.items():
            if orig_label in fgd.label_mapping:
                internal_idx = fgd.label_mapping[orig_label]
                targets[internal_idx] = proportion

        _, loss, _ = fgd.run_fpr(targets=targets, gamma=gamma, lr=lr,
                                  max_iter=max_iter, verbose=verbose)

        pr_array = fgd.pagerank(gamma=gamma)
        nodelist = list(G_labeled.nodes())
        pr = {nodelist[i]: pr_array[i] for i in range(len(nodelist))}

        unique_values = sorted(set(pr.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (pr[node], value_to_rank[pr[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def betas_adaptgd(self, ranking: List[int], beta_value: float,
                      gamma: float = 0.15, lr: float = 1.0,
                      max_iter: int = 50, verbose: bool = False) -> Tuple[Dict[int, Tuple[float, int]], float]:
        """
        AdaptGD with geometric base = 0.5 + beta_value for the target distribution.

        Args:
            ranking: List of group IDs in priority order
            beta_value: Beta value to add to the base 0.5
            gamma: PageRank damping factor (default 0.15)
            lr: Learning rate for gradient descent (default 1.0)
            max_iter: Maximum iterations (default 50)
            verbose: Print progress during optimization

        Returns:
            (Dict mapping node_id -> (pagerank, rank), runtime)
        """
        if not FAIRGD_AVAILABLE:
            raise ImportError("FairGD not available.")

        start_time = time.time()
        base = 0.5 + beta_value

        G_labeled = self._prepare_graph_for_fairgd()
        fgd = FairGD(G_labeled, attr_name='label')

        target_dist = self._build_fairgd_target_distribution(ranking, base=base)

        targets = np.zeros(fgd.n_labels)
        for orig_label, proportion in target_dist.items():
            if orig_label in fgd.label_mapping:
                internal_idx = fgd.label_mapping[orig_label]
                targets[internal_idx] = proportion

        _, loss, _ = fgd.run_cfpr(targets=targets, gamma=gamma, lr=lr,
                                    max_iter=max_iter, verbose=verbose)

        pr_array = fgd.pagerank(gamma=gamma)
        nodelist = list(G_labeled.nodes())
        pr = {nodelist[i]: pr_array[i] for i in range(len(nodelist))}

        unique_values = sorted(set(pr.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (pr[node], value_to_rank[pr[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def betas_fspr(self, ranking: List[int], beta_value: float,
                   gamma: float = 0.15) -> Optional[Tuple[Dict[int, Tuple[float, int]], float]]:
        """
        FSPR with geometric base = 0.5 + beta_value for the target distribution.
        Binary groups only (len(ranking) == 2).

        Args:
            ranking: List of group IDs in priority order (must have length 2)
            beta_value: Beta value to add to the base 0.5
            gamma: PageRank damping factor (default 0.15)

        Returns:
            (Dict mapping node_id -> (pagerank, rank), runtime) or None if skipped
        """
        if len(ranking) != 2:
            return None

        if not FAIRLAR_AVAILABLE:
            raise ImportError("FSPR requires cvxopt.")

        if not FAIRGD_AVAILABLE:
            raise ImportError("FSPR requires FairGD utilities.")

        start_time = time.time()
        base = 0.5 + beta_value

        target_dist = self._build_binary_target_distribution(ranking, base=base)
        if target_dist is None:
            return None

        G_labeled = self._prepare_graph_for_fairgd()
        fgd = FairGD(G_labeled, attr_name='label')
        N = fgd.N

        phi_by_internal_idx = {}
        for orig_group, target_prop in target_dist.items():
            if orig_group in fgd.label_mapping:
                internal_idx = fgd.label_mapping[orig_group]
                phi_by_internal_idx[internal_idx] = target_prop

        phi = phi_by_internal_idx.get(1, 0.5)
        index = [fgd.labels[i] == 1 for i in range(N)]

        M = np.zeros((N, N))
        nodelist = list(G_labeled.nodes())
        for i, node_i in enumerate(nodelist):
            for j, node_j in enumerate(nodelist):
                if G_labeled.has_edge(node_i, node_j):
                    M[i][j] = 1.0

        for i in range(N):
            if not M[i].any():
                M[i] = np.ones(N)

        d = np.reciprocal(M.sum(axis=1))
        D = np.diag(d)
        P = D.dot(M)
        Q = gamma * np.linalg.inv(np.eye(N) - (1 - gamma) * P)
        Q = Q.T

        p = Q.dot(np.ones(N) / N)

        G_ineq = matrix(-np.eye(N))
        h = matrix(np.zeros(N))

        A_eq = np.vstack([Q[index, :].sum(axis=0), np.ones(N)])
        A = matrix(A_eq)
        b = matrix([phi, 1.0])

        P_qp = matrix(Q.T @ Q)
        q = matrix(-Q.T @ p)

        try:
            sol = solvers.qp(P=P_qp, q=q, G=G_ineq, h=h, A=A, b=b)
            if sol['status'] != 'optimal':
                return None
            v_opt = np.array(sol['x']).flatten()
        except Exception:
            return None

        pr_array = Q @ v_opt
        pr = {nodelist[i]: pr_array[i] for i in range(N)}

        unique_values = sorted(set(pr.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (pr[node], value_to_rank[pr[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def betas_lfpr_u(self, ranking: List[int],
                     beta_value: float) -> Optional[Tuple[Dict[int, Tuple[float, int]], float]]:
        """
        LFPR_U with geometric base = 0.5 + beta_value for the target distribution.
        Binary groups only (len(ranking) == 2).

        Args:
            ranking: List of group IDs in priority order (must have length 2)
            beta_value: Beta value to add to the base 0.5

        Returns:
            (Dict mapping node_id -> (pagerank, rank), runtime) or None if skipped
        """
        if len(ranking) != 2:
            return None

        if not FAIRGD_AVAILABLE:
            raise ImportError("LFPR_U requires FairGD utilities.")

        start_time = time.time()
        base = 0.5 + beta_value

        target_dist = self._build_binary_target_distribution(ranking, base=base)
        if target_dist is None:
            return None

        G_labeled = self._prepare_graph_for_fairgd()
        fgd = FairGD(G_labeled, attr_name='label')

        phi_internal = {}
        for orig_group, target_prop in target_dist.items():
            if orig_group in fgd.label_mapping:
                internal_idx = fgd.label_mapping[orig_group]
                phi_internal[internal_idx] = target_prop

        _, new_P = fairgd_utils.w_diff_LFPRU_newP_optimized(G_labeled, phi_internal)
        pr_array, _ = fairgd_utils.get_pagerank(G_labeled, new_P)

        nodelist = list(G_labeled.nodes())
        pr = {nodelist[i]: pr_array[i] for i in range(len(nodelist))}

        unique_values = sorted(set(pr.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (pr[node], value_to_rank[pr[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def betas_lfpr_n(self, ranking: List[int],
                     beta_value: float) -> Optional[Tuple[Dict[int, Tuple[float, int]], float]]:
        """
        LFPR_N with geometric base = 0.5 + beta_value for the target distribution.
        Binary groups only (len(ranking) == 2).

        Args:
            ranking: List of group IDs in priority order (must have length 2)
            beta_value: Beta value to add to the base 0.5

        Returns:
            (Dict mapping node_id -> (pagerank, rank), runtime) or None if skipped
        """
        if len(ranking) != 2:
            return None

        if not FAIRGD_AVAILABLE:
            raise ImportError("LFPR_N requires FairGD utilities.")

        start_time = time.time()
        base = 0.5 + beta_value

        target_dist = self._build_binary_target_distribution(ranking, base=base)
        if target_dist is None:
            return None

        G_labeled = self._prepare_graph_for_fairgd()
        fgd = FairGD(G_labeled, attr_name='label')

        phi_internal = {}
        for orig_group, target_prop in target_dist.items():
            if orig_group in fgd.label_mapping:
                internal_idx = fgd.label_mapping[orig_group]
                phi_internal[internal_idx] = target_prop

        _, new_P = fairgd_utils.w_diff_LFPRN_newP_optimized(G_labeled, phi_internal)
        pr_array, _ = fairgd_utils.get_pagerank(G_labeled, new_P)

        nodelist = list(G_labeled.nodes())
        pr = {nodelist[i]: pr_array[i] for i in range(len(nodelist))}

        unique_values = sorted(set(pr.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (pr[node], value_to_rank[pr[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def betweenness_centrality(self) -> Dict[int, Tuple[float, int]]:
        """
        Betweenness centrality (group-agnostic)

        Returns:
            Dict mapping node_id -> (betweenness, rank)
        """
        start_time = time.time()
        
        betweenness = nx.betweenness_centrality(self.G)
        
        # Assign ranks (higher betweenness = better rank)
        unique_values = sorted(set(betweenness.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}
        
        result = {
            node: (betweenness[node], value_to_rank[betweenness[node]])
            for node in self.G.nodes()
        }
        
        runtime = time.time() - start_time
        
        return result, runtime
    
    def pagerank(self) -> Dict[int, Tuple[float, int]]:
        """
        PageRank centrality (group-agnostic)

        Returns:
            Dict mapping node_id -> (pagerank, rank)
        """
        start_time = time.time()

        pr = nx.pagerank(self.G)

        # Assign ranks (higher PageRank = better rank)
        unique_values = sorted(set(pr.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (pr[node], value_to_rank[pr[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time

        return result, runtime

    def pagerank_uniform_jump(self, ranking: List[int]) -> Tuple[Dict[int, Tuple[float, int]], float]:
        """
        PageRank with uniform jump distribution over nodes in ranking groups.

        The personalization (jump) distribution is:
        - 1 for nodes whose group appears in the ranking
        - 0 for nodes whose group does not appear in the ranking

        NetworkX normalizes the personalization vector internally.

        Args:
            ranking: List of group IDs in priority order

        Returns:
            (Dict mapping node_id -> (pagerank, rank), runtime)
        """
        start_time = time.time()

        ranking_groups = set(ranking)

        # Build personalization dict: uniform over nodes in ranking groups
        personalization = {}
        for node in self.G.nodes():
            if self.group_assignments[node] in ranking_groups:
                personalization[node] = 1.0
            else:
                personalization[node] = 0.0

        # Handle edge case: if no nodes are in ranking groups, fall back to standard PageRank
        if sum(personalization.values()) == 0:
            pr = nx.pagerank(self.G)
        else:
            pr = nx.pagerank(self.G, personalization=personalization)

        # Assign ranks (higher PageRank = better rank)
        unique_values = sorted(set(pr.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (pr[node], value_to_rank[pr[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time

        return result, runtime

    def pagerank_exponential_jump(self, ranking: List[int]) -> Tuple[Dict[int, Tuple[float, int]], float]:
        """
        PageRank with exponential decay jump distribution based on group rank.

        The personalization (jump) distribution is:
        - 0 if the node's group does not appear in the ranking
        - (0.5)^(rank(group) - 1) if the node's group is at position rank in the ranking
          (position 1 = first in the list, gets weight 1.0;
           position 2 gets weight 0.5;
           position 3 gets weight 0.25; etc.)

        NetworkX normalizes the personalization vector internally.

        Args:
            ranking: List of group IDs in priority order

        Returns:
            (Dict mapping node_id -> (pagerank, rank), runtime)
        """
        start_time = time.time()

        # Map group_id -> position (1-indexed)
        group_to_position = {group_id: pos + 1 for pos, group_id in enumerate(ranking)}

        # Build personalization dict: exponential decay based on group position
        personalization = {}
        for node in self.G.nodes():
            node_group = self.group_assignments[node]
            if node_group in group_to_position:
                position = group_to_position[node_group]
                personalization[node] = (0.5) ** (position - 1)
            else:
                personalization[node] = 0.0

        # Handle edge case: if no nodes are in ranking groups, fall back to standard PageRank
        if sum(personalization.values()) == 0:
            pr = nx.pagerank(self.G)
        else:
            pr = nx.pagerank(self.G, personalization=personalization)

        # Assign ranks (higher PageRank = better rank)
        unique_values = sorted(set(pr.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (pr[node], value_to_rank[pr[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time

        return result, runtime

    def _build_fairgd_target_distribution(self, ranking: List[int], base: float = 0.5) -> Dict[int, float]:
        """
        Build target distribution for FairGD/AdaptGD based on ranking.

        Distribution:
        - For group at position i (1-indexed) in ranking: weight = base^(i-1)
        - For groups NOT in ranking: weight = (0.5)^(s+1) where s = len(ranking)

        The weights are normalized to sum to 1.

        Args:
            ranking: List of group IDs in priority order
            base: Geometric base for ranked groups (default 0.5)

        Returns:
            Dict mapping group_id -> target_proportion (normalized)
        """
        s = len(ranking)
        ranking_groups = set(ranking)

        # Map group_id -> position (1-indexed)
        group_to_position = {group_id: pos + 1 for pos, group_id in enumerate(ranking)}

        # Calculate raw weights for each group
        weights = {}
        for group_id in self.groups:
            if group_id in ranking_groups:
                position = group_to_position[group_id]
                weights[group_id] = base ** (position - 1)
            else:
                weights[group_id] = (0.5) ** (s + 1)

        # Normalize to sum to 1
        total = sum(weights.values())
        normalized = {g: w / total for g, w in weights.items()}

        return normalized

    def _prepare_graph_for_fairgd(self) -> nx.Graph:
        """
        Create a copy of the graph with group labels as node attributes.

        FairGD expects node attributes named 'label' for group membership.

        Returns:
            NetworkX graph with 'label' attribute on each node
        """
        G_copy = self.G.copy()
        for node in G_copy.nodes():
            G_copy.nodes[node]['label'] = self.group_assignments[node]
        return G_copy

    def fairgd(self, ranking: List[int], gamma: float = 0.15, lr: float = 1.0,
               max_iter: int = 50, verbose: bool = False) -> Tuple[Dict[int, Tuple[float, int]], float]:
        """
        FairGD baseline: PageRank with edge reweighting to achieve target group distribution.

        Uses the FairGD algorithm from "Fair PageRank via Edge Reweighting".
        Target distribution:
        - Groups in ranking: (0.5)^(position - 1)
        - Groups not in ranking: (0.5)^(s + 1) where s = len(ranking)

        Args:
            ranking: List of group IDs in priority order
            gamma: PageRank damping factor (default 0.15)
            lr: Learning rate for gradient descent (default 1.0)
            max_iter: Maximum iterations (default 50)
            verbose: Print progress during optimization

        Returns:
            (Dict mapping node_id -> (pagerank, rank), runtime)
        """
        if not FAIRGD_AVAILABLE:
            raise ImportError("FairGD not available. Ensure the Fair-Pagerank-via-Edge-Rewighting "
                              "repository is in src/other_baselines/")

        start_time = time.time()

        # Prepare graph with labels
        G_labeled = self._prepare_graph_for_fairgd()

        # Create FairGD object
        fgd = FairGD(G_labeled, attr_name='label')

        # Build target distribution
        target_dist = self._build_fairgd_target_distribution(ranking)

        # Convert to array format expected by FairGD (indexed by internal label mapping)
        # FairGD uses its own label_mapping which maps original labels to 0-indexed integers
        targets = np.zeros(fgd.n_labels)
        for orig_label, proportion in target_dist.items():
            if orig_label in fgd.label_mapping:
                internal_idx = fgd.label_mapping[orig_label]
                targets[internal_idx] = proportion

        # Run FairGD optimization
        _, loss, _ = fgd.run_fpr(targets=targets, gamma=gamma, lr=lr,
                                  max_iter=max_iter, verbose=verbose)

        # Get final PageRank scores
        pr_array = fgd.pagerank(gamma=gamma)

        # Map back to node IDs
        # FairGD uses node order from G_labeled.nodes()
        nodelist = list(G_labeled.nodes())
        pr = {nodelist[i]: pr_array[i] for i in range(len(nodelist))}

        # Assign ranks (higher PageRank = better rank)
        unique_values = sorted(set(pr.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (pr[node], value_to_rank[pr[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time

        return result, runtime

    def adaptgd(self, ranking: List[int], gamma: float = 0.15, lr: float = 1.0,
                max_iter: int = 50, verbose: bool = False) -> Tuple[Dict[int, Tuple[float, int]], float]:
        """
        AdaptGD baseline: Group-adapted PageRank with edge reweighting.

        Uses the AdaptGD algorithm from "Fair PageRank via Edge Reweighting".
        Similar to FairGD but uses group-adapted restart vectors.
        Target distribution:
        - Groups in ranking: (0.5)^(position - 1)
        - Groups not in ranking: (0.5)^(s + 1) where s = len(ranking)

        Args:
            ranking: List of group IDs in priority order
            gamma: PageRank damping factor (default 0.15)
            lr: Learning rate for gradient descent (default 1.0)
            max_iter: Maximum iterations (default 50)
            verbose: Print progress during optimization

        Returns:
            (Dict mapping node_id -> (pagerank, rank), runtime)
        """
        if not FAIRGD_AVAILABLE:
            raise ImportError("FairGD not available. Ensure the Fair-Pagerank-via-Edge-Rewighting "
                              "repository is in src/other_baselines/")

        start_time = time.time()

        # Prepare graph with labels
        G_labeled = self._prepare_graph_for_fairgd()

        # Create FairGD object
        fgd = FairGD(G_labeled, attr_name='label')

        # Build target distribution
        target_dist = self._build_fairgd_target_distribution(ranking)

        # Convert to array format expected by FairGD (indexed by internal label mapping)
        targets = np.zeros(fgd.n_labels)
        for orig_label, proportion in target_dist.items():
            if orig_label in fgd.label_mapping:
                internal_idx = fgd.label_mapping[orig_label]
                targets[internal_idx] = proportion

        # Run AdaptGD optimization (run_cfpr)
        _, loss, _ = fgd.run_cfpr(targets=targets, gamma=gamma, lr=lr,
                                   max_iter=max_iter, verbose=verbose)

        # Get final PageRank scores
        pr_array = fgd.pagerank(gamma=gamma)

        # Map back to node IDs
        nodelist = list(G_labeled.nodes())
        pr = {nodelist[i]: pr_array[i] for i in range(len(nodelist))}

        # Assign ranks (higher PageRank = better rank)
        unique_values = sorted(set(pr.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (pr[node], value_to_rank[pr[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time

        return result, runtime

    def _build_binary_target_distribution(self, ranking: List[int], base: float = 0.5) -> Optional[Dict[int, float]]:
        """
        Build target distribution for FairLaR methods (binary groups only).

        Only works when len(ranking) == 2. Returns None otherwise.

        Distribution:
        - Group at position i (0-indexed): weight = base^i
        - Groups NOT in ranking: weight = (0.5)^(s+1)

        Normalized to sum to 1.

        Args:
            ranking: List of group IDs in priority order
            base: Geometric base for ranked groups (default 0.5)

        Returns:
            Dict mapping internal_group_index (0 or 1) -> target_proportion, or None if not applicable
        """
        if len(ranking) != 2:
            return None

        # Calculate raw weights
        s = len(ranking)  # = 2
        raw_weights = {}
        for pos, group_id in enumerate(ranking):
            raw_weights[group_id] = base ** pos

        # Groups not in ranking get (0.5)^(s+1)
        ranking_groups = set(ranking)
        for group_id in self.groups:
            if group_id not in ranking_groups:
                raw_weights[group_id] = (0.5) ** (s + 1)

        # Normalize
        total = sum(raw_weights.values())
        normalized = {g: w / total for g, w in raw_weights.items()}

        return normalized

    def fspr(self, ranking: List[int], gamma: float = 0.15) -> Optional[Tuple[Dict[int, Tuple[float, int]], float]]:
        """
        FSPR 

        Uses quadratic programming to find optimal jump vector that minimizes
        distance from uniform PageRank while satisfying fairness constraints.

        Only works when len(ranking) == 2 (binary groups). Returns None otherwise.

        Target distribution:
        - Groups in ranking: (0.5)^(position - 1)
        - Groups not in ranking: (0.5)^(s + 1) where s = len(ranking)

        Args:
            ranking: List of group IDs in priority order (must have length 2)
            gamma: PageRank damping factor (default 0.15)

        Returns:
            (Dict mapping node_id -> (pagerank, rank), runtime) or None if skipped
        """
        if len(ranking) != 2:
            return None

        if not FAIRLAR_AVAILABLE:
            raise ImportError("FSPR requires cvxopt. Install with: pip install cvxopt")

        if not FAIRGD_AVAILABLE:
            raise ImportError("FSPR requires FairGD utilities.")

        start_time = time.time()

        # Build target distribution
        target_dist = self._build_binary_target_distribution(ranking)
        if target_dist is None:
            return None

        # Prepare graph with labels
        G_labeled = self._prepare_graph_for_fairgd()

        # Create FairGD object to get label mapping and Q matrix
        fgd = FairGD(G_labeled, attr_name='label')
        N = fgd.N

        # Build index array: index[i] = True if node i belongs to the "protected" group (group at position 1 in ranking)
        # In FairLaR convention, group 1 is the "protected" group
        # We need to map our ranking to their convention
        # The group at ranking position 0 (highest priority) -> we want its phi value
        group_at_pos1 = ranking[0]  # First in ranking
        group_at_pos2 = ranking[1]  # Second in ranking

        # Get phi for the protected group (internal index 1 in FairGD)
        # FairGD maps our group IDs to 0, 1, ... We need to find which internal index corresponds to which ranking position
        phi_by_internal_idx = {}
        for orig_group, target_prop in target_dist.items():
            if orig_group in fgd.label_mapping:
                internal_idx = fgd.label_mapping[orig_group]
                phi_by_internal_idx[internal_idx] = target_prop

        # phi for protected group (internal index 1)
        phi = phi_by_internal_idx.get(1, 0.5)

        # Build index: True if node belongs to internal group 1
        index = [fgd.labels[i] == 1 for i in range(N)]

        # Build adjacency matrix
        M = np.zeros((N, N))
        nodelist = list(G_labeled.nodes())
        for i, node_i in enumerate(nodelist):
            for j, node_j in enumerate(nodelist):
                if G_labeled.has_edge(node_i, node_j):
                    M[i][j] = 1.0

        # Handle sink nodes (nodes with no outgoing edges)
        for i in range(N):
            if not M[i].any():
                M[i] = np.ones(N)

        # Compute Q matrix: P(v) = Qv where v is jump vector
        d = np.reciprocal(M.sum(axis=1))
        D = np.diag(d)
        P = D.dot(M)
        Q = gamma * np.linalg.inv(np.eye(N) - (1 - gamma) * P)
        Q = Q.T

        # Solve QP: minimize ||Qv - p||^2 subject to:
        #   - v >= 0
        #   - sum(v) = 1
        #   - sum(Q[index,:]) * v = phi
        p = Q.dot(np.ones(N) / N)  # Uniform PageRank

        # Inequality constraint: -v <= 0 (i.e., v >= 0)
        G_ineq = matrix(-np.eye(N))
        h = matrix(np.zeros(N))

        # Equality constraints
        A_eq = np.vstack([Q[index, :].sum(axis=0), np.ones(N)])
        A = matrix(A_eq)
        b = matrix([phi, 1.0])

        # Objective: minimize v^T Q^T Q v - 2 p^T Q v
        P_qp = matrix(Q.T @ Q)
        q = matrix(-Q.T @ p)

        # Solve
        try:
            sol = solvers.qp(P=P_qp, q=q, G=G_ineq, h=h, A=A, b=b)
            if sol['status'] != 'optimal':
                return None
            v_opt = np.array(sol['x']).flatten()
        except Exception:
            return None

        # Compute fair PageRank
        pr_array = Q @ v_opt

        # Map back to node IDs
        pr = {nodelist[i]: pr_array[i] for i in range(N)}

        # Assign ranks (higher PageRank = better rank)
        unique_values = sorted(set(pr.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (pr[node], value_to_rank[pr[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time

        return result, runtime

    def lfpr_u(self, ranking: List[int]) -> Optional[Tuple[Dict[int, Tuple[float, int]], float]]:
        """
        LFPR_U

        Modifies transition matrix to achieve target group PageRank by
        redistributing residual mass uniformly within each group.

        Only works when len(ranking) == 2 (binary groups). Returns None otherwise.

        Target distribution:
        - Groups in ranking: (0.5)^(position - 1)
        - Groups not in ranking: (0.5)^(s + 1) where s = len(ranking)

        Args:
            ranking: List of group IDs in priority order (must have length 2)

        Returns:
            (Dict mapping node_id -> (pagerank, rank), runtime) or None if skipped
        """
        if len(ranking) != 2:
            return None

        if not FAIRGD_AVAILABLE:
            raise ImportError("LFPR_U requires FairGD utilities.")

        start_time = time.time()

        # Build target distribution
        target_dist = self._build_binary_target_distribution(ranking)
        if target_dist is None:
            return None

        # Prepare graph with labels
        G_labeled = self._prepare_graph_for_fairgd()

        # Create FairGD object
        fgd = FairGD(G_labeled, attr_name='label')

        # Convert target distribution to internal indices
        phi_internal = {}
        for orig_group, target_prop in target_dist.items():
            if orig_group in fgd.label_mapping:
                internal_idx = fgd.label_mapping[orig_group]
                phi_internal[internal_idx] = target_prop

        # Call LFPR_U implementation
        _, new_P = fairgd_utils.w_diff_LFPRU_newP_optimized(G_labeled, phi_internal)

        # Get PageRank from modified transition matrix
        pr_array, _ = fairgd_utils.get_pagerank(G_labeled, new_P)

        # Map back to node IDs
        nodelist = list(G_labeled.nodes())
        pr = {nodelist[i]: pr_array[i] for i in range(len(nodelist))}

        # Assign ranks (higher PageRank = better rank)
        unique_values = sorted(set(pr.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (pr[node], value_to_rank[pr[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time

        return result, runtime

    def lfpr_n(self, ranking: List[int]) -> Optional[Tuple[Dict[int, Tuple[float, int]], float]]:
        """
        LFPR_N

        Modifies transition matrix to achieve target group PageRank by
        redistributing weight only among existing neighbors.

        Only works when len(ranking) == 2 (binary groups). Returns None otherwise.

        Target distribution:
        - Groups in ranking: (0.5)^(position - 1)
        - Groups not in ranking: (0.5)^(s + 1) where s = len(ranking)

        Args:
            ranking: List of group IDs in priority order (must have length 2)

        Returns:
            (Dict mapping node_id -> (pagerank, rank), runtime) or None if skipped
        """
        if len(ranking) != 2:
            return None

        if not FAIRGD_AVAILABLE:
            raise ImportError("LFPR_N requires FairGD utilities.")

        start_time = time.time()

        # Build target distribution
        target_dist = self._build_binary_target_distribution(ranking)
        if target_dist is None:
            return None

        # Prepare graph with labels
        G_labeled = self._prepare_graph_for_fairgd()

        # Create FairGD object
        fgd = FairGD(G_labeled, attr_name='label')

        # Convert target distribution to internal indices
        phi_internal = {}
        for orig_group, target_prop in target_dist.items():
            if orig_group in fgd.label_mapping:
                internal_idx = fgd.label_mapping[orig_group]
                phi_internal[internal_idx] = target_prop

        # Call LFPR_N implementation
        _, new_P = fairgd_utils.w_diff_LFPRN_newP_optimized(G_labeled, phi_internal)

        # Get PageRank from modified transition matrix
        pr_array, _ = fairgd_utils.get_pagerank(G_labeled, new_P)

        # Map back to node IDs
        nodelist = list(G_labeled.nodes())
        pr = {nodelist[i]: pr_array[i] for i in range(len(nodelist))}

        # Assign ranks (higher PageRank = better rank)
        unique_values = sorted(set(pr.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (pr[node], value_to_rank[pr[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time

        return result, runtime

    def random_baseline(self, ranking: List[int], seed_offset: int = 0) -> Dict[int, Tuple[float, int]]:
        """
        Random baseline - assign random scores

        Args:
            seed_offset: An additional offset for reproducibility
            ranking: is used ONLY to generate a unique seed

        Returns:
            Dict mapping node_id -> (random_score, rank)
        """
        start_time = time.time()

        # Create a unique seed based on the ranking (and a fixed base seed)
        # This ensures that 'random' results are different for different rankings
        # but reproducible for the same ranking.
        base_seed = 42 # A fixed base seed for reproducibility across runs
        seed = base_seed + hash(tuple(ranking)) % (2**32 - 1) + seed_offset # Combine base, ranking, and offset
        rng = np.random.default_rng(seed)


        scores = {node: rng.random() for node in self.G.nodes()}
        
        # Assign ranks
        unique_values = sorted(set(scores.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}
        
        result = {
            node: (scores[node], value_to_rank[scores[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time

        return result, runtime

    @staticmethod
    def compute_h_index(values: List[int]) -> int:
        """
        Compute the H-index of a list of values.

        The H-index is the maximum value h such that at least h values are >= h.

        Args:
            values: List of integer values

        Returns:
            The H-index
        """
        if not values:
            return 0
        sorted_values = sorted(values, reverse=True)
        h = 0
        for i, v in enumerate(sorted_values):
            if v >= i + 1:
                h = i + 1
            else:
                break
        return h

    def h1_index(self) -> Tuple[Dict[int, Tuple[int, int]], float]:
        """
        H1-index: For each node, compute the H-index of the degrees of its neighbors.

        Higher H1 value = better rank (rank 1 is best).

        Returns:
            Dict mapping node_id -> (h1_value, rank), runtime
        """
        start_time = time.time()

        # Get degrees for all nodes
        degrees = dict(self.G.degree())

        # Compute H1 for each node
        h1_values = {}
        for node in self.G.nodes():
            neighbor_degrees = [degrees[neighbor] for neighbor in self.G.neighbors(node)]
            h1_values[node] = self.compute_h_index(neighbor_degrees)

        # Assign ranks (higher H1 = better rank)
        unique_values = sorted(set(h1_values.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (h1_values[node], value_to_rank[h1_values[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def h2_index(self) -> Tuple[Dict[int, Tuple[int, int]], float]:
        """
        H2-index: For each node, compute the H-index of the H1 values of its neighbors.

        The chain: degrees -> H1 -> H2
        - H1(v) = H-index of {degree(u) : u in neighbors(v)}
        - H2(v) = H-index of {H1(u) : u in neighbors(v)}

        Higher H2 value = better rank (rank 1 is best).

        Returns:
            Dict mapping node_id -> (h2_value, rank), runtime
        """
        start_time = time.time()

        # Get degrees for all nodes
        degrees = dict(self.G.degree())

        # Compute H1 for each node
        h1_values = {}
        for node in self.G.nodes():
            neighbor_degrees = [degrees[neighbor] for neighbor in self.G.neighbors(node)]
            h1_values[node] = self.compute_h_index(neighbor_degrees)

        # Compute H2 for each node
        h2_values = {}
        for node in self.G.nodes():
            neighbor_h1 = [h1_values[neighbor] for neighbor in self.G.neighbors(node)]
            h2_values[node] = self.compute_h_index(neighbor_h1)

        # Assign ranks (higher H2 = better rank)
        unique_values = sorted(set(h2_values.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (h2_values[node], value_to_rank[h2_values[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def h3_index(self) -> Tuple[Dict[int, Tuple[int, int]], float]:
        """
        H3-index: For each node, compute the H-index of the H2 values of its neighbors.

        The chain: degrees -> H1 -> H2 -> H3
        - H1(v) = H-index of {degree(u) : u in neighbors(v)}
        - H2(v) = H-index of {H1(u) : u in neighbors(v)}
        - H3(v) = H-index of {H2(u) : u in neighbors(v)}

        Higher H3 value = better rank (rank 1 is best).

        Returns:
            Dict mapping node_id -> (h3_value, rank), runtime
        """
        start_time = time.time()

        # Get degrees for all nodes
        degrees = dict(self.G.degree())

        # Compute H1 for each node
        h1_values = {}
        for node in self.G.nodes():
            neighbor_degrees = [degrees[neighbor] for neighbor in self.G.neighbors(node)]
            h1_values[node] = self.compute_h_index(neighbor_degrees)

        # Compute H2 for each node
        h2_values = {}
        for node in self.G.nodes():
            neighbor_h1 = [h1_values[neighbor] for neighbor in self.G.neighbors(node)]
            h2_values[node] = self.compute_h_index(neighbor_h1)

        # Compute H3 for each node
        h3_values = {}
        for node in self.G.nodes():
            neighbor_h2 = [h2_values[neighbor] for neighbor in self.G.neighbors(node)]
            h3_values[node] = self.compute_h_index(neighbor_h2)

        # Assign ranks (higher H3 = better rank)
        unique_values = sorted(set(h3_values.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (h3_values[node], value_to_rank[h3_values[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def h4_index(self) -> Tuple[Dict[int, Tuple[int, int]], float]:
        """
        H4-index: For each node, compute the H-index of the H3 values of its neighbors.

        The chain: degrees -> H1 -> H2 -> H3 -> H4

        Higher H4 value = better rank (rank 1 is best).

        Returns:
            Dict mapping node_id -> (h4_value, rank), runtime
        """
        start_time = time.time()

        # Get degrees for all nodes
        degrees = dict(self.G.degree())

        # Compute H1 for each node
        h1_values = {}
        for node in self.G.nodes():
            neighbor_degrees = [degrees[neighbor] for neighbor in self.G.neighbors(node)]
            h1_values[node] = self.compute_h_index(neighbor_degrees)

        # Compute H2 for each node
        h2_values = {}
        for node in self.G.nodes():
            neighbor_h1 = [h1_values[neighbor] for neighbor in self.G.neighbors(node)]
            h2_values[node] = self.compute_h_index(neighbor_h1)

        # Compute H3 for each node
        h3_values = {}
        for node in self.G.nodes():
            neighbor_h2 = [h2_values[neighbor] for neighbor in self.G.neighbors(node)]
            h3_values[node] = self.compute_h_index(neighbor_h2)

        # Compute H4 for each node
        h4_values = {}
        for node in self.G.nodes():
            neighbor_h3 = [h3_values[neighbor] for neighbor in self.G.neighbors(node)]
            h4_values[node] = self.compute_h_index(neighbor_h3)

        # Assign ranks (higher H4 = better rank)
        unique_values = sorted(set(h4_values.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (h4_values[node], value_to_rank[h4_values[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def h5_index(self) -> Tuple[Dict[int, Tuple[int, int]], float]:
        """
        H5-index: For each node, compute the H-index of the H4 values of its neighbors.

        The chain: degrees -> H1 -> H2 -> H3 -> H4 -> H5

        Higher H5 value = better rank (rank 1 is best).

        Returns:
            Dict mapping node_id -> (h5_value, rank), runtime
        """
        start_time = time.time()

        # Get degrees for all nodes
        degrees = dict(self.G.degree())

        # Compute H1 for each node
        h1_values = {}
        for node in self.G.nodes():
            neighbor_degrees = [degrees[neighbor] for neighbor in self.G.neighbors(node)]
            h1_values[node] = self.compute_h_index(neighbor_degrees)

        # Compute H2 for each node
        h2_values = {}
        for node in self.G.nodes():
            neighbor_h1 = [h1_values[neighbor] for neighbor in self.G.neighbors(node)]
            h2_values[node] = self.compute_h_index(neighbor_h1)

        # Compute H3 for each node
        h3_values = {}
        for node in self.G.nodes():
            neighbor_h2 = [h2_values[neighbor] for neighbor in self.G.neighbors(node)]
            h3_values[node] = self.compute_h_index(neighbor_h2)

        # Compute H4 for each node
        h4_values = {}
        for node in self.G.nodes():
            neighbor_h3 = [h3_values[neighbor] for neighbor in self.G.neighbors(node)]
            h4_values[node] = self.compute_h_index(neighbor_h3)

        # Compute H5 for each node
        h5_values = {}
        for node in self.G.nodes():
            neighbor_h4 = [h4_values[neighbor] for neighbor in self.G.neighbors(node)]
            h5_values[node] = self.compute_h_index(neighbor_h4)

        # Assign ranks (higher H5 = better rank)
        unique_values = sorted(set(h5_values.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (h5_values[node], value_to_rank[h5_values[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def closeness_centrality(self) -> Tuple[Dict[int, Tuple[float, int]], float]:
        """
        Standard closeness centrality (group-agnostic).

        For each node, closeness = (n-1) / sum of distances to all reachable nodes.
        NetworkX handles disconnected graphs by normalizing over reachable nodes only.

        Higher closeness = better rank (rank 1 is best).

        Returns:
            Dict mapping node_id -> (closeness, rank), runtime
        """
        start_time = time.time()

        scores = nx.closeness_centrality(self.G)

        unique_values = sorted(set(scores.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (scores[node], value_to_rank[scores[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def gravity_centrality(self) -> Tuple[Dict[int, Tuple[float, int]], float]:
        """
        Gravity centrality based on k-core decomposition.

        For each node i:
            gc(i) = k_s(i) * sum_{j in phi(i)} k_s(j) / d(i,j)^2
        where phi(i) = {j : 1 <= d(i,j) <= 3} and k_s(.) is the coreness.

        Higher gc = better rank (rank 1 is best).

        Returns:
            Dict mapping node_id -> (gc_value, rank), runtime
        """
        start_time = time.time()

        core_number = nx.core_number(self.G)

        gc_values = {}
        for node in self.G.nodes():
            dist_dict = dict(nx.single_source_shortest_path_length(self.G, node, cutoff=3))
            gc_values[node] = core_number[node] * sum(
                core_number[j] / (d * d)
                for j, d in dist_dict.items()
                if j != node
            )

        unique_values = sorted(set(gc_values.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (gc_values[node], value_to_rank[gc_values[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def extended_gravity_centrality(self) -> Tuple[Dict[int, Tuple[float, int]], float]:
        """
        Extended gravity centrality: sum of neighbors' gravity centrality scores.

        For each node i:
            egc(i) = sum_{j in N(i)} gc(j)
        where gc(j) is the gravity_centrality of node j.

        Internally recomputes gravity_centrality (same pattern as h2_index recomputes h1).

        Higher egc = better rank (rank 1 is best).

        Returns:
            Dict mapping node_id -> (egc_value, rank), runtime
        """
        start_time = time.time()

        # Compute gravity values internally
        core_number = nx.core_number(self.G)
        gc_values = {}
        for node in self.G.nodes():
            dist_dict = dict(nx.single_source_shortest_path_length(self.G, node, cutoff=3))
            gc_values[node] = core_number[node] * sum(
                core_number[j] / (d * d)
                for j, d in dist_dict.items()
                if j != node
            )

        egc_values = {}
        for node in self.G.nodes():
            egc_values[node] = sum(gc_values[j] for j in self.G.neighbors(node))

        unique_values = sorted(set(egc_values.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (egc_values[node], value_to_rank[egc_values[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def local_gravity(self) -> Tuple[Dict[int, Tuple[float, int]], float]:
        """
        Local gravity centrality based on node degrees.

        For each node i:
            lg(i) = delta(i) * sum_{j in phi(i)} delta(j) / d(i,j)^2
        where phi(i) = {j : 1 <= d(i,j) <= 3} and delta(.) is the degree.

        Higher lg = better rank (rank 1 is best).

        Returns:
            Dict mapping node_id -> (lg_value, rank), runtime
        """
        start_time = time.time()

        degrees = dict(self.G.degree())

        lg_values = {}
        for node in self.G.nodes():
            dist_dict = dict(nx.single_source_shortest_path_length(self.G, node, cutoff=3))
            lg_values[node] = degrees[node] * sum(
                degrees[j] / (d * d)
                for j, d in dist_dict.items()
                if j != node
            )

        unique_values = sorted(set(lg_values.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (lg_values[node], value_to_rank[lg_values[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def gravity_h_index(self) -> Tuple[Dict[int, Tuple[float, int]], float]:
        """
        Gravity H-index centrality.

        For each node i:
            hv(i) = |{j in N(i) : deg(j) >= h1_index(i)}|
            p_{ij} = 1/delta(i)  for (i,j) in E
            c(i) = sum_{j in N(i)} [(1/delta(i)) * (1 + sum_{w in N(j) cap N(i)} 1/delta(w))]^2
            hvgc(i) = exp(-c(i)) * hv(i) * sum_{j in phi(i)} hv(j) / d(i,j)^2

        Higher hvgc = better rank (rank 1 is best).

        Returns:
            Dict mapping node_id -> (hvgc_value, rank), runtime
        """
        start_time = time.time()

        degrees = dict(self.G.degree())

        # Compute H1 index for each node
        h1_values = {}
        for node in self.G.nodes():
            neighbor_degrees = [degrees[neighbor] for neighbor in self.G.neighbors(node)]
            h1_values[node] = self.compute_h_index(neighbor_degrees)

        # Compute hv(i) = |{j in N(i) : deg(j) >= h1_index(i)}|
        hv_values = {}
        for node in self.G.nodes():
            h1 = h1_values[node]
            hv_values[node] = sum(1 for j in self.G.neighbors(node) if degrees[j] >= h1)

        # Compute c(i)
        c_values = {}
        for node in self.G.nodes():
            deg_i = degrees[node]
            if deg_i == 0:
                c_values[node] = 0.0
                continue
            neighbors_i = set(self.G.neighbors(node))
            c = 0.0
            inv_deg_i = 1.0 / deg_i
            for j in neighbors_i:
                common = set(self.G.neighbors(j)) & neighbors_i
                inner = inv_deg_i * (1.0 + sum(
                    1.0 / degrees[w] for w in common if degrees[w] > 0
                ))
                c += inner * inner
            c_values[node] = c

        # Compute hvgc(i) = exp(-c(i)) * hv(i) * sum_{j in phi(i)} hv(j) / d(i,j)^2
        hvgc_values = {}
        for node in self.G.nodes():
            if hv_values[node] == 0:
                hvgc_values[node] = 0.0
                continue
            dist_dict = dict(nx.single_source_shortest_path_length(self.G, node, cutoff=3))
            gravity_sum = sum(
                hv_values[j] / (d * d)
                for j, d in dist_dict.items()
                if j != node
            )
            hvgc_values[node] = np.exp(-c_values[node]) * hv_values[node] * gravity_sum

        unique_values = sorted(set(hvgc_values.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (hvgc_values[node], value_to_rank[hvgc_values[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def clustering_coefficient_centrality(self) -> Tuple[Dict[int, Tuple[float, int]], float]:
        """
        Clustering coefficient centrality.

        For each node i:
            ccc(i) = 2 * |E(i)| / (delta(i) * (delta(i) - 1))
        where |E(i)| is the number of edges in the subgraph induced by {i} union N(i).
        Set ccc(i) = 0 if delta(i) <= 1.

        Higher ccc = better rank (rank 1 is best).

        Returns:
            Dict mapping node_id -> (ccc_value, rank), runtime
        """
        start_time = time.time()

        degrees = dict(self.G.degree())

        ccc_values = {}
        for node in self.G.nodes():
            deg = degrees[node]
            if deg <= 1:
                ccc_values[node] = 0.0
                continue
            closed_neighborhood = [node] + list(self.G.neighbors(node))
            subgraph = self.G.subgraph(closed_neighborhood)
            E_i = subgraph.number_of_edges()
            ccc_values[node] = 2.0 * E_i / (deg * (deg - 1))

        unique_values = sorted(set(ccc_values.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (ccc_values[node], value_to_rank[ccc_values[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def local_global_centrality(self) -> Tuple[Dict[int, Tuple[float, int]], float]:
        """
        Local-global centrality combining degree and distance.

        For each node i:
            lgc(i) = (delta(i) / n) * sum_{j != i, j reachable} sqrt(delta(j)) / d(i,j)

        Unreachable nodes are excluded from the sum.
        Higher lgc = better rank (rank 1 is best).

        Returns:
            Dict mapping node_id -> (lgc_value, rank), runtime
        """
        start_time = time.time()

        degrees = dict(self.G.degree())
        n = self.n
        sqrt_degrees = {node: np.sqrt(degrees[node]) for node in self.G.nodes()}

        lgc_values = {}
        for node in self.G.nodes():
            deg_i = degrees[node]
            if deg_i == 0:
                lgc_values[node] = 0.0
                continue
            dist_dict = dict(nx.single_source_shortest_path_length(self.G, node))
            lgc_values[node] = (deg_i / n) * sum(
                sqrt_degrees[j] / d
                for j, d in dist_dict.items()
                if j != node
            )

        unique_values = sorted(set(lgc_values.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (lgc_values[node], value_to_rank[lgc_values[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def entropy_degree_distance_combination(self) -> Tuple[Dict[int, Tuple[float, int]], float]:
        """
        Entropy-degree-distance combination centrality.

        For each node i:
            p(j) = delta(j) / sum_{w in N(i)} delta(w)  for j in N(i)
            e(i) = -sum_{j in N(i)} p(j) * log2(p(j))   (0 if N(i) empty)
            eddc(i) = (delta(i) / n) * sum_{j != i, j reachable} sqrt(e(i) + e(j)) / d(i,j)

        Unreachable nodes are excluded from the sum.
        Higher eddc = better rank (rank 1 is best).

        Returns:
            Dict mapping node_id -> (eddc_value, rank), runtime
        """
        start_time = time.time()

        degrees = dict(self.G.degree())
        n = self.n

        # Compute entropy e(i) for each node
        e_values = {}
        for node in self.G.nodes():
            neighbors = list(self.G.neighbors(node))
            if not neighbors:
                e_values[node] = 0.0
                continue
            total_deg = sum(degrees[w] for w in neighbors)
            if total_deg == 0:
                e_values[node] = 0.0
                continue
            entropy = 0.0
            for j in neighbors:
                p = degrees[j] / total_deg
                if p > 0:
                    entropy -= p * np.log2(p)
            e_values[node] = entropy

        eddc_values = {}
        for node in self.G.nodes():
            deg_i = degrees[node]
            if deg_i == 0:
                eddc_values[node] = 0.0
                continue
            ei = e_values[node]
            dist_dict = dict(nx.single_source_shortest_path_length(self.G, node))
            eddc_values[node] = (deg_i / n) * sum(
                np.sqrt(ei + e_values[j]) / d
                for j, d in dist_dict.items()
                if j != node
            )

        unique_values = sorted(set(eddc_values.values()), reverse=True)
        value_to_rank = {val: rank + 1 for rank, val in enumerate(unique_values)}

        result = {
            node: (eddc_values[node], value_to_rank[eddc_values[node]])
            for node in self.G.nodes()
        }

        runtime = time.time() - start_time
        return result, runtime

    def get_top_spreader(self, method_result: Dict) -> int:
        """
        Get the node with rank 1 (best spreader) from method result
        
        If multiple nodes have rank 1, return the first one.
        
        Args:
            method_result: Dict from any method (node -> (value, rank))
        
        Returns:
            Node ID of top spreader
        """
        # Find all nodes with rank 1
        top_nodes = [node for node, (val, rank) in method_result.items() if rank == 1]
        
        if len(top_nodes) == 0:
            raise ValueError("No node with rank 1 found")
        
        # Return first one (arbitrary choice among ties)
        return top_nodes[0]

    def _combine_with_tiebreak(
        self,
        primary_result: Dict[int, Tuple[Any, int]],
        secondary_result: Dict[int, Tuple[Any, int]]
    ) -> Dict[int, Tuple[Any, int]]:
        """
        Combine two method results using the secondary as tie-breaker for the primary.

        For nodes with the same rank in primary_result, use their relative ordering
        in secondary_result to break ties.

        Args:
            primary_result: Dict mapping node -> (value, rank) from primary method
            secondary_result: Dict mapping node -> (value, rank) from secondary method

        Returns:
            Dict mapping node -> (primary_value, new_rank) with ties broken
        """
        # Group nodes by their primary rank
        nodes_by_primary_rank = {}
        for node, (value, rank) in primary_result.items():
            if rank not in nodes_by_primary_rank:
                nodes_by_primary_rank[rank] = []
            nodes_by_primary_rank[rank].append((node, value))

        # Build new ranking by processing primary ranks in order
        combined_result = {}
        current_rank = 1

        for primary_rank in sorted(nodes_by_primary_rank.keys()):
            nodes_in_group = nodes_by_primary_rank[primary_rank]

            if len(nodes_in_group) == 1:
                # No tie to break
                node, value = nodes_in_group[0]
                combined_result[node] = (value, current_rank)
                current_rank += 1
            else:
                # Sort nodes by their secondary rank
                nodes_with_secondary = []
                for node, value in nodes_in_group:
                    secondary_rank = secondary_result.get(node, (None, float('inf')))[1]
                    nodes_with_secondary.append((node, value, secondary_rank))

                # Sort by secondary rank (lower is better)
                nodes_with_secondary.sort(key=lambda x: x[2])

                # Assign new ranks, preserving ties in secondary
                prev_secondary_rank = None
                rank_for_group = current_rank

                for node, value, sec_rank in nodes_with_secondary:
                    if prev_secondary_rank is not None and sec_rank != prev_secondary_rank:
                        rank_for_group = current_rank
                    combined_result[node] = (value, rank_for_group)
                    prev_secondary_rank = sec_rank
                    current_rank += 1

        return combined_result

    def _restrict_to_ranking_groups(
        self,
        method_result: Dict[int, Tuple[Any, int]],
        ranking: List[int]
    ) -> Dict[int, Tuple[Any, int]]:
        """
        Restrict a method result by prioritizing nodes in ranking groups.

        Nodes belonging to groups in the ranking are moved to the top,
        preserving their relative ordering. Other nodes follow after.

        Args:
            method_result: Dict mapping node -> (value, rank) from any method
            ranking: List of group IDs that should be prioritized

        Returns:
            Dict mapping node -> (value, new_rank) with ranking group nodes first
        """
        ranking_groups = set(ranking)

        # Separate nodes into selected (in ranking groups) and non-selected
        selected_nodes = []  # (node, value, original_rank)
        non_selected_nodes = []  # (node, value, original_rank)

        for node, (value, rank) in method_result.items():
            group = self.group_assignments.get(node)
            if group in ranking_groups:
                selected_nodes.append((node, value, rank))
            else:
                non_selected_nodes.append((node, value, rank))

        # Sort both lists by original rank
        selected_nodes.sort(key=lambda x: x[2])
        non_selected_nodes.sort(key=lambda x: x[2])

        # Build new ranking: selected first, then non-selected
        restricted_result = {}
        current_rank = 1

        # Process selected nodes
        prev_original_rank = None
        rank_for_group = current_rank
        for node, value, orig_rank in selected_nodes:
            if prev_original_rank is not None and orig_rank != prev_original_rank:
                rank_for_group = current_rank
            restricted_result[node] = (value, rank_for_group)
            prev_original_rank = orig_rank
            current_rank += 1

        # Process non-selected nodes
        prev_original_rank = None
        rank_for_group = current_rank
        for node, value, orig_rank in non_selected_nodes:
            if prev_original_rank is not None and orig_rank != prev_original_rank:
                rank_for_group = current_rank
            restricted_result[node] = (value, rank_for_group)
            prev_original_rank = orig_rank
            current_rank += 1

        return restricted_result

    def _restrict_to_first_group(self, method_result: Dict[int, Tuple[Any, int]],
                                  ranking: List[int]) -> Dict[int, Tuple[Any, int]]:
        """
        Re-rank method results to prioritize nodes in the FIRST ranking group only.

        Like _restrict_to_ranking_groups but only considers ranking[0].
        """
        return self._restrict_to_ranking_groups(method_result, [ranking[0]])

    def restricted_coreness(self, ranking: List[int],
                            core_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Coreness with nodes in ranking groups prioritized"""
        start_time = time.time()

        if core_result is None:
            core_result, _ = self.coreness()
        restricted = self._restrict_to_ranking_groups(core_result, ranking)

        return restricted, time.time() - start_time

    def restricted_degree(self, ranking: List[int],
                          deg_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Degree with nodes in ranking groups prioritized"""
        start_time = time.time()

        if deg_result is None:
            deg_result, _ = self.degree_centrality()
        restricted = self._restrict_to_ranking_groups(deg_result, ranking)

        return restricted, time.time() - start_time

    def restricted_degree_lexicographic(self, ranking: List[int],
                                        deg_lex_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Lexicographic degree with nodes in ranking groups prioritized"""
        start_time = time.time()

        if deg_lex_result is None:
            deg_lex_result, _ = self.degree_centrality_lexicographic(ranking)
        restricted = self._restrict_to_ranking_groups(deg_lex_result, ranking)

        return restricted, time.time() - start_time

    def restricted_pagerank(self, ranking: List[int],
                            pr_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """PageRank with nodes in ranking groups prioritized"""
        start_time = time.time()

        if pr_result is None:
            pr_result, _ = self.pagerank()
        restricted = self._restrict_to_ranking_groups(pr_result, ranking)

        return restricted, time.time() - start_time

    def restricted_betweenness(self, ranking: List[int],
                               bet_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Betweenness with nodes in ranking groups prioritized"""
        start_time = time.time()

        if bet_result is None:
            bet_result, _ = self.betweenness_centrality()
        restricted = self._restrict_to_ranking_groups(bet_result, ranking)

        return restricted, time.time() - start_time

    def restricted_h1_index(self, ranking: List[int],
                            h1_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """H1-index with nodes in ranking groups prioritized"""
        start_time = time.time()

        if h1_result is None:
            h1_result, _ = self.h1_index()
        restricted = self._restrict_to_ranking_groups(h1_result, ranking)

        return restricted, time.time() - start_time

    def restricted_h2_index(self, ranking: List[int],
                            h2_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """H2-index with nodes in ranking groups prioritized"""
        start_time = time.time()

        if h2_result is None:
            h2_result, _ = self.h2_index()
        restricted = self._restrict_to_ranking_groups(h2_result, ranking)

        return restricted, time.time() - start_time

    def restricted_h3_index(self, ranking: List[int],
                            h3_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """H3-index with nodes in ranking groups prioritized"""
        start_time = time.time()

        if h3_result is None:
            h3_result, _ = self.h3_index()
        restricted = self._restrict_to_ranking_groups(h3_result, ranking)

        return restricted, time.time() - start_time

    def restricted_h4_index(self, ranking: List[int],
                            h4_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """H4-index with nodes in ranking groups prioritized"""
        start_time = time.time()

        if h4_result is None:
            h4_result, _ = self.h4_index()
        restricted = self._restrict_to_ranking_groups(h4_result, ranking)

        return restricted, time.time() - start_time

    def restricted_h5_index(self, ranking: List[int],
                            h5_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """H5-index with nodes in ranking groups prioritized"""
        start_time = time.time()

        if h5_result is None:
            h5_result, _ = self.h5_index()
        restricted = self._restrict_to_ranking_groups(h5_result, ranking)

        return restricted, time.time() - start_time

    # --- Restricted-first methods (restrict to first group only) ---

    def restricted_first_coreness(self, ranking: List[int],
                                  core_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Coreness with nodes in first ranking group prioritized"""
        start_time = time.time()
        if core_result is None:
            core_result, _ = self.coreness()
        restricted = self._restrict_to_first_group(core_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_degree(self, ranking: List[int],
                                deg_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Degree with nodes in first ranking group prioritized"""
        start_time = time.time()
        if deg_result is None:
            deg_result, _ = self.degree_centrality()
        restricted = self._restrict_to_first_group(deg_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_degree_lexicographic(self, ranking: List[int],
                                              deg_lex_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Lexicographic degree with nodes in first ranking group prioritized"""
        start_time = time.time()
        if deg_lex_result is None:
            deg_lex_result, _ = self.degree_centrality_lexicographic(ranking)
        restricted = self._restrict_to_first_group(deg_lex_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_pagerank(self, ranking: List[int],
                                  pr_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """PageRank with nodes in first ranking group prioritized"""
        start_time = time.time()
        if pr_result is None:
            pr_result, _ = self.pagerank()
        restricted = self._restrict_to_first_group(pr_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_betweenness(self, ranking: List[int],
                                     bet_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Betweenness with nodes in first ranking group prioritized"""
        start_time = time.time()
        if bet_result is None:
            bet_result, _ = self.betweenness_centrality()
        restricted = self._restrict_to_first_group(bet_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_h1_index(self, ranking: List[int],
                                  h1_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """H1-index with nodes in first ranking group prioritized"""
        start_time = time.time()
        if h1_result is None:
            h1_result, _ = self.h1_index()
        restricted = self._restrict_to_first_group(h1_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_h2_index(self, ranking: List[int],
                                  h2_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """H2-index with nodes in first ranking group prioritized"""
        start_time = time.time()
        if h2_result is None:
            h2_result, _ = self.h2_index()
        restricted = self._restrict_to_first_group(h2_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_h3_index(self, ranking: List[int],
                                  h3_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """H3-index with nodes in first ranking group prioritized"""
        start_time = time.time()
        if h3_result is None:
            h3_result, _ = self.h3_index()
        restricted = self._restrict_to_first_group(h3_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_h4_index(self, ranking: List[int],
                                  h4_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """H4-index with nodes in first ranking group prioritized"""
        start_time = time.time()
        if h4_result is None:
            h4_result, _ = self.h4_index()
        restricted = self._restrict_to_first_group(h4_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_h5_index(self, ranking: List[int],
                                  h5_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """H5-index with nodes in first ranking group prioritized"""
        start_time = time.time()
        if h5_result is None:
            h5_result, _ = self.h5_index()
        restricted = self._restrict_to_first_group(h5_result, ranking)
        return restricted, time.time() - start_time

    # --- Restricted methods for new baselines ---

    def restricted_closeness_centrality(self, ranking: List[int],
                                        closeness_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Closeness centrality with nodes in ranking groups prioritized"""
        start_time = time.time()
        if closeness_result is None:
            closeness_result, _ = self.closeness_centrality()
        restricted = self._restrict_to_ranking_groups(closeness_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_closeness_centrality(self, ranking: List[int],
                                              closeness_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Closeness centrality with nodes in first ranking group prioritized"""
        start_time = time.time()
        if closeness_result is None:
            closeness_result, _ = self.closeness_centrality()
        restricted = self._restrict_to_first_group(closeness_result, ranking)
        return restricted, time.time() - start_time

    def restricted_gravity_centrality(self, ranking: List[int],
                                      gravity_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Gravity centrality with nodes in ranking groups prioritized"""
        start_time = time.time()
        if gravity_result is None:
            gravity_result, _ = self.gravity_centrality()
        restricted = self._restrict_to_ranking_groups(gravity_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_gravity_centrality(self, ranking: List[int],
                                            gravity_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Gravity centrality with nodes in first ranking group prioritized"""
        start_time = time.time()
        if gravity_result is None:
            gravity_result, _ = self.gravity_centrality()
        restricted = self._restrict_to_first_group(gravity_result, ranking)
        return restricted, time.time() - start_time

    def restricted_extended_gravity_centrality(self, ranking: List[int],
                                               ext_gravity_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Extended gravity centrality with nodes in ranking groups prioritized"""
        start_time = time.time()
        if ext_gravity_result is None:
            ext_gravity_result, _ = self.extended_gravity_centrality()
        restricted = self._restrict_to_ranking_groups(ext_gravity_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_extended_gravity_centrality(self, ranking: List[int],
                                                     ext_gravity_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Extended gravity centrality with nodes in first ranking group prioritized"""
        start_time = time.time()
        if ext_gravity_result is None:
            ext_gravity_result, _ = self.extended_gravity_centrality()
        restricted = self._restrict_to_first_group(ext_gravity_result, ranking)
        return restricted, time.time() - start_time

    def restricted_local_gravity(self, ranking: List[int],
                                 local_gravity_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Local gravity with nodes in ranking groups prioritized"""
        start_time = time.time()
        if local_gravity_result is None:
            local_gravity_result, _ = self.local_gravity()
        restricted = self._restrict_to_ranking_groups(local_gravity_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_local_gravity(self, ranking: List[int],
                                       local_gravity_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Local gravity with nodes in first ranking group prioritized"""
        start_time = time.time()
        if local_gravity_result is None:
            local_gravity_result, _ = self.local_gravity()
        restricted = self._restrict_to_first_group(local_gravity_result, ranking)
        return restricted, time.time() - start_time

    def restricted_gravity_h_index(self, ranking: List[int],
                                   gravity_h_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Gravity H-index with nodes in ranking groups prioritized"""
        start_time = time.time()
        if gravity_h_result is None:
            gravity_h_result, _ = self.gravity_h_index()
        restricted = self._restrict_to_ranking_groups(gravity_h_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_gravity_h_index(self, ranking: List[int],
                                         gravity_h_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Gravity H-index with nodes in first ranking group prioritized"""
        start_time = time.time()
        if gravity_h_result is None:
            gravity_h_result, _ = self.gravity_h_index()
        restricted = self._restrict_to_first_group(gravity_h_result, ranking)
        return restricted, time.time() - start_time

    def restricted_clustering_coefficient_centrality(self, ranking: List[int],
                                                     clustering_cc_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Clustering coefficient centrality with nodes in ranking groups prioritized"""
        start_time = time.time()
        if clustering_cc_result is None:
            clustering_cc_result, _ = self.clustering_coefficient_centrality()
        restricted = self._restrict_to_ranking_groups(clustering_cc_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_clustering_coefficient_centrality(self, ranking: List[int],
                                                           clustering_cc_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Clustering coefficient centrality with nodes in first ranking group prioritized"""
        start_time = time.time()
        if clustering_cc_result is None:
            clustering_cc_result, _ = self.clustering_coefficient_centrality()
        restricted = self._restrict_to_first_group(clustering_cc_result, ranking)
        return restricted, time.time() - start_time

    def restricted_local_global_centrality(self, ranking: List[int],
                                           lgc_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Local-global centrality with nodes in ranking groups prioritized"""
        start_time = time.time()
        if lgc_result is None:
            lgc_result, _ = self.local_global_centrality()
        restricted = self._restrict_to_ranking_groups(lgc_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_local_global_centrality(self, ranking: List[int],
                                                 lgc_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Local-global centrality with nodes in first ranking group prioritized"""
        start_time = time.time()
        if lgc_result is None:
            lgc_result, _ = self.local_global_centrality()
        restricted = self._restrict_to_first_group(lgc_result, ranking)
        return restricted, time.time() - start_time

    def restricted_entropy_degree_distance_combination(self, ranking: List[int],
                                                       eddc_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Entropy-degree-distance combination with nodes in ranking groups prioritized"""
        start_time = time.time()
        if eddc_result is None:
            eddc_result, _ = self.entropy_degree_distance_combination()
        restricted = self._restrict_to_ranking_groups(eddc_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_entropy_degree_distance_combination(self, ranking: List[int],
                                                             eddc_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Entropy-degree-distance combination with nodes in first ranking group prioritized"""
        start_time = time.time()
        if eddc_result is None:
            eddc_result, _ = self.entropy_degree_distance_combination()
        restricted = self._restrict_to_first_group(eddc_result, ranking)
        return restricted, time.time() - start_time

    # --- Restricted versions of ranking-dependent methods ---

    def restricted_pagerank_uniform_jump(self, ranking: List[int],
                                         base_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """PageRank uniform jump with nodes in ranking groups prioritized"""
        start_time = time.time()
        if base_result is None:
            base_result, _ = self.pagerank_uniform_jump(ranking)
        restricted = self._restrict_to_ranking_groups(base_result, ranking)
        return restricted, time.time() - start_time

    def restricted_pagerank_exponential_jump(self, ranking: List[int],
                                             base_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """PageRank exponential jump with nodes in ranking groups prioritized"""
        start_time = time.time()
        if base_result is None:
            base_result, _ = self.pagerank_exponential_jump(ranking)
        restricted = self._restrict_to_ranking_groups(base_result, ranking)
        return restricted, time.time() - start_time

    def restricted_fairgd(self, ranking: List[int],
                          base_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """FairGD with nodes in ranking groups prioritized"""
        start_time = time.time()
        if base_result is None:
            base_result, _ = self.fairgd(ranking)
        restricted = self._restrict_to_ranking_groups(base_result, ranking)
        return restricted, time.time() - start_time

    def restricted_adaptgd(self, ranking: List[int],
                           base_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """AdaptGD with nodes in ranking groups prioritized"""
        start_time = time.time()
        if base_result is None:
            base_result, _ = self.adaptgd(ranking)
        restricted = self._restrict_to_ranking_groups(base_result, ranking)
        return restricted, time.time() - start_time

    def restricted_fspr(self, ranking: List[int],
                        base_result: Dict[int, Tuple[Any, int]] = None) -> Optional[Tuple[Dict[int, Tuple[Any, int]], float]]:
        """FSPR with nodes in ranking groups prioritized"""
        start_time = time.time()
        if base_result is None:
            result = self.fspr(ranking)
            if result is None:
                return None
            base_result, _ = result
        restricted = self._restrict_to_ranking_groups(base_result, ranking)
        return restricted, time.time() - start_time

    def restricted_lfpr_u(self, ranking: List[int],
                          base_result: Dict[int, Tuple[Any, int]] = None) -> Optional[Tuple[Dict[int, Tuple[Any, int]], float]]:
        """LFPR_U with nodes in ranking groups prioritized"""
        start_time = time.time()
        if base_result is None:
            result = self.lfpr_u(ranking)
            if result is None:
                return None
            base_result, _ = result
        restricted = self._restrict_to_ranking_groups(base_result, ranking)
        return restricted, time.time() - start_time

    def restricted_lfpr_n(self, ranking: List[int],
                          base_result: Dict[int, Tuple[Any, int]] = None) -> Optional[Tuple[Dict[int, Tuple[Any, int]], float]]:
        """LFPR_N with nodes in ranking groups prioritized"""
        start_time = time.time()
        if base_result is None:
            result = self.lfpr_n(ranking)
            if result is None:
                return None
            base_result, _ = result
        restricted = self._restrict_to_ranking_groups(base_result, ranking)
        return restricted, time.time() - start_time

    # --- Restricted-first versions of ranking-dependent methods ---

    def restricted_first_pagerank_uniform_jump(self, ranking: List[int],
                                               base_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """PageRank uniform jump with nodes in first ranking group prioritized"""
        start_time = time.time()
        if base_result is None:
            base_result, _ = self.pagerank_uniform_jump(ranking)
        restricted = self._restrict_to_first_group(base_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_pagerank_exponential_jump(self, ranking: List[int],
                                                   base_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """PageRank exponential jump with nodes in first ranking group prioritized"""
        start_time = time.time()
        if base_result is None:
            base_result, _ = self.pagerank_exponential_jump(ranking)
        restricted = self._restrict_to_first_group(base_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_fairgd(self, ranking: List[int],
                                base_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """FairGD with nodes in first ranking group prioritized"""
        start_time = time.time()
        if base_result is None:
            base_result, _ = self.fairgd(ranking)
        restricted = self._restrict_to_first_group(base_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_adaptgd(self, ranking: List[int],
                                 base_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """AdaptGD with nodes in first ranking group prioritized"""
        start_time = time.time()
        if base_result is None:
            base_result, _ = self.adaptgd(ranking)
        restricted = self._restrict_to_first_group(base_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_fspr(self, ranking: List[int],
                              base_result: Dict[int, Tuple[Any, int]] = None) -> Optional[Tuple[Dict[int, Tuple[Any, int]], float]]:
        """FSPR with nodes in first ranking group prioritized"""
        start_time = time.time()
        if base_result is None:
            result = self.fspr(ranking)
            if result is None:
                return None
            base_result, _ = result
        restricted = self._restrict_to_first_group(base_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_lfpr_u(self, ranking: List[int],
                                base_result: Dict[int, Tuple[Any, int]] = None) -> Optional[Tuple[Dict[int, Tuple[Any, int]], float]]:
        """LFPR_U with nodes in first ranking group prioritized"""
        start_time = time.time()
        if base_result is None:
            result = self.lfpr_u(ranking)
            if result is None:
                return None
            base_result, _ = result
        restricted = self._restrict_to_first_group(base_result, ranking)
        return restricted, time.time() - start_time

    def restricted_first_lfpr_n(self, ranking: List[int],
                                base_result: Dict[int, Tuple[Any, int]] = None) -> Optional[Tuple[Dict[int, Tuple[Any, int]], float]]:
        """LFPR_N with nodes in first ranking group prioritized"""
        start_time = time.time()
        if base_result is None:
            result = self.lfpr_n(ranking)
            if result is None:
                return None
            base_result, _ = result
        restricted = self._restrict_to_first_group(base_result, ranking)
        return restricted, time.time() - start_time

    def lexipeeling_coreness(self, ranking: List[int],
                             lex_result: Dict[int, Tuple[Any, int]] = None,
                             core_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Lexipeeling with coreness as tie-breaker"""
        start_time = time.time()

        if lex_result is None:
            lex_result, _ = self.lexipeeling(ranking)
        if core_result is None:
            core_result, _ = self.coreness()

        combined = self._combine_with_tiebreak(lex_result, core_result)

        return combined, time.time() - start_time

    def lexipeeling_degree(self, ranking: List[int],
                           lex_result: Dict[int, Tuple[Any, int]] = None,
                           deg_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Lexipeeling with degree as tie-breaker"""
        start_time = time.time()

        if lex_result is None:
            lex_result, _ = self.lexipeeling(ranking)
        if deg_result is None:
            deg_result, _ = self.degree_centrality()

        combined = self._combine_with_tiebreak(lex_result, deg_result)

        return combined, time.time() - start_time

    def lexipeeling_degree_lexicographic(self, ranking: List[int],
                                         lex_result: Dict[int, Tuple[Any, int]] = None,
                                         deg_lex_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Lexipeeling with lexicographic degree as tie-breaker"""
        start_time = time.time()

        if lex_result is None:
            lex_result, _ = self.lexipeeling(ranking)
        if deg_lex_result is None:
            deg_lex_result, _ = self.degree_centrality_lexicographic(ranking)

        combined = self._combine_with_tiebreak(lex_result, deg_lex_result)

        return combined, time.time() - start_time

    def coreness_lexipeeling(self, ranking: List[int],
                             core_result: Dict[int, Tuple[Any, int]] = None,
                             lex_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Coreness with lexipeeling as tie-breaker"""
        start_time = time.time()

        if core_result is None:
            core_result, _ = self.coreness()
        if lex_result is None:
            lex_result, _ = self.lexipeeling(ranking)

        combined = self._combine_with_tiebreak(core_result, lex_result)

        return combined, time.time() - start_time

    def degree_lexipeeling(self, ranking: List[int],
                           deg_result: Dict[int, Tuple[Any, int]] = None,
                           lex_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Degree with lexipeeling as tie-breaker"""
        start_time = time.time()

        if deg_result is None:
            deg_result, _ = self.degree_centrality()
        if lex_result is None:
            lex_result, _ = self.lexipeeling(ranking)

        combined = self._combine_with_tiebreak(deg_result, lex_result)

        return combined, time.time() - start_time

    def degree_lexicographic_lexipeeling(self, ranking: List[int],
                                         deg_lex_result: Dict[int, Tuple[Any, int]] = None,
                                         lex_result: Dict[int, Tuple[Any, int]] = None) -> Tuple[Dict[int, Tuple[Any, int]], float]:
        """Lexicographic degree with lexipeeling as tie-breaker"""
        start_time = time.time()

        if deg_lex_result is None:
            deg_lex_result, _ = self.degree_centrality_lexicographic(ranking)
        if lex_result is None:
            lex_result, _ = self.lexipeeling(ranking)

        combined = self._combine_with_tiebreak(deg_lex_result, lex_result)

        return combined, time.time() - start_time


    # List of ranking-independent method names (for reference)
    RANKING_INDEPENDENT_METHODS = [
        'coreness', 'degree', 'betweenness', 'pagerank',
        'h1_index', 'h2_index', 'h3_index', 'h4_index', 'h5_index',
        'closeness_centrality',
        'gravity_centrality', 'extended_gravity_centrality',
        'local_gravity', 'gravity_h_index',
        'clustering_coefficient_centrality',
        'local_global_centrality',
        'entropy_degree_distance_combination',
    ]

    # List of ranking-dependent method names (for reference)
    RANKING_DEPENDENT_METHODS = [
        'lexipeeling', 'lexipeeling_extended', 'degree_lexicographic', 'random',
        'pagerank_uniform_jump', 'pagerank_exponential_jump',
        'fairgd', 'adaptgd',
        'fspr', 'lfpr_u', 'lfpr_n',  # FairLaR methods (binary groups only)
        'lexipeeling_coreness', 'lexipeeling_degree', 'lexipeeling_degree_lexicographic',
        'coreness_lexipeeling', 'degree_lexipeeling', 'degree_lexicographic_lexipeeling',
        'restricted_coreness', 'restricted_degree', 'restricted_degree_lexicographic',
        'restricted_pagerank', 'restricted_betweenness',
        'restricted_h1_index', 'restricted_h2_index', 'restricted_h3_index',
        'restricted_h4_index', 'restricted_h5_index',
        'restricted_first_coreness', 'restricted_first_degree', 'restricted_first_degree_lexicographic',
        'restricted_first_pagerank', 'restricted_first_betweenness',
        'restricted_first_h1_index', 'restricted_first_h2_index', 'restricted_first_h3_index',
        'restricted_first_h4_index', 'restricted_first_h5_index',
        'restricted_pagerank_uniform_jump', 'restricted_pagerank_exponential_jump',
        'restricted_fairgd', 'restricted_adaptgd',
        'restricted_fspr', 'restricted_lfpr_u', 'restricted_lfpr_n',
        'restricted_first_pagerank_uniform_jump', 'restricted_first_pagerank_exponential_jump',
        'restricted_first_fairgd', 'restricted_first_adaptgd',
        'restricted_first_fspr', 'restricted_first_lfpr_u', 'restricted_first_lfpr_n',
        # Restricted variants of new baselines
        'restricted_closeness_centrality', 'restricted_first_closeness_centrality',
        'restricted_gravity_centrality', 'restricted_first_gravity_centrality',
        'restricted_extended_gravity_centrality', 'restricted_first_extended_gravity_centrality',
        'restricted_local_gravity', 'restricted_first_local_gravity',
        'restricted_gravity_h_index', 'restricted_first_gravity_h_index',
        'restricted_clustering_coefficient_centrality', 'restricted_first_clustering_coefficient_centrality',
        'restricted_local_global_centrality', 'restricted_first_local_global_centrality',
        'restricted_entropy_degree_distance_combination', 'restricted_first_entropy_degree_distance_combination',
    ]

    def run_ranking_independent_methods(self) -> Dict[str, Dict]:
        """
        Run methods that don't depend on ranking.
        These only need to be computed once per graph.

        Returns:
            Dict mapping method_name -> {result, runtime, top_spreader}
        """
        results = {}

        print("Running Coreness...")
        core_result, core_time = self.coreness()
        results['coreness'] = {
            'result': core_result,
            'runtime': core_time,
            'top_spreader': self.get_top_spreader(core_result)
        }

        print("Running Degree...")
        deg_result, deg_time = self.degree_centrality()
        results['degree'] = {
            'result': deg_result,
            'runtime': deg_time,
            'top_spreader': self.get_top_spreader(deg_result)
        }

        print("Running Betweenness...")
        bet_result, bet_time = self.betweenness_centrality()
        results['betweenness'] = {
            'result': bet_result,
            'runtime': bet_time,
            'top_spreader': self.get_top_spreader(bet_result)
        }

        print("Running PageRank...")
        pr_result, pr_time = self.pagerank()
        results['pagerank'] = {
            'result': pr_result,
            'runtime': pr_time,
            'top_spreader': self.get_top_spreader(pr_result)
        }

        print("Running H1-index...")
        h1_result, h1_time = self.h1_index()
        results['h1_index'] = {
            'result': h1_result,
            'runtime': h1_time,
            'top_spreader': self.get_top_spreader(h1_result)
        }

        print("Running H2-index...")
        h2_result, h2_time = self.h2_index()
        results['h2_index'] = {
            'result': h2_result,
            'runtime': h2_time,
            'top_spreader': self.get_top_spreader(h2_result)
        }

        print("Running H3-index...")
        h3_result, h3_time = self.h3_index()
        results['h3_index'] = {
            'result': h3_result,
            'runtime': h3_time,
            'top_spreader': self.get_top_spreader(h3_result)
        }

        print("Running H4-index...")
        h4_result, h4_time = self.h4_index()
        results['h4_index'] = {
            'result': h4_result,
            'runtime': h4_time,
            'top_spreader': self.get_top_spreader(h4_result)
        }

        print("Running H5-index...")
        h5_result, h5_time = self.h5_index()
        results['h5_index'] = {
            'result': h5_result,
            'runtime': h5_time,
            'top_spreader': self.get_top_spreader(h5_result)
        }

        print("Running Closeness Centrality...")
        closeness_result, closeness_time = self.closeness_centrality()
        results['closeness_centrality'] = {
            'result': closeness_result,
            'runtime': closeness_time,
            'top_spreader': self.get_top_spreader(closeness_result)
        }

        print("Running Gravity Centrality...")
        gravity_result, gravity_time = self.gravity_centrality()
        results['gravity_centrality'] = {
            'result': gravity_result,
            'runtime': gravity_time,
            'top_spreader': self.get_top_spreader(gravity_result)
        }

        print("Running Extended Gravity Centrality...")
        ext_gravity_result, ext_gravity_time = self.extended_gravity_centrality()
        results['extended_gravity_centrality'] = {
            'result': ext_gravity_result,
            'runtime': ext_gravity_time,
            'top_spreader': self.get_top_spreader(ext_gravity_result)
        }

        print("Running Local Gravity...")
        local_gravity_result, local_gravity_time = self.local_gravity()
        results['local_gravity'] = {
            'result': local_gravity_result,
            'runtime': local_gravity_time,
            'top_spreader': self.get_top_spreader(local_gravity_result)
        }

        print("Running Gravity H-index...")
        gravity_h_result, gravity_h_time = self.gravity_h_index()
        results['gravity_h_index'] = {
            'result': gravity_h_result,
            'runtime': gravity_h_time,
            'top_spreader': self.get_top_spreader(gravity_h_result)
        }

        print("Running Clustering Coefficient Centrality...")
        clustering_cc_result, clustering_cc_time = self.clustering_coefficient_centrality()
        results['clustering_coefficient_centrality'] = {
            'result': clustering_cc_result,
            'runtime': clustering_cc_time,
            'top_spreader': self.get_top_spreader(clustering_cc_result)
        }

        print("Running Local-Global Centrality...")
        lgc_result, lgc_time = self.local_global_centrality()
        results['local_global_centrality'] = {
            'result': lgc_result,
            'runtime': lgc_time,
            'top_spreader': self.get_top_spreader(lgc_result)
        }

        print("Running Entropy-Degree-Distance Combination...")
        eddc_result, eddc_time = self.entropy_degree_distance_combination()
        results['entropy_degree_distance_combination'] = {
            'result': eddc_result,
            'runtime': eddc_time,
            'top_spreader': self.get_top_spreader(eddc_result)
        }

        return results

    def _run_with_timeout(self, method_name: str, method_func: Callable, timeout: int) -> Tuple[Optional[Tuple], bool]:
        """
        Run a method with a timeout.

        Args:
            method_name: Name of the method (for logging)
            method_func: Callable that returns (result, runtime) or None
            timeout: Timeout in seconds

        Returns:
            Tuple of (output, timed_out) where:
                - output: Method output (result, runtime) or None if timed out or failed
                - timed_out: True if the method exceeded the timeout
        """
        result_box = [None]
        exc_box = [None]

        def target():
            try:
                result_box[0] = method_func()
            except Exception as e:
                exc_box[0] = e

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            # Thread still running — timed out. Being a daemon thread it will be
            # silently abandoned when the process exits (no atexit join).
            print(f"Warning: {method_name} exceeded timeout ({timeout}s), skipping")
            return None, True

        if exc_box[0] is not None:
            print(f"Warning: {method_name} failed with error: {exc_box[0]}, skipping")
            return None, False

        return result_box[0], False

    def run_ranking_dependent_methods(self, ranking: List[int],
                                       ranking_independent_results: Dict[str, Dict] = None,
                                       timeout: int = DEFAULT_METHOD_TIMEOUT,
                                       skip_methods: set = None) -> Tuple[Dict[str, Dict], set]:
        """
        Run methods that depend on ranking.
        These need to be computed for each ranking.

        Delegates to run_selected_methods with all ranking-dependent methods.

        Args:
            ranking: List of group IDs in priority order
            ranking_independent_results: Optional pre-computed results from run_ranking_independent_methods().
                                         If provided, these will be reused for composite/restricted methods.
            timeout: Maximum time in seconds for each method (default: 3600 = 1 hour).
                     Methods exceeding this time will be skipped.
            skip_methods: Set of method names to skip (e.g., methods that timed out on a
                          previous ranking for this graph).

        Returns:
            Tuple of (results, timed_out) where:
                - results: Dict mapping method_name -> {result, runtime, top_spreader}
                - timed_out: Set of method names that timed out during this call
        """
        return self.run_selected_methods(
            ranking, self.RANKING_DEPENDENT_METHODS,
            ranking_independent_results=ranking_independent_results,
            timeout=timeout, skip_methods=skip_methods
        )

    def run_selected_methods(self, ranking: List[int], method_names: List[str],
                              ranking_independent_results: Dict[str, Dict] = None,
                              timeout: int = DEFAULT_METHOD_TIMEOUT,
                              skip_methods: set = None) -> Tuple[Dict[str, Dict], set]:
        """
        Run selected methods for a given ranking

        Args:
            ranking: List of group IDs in priority order
            method_names: List of method names to run
            ranking_independent_results: Optional pre-computed results from run_ranking_independent_methods().
                                         If provided, these will be reused for composite/restricted methods.
            timeout: Maximum time in seconds for each method (default: 3600 = 1 hour).
                     Methods exceeding this time will be skipped.
            skip_methods: Set of method names to skip (e.g., methods that timed out on a
                          previous ranking for this graph). Dependent methods are also
                          automatically skipped.

        Returns:
            Tuple of (results, timed_out) where:
                - results: Dict mapping method_name -> {result, runtime, top_spreader}
                - timed_out: Set of method names that timed out during this call
        """
        results = {}
        timed_out = set()
        effective_skip = set(skip_methods) if skip_methods else set()

        # Extract pre-computed ranking-independent results if available
        core_result = None
        deg_result = None
        bet_result = None
        pr_result = None
        h1_result = None
        h2_result = None
        h3_result = None
        h4_result = None
        h5_result = None
        closeness_result = None
        gravity_result = None
        ext_gravity_result = None
        local_gravity_result = None
        gravity_h_result = None
        clustering_cc_result = None
        lgc_result = None
        eddc_result = None
        if ranking_independent_results:
            if 'coreness' in ranking_independent_results:
                core_result = ranking_independent_results['coreness']['result']
            if 'degree' in ranking_independent_results:
                deg_result = ranking_independent_results['degree']['result']
            if 'betweenness' in ranking_independent_results:
                bet_result = ranking_independent_results['betweenness']['result']
            if 'pagerank' in ranking_independent_results:
                pr_result = ranking_independent_results['pagerank']['result']
            if 'h1_index' in ranking_independent_results:
                h1_result = ranking_independent_results['h1_index']['result']
            if 'h2_index' in ranking_independent_results:
                h2_result = ranking_independent_results['h2_index']['result']
            if 'h3_index' in ranking_independent_results:
                h3_result = ranking_independent_results['h3_index']['result']
            if 'h4_index' in ranking_independent_results:
                h4_result = ranking_independent_results['h4_index']['result']
            if 'h5_index' in ranking_independent_results:
                h5_result = ranking_independent_results['h5_index']['result']
            if 'closeness_centrality' in ranking_independent_results:
                closeness_result = ranking_independent_results['closeness_centrality']['result']
            if 'gravity_centrality' in ranking_independent_results:
                gravity_result = ranking_independent_results['gravity_centrality']['result']
            if 'extended_gravity_centrality' in ranking_independent_results:
                ext_gravity_result = ranking_independent_results['extended_gravity_centrality']['result']
            if 'local_gravity' in ranking_independent_results:
                local_gravity_result = ranking_independent_results['local_gravity']['result']
            if 'gravity_h_index' in ranking_independent_results:
                gravity_h_result = ranking_independent_results['gravity_h_index']['result']
            if 'clustering_coefficient_centrality' in ranking_independent_results:
                clustering_cc_result = ranking_independent_results['clustering_coefficient_centrality']['result']
            if 'local_global_centrality' in ranking_independent_results:
                lgc_result = ranking_independent_results['local_global_centrality']['result']
            if 'entropy_degree_distance_combination' in ranking_independent_results:
                eddc_result = ranking_independent_results['entropy_degree_distance_combination']['result']

        # Pre-compute ranking-dependent results that are reused by multiple methods
        # These are computed once per ranking and shared across combined methods
        method_names_set = set(method_names)

        # Methods that need lexipeeling result
        methods_needing_lex = {
            'lexipeeling', 'lexipeeling_coreness', 'lexipeeling_degree',
            'lexipeeling_degree_lexicographic', 'coreness_lexipeeling',
            'degree_lexipeeling', 'degree_lexicographic_lexipeeling'
        }
        # Methods that need degree_lexicographic result
        methods_needing_deg_lex = {
            'degree_lexicographic', 'lexipeeling_degree_lexicographic',
            'degree_lexicographic_lexipeeling', 'restricted_degree_lexicographic',
            'restricted_first_degree_lexicographic'
        }

        # Propagate skips: if a base method is skipped, skip all methods that depend on it
        if effective_skip & methods_needing_lex:
            # If lexipeeling itself is skipped, skip all methods that depend on it
            if 'lexipeeling' in effective_skip:
                effective_skip.update(methods_needing_lex & method_names_set)
        if effective_skip & methods_needing_deg_lex:
            if 'degree_lexicographic' in effective_skip:
                effective_skip.update(methods_needing_deg_lex & method_names_set)

        # Propagate timeouts of base methods to their restricted variants.
        # If a base timed out externally (passed via skip_methods), its restricted/restricted_first
        # variants must also be skipped — otherwise they would recompute the base with no timeout.
        # This covers both ranking-independent bases (passed from run_methods.py) and
        # ranking-dependent bases (timed out on a previous ranking and carried via skip_methods).
        _base_to_restricted = {
            # Ranking-independent bases
            'betweenness':           {'restricted_betweenness',              'restricted_first_betweenness'},
            'coreness':              {'restricted_coreness',                 'restricted_first_coreness'},
            'degree':                {'restricted_degree',                   'restricted_first_degree'},
            'pagerank':              {'restricted_pagerank',                 'restricted_first_pagerank'},
            'h1_index':              {'restricted_h1_index',                 'restricted_first_h1_index'},
            'h2_index':              {'restricted_h2_index',                 'restricted_first_h2_index'},
            'h3_index':              {'restricted_h3_index',                 'restricted_first_h3_index'},
            'h4_index':              {'restricted_h4_index',                 'restricted_first_h4_index'},
            'h5_index':              {'restricted_h5_index',                 'restricted_first_h5_index'},
            # New baseline bases
            'closeness_centrality':               {'restricted_closeness_centrality',               'restricted_first_closeness_centrality'},
            'gravity_centrality':                 {'restricted_gravity_centrality',                 'restricted_first_gravity_centrality'},
            'extended_gravity_centrality':        {'restricted_extended_gravity_centrality',        'restricted_first_extended_gravity_centrality'},
            'local_gravity':                      {'restricted_local_gravity',                      'restricted_first_local_gravity'},
            'gravity_h_index':                    {'restricted_gravity_h_index',                    'restricted_first_gravity_h_index'},
            'clustering_coefficient_centrality':  {'restricted_clustering_coefficient_centrality',  'restricted_first_clustering_coefficient_centrality'},
            'local_global_centrality':            {'restricted_local_global_centrality',            'restricted_first_local_global_centrality'},
            'entropy_degree_distance_combination':{'restricted_entropy_degree_distance_combination','restricted_first_entropy_degree_distance_combination'},
            # Ranking-dependent bases
            'pagerank_uniform_jump':    {'restricted_pagerank_uniform_jump',    'restricted_first_pagerank_uniform_jump'},
            'pagerank_exponential_jump':{'restricted_pagerank_exponential_jump', 'restricted_first_pagerank_exponential_jump'},
            'fairgd':                {'restricted_fairgd',                   'restricted_first_fairgd'},
            'adaptgd':               {'restricted_adaptgd',                  'restricted_first_adaptgd'},
            'fspr':                  {'restricted_fspr',                     'restricted_first_fspr'},
            'lfpr_u':                {'restricted_lfpr_u',                   'restricted_first_lfpr_u'},
            'lfpr_n':                {'restricted_lfpr_n',                   'restricted_first_lfpr_n'},
        }
        for base, dependents in _base_to_restricted.items():
            if base in effective_skip:
                effective_skip.update(dependents & method_names_set)

        # Pre-compute lexipeeling if needed by any non-skipped method
        lex_result = None
        lex_time = 0
        methods_needing_lex_active = (method_names_set & methods_needing_lex) - effective_skip
        if methods_needing_lex_active:
            output, was_timeout = self._run_with_timeout(
                'lexipeeling (pre-computation)', lambda: self.lexipeeling(ranking), timeout)
            if was_timeout:
                timed_out.add('lexipeeling')
                effective_skip.update(methods_needing_lex & method_names_set)
            elif output is not None:
                lex_result, lex_time = output

        # Pre-compute degree_lexicographic if needed by any non-skipped method
        deg_lex_result = None
        deg_lex_time = 0
        methods_needing_deg_lex_active = (method_names_set & methods_needing_deg_lex) - effective_skip
        if methods_needing_deg_lex_active:
            output, was_timeout = self._run_with_timeout(
                'degree_lexicographic (pre-computation)',
                lambda: self.degree_centrality_lexicographic(ranking), timeout)
            if was_timeout:
                timed_out.add('degree_lexicographic')
                effective_skip.update(methods_needing_deg_lex & method_names_set)
            elif output is not None:
                deg_lex_result, deg_lex_time = output

        # Pre-compute ranking-dependent base methods if any restricted variant needs them
        # This avoids double computation when both base and restricted are requested
        _restricted_pr_uj = {'restricted_pagerank_uniform_jump', 'restricted_first_pagerank_uniform_jump'}
        _restricted_pr_ej = {'restricted_pagerank_exponential_jump', 'restricted_first_pagerank_exponential_jump'}
        _restricted_fairgd = {'restricted_fairgd', 'restricted_first_fairgd'}
        _restricted_adaptgd = {'restricted_adaptgd', 'restricted_first_adaptgd'}
        _restricted_fspr = {'restricted_fspr', 'restricted_first_fspr'}
        _restricted_lfpr_u = {'restricted_lfpr_u', 'restricted_first_lfpr_u'}
        _restricted_lfpr_n = {'restricted_lfpr_n', 'restricted_first_lfpr_n'}

        pr_uj_result = None
        pr_uj_time = 0
        if (method_names_set & (_restricted_pr_uj | {'pagerank_uniform_jump'})) - effective_skip:
            needs_precompute = bool(method_names_set & _restricted_pr_uj - effective_skip)
            if needs_precompute:
                output, was_timeout = self._run_with_timeout(
                    'pagerank_uniform_jump (pre-computation)', lambda: self.pagerank_uniform_jump(ranking), timeout)
                if was_timeout:
                    timed_out.add('pagerank_uniform_jump')
                    effective_skip.update(_restricted_pr_uj & method_names_set)
                    effective_skip.add('pagerank_uniform_jump')
                elif output is not None:
                    pr_uj_result, pr_uj_time = output

        pr_ej_result = None
        pr_ej_time = 0
        if (method_names_set & (_restricted_pr_ej | {'pagerank_exponential_jump'})) - effective_skip:
            needs_precompute = bool(method_names_set & _restricted_pr_ej - effective_skip)
            if needs_precompute:
                output, was_timeout = self._run_with_timeout(
                    'pagerank_exponential_jump (pre-computation)', lambda: self.pagerank_exponential_jump(ranking), timeout)
                if was_timeout:
                    timed_out.add('pagerank_exponential_jump')
                    effective_skip.update(_restricted_pr_ej & method_names_set)
                    effective_skip.add('pagerank_exponential_jump')
                elif output is not None:
                    pr_ej_result, pr_ej_time = output

        fairgd_result = None
        fairgd_time = 0
        if FAIRGD_AVAILABLE and (method_names_set & (_restricted_fairgd | {'fairgd'})) - effective_skip:
            needs_precompute = bool(method_names_set & _restricted_fairgd - effective_skip)
            if needs_precompute:
                output, was_timeout = self._run_with_timeout(
                    'fairgd (pre-computation)', lambda: self.fairgd(ranking), timeout)
                if was_timeout:
                    timed_out.add('fairgd')
                    effective_skip.update(_restricted_fairgd & method_names_set)
                    effective_skip.add('fairgd')
                elif output is not None:
                    fairgd_result, fairgd_time = output

        adaptgd_result = None
        adaptgd_time = 0
        if FAIRGD_AVAILABLE and (method_names_set & (_restricted_adaptgd | {'adaptgd'})) - effective_skip:
            needs_precompute = bool(method_names_set & _restricted_adaptgd - effective_skip)
            if needs_precompute:
                output, was_timeout = self._run_with_timeout(
                    'adaptgd (pre-computation)', lambda: self.adaptgd(ranking), timeout)
                if was_timeout:
                    timed_out.add('adaptgd')
                    effective_skip.update(_restricted_adaptgd & method_names_set)
                    effective_skip.add('adaptgd')
                elif output is not None:
                    adaptgd_result, adaptgd_time = output

        fspr_result = None
        fspr_time = 0
        if FAIRLAR_AVAILABLE and FAIRGD_AVAILABLE and len(ranking) == 2 and (method_names_set & (_restricted_fspr | {'fspr'})) - effective_skip:
            needs_precompute = bool(method_names_set & _restricted_fspr - effective_skip)
            if needs_precompute:
                output, was_timeout = self._run_with_timeout(
                    'fspr (pre-computation)', lambda: self.fspr(ranking), timeout)
                if was_timeout:
                    timed_out.add('fspr')
                    effective_skip.update(_restricted_fspr & method_names_set)
                    effective_skip.add('fspr')
                elif output is not None and output is not None:
                    fspr_result, fspr_time = output

        lfpr_u_result = None
        lfpr_u_time = 0
        if FAIRGD_AVAILABLE and len(ranking) == 2 and (method_names_set & (_restricted_lfpr_u | {'lfpr_u'})) - effective_skip:
            needs_precompute = bool(method_names_set & _restricted_lfpr_u - effective_skip)
            if needs_precompute:
                output, was_timeout = self._run_with_timeout(
                    'lfpr_u (pre-computation)', lambda: self.lfpr_u(ranking), timeout)
                if was_timeout:
                    timed_out.add('lfpr_u')
                    effective_skip.update(_restricted_lfpr_u & method_names_set)
                    effective_skip.add('lfpr_u')
                elif output is not None:
                    lfpr_u_result, lfpr_u_time = output

        lfpr_n_result = None
        lfpr_n_time = 0
        if FAIRGD_AVAILABLE and len(ranking) == 2 and (method_names_set & (_restricted_lfpr_n | {'lfpr_n'})) - effective_skip:
            needs_precompute = bool(method_names_set & _restricted_lfpr_n - effective_skip)
            if needs_precompute:
                output, was_timeout = self._run_with_timeout(
                    'lfpr_n (pre-computation)', lambda: self.lfpr_n(ranking), timeout)
                if was_timeout:
                    timed_out.add('lfpr_n')
                    effective_skip.update(_restricted_lfpr_n & method_names_set)
                    effective_skip.add('lfpr_n')
                elif output is not None:
                    lfpr_n_result, lfpr_n_time = output

        # Mapping from method names to functions (using pre-computed results)
        method_funcs = {
            'lexipeeling': lambda: (lex_result, lex_time),
            'lexipeeling_extended': lambda: self.lexipeeling_extended(ranking),
            'degree_lexicographic': lambda: (deg_lex_result, deg_lex_time),
            'coreness': lambda: self.coreness(),
            'degree': lambda: self.degree_centrality(),
            'betweenness': lambda: self.betweenness_centrality(),
            'pagerank': lambda: self.pagerank(),
            'random': lambda: self.random_baseline(ranking),
            'pagerank_uniform_jump': lambda: (pr_uj_result, pr_uj_time) if pr_uj_result is not None else self.pagerank_uniform_jump(ranking),
            'pagerank_exponential_jump': lambda: (pr_ej_result, pr_ej_time) if pr_ej_result is not None else self.pagerank_exponential_jump(ranking),
            # FairGD-based methods (conditional on availability)
            'fairgd': lambda: (fairgd_result, fairgd_time) if fairgd_result is not None else (self.fairgd(ranking) if FAIRGD_AVAILABLE else None),
            'adaptgd': lambda: (adaptgd_result, adaptgd_time) if adaptgd_result is not None else (self.adaptgd(ranking) if FAIRGD_AVAILABLE else None),
            # FairLaR methods (binary groups only, conditional on availability)
            'fspr': lambda: (fspr_result, fspr_time) if fspr_result is not None else (self.fspr(ranking) if (FAIRLAR_AVAILABLE and FAIRGD_AVAILABLE and len(ranking) == 2) else None),
            'lfpr_u': lambda: (lfpr_u_result, lfpr_u_time) if lfpr_u_result is not None else (self.lfpr_u(ranking) if (FAIRGD_AVAILABLE and len(ranking) == 2) else None),
            'lfpr_n': lambda: (lfpr_n_result, lfpr_n_time) if lfpr_n_result is not None else (self.lfpr_n(ranking) if (FAIRGD_AVAILABLE and len(ranking) == 2) else None),
            # H-index based methods
            'h1_index': lambda: self.h1_index(),
            'h2_index': lambda: self.h2_index(),
            'h3_index': lambda: self.h3_index(),
            'h4_index': lambda: self.h4_index(),
            'h5_index': lambda: self.h5_index(),
            # Combined methods with tie-breaking (reusing pre-computed results)
            'lexipeeling_coreness': lambda: self.lexipeeling_coreness(ranking, lex_result=lex_result, core_result=core_result),
            'lexipeeling_degree': lambda: self.lexipeeling_degree(ranking, lex_result=lex_result, deg_result=deg_result),
            'lexipeeling_degree_lexicographic': lambda: self.lexipeeling_degree_lexicographic(ranking, lex_result=lex_result, deg_lex_result=deg_lex_result),
            'coreness_lexipeeling': lambda: self.coreness_lexipeeling(ranking, core_result=core_result, lex_result=lex_result),
            'degree_lexipeeling': lambda: self.degree_lexipeeling(ranking, deg_result=deg_result, lex_result=lex_result),
            'degree_lexicographic_lexipeeling': lambda: self.degree_lexicographic_lexipeeling(ranking, deg_lex_result=deg_lex_result, lex_result=lex_result),
            # Restricted methods (reusing pre-computed results)
            'restricted_coreness': lambda: self.restricted_coreness(ranking, core_result=core_result),
            'restricted_degree': lambda: self.restricted_degree(ranking, deg_result=deg_result),
            'restricted_degree_lexicographic': lambda: self.restricted_degree_lexicographic(ranking, deg_lex_result=deg_lex_result),
            'restricted_pagerank': lambda: self.restricted_pagerank(ranking, pr_result=pr_result),
            'restricted_betweenness': lambda: self.restricted_betweenness(ranking, bet_result=bet_result),
            # Restricted H-index methods (reusing pre-computed results)
            'restricted_h1_index': lambda: self.restricted_h1_index(ranking, h1_result=h1_result),
            'restricted_h2_index': lambda: self.restricted_h2_index(ranking, h2_result=h2_result),
            'restricted_h3_index': lambda: self.restricted_h3_index(ranking, h3_result=h3_result),
            'restricted_h4_index': lambda: self.restricted_h4_index(ranking, h4_result=h4_result),
            'restricted_h5_index': lambda: self.restricted_h5_index(ranking, h5_result=h5_result),
            # Restricted-first methods (first group only, reusing pre-computed results)
            'restricted_first_coreness': lambda: self.restricted_first_coreness(ranking, core_result=core_result),
            'restricted_first_degree': lambda: self.restricted_first_degree(ranking, deg_result=deg_result),
            'restricted_first_degree_lexicographic': lambda: self.restricted_first_degree_lexicographic(ranking, deg_lex_result=deg_lex_result),
            'restricted_first_pagerank': lambda: self.restricted_first_pagerank(ranking, pr_result=pr_result),
            'restricted_first_betweenness': lambda: self.restricted_first_betweenness(ranking, bet_result=bet_result),
            'restricted_first_h1_index': lambda: self.restricted_first_h1_index(ranking, h1_result=h1_result),
            'restricted_first_h2_index': lambda: self.restricted_first_h2_index(ranking, h2_result=h2_result),
            'restricted_first_h3_index': lambda: self.restricted_first_h3_index(ranking, h3_result=h3_result),
            'restricted_first_h4_index': lambda: self.restricted_first_h4_index(ranking, h4_result=h4_result),
            'restricted_first_h5_index': lambda: self.restricted_first_h5_index(ranking, h5_result=h5_result),
            # Restricted ranking-dependent methods (reusing pre-computed results)
            'restricted_pagerank_uniform_jump': lambda: self.restricted_pagerank_uniform_jump(ranking, base_result=pr_uj_result),
            'restricted_pagerank_exponential_jump': lambda: self.restricted_pagerank_exponential_jump(ranking, base_result=pr_ej_result),
            'restricted_fairgd': lambda: self.restricted_fairgd(ranking, base_result=fairgd_result) if FAIRGD_AVAILABLE else None,
            'restricted_adaptgd': lambda: self.restricted_adaptgd(ranking, base_result=adaptgd_result) if FAIRGD_AVAILABLE else None,
            'restricted_fspr': lambda: self.restricted_fspr(ranking, base_result=fspr_result) if (FAIRLAR_AVAILABLE and FAIRGD_AVAILABLE and len(ranking) == 2) else None,
            'restricted_lfpr_u': lambda: self.restricted_lfpr_u(ranking, base_result=lfpr_u_result) if (FAIRGD_AVAILABLE and len(ranking) == 2) else None,
            'restricted_lfpr_n': lambda: self.restricted_lfpr_n(ranking, base_result=lfpr_n_result) if (FAIRGD_AVAILABLE and len(ranking) == 2) else None,
            # Restricted-first ranking-dependent methods (reusing pre-computed results)
            'restricted_first_pagerank_uniform_jump': lambda: self.restricted_first_pagerank_uniform_jump(ranking, base_result=pr_uj_result),
            'restricted_first_pagerank_exponential_jump': lambda: self.restricted_first_pagerank_exponential_jump(ranking, base_result=pr_ej_result),
            'restricted_first_fairgd': lambda: self.restricted_first_fairgd(ranking, base_result=fairgd_result) if FAIRGD_AVAILABLE else None,
            'restricted_first_adaptgd': lambda: self.restricted_first_adaptgd(ranking, base_result=adaptgd_result) if FAIRGD_AVAILABLE else None,
            'restricted_first_fspr': lambda: self.restricted_first_fspr(ranking, base_result=fspr_result) if (FAIRLAR_AVAILABLE and FAIRGD_AVAILABLE and len(ranking) == 2) else None,
            'restricted_first_lfpr_u': lambda: self.restricted_first_lfpr_u(ranking, base_result=lfpr_u_result) if (FAIRGD_AVAILABLE and len(ranking) == 2) else None,
            'restricted_first_lfpr_n': lambda: self.restricted_first_lfpr_n(ranking, base_result=lfpr_n_result) if (FAIRGD_AVAILABLE and len(ranking) == 2) else None,
            # Restricted methods for new baselines
            'restricted_closeness_centrality': lambda: self.restricted_closeness_centrality(ranking, closeness_result=closeness_result),
            'restricted_first_closeness_centrality': lambda: self.restricted_first_closeness_centrality(ranking, closeness_result=closeness_result),
            'restricted_gravity_centrality': lambda: self.restricted_gravity_centrality(ranking, gravity_result=gravity_result),
            'restricted_first_gravity_centrality': lambda: self.restricted_first_gravity_centrality(ranking, gravity_result=gravity_result),
            'restricted_extended_gravity_centrality': lambda: self.restricted_extended_gravity_centrality(ranking, ext_gravity_result=ext_gravity_result),
            'restricted_first_extended_gravity_centrality': lambda: self.restricted_first_extended_gravity_centrality(ranking, ext_gravity_result=ext_gravity_result),
            'restricted_local_gravity': lambda: self.restricted_local_gravity(ranking, local_gravity_result=local_gravity_result),
            'restricted_first_local_gravity': lambda: self.restricted_first_local_gravity(ranking, local_gravity_result=local_gravity_result),
            'restricted_gravity_h_index': lambda: self.restricted_gravity_h_index(ranking, gravity_h_result=gravity_h_result),
            'restricted_first_gravity_h_index': lambda: self.restricted_first_gravity_h_index(ranking, gravity_h_result=gravity_h_result),
            'restricted_clustering_coefficient_centrality': lambda: self.restricted_clustering_coefficient_centrality(ranking, clustering_cc_result=clustering_cc_result),
            'restricted_first_clustering_coefficient_centrality': lambda: self.restricted_first_clustering_coefficient_centrality(ranking, clustering_cc_result=clustering_cc_result),
            'restricted_local_global_centrality': lambda: self.restricted_local_global_centrality(ranking, lgc_result=lgc_result),
            'restricted_first_local_global_centrality': lambda: self.restricted_first_local_global_centrality(ranking, lgc_result=lgc_result),
            'restricted_entropy_degree_distance_combination': lambda: self.restricted_entropy_degree_distance_combination(ranking, eddc_result=eddc_result),
            'restricted_first_entropy_degree_distance_combination': lambda: self.restricted_first_entropy_degree_distance_combination(ranking, eddc_result=eddc_result),
        }

        for method_name in method_names:
            if method_name not in method_funcs:
                print(f"Warning: Unknown method '{method_name}', skipping")
                continue

            if method_name in effective_skip:
                print(f"Skipping {method_name} (timed out on this graph)")
                continue

            print(f"Running {method_name}...")

            # Run method with timeout
            output, was_timeout = self._run_with_timeout(method_name, method_funcs[method_name], timeout)

            if was_timeout:
                timed_out.add(method_name)
                continue

            # Handle methods that can return None (e.g., lexipeeling_extended or unavailable methods)
            if output is None:
                continue

            method_result, method_time = output
            results[method_name] = {
                'result': method_result,
                'runtime': method_time,
                'top_spreader': self.get_top_spreader(method_result)
            }

        return results, timed_out


def load_graph(edge_file: str, group_file: str) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Load graph from edge and group files

    Args:
        edge_file: Path to edge list file
        group_file: Path to group assignment file

    Returns:
        (G, group_assignments)
    """
    # Load edges
    G = nx.Graph()
    with open(edge_file, 'r') as f:
        for line in f:
            u, v = map(int, line.strip().split())
            G.add_edge(u, v)

    # Remove self-loops (not meaningful for influence maximization)
    num_self_loops = nx.number_of_selfloops(G)
    if num_self_loops > 0:
        G.remove_edges_from(nx.selfloop_edges(G))

    # Load group assignments
    group_assignments = {}
    with open(group_file, 'r') as f:
        for line in f:
            node, group = map(int, line.strip().split())
            group_assignments[node] = group

    # Add isolated nodes to graph (nodes with no edges)
    for node in group_assignments.keys():
        if node not in G.nodes():
            G.add_node(node)

    return G, group_assignments    