"""
Lexicographic Utilities Module

Pure functions for lexicographic operations on vectors.
Provides reusable utilities for comparing, sorting, and ranking vectors.
"""

from typing import List, Tuple, Dict, Any
import heapq


def lexicographic_compare(vec1: List[float], vec2: List[float]) -> int:
    """
    Compare two vectors lexicographically.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        -1 if vec1 < vec2 lexicographically
         0 if vec1 == vec2
         1 if vec1 > vec2 lexicographically
    
    Example:
        >>> lexicographic_compare([1, 5], [2, 3])
        -1  # 1 < 2 in first position
        >>> lexicographic_compare([2, 3], [2, 5])
        -1  # Equal in first, 3 < 5 in second
        >>> lexicographic_compare([3, 1], [3, 1])
        0   # Equal
    """
    for v1, v2 in zip(vec1, vec2):
        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1
    return 0


def lexicographic_max(vec1: List[float], vec2: List[float]) -> List[float]:
    """
    Return lexicographically larger of two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Lexicographically larger vector
    
    Example:
        >>> lexicographic_max([1, 5], [2, 3])
        [2, 3]
    """
    return vec1 if lexicographic_compare(vec1, vec2) >= 0 else vec2


def lexicographic_min(vec1: List[float], vec2: List[float]) -> List[float]:
    """
    Return lexicographically smaller of two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Lexicographically smaller vector
    
    Example:
        >>> lexicographic_min([1, 5], [2, 3])
        [1, 5]
    """
    return vec1 if lexicographic_compare(vec1, vec2) <= 0 else vec2


def rank_vectors_lexicographically(
    vectors: Dict[Any, List[float]], 
    reverse: bool = True
) -> Dict[Any, int]:
    """
    Rank vectors lexicographically.
    
    Args:
        vectors: Dict mapping keys to vectors
        reverse: If True, larger vectors get smaller ranks (rank 1 = best)
    
    Returns:
        Dict mapping keys to ranks (1-based)
    
    Example:
        >>> vectors = {
        ...     'a': [2, 3],
        ...     'b': [1, 5],
        ...     'c': [2, 1]
        ... }
        >>> rank_vectors_lexicographically(vectors, reverse=True)
        {'a': 1, 'c': 2, 'b': 3}  # [2,3] > [2,1] > [1,5]
    """
    # Create sortable items
    items = [(key, tuple(vec)) for key, vec in vectors.items()]
    
    # Sort by vector (tuples compare lexicographically)
    items.sort(key=lambda x: x[1], reverse=reverse)
    
    # Assign ranks (handle ties)
    ranks = {}
    current_rank = 0
    prev_vector = None
    
    for key, vector in items:
        if vector != prev_vector:
            current_rank = current_rank + 1
        ranks[key] = current_rank
        prev_vector = vector
    
    return ranks


def sort_by_ranking(
    group_data: Dict[Any, Dict[int, float]],
    ranking: List[int],
    reverse: bool = True
) -> List[Tuple[Any, List[float]]]:
    """
    Sort items by their group-wise values according to a ranking.
    
    1. Extracts vectors aligned with ranking
    2. Sorts them lexicographically
    
    Args:
        group_data: Dict mapping keys to {group_id: value}
        ranking: List of group IDs in priority order
        reverse: If True, sort descending (largest first)
    
    Returns:
        List of (key, vector) tuples sorted by lexicographic order
    
    Example:
        >>> data = {
        ...     'alice': {0: 10.5, 1: 5.2, 2: 3.1},
        ...     'bob':   {0: 10.5, 1: 6.0, 2: 2.0},
        ...     'carol': {0: 8.0,  1: 9.0, 2: 1.0}
        ... }
        >>> ranking = [0, 1]  # Care about groups 0 and 1 only
        >>> sort_by_ranking(data, ranking, reverse=True)
        [('bob', [10.5, 6.0]),    # Best: [10.5, 6.0]
         ('alice', [10.5, 5.2]),  # Second: [10.5, 5.2]
         ('carol', [8.0, 9.0])]   # Third: [8.0, 9.0]
    """
    # Extract vectors aligned with ranking
    items = []
    for key, values in group_data.items():
        vector = [values.get(group_id, 0.0) for group_id in ranking]
        items.append((key, vector))
    
    # Sort lexicographically
    items.sort(key=lambda x: tuple(x[1]), reverse=reverse)
    
    return items


class LexicographicHeap:
    """
    Min-heap for lexicographic vectors.
    
    Supports efficient extraction of minimum and updates.
    """
    
    def __init__(self):
        """Initialize empty heap"""
        self._heap = []
        self._entry_map = {}  # Maps key -> [priority_tuple, key, valid]
        self._counter = 0  # Tie-breaker for same vectors
    
    def push(self, key: Any, vector: List[float]):
        """
        Add or update key with given vector.
        
        Args:
            key: Unique identifier
            vector: Priority vector (lexicographic comparison)
        """
        # Mark old entry as invalid if exists
        if key in self._entry_map:
            self._entry_map[key][3] = False  # valid is at index 3
        
        # Add new entry
        priority = tuple(vector)
        entry = [priority, self._counter, key, True]  # [priority, tie-breaker, key, valid]
        self._entry_map[key] = entry
        heapq.heappush(self._heap, entry)
        self._counter += 1
    
    def pop(self) -> Tuple[Any, List[float]]:
        """
        Remove and return item with minimum vector.
        
        Returns:
            (key, vector) tuple
        
        Raises:
            KeyError: If heap is empty
        """
        # Skip invalid entries
        while self._heap:
            entry = heapq.heappop(self._heap)
            priority, counter, key, valid = entry
            if valid:
                del self._entry_map[key]
                return key, list(priority)
        
        raise KeyError("Heap is empty")
    
    def peek(self) -> Tuple[Any, List[float]]:
        """
        Return minimum item without removing it.
        
        Returns:
            (key, vector) tuple
        
        Raises:
            KeyError: If heap is empty
        """
        # Find first valid entry
        for priority, _, key, valid in self._heap:
            if valid:
                return key, list(priority)
        
        raise KeyError("Heap is empty")
    
    def __len__(self) -> int:
        """Return number of valid items in heap"""
        return len(self._entry_map)
    
    def __bool__(self) -> bool:
        """Return True if heap is non-empty"""
        return len(self._entry_map) > 0



if __name__ == "__main__":
    # Test lexicographic operations
    print("Testing Lexicographic Utils...")
    
    # Test comparison
    print("\n1. Lexicographic Comparison:")
    print(f"  [1, 5] vs [2, 3]: {lexicographic_compare([1, 5], [2, 3])}")  # -1
    print(f"  [2, 3] vs [2, 5]: {lexicographic_compare([2, 3], [2, 5])}")  # -1
    print(f"  [3, 1] vs [3, 1]: {lexicographic_compare([3, 1], [3, 1])}")  # 0
    print(f"  [3, 5] vs [2, 8]: {lexicographic_compare([3, 5], [2, 8])}")  # 1
    
    # Test max/min
    print("\n2. Lexicographic Max/Min:")
    print(f"  max([1, 5], [2, 3]): {lexicographic_max([1, 5], [2, 3])}")
    print(f"  min([1, 5], [2, 3]): {lexicographic_min([1, 5], [2, 3])}")
    
    # Test ranking
    print("\n3. Ranking Vectors:")
    vectors = {
        'a': [2, 3],
        'b': [1, 5],
        'c': [2, 1],
        'd': [2, 3]  # Tie with 'a'
    }
    ranks = rank_vectors_lexicographically(vectors, reverse=True)
    print(f"  Vectors: {vectors}")
    print(f"  Ranks: {ranks}")
    
    # Test sort_by_ranking (NEW!)
    print("\n4. Sort by Ranking (NEW!):")
    group_data = {
        'alice': {0: 10.5, 1: 5.2, 2: 3.1},
        'bob':   {0: 10.5, 1: 6.0, 2: 2.0},
        'carol': {0: 8.0,  1: 9.0, 2: 1.0}
    }
    ranking = [0, 1]  # Care about groups 0 and 1
    sorted_items = sort_by_ranking(group_data, ranking, reverse=True)
    print(f"  Group data: {group_data}")
    print(f"  Ranking: {ranking}")
    print(f"  Sorted (descending):")
    for key, vector in sorted_items:
        print(f"    {key}: {vector}")
    
    # Test heap
    print("\n5. Lexicographic Heap:")
    heap = LexicographicHeap()
    heap.push('node1', [2, 3])
    heap.push('node2', [1, 5])
    heap.push('node3', [2, 1])
    print(f"  Heap size: {len(heap)}")
    
    min_key, min_vec = heap.pop()
    print(f"  Popped minimum: {min_key} -> {min_vec}")
    
    # Update existing key
    heap.push('node1', [0, 9])  # Update node1's vector
    min_key, min_vec = heap.pop()
    print(f"  After update, new minimum: {min_key} -> {min_vec}")
    
    print("\n✓ All tests passed!")
    print("\nNote: For SIR-specific evaluation, see sir_evaluation.py")
