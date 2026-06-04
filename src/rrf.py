from qdrant_client.models import ScoredPoint

import numpy as np

def rrf(rank_lists, k=60, default_rank=1000):
    """
    Reciprocal Rank Fusion (RRF).
    """
    all_items = set(item for rank_list in rank_lists for item, _ in rank_list)
    item_to_index = {item: idx for idx, item in enumerate(all_items)}

    rank_matrix = np.full((len(all_items), len(rank_lists)), default_rank)

    for list_idx, rank_list in enumerate(rank_lists):
        for item, rank in rank_list:
            rank_matrix[item_to_index[item], list_idx] = rank

    rrf_scores = np.sum(1.0 / (k + rank_matrix), axis=1)

    sorted_indices = np.argsort(-rrf_scores)
    index_to_item = {idx: item for item, idx in item_to_index.items()}
    sorted_items = [(index_to_item[idx], rrf_scores[idx]) for idx in sorted_indices]

    return sorted_items


def rank_list(search_result: list[ScoredPoint]):
    """Convert Qdrant search results to (id, rank) pairs."""
    return [(point.id, rank + 1) for rank, point in enumerate(search_result)]
