"""Embedding helpers for input-only component identities.

The loader attaches a deterministic ``component_vocab_id`` to each of the
first four component graphs.  The vocabulary is derived from the active input
CSV only, not from labels or an external evaluation set.
"""

import torch
import torch.nn as nn


# Backwards-compatible defaults for historical configurations.  New
# OneHotEmbedGPS runs supply input-derived sizes through ``cfg``.
COMPONENT_VOCAB_SIZES = {
    1: 2,   # IL_SMILE: 2 unique molecules
    2: 3,   # HL_SMILE: 3 unique molecules
    3: 2,   # Chol_SMILE: 2 unique molecules
    4: 3,   # PEG_SMILE: 3 unique molecules
}

# Atom-count fallback for legacy processed datasets that do not contain
# ``component_vocab_id``. New input-only runs never use this mapping.
COMPONENT_ATOM_TO_ID = {
    1: {149: 0, 137: 1},                    # IL
    2: {129: 0, 138: 1, 142: 2},            # HL
    3: {74: 0, 80: 1},                       # Chol
    4: {103: 0, 119: 1, 143: 2},            # PEG
}


def get_vocab_id(component_idx, num_nodes):
    """Map atom count to vocabulary ID for a given component (1-4).

    Args:
        component_idx: int in {1, 2, 3, 4}
        num_nodes: tensor of shape [batch_size] with atom counts

    Returns:
        tensor of shape [batch_size] with vocabulary IDs, or -1 for unknown.
    """
    mapping = COMPONENT_ATOM_TO_ID[component_idx]
    ids = torch.full_like(num_nodes, -1, dtype=torch.long)
    for atom_count, vocab_id in mapping.items():
        ids = torch.where(num_nodes == atom_count, vocab_id, ids)
    return ids


def build_component_embeddings(hidden_dim, vocab_sizes=None):
    """Build embedding layers for components 1-4.

    Returns:
        nn.ModuleDict with keys 'comp1'..'comp4', each an nn.Embedding.
    """
    if vocab_sizes is None:
        vocab_sizes = [COMPONENT_VOCAB_SIZES[i] for i in range(1, 5)]
    if len(vocab_sizes) != 4 or any(int(size) <= 0 for size in vocab_sizes):
        raise ValueError('vocab_sizes must contain four positive sizes.')
    return nn.ModuleDict({
        f'comp{i}': nn.Embedding(int(vocab_sizes[i - 1]), hidden_dim)
        for i in range(1, 5)
    })
