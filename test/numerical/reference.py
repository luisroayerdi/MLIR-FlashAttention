"""Reference (ground-truth) attention implementation for numerical validation.

Plain numpy, computed the naive unfused way (matches the mathematical
definition, not the online-softmax algorithm Pass 2 implements) so it is an
independent check rather than a re-implementation of the same algorithm.
"""

import numpy as np


def attention_reference(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         scale: float, mask: np.ndarray | None = None) -> np.ndarray:
    """Standard scaled dot-product attention.

    Q: [seq_q, head_dim]   K, V: [seq_k, head_dim]
    mask: optional [seq_q, seq_k] bool array; True = masked out (-inf).
    Returns: [seq_q, head_dim]
    """
    S = (Q @ K.T) * scale
    if mask is not None:
        S = np.where(mask, -np.inf, S)
    S = S - np.max(S, axis=-1, keepdims=True)
    P = np.exp(S)
    P = P / np.sum(P, axis=-1, keepdims=True)
    return P @ V
