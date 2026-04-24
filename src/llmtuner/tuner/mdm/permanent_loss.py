"""
Permanent-based Sudoku rule loss.

For a peer set of 9 cells with per-cell digit distributions stacked into a 9x9
matrix M (M[i, d] = P(cell_i = digit d)), the permanent

    perm(M) = sum over σ in S_9 of prod_i M[i, σ(i)]

equals the probability that independent marginal sampling produces a valid
peer set (a permutation of digits 1..9). perm(M) ∈ [0, 1] with 1 at one-hot
permutation matrices.

Loss per peer set: -log(perm(M)). Total loss: sum over the 27 peer sets, mean
over batch.

Computed via Ryser's formula (O(n · 2ⁿ), fully vectorised, differentiable),
in float32 for numerical stability of the alternating sum.
"""
from typing import Optional

import torch
import torch.nn.functional as F

from llmtuner.tuner.mdm.rule_loss import _get_groups


# ---------------------------------------------------------------------------
# Precomputed Ryser subset masks for n = 9.
# subset_mask[s, j] == 1 iff column j is in subset s.
# signs[s] = (-1)^(n - popcount(s)).
# Each is a 1-D / 2-D tensor of fixed shape, independent of data.
# ---------------------------------------------------------------------------
def _build_ryser_tables(n: int = 9):
    num_subsets = 1 << n  # 2^n
    subset_mask = torch.zeros(num_subsets, n, dtype=torch.float32)
    signs = torch.empty(num_subsets, dtype=torch.float32)
    for s in range(num_subsets):
        popcount = bin(s).count("1")
        for j in range(n):
            if (s >> j) & 1:
                subset_mask[s, j] = 1.0
        signs[s] = 1.0 if (n - popcount) % 2 == 0 else -1.0
    return subset_mask, signs


_SUBSET_MASK, _SIGNS = _build_ryser_tables(9)
_RYSER_CACHE: dict = {}


def _get_ryser(device: torch.device):
    key = str(device)
    if key not in _RYSER_CACHE:
        _RYSER_CACHE[key] = (_SUBSET_MASK.to(device), _SIGNS.to(device))
    return _RYSER_CACHE[key]


def ryser_permanent(M: torch.Tensor) -> torch.Tensor:
    """
    Compute the permanent of batched 9x9 matrices via Ryser's formula.

    M: [..., 9, 9] float tensor (any batch shape, last two dims are the matrix).
    Returns: [...] scalar per batch element.

    Ryser (shifted sign form, summing the empty subset which contributes 0):
        perm(M) = sum_S (-1)^(n - |S|) * prod_i (sum_{j in S} M[i, j])
    """
    n = M.size(-1)
    assert M.size(-2) == n, f"Ryser expects square last dims, got {M.shape}"
    subset_mask, signs = _get_ryser(M.device)
    # Row-sums over the columns in each subset.
    # R[..., s, i] = sum_j M[..., i, j] * subset_mask[s, j]
    R = torch.einsum("...ij,sj->...si", M, subset_mask)  # [..., 2^n, n]
    P = R.prod(dim=-1)                                    # [..., 2^n]
    return (signs * P).sum(dim=-1)                        # [...]


def compute_permanent_loss(
    logits: torch.Tensor,
    src_mask: torch.Tensor,
    tokenizer,
    target_len: int = 81,
    digit_ids: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Args:
        logits: [B, L, V] — already shift-corrected (same contract as
                compute_rule_loss).
        src_mask: [B, L] — 1/True at prompt + [SEP] positions.
        tokenizer: CustomTokenizer — used to find digit token ids (1..9).
        target_len: 81 for standard Sudoku.
        digit_ids: optional [9] tensor to skip tokenizer lookup.
        eps: clamp floor on perm(M) before the log, to guard against
             sharp-distribution edge cases where perm(M) would otherwise be 0.

    Returns:
        Scalar tensor: sum over 27 peer sets of -log(perm(M_g)), averaged
        over batch. In [0, ∞), 0 only if every peer-set distribution is a
        one-hot permutation (fully-solved Sudoku).
    """
    device = logits.device
    B, L, V = logits.shape

    if digit_ids is None:
        from llmtuner.tuner.mdm.rule_loss import digit_token_ids
        digit_ids = torch.tensor(digit_token_ids(tokenizer), device=device, dtype=torch.long)
    elif digit_ids.device != device:
        digit_ids = digit_ids.to(device)

    # Locate the 81 target cells (same as in compute_rule_loss).
    src_mask_bool = src_mask.bool()
    target_start = src_mask_bool.sum(dim=1)                      # [B]
    offsets = torch.arange(target_len, device=device).unsqueeze(0)  # [1, 81]
    target_positions = (target_start.unsqueeze(1) + offsets).clamp(max=L - 1)
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, target_len)
    cell_logits = logits[batch_idx, target_positions]            # [B, 81, V]

    # Promote to fp32 to keep the alternating Ryser sum numerically stable.
    cell_logits = cell_logits.float()
    digit_logits = cell_logits.index_select(-1, digit_ids)       # [B, 81, 9]
    p = F.softmax(digit_logits, dim=-1)                          # [B, 81, 9]

    groups = _get_groups(device)                                  # [27, 9]
    M = p[:, groups, :]                                           # [B, 27, 9, 9]

    perm = ryser_permanent(M)                                     # [B, 27]
    neglog = -torch.log(perm.clamp(min=eps))                      # [B, 27]

    # Sum over 27 peer sets, mean over batch.
    return neglog.sum(dim=1).mean()
