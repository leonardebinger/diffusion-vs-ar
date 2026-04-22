"""
Sudoku rule-based auxiliary loss for MDM training.

Computes per-pair expected collision probability across the 27 Sudoku groups
(9 rows + 9 columns + 9 3x3 boxes) from the model's predicted digit distributions
at the 81 target cells. Valid solved sudokus have collision = 0 everywhere.

Used as an auxiliary term: total_loss = ce_loss + lambda * rule_loss.
"""
from typing import Optional

import torch
import torch.nn.functional as F


def _sudoku_group_indices() -> torch.Tensor:
    """Return a [27, 9] int64 tensor whose rows list the 9 cell indices in each
    of the 27 Sudoku groups (9 rows, then 9 cols, then 9 boxes)."""
    rows = [[r * 9 + c for c in range(9)] for r in range(9)]
    cols = [[r * 9 + c for r in range(9)] for c in range(9)]
    boxes = []
    for br in range(3):
        for bc in range(3):
            cells = [(br * 3 + dr) * 9 + (bc * 3 + dc)
                     for dr in range(3) for dc in range(3)]
            boxes.append(cells)
    return torch.tensor(rows + cols + boxes, dtype=torch.long)


_GROUP_CACHE: dict = {}


def _get_groups(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _GROUP_CACHE:
        _GROUP_CACHE[key] = _sudoku_group_indices().to(device)
    return _GROUP_CACHE[key]


def digit_token_ids(tokenizer) -> list:
    """Token ids for the 9 valid solution digits '1'..'9'."""
    return [tokenizer._convert_token_to_id(str(d)) for d in range(1, 10)]


def compute_rule_loss(
    logits: torch.Tensor,
    src_mask: torch.Tensor,
    tokenizer,
    target_len: int = 81,
    digit_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Args:
        logits: [B, L, V] — already shift-corrected to align with target positions
                (parent's inner_forward applies: logits = cat([logits[:,0:1], logits[:,:-1]])).
        src_mask: [B, L] — 1/True at prompt + [SEP] positions, 0/False after.
        tokenizer: CustomTokenizer — used to find digit token ids (1..9).
        target_len: number of target cells (81 for standard Sudoku).
        digit_ids: optional [9] tensor, precomputed to avoid repeated lookups.

    Returns:
        Scalar tensor in [0, 1] — mean per-pair expected digit collision across
        27 groups × batch. 0 for one-hot distributions matching a valid solution.
    """
    device = logits.device
    B, L, V = logits.shape

    if digit_ids is None:
        digit_ids = torch.tensor(digit_token_ids(tokenizer), device=device, dtype=torch.long)
    elif digit_ids.device != device:
        digit_ids = digit_ids.to(device)

    # Target start: first False in src_mask = count of True positions = len(src) + 1 (for [SEP])
    src_mask_bool = src_mask.bool()
    target_start = src_mask_bool.sum(dim=1)  # [B]

    offsets = torch.arange(target_len, device=device).unsqueeze(0)  # [1, target_len]
    target_positions = target_start.unsqueeze(1) + offsets          # [B, target_len]
    target_positions = torch.clamp(target_positions, max=L - 1)

    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, target_len)
    cell_logits = logits[batch_idx, target_positions]  # [B, target_len, V]

    digit_logits = cell_logits.index_select(-1, digit_ids)  # [B, target_len, 9]
    # Softmax restricted to the 9 digit classes (other tokens are invalid solution values).
    p = F.softmax(digit_logits, dim=-1)  # [B, target_len, 9]

    groups = _get_groups(device)         # [27, 9]
    p_g = p[:, groups, :]                # [B, 27, 9, 9]

    # Expected collision pairs per group = sum_{i<j} <p_i, p_j>
    #   = 0.5 * (||sum_i p_i||^2 - sum_i ||p_i||^2)
    sum_p = p_g.sum(dim=2)                    # [B, 27, 9]
    sum_sq = (sum_p * sum_p).sum(-1)          # [B, 27]
    self_sq = (p_g * p_g).sum(-1).sum(-1)     # [B, 27]
    expected_pairs = 0.5 * (sum_sq - self_sq) # [B, 27]

    # C(9, 2) = 36 pairs per group → normalize to per-pair probability in [0, 1]
    per_group = expected_pairs / 36.0         # [B, 27]

    return per_group.mean()
