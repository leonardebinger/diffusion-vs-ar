"""
Unit tests for the Sudoku rule loss. CPU-only, no GPU/model needed.

Run:  python -m pytest src/llmtuner/tuner/mdm/test_rule_loss.py -v
Or:   python src/llmtuner/tuner/mdm/test_rule_loss.py   (runs asserts directly)
"""
import math

import torch

from llmtuner.tuner.mdm.rule_loss import (
    _sudoku_group_indices,
    compute_rule_loss,
    digit_token_ids,
)


class _StubTokenizer:
    """Mimics CustomTokenizer's token-id scheme: specials 0..4, then vocab chars 5+."""
    _vocab = [str(d) for d in range(10)]  # '0'..'9'

    def _convert_token_to_id(self, tok):
        # '0' -> 5, '1' -> 6, ..., '9' -> 14 (matches model_config_tiny/tokenizer_config.json)
        return 5 + self._vocab.index(tok)


def _make_logits_from_probs(probs_9: torch.Tensor, L: int, V: int = 31,
                            target_start: int = 82, target_len: int = 81):
    """
    Build a [1, L, V] logits tensor whose post-shift softmax over digit channels (ids 6..14)
    at target positions 82..162 equals probs_9.

    probs_9: [target_len, 9] desired per-cell distribution over digits 1..9.

    We place large log-probs at positions target_start-1..target_start+target_len-2 so that
    after the right-shift (logits = cat([logits[:,0:1], logits[:,:-1]])) they land at
    target_start..target_start+target_len-1. But to keep the test self-contained we skip
    the shift and pass already-shift-aligned logits directly (the trainer does the shift
    before calling compute_rule_loss).
    """
    logits = torch.full((1, L, V), -1e4)
    # Digit token ids for '1'..'9' are 6..14
    digit_ids = list(range(6, 15))
    # Work in log space to produce desired softmax over the 9 digit channels.
    # Since all non-digit channels are at -1e4 ≈ 0 probability, softmax over full vocab
    # ≈ softmax over digit-only subset.
    log_probs = probs_9.clamp(min=1e-30).log()  # [target_len, 9]
    for cell_idx in range(target_len):
        pos = target_start + cell_idx
        for k, tok_id in enumerate(digit_ids):
            logits[0, pos, tok_id] = log_probs[cell_idx, k].item()
    return logits


def _src_mask(L: int = 164, sep_pos: int = 81):
    """src_mask: True for prompt + [SEP] = first sep_pos+1 positions."""
    m = torch.zeros(1, L, dtype=torch.bool)
    m[0, : sep_pos + 1] = True
    return m


def test_groups_structure():
    groups = _sudoku_group_indices()
    assert groups.shape == (27, 9)
    # Every cell (0..80) appears in exactly 3 groups (1 row, 1 col, 1 box)
    flat = groups.flatten().tolist()
    for cell in range(81):
        assert flat.count(cell) == 3, f"cell {cell} appears {flat.count(cell)} times, expected 3"


def test_uniform_distribution_gives_1_over_9():
    """Every cell has uniform 1/9 probability over digits → per-pair collision = 1/9."""
    tok = _StubTokenizer()
    assert digit_token_ids(tok) == list(range(6, 15))
    probs = torch.full((81, 9), 1.0 / 9)
    logits = _make_logits_from_probs(probs, L=164)
    loss = compute_rule_loss(logits, _src_mask(), tok)
    assert abs(loss.item() - 1.0 / 9) < 1e-4, f"expected 1/9, got {loss.item()}"


def test_valid_solved_sudoku_gives_zero():
    """Cells set to one-hot matching a valid solved Sudoku → collision = 0."""
    tok = _StubTokenizer()
    # Hand-written valid Sudoku solution (9 digits per row, valid across rows/cols/boxes)
    solution = [
        [5,3,4, 6,7,8, 9,1,2],
        [6,7,2, 1,9,5, 3,4,8],
        [1,9,8, 3,4,2, 5,6,7],
        [8,5,9, 7,6,1, 4,2,3],
        [4,2,6, 8,5,3, 7,9,1],
        [7,1,3, 9,2,4, 8,5,6],
        [9,6,1, 5,3,7, 2,8,4],
        [2,8,7, 4,1,9, 6,3,5],
        [3,4,5, 2,8,6, 1,7,9],
    ]
    # one-hot over digits 1..9 (index 0 → digit 1, index 8 → digit 9)
    probs = torch.zeros(81, 9)
    for r in range(9):
        for c in range(9):
            probs[r * 9 + c, solution[r][c] - 1] = 1.0

    logits = _make_logits_from_probs(probs, L=164)
    loss = compute_rule_loss(logits, _src_mask(), tok)
    assert loss.item() < 1e-4, f"expected ~0, got {loss.item()}"


def test_all_same_digit_row_hits_ceiling():
    """All 9 cells in one row put mass 1 on the same digit → that row's collision = 1 per pair."""
    tok = _StubTokenizer()
    probs = torch.full((81, 9), 1.0 / 9)
    # Make row 0 degenerate: every cell picks digit 1 (index 0)
    for c in range(9):
        probs[c] = 0
        probs[c, 0] = 1.0
    logits = _make_logits_from_probs(probs, L=164)
    loss = compute_rule_loss(logits, _src_mask(), tok)
    # Row 0 contributes 1.0 (max collision). The same 9 cells also participate in 9 cols and 3 boxes.
    # Those other groups see 1/9 uniform in their remaining cells and a deterministic cell from row 0.
    # So the overall mean should be noticeably > 1/9.
    assert loss.item() > 1.0 / 9, f"expected > 1/9 when a row is fully degenerate, got {loss.item()}"


def test_gradient_flows():
    """Rule loss should be differentiable wrt logits."""
    tok = _StubTokenizer()
    probs = torch.full((81, 9), 1.0 / 9)
    logits = _make_logits_from_probs(probs, L=164)
    logits = logits.clone().requires_grad_(True)
    loss = compute_rule_loss(logits, _src_mask(), tok)
    loss.backward()
    assert logits.grad is not None
    # Gradient should be non-zero at target positions, zero elsewhere
    grad_target = logits.grad[0, 82:163].abs().sum().item()
    grad_non_target = logits.grad[0, :82].abs().sum().item() + logits.grad[0, 163:].abs().sum().item()
    assert grad_target > 0, "expected nonzero gradient on target positions"
    assert grad_non_target < 1e-6, f"expected zero gradient outside target, got {grad_non_target}"


if __name__ == "__main__":
    tests = [
        test_groups_structure,
        test_uniform_distribution_gives_1_over_9,
        test_valid_solved_sudoku_gives_zero,
        test_all_same_digit_row_hits_ceiling,
        test_gradient_flows,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print("all rule_loss tests passed")
