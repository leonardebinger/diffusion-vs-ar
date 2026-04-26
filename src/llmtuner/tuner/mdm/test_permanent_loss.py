"""
Unit tests for the permanent-based Sudoku rule loss. CPU-only.

Run:  python -m pytest src/llmtuner/tuner/mdm/test_permanent_loss.py -v
Or:   python src/llmtuner/tuner/mdm/test_permanent_loss.py
"""
import itertools
import math

import torch

from llmtuner.tuner.mdm.permanent_loss import (
    ryser_permanent,
    compute_permanent_loss,
    _build_ryser_tables,
)


class _StubTokenizer:
    _vocab = [str(d) for d in range(10)]

    def _convert_token_to_id(self, tok):
        return 5 + self._vocab.index(tok)  # '1' -> 6, ..., '9' -> 14


def _brute_force_permanent(M: torch.Tensor) -> float:
    """O(n! · n) brute-force permanent for small n, for cross-checking Ryser."""
    n = M.size(-1)
    total = 0.0
    for perm in itertools.permutations(range(n)):
        prod = 1.0
        for i, j in enumerate(perm):
            prod *= M[i, j].item()
        total += prod
    return total


def test_identity_gives_perm_1():
    """perm(I_n) = 1 since only the identity permutation contributes."""
    I = torch.eye(9)
    p = ryser_permanent(I).item()
    assert abs(p - 1.0) < 1e-5, f"perm(I_9) = {p}, expected 1"


def test_uniform_J_over_n_gives_factorial_over_n_n():
    """perm((1/n) J_n) = n! / n^n."""
    n = 9
    M = torch.full((n, n), 1.0 / n)
    p = ryser_permanent(M).item()
    expected = math.factorial(n) / (n ** n)  # 9!/9^9 ≈ 9.4e-4
    assert abs(p - expected) / expected < 1e-3, f"perm(J_9/9) = {p}, expected {expected}"


def test_ryser_matches_brute_force_small():
    """Cross-check Ryser vs brute-force on small random n=4 matrices."""
    # n=4: 4! = 24 permutations. Ryser table is 2^4=16.
    # Rebuild tables for n=4 and compare.
    torch.manual_seed(0)
    for _ in range(5):
        M = torch.rand(4, 4)
        # Ryser at n=4:
        subset_mask, signs = _build_ryser_tables(4)
        R = torch.einsum("ij,sj->si", M, subset_mask)
        P = R.prod(dim=-1)
        ryser_val = (signs * P).sum().item()
        bf_val = _brute_force_permanent(M)
        assert abs(ryser_val - bf_val) < 1e-5, \
            f"Ryser {ryser_val} != brute force {bf_val} on {M}"


def test_random_9x9_matches_brute_force():
    """Cross-check 9x9: n=9 has 362 880 permutations, still feasible once.
    Tolerance 1e-3 because float32 catastrophic cancellation in the
    alternating-sign Ryser sum dominates at this matrix scale (~1e-7 permanents)."""
    torch.manual_seed(1)
    M = torch.rand(9, 9) * 0.1  # small entries → tiny permanent (~1e-7), float32 limit
    ryser_val = ryser_permanent(M).item()
    bf_val = _brute_force_permanent(M)
    rel_err = abs(ryser_val - bf_val) / max(abs(bf_val), 1e-20)
    assert rel_err < 1e-3, f"Ryser {ryser_val} vs brute force {bf_val} (rel_err {rel_err})"


def _make_logits_from_probs(probs_9: torch.Tensor, L: int = 164, V: int = 31,
                            target_start: int = 82, target_len: int = 81):
    """Build [1, L, V] logits so softmax over digit channels at target
    positions reproduces probs_9 exactly (non-digit channels at -1e4)."""
    logits = torch.full((1, L, V), -1e4)
    digit_ids = list(range(6, 15))  # tokens '1'..'9'
    log_probs = probs_9.clamp(min=1e-30).log()
    for cell_idx in range(target_len):
        pos = target_start + cell_idx
        for k, tok_id in enumerate(digit_ids):
            logits[0, pos, tok_id] = log_probs[cell_idx, k].item()
    return logits


def _src_mask(L: int = 164, sep_pos: int = 81):
    m = torch.zeros(1, L, dtype=torch.bool)
    m[0, : sep_pos + 1] = True
    return m


def test_solved_sudoku_gives_zero_loss():
    """One-hot matching a valid solved Sudoku → perm = 1 per group → loss = 0."""
    tok = _StubTokenizer()
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
    probs = torch.zeros(81, 9)
    for r in range(9):
        for c in range(9):
            probs[r * 9 + c, solution[r][c] - 1] = 1.0
    logits = _make_logits_from_probs(probs)
    loss = compute_permanent_loss(logits, _src_mask(), tok).item()
    assert loss < 1e-3, f"Expected ≈ 0, got {loss}"


def test_uniform_marginals_match_theory():
    """Uniform 1/9 at every cell → each peer-set permanent is 9!/9^9."""
    tok = _StubTokenizer()
    probs = torch.full((81, 9), 1.0 / 9)
    logits = _make_logits_from_probs(probs)
    loss = compute_permanent_loss(logits, _src_mask(), tok).item()
    expected_per_group = math.factorial(9) / (9 ** 9)
    expected_total = 27 * (-math.log(expected_per_group))
    rel_err = abs(loss - expected_total) / expected_total
    assert rel_err < 1e-3, f"Expected {expected_total}, got {loss} (rel_err {rel_err})"


def test_gradient_flows_to_target_only():
    """Gradient of permanent loss should touch target positions only."""
    tok = _StubTokenizer()
    probs = torch.full((81, 9), 1.0 / 9)
    logits = _make_logits_from_probs(probs).clone().requires_grad_(True)
    loss = compute_permanent_loss(logits, _src_mask(), tok)
    loss.backward()
    assert logits.grad is not None
    grad_target = logits.grad[0, 82:163].abs().sum().item()
    grad_non_target = (logits.grad[0, :82].abs().sum().item()
                       + logits.grad[0, 163:].abs().sum().item())
    assert grad_target > 0, "expected nonzero gradient on target positions"
    assert grad_non_target < 1e-6, \
        f"expected zero gradient outside target, got {grad_non_target}"


if __name__ == "__main__":
    tests = [
        test_identity_gives_perm_1,
        test_uniform_J_over_n_gives_factorial_over_n_n,
        test_ryser_matches_brute_force_small,
        test_random_9x9_matches_brute_force,
        test_solved_sudoku_gives_zero_loss,
        test_uniform_marginals_match_theory,
        test_gradient_flows_to_target_only,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print("all permanent_loss tests passed")
