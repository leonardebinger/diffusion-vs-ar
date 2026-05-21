#!/usr/bin/env python3
"""Convert Shah et al. 2024 Sudoku NPY files to Ye codebase CSV format.

Shah NPY format (each row: 325 ints):
    row[0]                     = number of given cells in the puzzle
    row[1+4i .. 1+4i+3], i=0..80 = (cell_row, cell_col, correct_value, strategy)
        strategy == 0  -> cell is GIVEN in the puzzle
        strategy != 0  -> cell is EMPTY in the puzzle (value is the answer)

Ye CSV format (two columns):
    quizzes   = 81-char string, '0' = blank, '1'-'9' = given digit
    solutions = 81-char string, fully solved puzzle

Usage:
    # full conversion (1.8M train, ~100k test)
    python3 convert_shah_sudoku.py <input_dir> <output_dir>

    # subsampled
    python3 convert_shah_sudoku.py <input_dir> <output_dir> \\
        --train-samples 100000 --test-samples 10000
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def npy_to_csv(npy_path: Path, csv_path: Path, n_samples=None, seed: int = 42) -> None:
    print(f"Loading {npy_path}...", flush=True)
    data = np.load(npy_path, allow_pickle=True)
    n_total = len(data)
    print(f"  total puzzles: {n_total}", flush=True)

    if n_samples is not None and n_samples < n_total:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, n_samples, replace=False)
        idx.sort()
        data = data[idx]
        print(f"  subsampled to: {n_samples} (seed={seed})", flush=True)

    n = len(data)
    print(f"Writing {csv_path}...", flush=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quizzes", "solutions"])

        for i, row in enumerate(data):
            if i and i % 100_000 == 0:
                print(f"  {i:,}/{n:,}", flush=True)

            puzzle = ["0"] * 81
            solution = ["0"] * 81

            for k in range(81):
                base = 1 + k * 4
                r = int(row[base])
                c = int(row[base + 1])
                v = int(row[base + 2])
                strategy = int(row[base + 3])
                pos = r * 9 + c
                solution[pos] = str(v)
                if strategy == 0:
                    puzzle[pos] = str(v)

            writer.writerow(["".join(puzzle), "".join(solution)])

    print(f"  done: {n:,} rows -> {csv_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input_dir", type=Path,
                   help="Directory containing sudoku-train-data.npy and sudoku-test-data.npy")
    p.add_argument("output_dir", type=Path,
                   help="Directory to write sudoku_train.csv and sudoku_test.csv")
    p.add_argument("--train-samples", type=int, default=None,
                   help="Optional subsample size for training set (default: all)")
    p.add_argument("--test-samples", type=int, default=None,
                   help="Optional subsample size for test set (default: all)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    npy_to_csv(
        args.input_dir / "sudoku-train-data.npy",
        args.output_dir / "sudoku_train.csv",
        n_samples=args.train_samples, seed=args.seed,
    )
    npy_to_csv(
        args.input_dir / "sudoku-test-data.npy",
        args.output_dir / "sudoku_test.csv",
        n_samples=args.test_samples, seed=args.seed,
    )


if __name__ == "__main__":
    main()