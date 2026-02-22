import argparse
import os
import torch
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Tokenize corpus and build train/val/test splits.")
    parser.add_argument("--corpus", required=True, help="Path to corpus text file")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--out", required=True, help="Output path for full ids tensor (.pt)")
    parser.add_argument("--splits-out", required=True, help="Output path for split tensors (.pt)")
    parser.add_argument("--train-split", type=float, default=0.98)
    parser.add_argument("--val-split", type=float, default=0.01)
    parser.add_argument("--test-split", type=float, default=0.01)
    args = parser.parse_args()

    if abs(args.train_split + args.val_split + args.test_split - 1.0) > 1e-6:
        raise ValueError("train/val/test splits must sum to 1.0")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    with open(args.corpus, "r", encoding="utf-8") as f:
        text = f.read()

    ids = tokenizer.encode(text).ids
    ids_t = torch.tensor(ids, dtype=torch.long)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.splits_out), exist_ok=True)
    torch.save(ids_t, args.out)

    n_total = len(ids_t)
    n_train = int(n_total * args.train_split)
    n_val = int(n_total * args.val_split)
    n_test = n_total - n_train - n_val

    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("One of the splits is empty. Adjust split ratios.")

    splits = {
        "train": ids_t[:n_train],
        "val": ids_t[n_train:n_train + n_val],
        "test": ids_t[n_train + n_val:]
    }
    torch.save(splits, args.splits_out)

    print(f"Total tokens: {n_total}")
    print(f"Train/Val/Test: {n_train}/{n_val}/{n_test}")
    print(f"Saved full ids to {args.out}")
    print(f"Saved splits to {args.splits_out}")


if __name__ == "__main__":
    main()
