import argparse
import math
import os
from collections import Counter
import torch

from src.utils import load_config, save_json


def build_counts(ids, context_len):
    ctx_counts = Counter()
    ngram_counts = Counter()
    for i in range(context_len, len(ids)):
        ctx = tuple(ids[i - context_len:i])
        ngram = ctx + (ids[i],)
        ctx_counts[ctx] += 1
        ngram_counts[ngram] += 1
    return ctx_counts, ngram_counts


def cross_entropy(ids, context_len, ctx_counts, ngram_counts, vocab_size, k):
    total_logprob = 0.0
    total = 0
    for i in range(context_len, len(ids)):
        ctx = tuple(ids[i - context_len:i])
        ngram = ctx + (ids[i],)
        num = ngram_counts.get(ngram, 0) + k
        den = ctx_counts.get(ctx, 0) + k * vocab_size
        p = num / den
        total_logprob += -math.log(p)
        total += 1
    return total_logprob / max(1, total)


def main():
    parser = argparse.ArgumentParser(description="Add-k smoothed n-gram baseline.")
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--k", type=float, default=0.1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = torch.load(cfg["splits_path"], map_location="cpu")

    train_ids = splits["train"].tolist()
    val_ids = splits["val"].tolist()
    test_ids = splits["test"].tolist()

    ctx_counts, ngram_counts = build_counts(train_ids, cfg["context_len"])

    val_ce = cross_entropy(val_ids, cfg["context_len"], ctx_counts, ngram_counts, cfg["vocab_size"], args.k)
    test_ce = cross_entropy(test_ids, cfg["context_len"], ctx_counts, ngram_counts, cfg["vocab_size"], args.k)

    metrics = {
        "k": args.k,
        "val_loss": val_ce,
        "val_ppl": math.exp(val_ce),
        "test_loss": test_ce,
        "test_ppl": math.exp(test_ce)
    }

    os.makedirs(cfg["artifacts_dir"], exist_ok=True)
    save_json(os.path.join(cfg["artifacts_dir"], "baseline_metrics.json"), metrics)

    print(f"Val loss={val_ce:.4f}, ppl={metrics['val_ppl']:.3f}")
    print(f"Test loss={test_ce:.4f}, ppl={metrics['test_ppl']:.3f}")


if __name__ == "__main__":
    main()
