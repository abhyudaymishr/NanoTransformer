import argparse
import os
import time
import numpy as np
import torch

from .model import MiniTransformerLM
from .utils import load_config, save_json


def measure(model, device, vocab_size, context_len, iters, warmup):
    model.eval()
    times = []

    with torch.inference_mode():
        for _ in range(warmup):
            ctx = torch.randint(0, vocab_size, (1, context_len), device=device)
            _ = model(ctx)
            if device.type == "cuda":
                torch.cuda.synchronize()

        for _ in range(iters):
            ctx = torch.randint(0, vocab_size, (1, context_len), device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(ctx)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99))
    }


def main():
    parser = argparse.ArgumentParser(description="Measure single-token latency.")
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = MiniTransformerLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        ff_mult=cfg["ff_mult"],
        context_len=cfg["context_len"],
        dropout=cfg["dropout"]
    ).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])

    metrics = measure(model, device, cfg["vocab_size"], cfg["context_len"], args.iters, args.warmup)

    os.makedirs(cfg["artifacts_dir"], exist_ok=True)
    save_json(os.path.join(cfg["artifacts_dir"], "latency.json"), metrics)

    print(f"mean={metrics['mean_ms']:.3f}ms p50={metrics['p50_ms']:.3f}ms p95={metrics['p95_ms']:.3f}ms p99={metrics['p99_ms']:.3f}ms")


if __name__ == "__main__":
    main()
