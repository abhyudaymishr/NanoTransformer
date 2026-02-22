import argparse
import math
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.dataset import TrigramDataset
from src.model import MiniTransformerLM
from src.utils import load_config, set_seed, count_params, save_json


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
    return total_loss / max(1, total)


def make_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="Train NanoTransformer.")
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    splits = torch.load(cfg["splits_path"], map_location="cpu")
    train_ds = TrigramDataset(splits["train"], context_len=cfg["context_len"])
    val_ds = TrigramDataset(splits["val"], context_len=cfg["context_len"])

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory
    )

    model = MiniTransformerLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        ff_mult=cfg["ff_mult"],
        context_len=cfg["context_len"],
        dropout=cfg["dropout"]
    ).to(device)

    param_count = count_params(model)
    print(f"Parameter count: {param_count}")

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = cfg["max_epochs"] * max(1, len(train_loader))
    scheduler = make_scheduler(optimizer, cfg["warmup_steps"], total_steps)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["artifacts_dir"], exist_ok=True)

    best_val = float("inf")
    best_epoch = -1
    global_step = 0

    for epoch in range(cfg["max_epochs"]):
        model.train()
        epoch_loss = 0.0
        total = 0
        start = time.time()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            if cfg["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * x.size(0)
            total += x.size(0)
            global_step += 1

        train_loss = epoch_loss / max(1, total)
        val_loss = evaluate(model, val_loader, device, criterion)
        val_ppl = math.exp(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start

        print(f"Epoch {epoch + 1}/{cfg['max_epochs']} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.3f} | lr={lr:.2e} | {elapsed:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "param_count": param_count,
                "epoch": best_epoch,
                "global_step": global_step
            }
            torch.save(ckpt, os.path.join(cfg["checkpoint_dir"], "best.pt"))

    metrics = {
        "best_val_loss": best_val,
        "best_val_ppl": math.exp(best_val),
        "best_epoch": best_epoch,
        "param_count": param_count
    }
    save_json(os.path.join(cfg["artifacts_dir"], "metrics.json"), metrics)
    save_json(os.path.join(cfg["artifacts_dir"], "model_config.json"), cfg)
    print("Training complete. Metrics saved to artifacts/metrics.json")


if __name__ == "__main__":
    main()
