import argparse
import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import TrigramDataset
from src.model import MiniTransformerLM
from src.utils import load_config, save_json


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate NanoTransformer.")
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    splits = torch.load(cfg["splits_path"], map_location="cpu")
    val_ds = TrigramDataset(splits["val"], context_len=cfg["context_len"])
    test_ds = TrigramDataset(splits["test"], context_len=cfg["context_len"])

    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = MiniTransformerLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        ff_mult=cfg["ff_mult"],
        context_len=cfg["context_len"],
        dropout=cfg["dropout"]
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    criterion = nn.CrossEntropyLoss()
    val_loss = evaluate(model, val_loader, device, criterion)
    test_loss = evaluate(model, test_loader, device, criterion)

    metrics = {
        "val_loss": val_loss,
        "val_ppl": math.exp(val_loss),
        "test_loss": test_loss,
        "test_ppl": math.exp(test_loss)
    }
    os.makedirs(cfg["artifacts_dir"], exist_ok=True)
    save_json(os.path.join(cfg["artifacts_dir"], "eval_metrics.json"), metrics)

    print(f"Val loss={val_loss:.4f}, ppl={metrics['val_ppl']:.3f}")
    print(f"Test loss={test_loss:.4f}, ppl={metrics['test_ppl']:.3f}")


if __name__ == "__main__":
    main()
