import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x):
        t = x.size(1)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        out, _ = self.mha(x, x, x, attn_mask=mask, need_weights=False)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.ff(self.ln2(x))
        return x


class MiniTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4, ff_mult=4, context_len=3, dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(context_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_mult=ff_mult, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self.context_len = context_len

    def forward(self, x):
        bsz, t = x.size()
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(bsz, t)
        x = self.embed(x) + self.pos(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x[:, -1, :])
        return logits
