
import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding

class TransformerEncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(nn.Linear(model_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, model_dim))
        self.ln2 = nn.LayerNorm(model_dim)
        self._last_attn = None
    def forward(self, x):
        attn_out, attn_w = self.mha(x, x, x, need_weights=True)
        # attn_w shape: (batch, num_heads, seq_len, seq_len) in newer torch or (batch, seq_len, seq_len)
        self._last_attn = attn_w.detach().cpu()
        x = self.ln1(x + attn_out)
        x2 = self.ff(x)
        x = self.ln2(x + x2)
        return x
    def get_attn(self):
        return self._last_attn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, seq_len, feature_dim=1, model_dim=64, num_heads=4, num_layers=2, ff_dim=128, pred_len=1):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, model_dim)
        self.pos = PositionalEncoding(model_dim, max_len=seq_len+10)
        self.layers = nn.ModuleList([TransformerEncoderBlock(model_dim, num_heads, ff_dim) for _ in range(num_layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(model_dim, pred_len)
    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        x = self.input_proj(x)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x)
        x_p = x.permute(0,2,1)
        pooled = self.pool(x_p).squeeze(-1)
        out = self.fc(pooled)
        return out.unsqueeze(1)
    def get_all_attentions(self):
        return [l.get_attn() for l in self.layers]
