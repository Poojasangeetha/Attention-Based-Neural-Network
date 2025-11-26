
import torch
from torch import nn
from .positional_encoding import PositionalEncoding

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.attention_weights = None

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        encoded = self.encoder(x)
        return self.decoder(encoded[:, -1, :])
