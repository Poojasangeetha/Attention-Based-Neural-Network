
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, seq_len, feature_dim, model_dim, num_heads, num_layers):
        super().__init__()
        self.embed=nn.Linear(feature_dim, model_dim)
        layer=nn.TransformerEncoderLayer(model_dim,num_heads)
        self.encoder=nn.TransformerEncoder(layer, num_layers)
        self.fc=nn.Linear(model_dim,1)

    def forward(self,x):
        x=self.embed(x)
        x=self.encoder(x)
        return self.fc(x[:,-1])

    def get_attention(self):
        return None  # simplified
