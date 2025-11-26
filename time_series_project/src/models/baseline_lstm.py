
import torch
import torch.nn as nn

class BaselineLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm=nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc=nn.Linear(hidden_dim,1)

    def forward(self,x):
        out,_=self.lstm(x)
        return self.fc(out[:,-1])
