
import torch
import numpy as np
from data.dataset import load_complex_dataset
from analysis.eda import perform_eda
from models.transformer_model import TimeSeriesTransformer
from evaluation.metrics import mase, rmse, mape
from tuning.hyperparameter_search import hyperparameter_search
from interpretation.attention import extract_attention_weights

df = load_complex_dataset()
perform_eda(df)

series = df['value'].values
window = 50
X, y = [], []
for i in range(len(series) - window):
    X.append(series[i:i+window])
    y.append(series[i+window])

X = torch.tensor(np.array(X)).float().unsqueeze(-1)
y = torch.tensor(np.array(y)).float()

params = hyperparameter_search()
model = TimeSeriesTransformer(**params)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

for epoch in range(5):
    opt.zero_grad()
    pred = model(X)
    loss = loss_fn(pred.squeeze(), y)
    loss.backward()
    opt.step()

pred_np = pred.detach().numpy().squeeze()
y_np = y.numpy()

print("MASE:", mase(y_np, pred_np))
print("RMSE:", rmse(y_np, pred_np))
print("MAPE:", mape(y_np, pred_np))

att = extract_attention_weights(model)
print("Attention:", att)
