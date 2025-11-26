
import os
import numpy as np
import torch
from config import Config
from data.dataset import load_or_generate
from preprocessing.pipeline import scale_series, create_sequences
from analysis.eda import run_eda
from models.transformer import TimeSeriesTransformer
from models.lstm_baseline import BaselineLSTM
from training.train import train_loop
from evaluation.metrics import basic_eval
from tuning.grid_search import grid_search
from interpretation.attention_viz import save_attention_maps

os.makedirs(Config.SAVE_DIR, exist_ok=True)
os.makedirs(Config.REPORTS_DIR, exist_ok=True)

# 1) Load data
df = load_or_generate()
# 2) EDA
adf_summary = run_eda(df, freq=24, out_dir=Config.REPORTS_DIR)
print("ADF summary:", adf_summary)

# 3) Preprocess
series = df['value'].values
scaled, scaler = scale_series(series)
X, y = create_sequences(scaled, seq_len=Config.SEQ_LEN, pred_len=Config.PRED_LEN)
# reshape for model: (N, seq_len, 1)
X = torch.tensor(X).float().unsqueeze(-1)
y = torch.tensor(y).float().squeeze(-1)  # (N, pred_len) -> (N,)

# 4) Train/val/test split
n = len(X)
test_n = int(n * Config.TEST_RATIO)
val_n = int(n * Config.VAL_RATIO)
train_n = n - test_n - val_n
X_train, X_val, X_test = X[:train_n], X[train_n:train_n+val_n], X[train_n+val_n:]
y_train, y_val, y_test = y[:train_n], y[train_n:train_n+val_n], y[train_n+val_n:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 5) Hyperparameter grid search (small)
param_grid = {
    'model_dim': [32, 64],
    'num_heads': [2, 4],
    'num_layers': [1, 2]
}
def build_fn(cfg):
    return TimeSeriesTransformer(seq_len=Config.SEQ_LEN, feature_dim=1, model_dim=cfg['model_dim'],
                                 num_heads=cfg['num_heads'], num_layers=cfg['num_layers'], ff_dim=Config.HIDDEN_DIM)
best_cfg = grid_search(param_grid, build_fn, X_train, y_train, X_val, y_val, Config, device=device)
print("Best cfg from grid:", best_cfg)

# 6) Build final models
transformer = TimeSeriesTransformer(seq_len=Config.SEQ_LEN, feature_dim=1, model_dim=best_cfg['model_dim'],
                                   num_heads=best_cfg['num_heads'], num_layers=best_cfg['num_layers'], ff_dim=Config.HIDDEN_DIM)
lstm = BaselineLSTM(input_dim=1, hidden_dim=Config.HIDDEN_DIM, out_dim=Config.PRED_LEN)

# 7) Train (longer)
train_loop(transformer, X_train, y_train, X_val, y_val, config=Config, device=device, save_path=os.path.join(Config.SAVE_DIR,"transformer.pth"))
train_loop(lstm, X_train, y_train, X_val, y_val, config=Config, device=device, save_path=os.path.join(Config.SAVE_DIR,"lstm.pth"))

# 8) Evaluation
transformer.eval(); lstm.eval()
with torch.no_grad():
    tpred = transformer(X_test.to(device)).squeeze(1).cpu().numpy()
    lpred = lstm(X_test.to(device)).squeeze(1).cpu().numpy()
    y_true = y_test.cpu().numpy()

# inverse scale
tpred_inv = scaler.inverse_transform(tpred.reshape(-1,1)).squeeze(-1)
lpred_inv = scaler.inverse_transform(lpred.reshape(-1,1)).squeeze(-1)
y_inv = scaler.inverse_transform(y_true.reshape(-1,1)).squeeze(-1)

print("Transformer eval:", basic_eval(y_inv, tpred_inv))
print("LSTM eval:", basic_eval(y_inv, lpred_inv))

# 9) Attention analysis
save_attention_maps(transformer, X_test[:8], out_dir=os.path.join(Config.REPORTS_DIR,"attention"), device=device)

# 10) Generate textual report
report = []
report.append("Dataset: synthetic generated or loaded from data/dataset.csv")
report.append(f"Samples: {len(df)}; Seq len: {Config.SEQ_LEN}")
report.append("Preprocessing: MinMax scaling applied to full series before sequencing.")
report.append("Hyperparameter search grid: " + str(param_grid))
report.append("Best cfg: " + str(best_cfg))
report.append("Metrics (Transformer): " + str(basic_eval(y_inv, tpred_inv)))
report.append("Metrics (LSTM): " + str(basic_eval(y_inv, lpred_inv)))

with open(os.path.join(Config.REPORTS_DIR,"report.txt"), "w") as f:
    f.write("\n".join(report))

print("Report written to", os.path.join(Config.REPORTS_DIR,"report.txt"))
