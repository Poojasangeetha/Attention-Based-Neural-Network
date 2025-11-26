
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

def train_loop(model, X_train, y_train, X_val=None, y_val=None, config=None, device='cpu', save_path=None):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_fn = torch.nn.MSELoss()
    best = float('inf')
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    for epoch in range(config.EPOCHS):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            preds_s = preds.squeeze(1) if preds.dim()==3 else preds
            yb_s = yb.squeeze(1) if yb.dim()==3 else yb
            loss = loss_fn(preds_s, yb_s)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        epoch_loss = running / len(dataset)
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                vpred = model(X_val.to(device))
                vpred_s = vpred.squeeze(1) if vpred.dim()==3 else vpred
                vloss = loss_fn(vpred_s, y_val.to(device))
                val_loss = vloss.item()
            print(f"Epoch {epoch+1}/{config.EPOCHS} train_loss={epoch_loss:.6f} val_loss={val_loss:.6f}")
            if val_loss < best and save_path:
                best = val_loss
                torch.save(model.state_dict(), save_path)
        else:
            print(f"Epoch {epoch+1}/{config.EPOCHS} train_loss={epoch_loss:.6f}")
    return model
