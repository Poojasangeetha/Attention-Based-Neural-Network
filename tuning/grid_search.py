
import itertools
import torch
from training.train import train_loop

def grid_search(param_grid, build_model_fn, X_train, y_train, X_val, y_val, config, device='cpu'):
    best_cfg = None
    best_loss = float('inf')
    for combo in itertools.product(*param_grid.values()):
        cfg = dict(zip(param_grid.keys(), combo))
        model = build_model_fn(cfg)
        # quick train for few epochs to evaluate
        tmp_cfg = type('C', (), dict(**config.__dict__))
        tmp_cfg.EPOCHS = max(5, int(config.EPOCHS/5))
        train_loop(model, X_train, y_train, X_val=X_val, y_val=y_val, config=tmp_cfg, device=device)
        # evaluate val loss
        model.eval()
        with torch.no_grad():
            vpred = model(X_val.to(device))
            vpred_s = vpred.squeeze(1) if vpred.dim()==3 else vpred
            loss = torch.nn.functional.mse_loss(vpred_s, y_val.to(device)).item()
        print("Grid combo", cfg, "val_loss", loss)
        if loss < best_loss:
            best_loss = loss
            best_cfg = cfg
    return best_cfg
