
import torch
import numpy as np

def evaluate_model(model,X,y,scaler):
    model.eval()
    preds=model(X).detach().numpy()
    return preds
