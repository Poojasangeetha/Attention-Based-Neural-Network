
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def save_attention_maps(model, X_samples, out_dir="reports/attention", device='cpu'):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        _ = model(X_samples.to(device))
        mats = model.get_all_attentions()
    for layer_idx, mat in enumerate(mats):
        if mat is None:
            continue
        arr = mat.numpy()
        # handle (batch, seq, seq) or (batch, heads, seq, seq)
        for i in range(arr.shape[0]):
            mat2 = arr[i]
            if mat2.ndim == 3:
                # average heads
                mat2 = mat2.mean(axis=0)
            plt.figure(figsize=(6,5))
            sns.heatmap(mat2, vmin=0, vmax=1)
            plt.title(f"Layer {layer_idx+1} Sample {i}")
            plt.savefig(f"{out_dir}/layer{layer_idx+1}_sample{i}.png")
            plt.close()
    print("Saved attention maps to", out_dir)
