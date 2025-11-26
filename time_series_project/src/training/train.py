
import torch

def train_model(model,X,y,Config):
    opt=torch.optim.Adam(model.parameters(), lr=Config.LR)
    loss_fn=torch.nn.MSELoss()

    for epoch in range(Config.EPOCHS):
        model.train()
        opt.zero_grad()
        pred=model(X)
        loss=loss_fn(pred,y)
        loss.backward()
        opt.step()
        print("Epoch",epoch,"Loss:",loss.item())
