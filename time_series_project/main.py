
import pandas as pd
import torch
from config import Config
from src.eda import run_eda
from src.data_pipeline import scale_data, create_dataset, train_test_split
from src.models.baseline_lstm import BaselineLSTM
from src.models.transformer_model import TransformerModel
from src.training.train import train_model
from src.training.evaluation import evaluate_model

def main():
    df=pd.DataFrame({'value':range(600)})
    run_eda(df)

    scaled,scaler=scale_data(df.values)
    X,y=create_dataset(scaled, Config.SEQ_LEN)
    Xtr,Xte,ytr,yte=train_test_split(X,y)

    Xtr=torch.tensor(Xtr,dtype=torch.float32)
    ytr=torch.tensor(ytr,dtype=torch.float32)
    Xte=torch.tensor(Xte,dtype=torch.float32)
    yte=torch.tensor(yte,dtype=torch.float32)

    lstm=BaselineLSTM(1,32)
    train_model(lstm,Xtr,ytr,Config)
    evaluate_model(lstm,Xte,yte,scaler)

    tr=TransformerModel(Config.SEQ_LEN,1,64,4,2)
    train_model(tr,Xtr,ytr,Config)
    evaluate_model(tr,Xte,yte,scaler)

if __name__=="__main__":
    main()
