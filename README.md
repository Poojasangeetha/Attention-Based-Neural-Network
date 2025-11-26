# Time Series Forecasting Project

## Overview
This project builds a Transformer-based time series forecasting model with full preprocessing, EDA, baseline comparison, hyperparameter tuning, and attention-weight interpretation.

## Dataset
- Synthetic dataset with 2000 time steps
- Includes trend + seasonality + noise
- Stored in: `data/data.csv`

## Project Structure
```
time_series_project_v3/
│
├── data/
│   └── data.csv
│
├── preprocessing/
│   ├── scaling.py
│   └── windowing.py
│
├── eda/
│   └── eda.py
│
├── baselines/
│   ├── lstm_baseline.py
│   └── arima_baseline.py
│
├── models/
│   ├── transformer.py
│   ├── train.py
│   ├── evaluate.py
│   └── hyperparameter_search.py
│
├── interpretation/
│   └── attention_analysis.py
│
└── main.py
```

## Features Implemented
### ✅ 1. Full EDA
- Trend, seasonality, ACF, PACF
- Stationarity tests (ADF)
- Rolling mean/variance
- Normality check

### ✅ 2. Preprocessing
- MinMaxScaler
- Sliding window sequence generation
- Train/Test split

### ✅ 3. Baseline Models
- ARIMA  
- LSTM  

### ✅ 4. Transformer Model
- Positional Encoding  
- Multi-head attention  
- Custom forward pass  
- Optimizer + Scheduler  

### ✅ 5. Hyperparameter Tuning
- Learning rate search  
- Batch size search  
- Hidden dimension sweep  

### ✅ 6. Evaluation Metrics
- RMSE  
- MAE  
- MAPE  
- MASE  

### ✅ 7. Attention Weight Analysis
- Extract attention weights from multi-head attention
- Visualize attention over input time steps

## How to Run

### Install Requirements
```bash
pip install -r requirements.txt
```

### Run EDA
```bash
python eda/eda.py
```

### Train Baseline LSTM
```bash
python baselines/lstm_baseline.py
```

### Train Transformer
```bash
python models/train.py
```

### Evaluate Transformer
```bash
python models/evaluate.py
```

### Run Hyperparameter Search
```bash
python models/hyperparameter_search.py
```

### Attention Visualization
```bash
python interpretation/attention_analysis.py
```

## Outputs
- `plots/` contains EDA + attention heatmaps  
- `models/` contains trained model weights  
- `results.json` contains all evaluation metrics  

## Future Improvements
- Add probabilistic forecasting  
- Add Prophet baseline  
- Add multivariate support  
- Cross-validation  
