# Stock Direction Prediction using LSTM

A regime-aware LSTM model for predicting short-term (9-minute) price direction of Reliance Industries using 3-minute candlestick data.

## Overview

This project implements a **binary classification** approach to intraday stock prediction. Instead of forecasting exact prices (regression), the model predicts whether the stock price will move **UP or DOWN** over the next 9 minutes.

### Key Results (Out-of-Sample: Jan – Mar 2026)
- **Accuracy**: 52% (above 50% random baseline)
- **DOWN Recall**: 76% (strong crash detection)
- **UP Recall**: 27% (selective, high-conviction entries)
- **Net Profit**: Positive over 505 serialized trades

## Dataset

The dataset consists of 3-minute OHLCV candlestick data for **Reliance Industries (NSE: RELIANCE)**, sourced via the AngelOne SmartAPI.

📥 **Download**: Available under [Releases](../../releases) — download `train_data.csv` and `test_data.csv`.

Place the files in the `data/` directory:
```
data/
├── train_data.csv    # Jan 2024 – Dec 2025
└── test_data.csv     # Jan 2026 – Mar 2026
```

## Project Structure

```
stock-prediction/
├── data/
│   ├── train_data.csv         # Training data (2024–2025)
│   ├── test_data.csv          # Test data (Jan–Mar 2026)
│   ├── best_stock_model.pth   # Trained model weights [not tracked]
│   └── x_scaler.pkl           # Fitted StandardScaler [not tracked]
├── src/
│   ├── features.py            # Feature engineering (RSI, BB Width, ATR)
│   ├── dataloader.py          # Sequence creation & data loading
│   └── model.py               # LSTM architecture definition
├── scripts/
│   ├── train.py               # Model training script
│   ├── test.py                # Evaluation & result generation
│   └── visualize.py           # Paper figure generation
├── plots/                     # Test result visualizations
├── paper_images/              # Research paper figures
└── README.md
```

## Features (10-Dimensional Input)

| Category | Feature | Description |
|----------|---------|-------------|
| Price | Open, High, Low, Close | Raw OHLC values |
| Volume | Volume | Trading activity per candle |
| Velocity | Diff_1, Diff_5 | 3-min and 15-min price change |
| Momentum | RSI (14) | Overbought/oversold detection |
| Tightness | Bollinger Width | Squeeze / breakout detection |
| Volatility | ATR (14) | Market noise measurement |

## Model Architecture

- **Type**: 2-layer stacked LSTM (128 hidden units)
- **Dropout**: 0.3 between LSTM layers
- **Head**: BatchNorm → Linear(128→64) → LeakyReLU → Linear(64→1) → Sigmoid
- **Loss**: Binary Cross-Entropy
- **Lookback**: 60 bars (3 hours of history)

## Usage

### Train
```bash
python3 scripts/train.py
```

### Evaluate
```bash
python3 scripts/test.py
```

### Generate Graphs
```bash
python3 scripts/visualize.py
```

## Requirements

```
torch
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```
