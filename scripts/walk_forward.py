import os
import sys
import torch
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Root-level imports
from src.features import calculate_features, prepare_target
from src.model import LSTMModel
from src.dataloader import create_sequences

def train_wf(X_train, y_train, input_dim, device):
    model = LSTMModel(input_dim=input_dim, hidden_dim=32, num_layers=1).to(device).to(torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=False)
    
    model.train()
    for epoch in range(10): # Quick training per window
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
    return model

def run_walk_forward(data_path, window_size=5000, step_size=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🚀 Starting Walk-Forward Validation on {device}")
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Data not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    df_feat = calculate_features(df)
    df_feat = prepare_target(df_feat, horizon=3, threshold_pct=0.001)
    
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume', 
        'diff_1', 'diff_5', 'rsi', 'rsi_fast', 'bb_width', 'atr',
        'ema_ratio', 'price_ema_dist', 'return_1', 'return_5', 'return_10',
        'candle_body', 'hl_range', 'vol_ratio', 'rolling_std',
        'raw_body', 'raw_range'
    ]
    
    results = []
    total_len = len(df_feat)
    lookback = 60
    
    # Walk-forward loop
    for i in range(window_size, total_len - step_size, step_size):
        train_df = df_feat.iloc[i-window_size : i]
        test_df = df_feat.iloc[i : i+step_size]
        
        print(f"Training window: {train_df['datetime'].iloc[0].date()} -> {train_df['datetime'].iloc[-1].date()}")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_df[feature_cols])
        X_test_scaled = scaler.transform(test_df[feature_cols])
        
        X_tr_seq, y_tr_seq = create_sequences(X_train_scaled, train_df['target'].values, lookback)
        X_ts_seq, y_ts_seq = create_sequences(X_test_scaled, test_df['target'].values, lookback)
        
        # Train
        model = train_wf(X_tr_seq, y_tr_seq, len(feature_cols), device)
        
        # Test
        model.eval()
        with torch.no_grad():
            logits = model(X_ts_seq.to(device))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            y_true = y_ts_seq.numpy()
            
            # Simple win rate (at 0.5 threshold) for the window
            y_pred = (probs > 0.5).astype(int)
            correct = (y_pred == y_true).sum()
            acc = correct / len(y_true)
            results.append(acc)
            print(f"Window Accuracy: {acc:.4f}")

    print("\n" + "="*55)
    if len(results) > 0:
        print(f"🏁 Walk-Forward Complete. Avg Window Accuracy: {np.mean(results):.4f}")
    else:
        print("🏁 Walk-Forward Complete. No windows were processed.")
    print("="*55)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Path to training CSV for Walk-Forward")
    parser.add_argument('--window', type=int, default=5000)
    parser.add_argument('--step', type=int, default=1000)
    args = parser.parse_args()
    
    run_walk_forward(data_path=args.data, window_size=args.window, step_size=args.step)
