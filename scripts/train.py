import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import joblib 
import sys
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Root-level imports
from src.features import calculate_features, prepare_target
from src.model import LSTMModel
from src.dataloader import create_sequences

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train_model(data_path, output_dir='data', lookback=60, epochs=50, lr=0.0001, batch_size=64):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using device: {device}")
    
    # 1. Load & Process Data
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Calculate Features
    print("🛠 Engineering features...")
    df_feat = calculate_features(df)
    df_feat = prepare_target(df_feat, horizon=3, threshold_pct=0.0005)
    
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume', 
        'diff_1', 'diff_5', 'rsi', 'rsi_fast', 'bb_width', 'atr',
        'ema_ratio', 'price_ema_dist', 'return_1', 'return_5', 'return_10',
        'candle_body', 'hl_range', 'vol_ratio', 'rolling_std',
        'raw_body', 'raw_range'
    ]
    
    # 2. Time-based Split (80% Train, 20% Val)
    split_idx = int(len(df_feat) * 0.8)
    train_df = df_feat.iloc[:split_idx].copy()
    val_df = df_feat.iloc[split_idx:].copy()
    
    # Scale Features (Fit ONLY on training data)
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(train_df[feature_cols])
    X_val_scaled = x_scaler.transform(val_df[feature_cols])
    
    y_train = train_df['target'].values
    y_val = val_df['target'].values
    
    # 3. Create Sequences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, lookback)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, lookback)
    
    # Dataloaders (NO SHUFFLE for time-series)
    train_loader = DataLoader(TensorDataset(X_train_seq, y_train_seq), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val_seq, y_val_seq), batch_size=batch_size, shuffle=False)
    
    # Save Scaler
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, 'x_scaler.pkl')
    joblib.dump(x_scaler, scaler_path)
    print(f"✅ Scaler saved to {scaler_path}. Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    # 4. Class Weights (Balanced focus)
    pos_weight = torch.tensor([1.0], dtype=torch.float32).to(device)
    print(f"⚖️ Using Balanced Pos weight: {pos_weight.item():.2f}")
    
    # 5. Model & Optimizer (Leaner model)
    model = LSTMModel(input_dim=len(feature_cols), hidden_dim=32, num_layers=1).to(device).to(torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 6. Training Loop
    best_val_loss = float('inf')
    early_stop_patience = 8
    counter = 0
    model_path = os.path.join(output_dir, 'best_stock_model.pth')
    
    print("🚀 Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device).to(torch.float32)
            y_batch = y_batch.to(device).to(torch.float32).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device).to(torch.float32)
                y_batch = y_batch.to(device).to(torch.float32).unsqueeze(1)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        
        if avg_val_loss < (best_val_loss - 1e-4):
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            counter = 0
            print(f"🌟 New best model saved to {model_path}")
        else:
            counter += 1
            if counter >= early_stop_patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
                
    print("✅ Training complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Path to training CSV")
    parser.add_argument('--output_dir', type=str, default='data', help="Directory to save scaler and model")
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    train_model(args.data, output_dir=args.output_dir, epochs=args.epochs)
