import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import joblib 
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features import calculate_features, prepare_target
from src.model import LSTMModel
from src.dataloader import get_train_loader

SEED = 40
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_model(data_path, lookback=60, epochs=30, lr=0.0001):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using device: {device}")
    
    # 1. Load & Process Data
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Calculate Features (Regime Modeling)
    df_feat = calculate_features(df)
    df_feat = prepare_target(df_feat, horizon=3) # 9-Min Horizon
    
    # 10 Feature Columns
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume', 
        'diff_1', 'diff_5', 'rsi', 'bb_width', 'atr'
    ]
    
    # 2. Get DataLoaders (X_scaler for input features, Y_scaler for delta target)
    # Binary Classification: No need for y_scaler
    X_data = df_feat[feature_cols].values
    y_data = df_feat['target'].values
    
    from sklearn.preprocessing import StandardScaler
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X_data)
    
    from src.dataloader import create_sequences
    from torch.utils.data import DataLoader, TensorDataset
    X_torch, y_torch = create_sequences(X_scaled, y_data, lookback)
    train_loader = DataLoader(TensorDataset(X_torch, y_torch), batch_size=64, shuffle=True)
    
    # Save the fitted scaler
    os.makedirs('data', exist_ok=True)
    joblib.dump(x_scaler, 'data/x_scaler.pkl')
    print("✅ Status: Input Scaler saved to data/")
    
    # 3. Model Init & Criterion
    model = LSTMModel(input_dim=len(feature_cols), hidden_dim=128).to(device)
    # Binary switch: BCELoss on probability
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 4. Training Loop
    best_loss = float('inf')
    patience = 5
    counter = 0
    
    print("🚀 Starting training")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} -> Binary BCE Loss: {avg_train_loss:.6f}")
        
        if avg_train_loss < (best_loss - 1e-4):
            best_loss = avg_train_loss
            torch.save(model.state_dict(), 'data/best_stock_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"🛑 Training stopped at epoch {epoch+1} (Stasis reached)")
                break
                
    print("✅ Training complete. Binary model saved.")

if __name__ == "__main__":
    train_model('data/train_data.csv', lookback=60, epochs=30)
