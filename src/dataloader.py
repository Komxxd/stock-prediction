import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

def create_sequences(X, y, lookback=60):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:(i + lookback)])
        ys.append(y[i + lookback])
    return torch.tensor(np.array(Xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.float32)

def get_train_loader(df, features_list, target_col='target', lookback=60, batch_size=64):
    """
    Final Training DataLoader (Scaler is fitted here)
    """
    X_data = df[features_list].values
    y_data = df[target_col].values
    
    # We use separate scalers to allow for inverse transformation on the target only
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_scaled = x_scaler.fit_transform(X_data)
    y_scaled = y_scaler.fit_transform(y_data.reshape(-1, 1)).flatten()
    
    X_torch, y_torch = create_sequences(X_scaled, y_scaled, lookback)
    dataset = TensorDataset(X_torch, y_torch)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return loader, x_scaler, y_scaler

def get_test_loader(df, features_list, x_scaler, y_scaler, target_col='target', lookback=60, batch_size=64):
    """
    Evaluation DataLoader (Uses pre-fitted scalers)
    """
    X_data = df[features_list].values
    y_data = df[target_col].values
    
    # Use pre-fitted scalers (Leakage prevention)
    X_scaled = x_scaler.transform(X_data)
    y_scaled = y_scaler.transform(y_data.reshape(-1, 1)).flatten()
    
    X_torch, y_torch = create_sequences(X_scaled, y_scaled, lookback)
    dataset = TensorDataset(X_torch, y_torch)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return loader
