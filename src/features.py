import pandas as pd
import numpy as np

def calculate_features(df):
    """
    10 Regime Features (The 'Eye' of the model).
    """
    df = df.copy()
    
    # 1. Base Multi-OHLCV diffs (Velocity)
    df['diff_1'] = df['close'].diff(1)
    df['diff_5'] = df['close'].diff(5)
    
    # 2. RSI (14-period window)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. Bollinger Bands (20-period)
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['std20'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['ma20'] + (2 * df['std20'])
    df['bb_lower'] = df['ma20'] - (2 * df['std20'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['ma20'] + 1e-9)
    
    # 4. ATR (14-period Average True Range)
    df['atr_h_l'] = df['high'] - df['low']
    df['atr_h_pc'] = abs(df['high'] - df['close'].shift(1))
    df['atr_l_pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['atr_h_l', 'atr_h_pc', 'atr_h_pc']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume', 
        'diff_1', 'diff_5', 'rsi', 'bb_width', 'atr'
    ]
    
    return df[['datetime'] + feature_cols].copy().dropna()

def prepare_target(df, horizon=3):
    """
    9-Minute Directional Logic:
    - 1 if Price in 9 minutes > Price now.
    - 0 if Price in 9 minutes < Price now.
    - We ignore 'flats' to force the model to learn the alpha.
    """
    df['future_close'] = df['close'].shift(-horizon)
    diff = df['future_close'] - df['close']
    
    # Focus only on UP or DOWN moves, discarding price stagnation
    df.loc[diff > 0, 'target'] = 1
    df.loc[diff < 0, 'target'] = 0
    
    return df.dropna(subset=['target'])
