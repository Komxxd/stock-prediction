import pandas as pd
import numpy as np

def calculate_features(df):
    """
    Enhanced Regime Features (The 'Eye' of the model).
    - Captures Trend, Momentum, Volatility, and Volume Dynamics.
    """
    df = df.copy()
    
    # 1. Base Multi-OHLCV diffs (Velocity)
    df['diff_1'] = df['close'].diff(1)
    df['diff_5'] = df['close'].diff(5)
    
    # 2. RSI (14-period and 7-period windows)
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    df['rsi'] = calc_rsi(df['close'], 14)
    df['rsi_fast'] = calc_rsi(df['close'], 7)
    
    # 3. Bollinger Bands & Volatility
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['std20'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['ma20'] + (2 * df['std20'])
    df['bb_lower'] = df['ma20'] - (2 * df['std20'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['ma20'] + 1e-9)
    df['rolling_std'] = df['close'].pct_change().rolling(window=20).std()
    
    # 4. ATR (Fixed Bug: atr_l_pc)
    df['atr_h_l'] = df['high'] - df['low']
    df['atr_h_pc'] = abs(df['high'] - df['close'].shift(1))
    df['atr_l_pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['atr_h_l', 'atr_h_pc', 'atr_l_pc']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # 5. Trend: EMA Ratios
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_ratio'] = df['ema_10'] / (df['ema_20'] + 1e-9)
    df['price_ema_dist'] = (df['close'] - df['ema_20']) / (df['ema_20'] + 1e-9)
    
    # 6. Momentum: Returns
    df['return_1'] = df['close'].pct_change(1)
    df['return_5'] = df['close'].pct_change(5)
    df['return_10'] = df['close'].pct_change(10)
    
    # 7. Market Structure
    df['candle_body'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
    df['hl_range'] = (df['high'] - df['low']) / (df['close'] + 1e-9)
    df['raw_body'] = df['close'] - df['open']
    df['raw_range'] = df['high'] - df['low']
    
    # 8. Volume Dynamics
    df['vol_ma20'] = df['volume'].rolling(window=20).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma20'] + 1e-9)
    
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume', 
        'diff_1', 'diff_5', 'rsi', 'rsi_fast', 'bb_width', 'atr',
        'ema_ratio', 'price_ema_dist', 'return_1', 'return_5', 'return_10',
        'candle_body', 'hl_range', 'vol_ratio', 'rolling_std',
        'raw_body', 'raw_range'
    ]
    
    return df[['datetime'] + feature_cols].copy().dropna()

def prepare_target(df, horizon=3, threshold_pct=0.0005):
    """
    Directional Logic with Threshold:
    - 1 if future move > threshold (UP)
    - 0 if future move < -threshold (DOWN)
    - Discards moves within [-threshold, threshold] as noise
    """
    df = df.copy()
    future_close = df['close'].shift(-horizon)
    price_diff_pct = (future_close - df['close']) / (df['close'] + 1e-9)
    
    df['target'] = np.nan
    df.loc[price_diff_pct > threshold_pct, 'target'] = 1
    df.loc[price_diff_pct < -threshold_pct, 'target'] = 0
    
    # We drop NaNs (rows within the threshold corridor and the last 'horizon' rows)
    return df.dropna(subset=['target'])
