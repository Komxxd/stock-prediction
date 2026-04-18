import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Root-level imports
from src.features import calculate_features, prepare_target
from src.model import LSTMModel
from src.dataloader import create_sequences

def run_test(data_path, model_path, x_scaler_path, output_dir='plots', lookback=60, threshold=0.5):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using device: {device}")

    # 1. Load & Prepare Data
    if not os.path.exists(data_path):
        print(f"❌ Error: Test data not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print("🛠 Engineering features for test data...")
    df_feat = calculate_features(df)
    df_feat = prepare_target(df_feat, horizon=3, threshold_pct=0.0005)

    test_dates = df_feat.iloc[lookback:]['datetime'].values
    actual_prices = df_feat.iloc[lookback:]['close'].values
    next_price_3 = df_feat['close'].shift(-3).iloc[lookback:].values

    feature_cols = [
        'open', 'high', 'low', 'close', 'volume', 
        'diff_1', 'diff_5', 'rsi', 'rsi_fast', 'bb_width', 'atr',
        'ema_ratio', 'price_ema_dist', 'return_1', 'return_5', 'return_10',
        'candle_body', 'hl_range', 'vol_ratio', 'rolling_std',
        'raw_body', 'raw_range'
    ]

    # 2. Load Scaler & Model
    if not os.path.exists(x_scaler_path):
        print(f"❌ Error: Scaler not found at {x_scaler_path}")
        return
    x_scaler = joblib.load(x_scaler_path)
    
    model = LSTMModel(input_dim=len(feature_cols), hidden_dim=32, num_layers=1).to(device).to(torch.float32)
    if not os.path.exists(model_path):
        print(f"❌ Error: Model weights not found at {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Inference
    X_scaled = x_scaler.transform(df_feat[feature_cols])
    y_true_full = df_feat['target'].values
    X_torch, y_torch = create_sequences(X_scaled, y_true_full, lookback)
    test_loader = DataLoader(TensorDataset(X_torch, y_torch), batch_size=64, shuffle=False)

    preds_prob = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device).to(torch.float32)
            logits = model(X_batch)
            probs = torch.sigmoid(logits) # Apply Sigmoid here
            preds_prob.extend(probs.cpu().numpy())

    y_prob = np.array(preds_prob).flatten()
    y_true = y_torch.numpy()
    
    # 4. Threshold Optimization
    print("\n" + "=" * 55)
    print("  THRESHOLD OPTIMIZATION")
    print("=" * 55)
    print(f"{'Threshold':<12} | {'AUC':<6} | {'Recall(UP)':<10} | {'Recall(DN)':<10} | {'F1':<5}")
    print("-" * 55)
    
    best_f1 = 0
    best_t = 0.5
    for t in np.arange(0.35, 0.65, 0.05):
        y_p = (y_prob > t).astype(int)
        report = classification_report(y_true, y_p, target_names=['DOWN (0)', 'UP (1)'], output_dict=True, zero_division=0)
        
        f1 = report['macro avg']['f1-score']
        r_up = report.get('UP (1)', {}).get('recall', 0.0)
        r_dn = report.get('DOWN (0)', {}).get('recall', 0.0)
        auc_val = roc_auc_score(y_true, y_prob)
        print(f"{t:<12.2f} | {auc_val:<6.3f} | {r_up:<10.2f} | {r_dn:<10.2f} | {f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print("-" * 55)
    print(f"✅ Best F1 Threshold: {best_t:.2f}")

    # Use the user-requested threshold for the final report or the best found
    final_t = threshold if threshold != 0.5 else best_t
    y_pred = (y_prob > final_t).astype(int)

    # 4. Final Evaluation Metrics
    auc = roc_auc_score(y_true, y_prob)
    print("\n" + "=" * 55)
    print(f"  OOS EVALUATION (Selected Threshold: {final_t:.2f})")
    print("=" * 55)
    print(f"  AUC-ROC: {auc:.4f}")
    report_text = classification_report(y_true, y_pred, target_names=['DOWN (0)', 'UP (1)'])
    print(report_text)

    # 5. Trading Simulation (Strict No-Trade Zone Strategy)
    trades = []
    # Strict no-trade zone: only trade if prob > 0.65 (LONG) or < 0.35 (SHORT)
    COST_PER_TRADE = 0.0
    
    for i in range(len(y_prob)):
        if y_prob[i] > 0.65:
            entry_p = actual_prices[i]
            exit_p = next_price_3[i]
            if np.isnan(exit_p): continue
            trades.append({
                'idx': i, 'type': 'LONG', 'entry': entry_p, 'exit': exit_p,
                'profit': exit_p - entry_p,
                'win': 1 if exit_p > entry_p else 0
            })
        elif y_prob[i] < 0.35:
            entry_p = actual_prices[i]
            exit_p = next_price_3[i]
            if np.isnan(exit_p): continue
            trades.append({
                'idx': i, 'type': 'SHORT', 'entry': entry_p, 'exit': exit_p,
                'profit': entry_p - exit_p,
                'win': 1 if exit_p < entry_p else 0
            })

    wins = sum(t['win'] for t in trades)
    total = len(trades)
    win_rate = (wins / total * 100) if total > 0 else 0
    profits = [t['profit'] for t in trades]
    cum_profit = np.cumsum(profits)
    net = cum_profit[-1] if len(cum_profit) > 0 else 0
    
    sharpe = (np.mean(profits) / (np.std(profits) + 1e-9)) * np.sqrt(252 * 50) if total > 5 else 0

    print("=" * 55)
    print("  QUANT BACKTEST STATS (No-Trade Zone enabled)")
    print("=" * 55)
    print(f"  Total Trades : {total}")
    print(f"  Win Rate      : {win_rate:.2f}%")
    print(f"  Net Profit    : ₹{net:.2f}")
    print(f"  Sharpe Ratio  : {sharpe:.2f}")
    print("=" * 55)

    # 6. Generate Plots
    os.makedirs(output_dir, exist_ok=True)
    
    # 6a. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
    plt.title(f"Confusion Matrix (Threshold: {final_t:.2f})")
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # 6b. Prob Hist
    plt.figure(figsize=(10, 6))
    plt.hist(y_prob[y_true == 0], bins=50, alpha=0.5, label='Actual DOWN', color='red')
    plt.hist(y_prob[y_true == 1], bins=50, alpha=0.5, label='Actual UP', color='green')
    plt.axvline(final_t, color='black', linestyle='--')
    plt.legend()
    plt.title("Probability Distribution")
    plt.savefig(os.path.join(output_dir, 'probability_histogram.png'))
    plt.close()

    # 6c. Equity Curve
    if len(cum_profit) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(cum_profit, label='Strategy Equity')
        plt.title(f"Cumulative Profit (No-Trade Zone >0.6 / <0.4)")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'equity_curve.png'))
        plt.close()

    print("\n✅ Evaluation complete. Plots saved.")
    
    return {
        'auc': auc,
        'report': report_text,
        'stats': {
            'total_trades': total,
            'win_rate': f"{win_rate:.2f}%",
            'net_profit': f"₹{net:.2f}",
            'sharpe': f"{sharpe:.2f}"
        },
        'test_size': len(y_true)
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Path to test CSV")
    parser.add_argument('--model', type=str, required=True, help="Path to .pth model")
    parser.add_argument('--scaler', type=str, required=True, help="Path to .pkl scaler")
    parser.add_argument('--output_dir', type=str, default='plots', help="Directory to save plots")
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    
    run_test(
        data_path=args.data,
        model_path=args.model,
        x_scaler_path=args.scaler,
        output_dir=args.output_dir,
        threshold=args.threshold
    )
