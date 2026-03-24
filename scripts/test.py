"""
test.py - Evaluate the trained LSTM model on unseen test data.

Produces:
  1. Classification Report (console + plots/classification_report.png)
  2. Confusion Matrix (plots/confusion_matrix.png)
  3. Probability Histogram (plots/probability_histogram.png)
  4. Equity Curve from serialized trading simulation (plots/equity_curve.png)
  5. Directional Overlay on last 100 candles (plots/directional_overlay.png)
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features import calculate_features, prepare_target
from src.model import LSTMModel
from src.dataloader import create_sequences


def run_test(data_path='data/test_data.csv', model_path='data/best_stock_model.pth',
             x_scaler_path='data/x_scaler.pkl', lookback=60):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using device: {device}")

    # ── 1. Load & Prepare Data ──────────────────────────────
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    df_feat = calculate_features(df)
    df_feat = prepare_target(df_feat, horizon=3)  # 9-minute horizon

    test_dates = df_feat.iloc[lookback:]['datetime'].values
    actual_prices = df_feat.iloc[lookback:]['close'].values
    next_price_3 = df_feat['close'].shift(-3).iloc[lookback:].values

    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'diff_1', 'diff_5', 'rsi', 'bb_width', 'atr'
    ]

    # ── 2. Load Scaler & Model ──────────────────────────────
    x_scaler = joblib.load(x_scaler_path)
    model = LSTMModel(input_dim=len(feature_cols), hidden_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ── 3. Inference ────────────────────────────────────────
    X_scaled = x_scaler.transform(df_feat[feature_cols].values)
    y_true_full = df_feat['target'].values
    X_torch, y_torch = create_sequences(X_scaled, y_true_full, lookback)
    test_loader = DataLoader(TensorDataset(X_torch, y_torch), batch_size=64, shuffle=False)

    preds_prob = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            out = model(X_batch)
            preds_prob.extend(out.cpu().numpy())

    y_prob = np.array(preds_prob).flatten()
    y_true = y_torch.numpy()
    y_pred = (y_prob > 0.50).astype(int)

    # ── 4. Classification Report ────────────────────────────
    print("\n" + "=" * 55)
    print("  CLASSIFICATION REPORT (Out-of-Sample Test)")
    print("=" * 55)
    report_text = classification_report(y_true, y_pred, target_names=['DOWN (0)', 'UP (1)'])
    print(report_text)

    # ── 5. Serialized Trading Simulation ────────────────────
    trades = []
    cooldown = 0
    for i in range(len(y_pred)):
        if cooldown > 0:
            cooldown -= 1
            continue
        if y_pred[i] == 1:
            entry_p = actual_prices[i]
            exit_p = next_price_3[i]
            if np.isnan(exit_p):
                continue
            trades.append({
                'idx': i, 'entry': entry_p, 'exit': exit_p,
                'profit': exit_p - entry_p - 0.05,
                'win': 1 if exit_p > entry_p else 0
            })
            cooldown = 3

    wins = sum(t['win'] for t in trades)
    total = len(trades)
    win_rate = (wins / total * 100) if total > 0 else 0
    cum_profit = np.cumsum([t['profit'] for t in trades])
    net = cum_profit[-1] if len(cum_profit) > 0 else 0

    print("=" * 55)
    print("  STRATEGY AUDIT (Serialized, 1-Trade-at-a-Time)")
    print("=" * 55)
    print(f"  Total Trades : {total}")
    print(f"  Winning       : {wins}")
    print(f"  Win Rate      : {win_rate:.2f}%")
    print(f"  Net Profit    : ₹{net:.2f}")
    print("=" * 55)

    # ── 6. Generate Plots ───────────────────────────────────
    os.makedirs('plots', exist_ok=True)

    # 6a. Classification Report Table
    report_dict = classification_report(y_true, y_pred, target_names=['DOWN (0)', 'UP (1)'], output_dict=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    table_data = [
        ['DOWN (0)', f"{report_dict['DOWN (0)']['precision']:.2f}",
         f"{report_dict['DOWN (0)']['recall']:.2f}", f"{report_dict['DOWN (0)']['f1-score']:.2f}",
         f"{int(report_dict['DOWN (0)']['support'])}"],
        ['UP (1)', f"{report_dict['UP (1)']['precision']:.2f}",
         f"{report_dict['UP (1)']['recall']:.2f}", f"{report_dict['UP (1)']['f1-score']:.2f}",
         f"{int(report_dict['UP (1)']['support'])}"],
        ['Accuracy', '', '', f"{report_dict['accuracy']:.2f}",
         f"{int(report_dict['macro avg']['support'])}"],
    ]
    tbl = ax.table(cellText=table_data,
                   colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.3, 2.0)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        else:
            cell.set_facecolor('#f8f9fa' if row % 2 == 0 else '#ffffff')
    plt.title("Classification Report (Jan – Mar 2026)", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('plots/classification_report.png', dpi=300)
    plt.close()

    # 6b. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted DOWN', 'Predicted UP'],
                yticklabels=['Actual DOWN', 'Actual UP'],
                annot_kws={"size": 18}, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=300)
    plt.close()

    # 6c. Probability Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_prob[y_true == 0], bins=50, alpha=0.6, color='red', label='Actual DOWN', density=True)
    ax.hist(y_prob[y_true == 1], bins=50, alpha=0.6, color='green', label='Actual UP', density=True)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.50)')
    ax.set_title("Probability Distribution by Actual Outcome", fontsize=14, fontweight='bold')
    ax.set_xlabel("Predicted Probability (UP)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('plots/probability_histogram.png', dpi=300)
    plt.close()

    # 6d. Equity Curve
    trade_dates = [test_dates[t['idx']] for t in trades]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(trade_dates, cum_profit, color='#2ecc71', linewidth=2, label='Cumulative Profit')
    ax.fill_between(trade_dates, 0, cum_profit,
                    where=[p >= 0 for p in cum_profit], color='green', alpha=0.1)
    ax.fill_between(trade_dates, 0, cum_profit,
                    where=[p < 0 for p in cum_profit], color='red', alpha=0.1)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title("Equity Curve: Serialized Trading (Jan – Mar 2026)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Timeline", fontsize=12)
    ax.set_ylabel("Cumulative Net Profit (₹)", fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.legend()
    stats = f"Trades: {total}  |  Win Rate: {win_rate:.1f}%  |  Net: ₹{net:.2f}"
    ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig('plots/equity_curve.png', dpi=300)
    plt.close()

    # 6e. Directional Overlay (Last 100 candles)
    zoom = 100
    dates_z = test_dates[-zoom:]
    prices_z = actual_prices[-zoom:]
    preds_z = y_pred[-zoom:]
    prob_z = y_prob[-zoom:]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(dates_z, prices_z, color='black', alpha=0.3, linewidth=2)
    for i in range(zoom):
        c = 'green' if preds_z[i] == 1 else 'red'
        s = 20 + abs(prob_z[i] - 0.5) * 200
        ax.scatter(dates_z[i], prices_z[i], color=c, s=s, alpha=0.7)
    from matplotlib.lines import Line2D
    legend_els = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),
                  Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)]
    ax.legend(legend_els, ['Predicted UP', 'Predicted DOWN'])
    ax.set_title("Directional Overlay (Last 100 Candles)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Price (₹)", fontsize=12)
    ax.set_xlabel("Timeline", fontsize=12)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('plots/directional_overlay.png', dpi=300)
    plt.close()

    print("\n✅ All plots saved to plots/")


if __name__ == "__main__":
    run_test()
