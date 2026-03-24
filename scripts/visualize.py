"""
visualize.py - Generate all supplementary figures for the research paper.

Produces (in paper_images/):
  1. full_price_path.png       - Training data price overview
  2. data_sample_table.png     - Raw dataset sample table
  3. test_price_path.png       - Test period price path
  4. indicator_rsi.png         - RSI indicator audit
  5. indicator_bb_width.png    - Bollinger Width audit
  6. indicator_atr.png         - ATR indicator audit
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features import calculate_features


def to_ist(df):
    """Convert timezone-aware datetime to naive IST for correct plotting."""
    df['datetime'] = pd.to_datetime(df['datetime'])
    if df['datetime'].dt.tz is not None:
        df['datetime'] = df['datetime'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    return df


def generate_all(train_path='data/train_data.csv', test_path='data/test_data.csv'):
    os.makedirs('paper_images', exist_ok=True)
    time_fmt = mdates.DateFormatter('%H:%M')

    # ── Load Data ───────────────────────────────────────────
    df_train = to_ist(pd.read_csv(train_path)).sort_values('datetime').reset_index(drop=True)
    df_test = to_ist(pd.read_csv(test_path)).sort_values('datetime').reset_index(drop=True)

    # ── 1. Full Training Price Path ─────────────────────────
    plt.figure(figsize=(12, 6))
    plt.plot(df_train['datetime'], df_train['close'], color='black', linewidth=1)
    plt.title("Historical Price Path: Reliance Industries (2024–2025)", fontsize=14, fontweight='bold')
    plt.xlabel("Timeline", fontsize=12)
    plt.ylabel("Price (₹)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('paper_images/full_price_path.png', dpi=300)
    plt.close()
    print("✅ full_price_path.png")

    # ── 2. Data Sample Table ────────────────────────────────
    sample = df_train.head(10).copy()
    sample['datetime'] = sample['datetime'].dt.strftime('%Y-%m-%d %H:%M')
    table_data = sample[['datetime', 'open', 'high', 'low', 'close', 'volume']].round(2)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    tbl = ax.table(cellText=table_data.values,
                   colLabels=table_data.columns,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#333333')
        else:
            cell.set_facecolor('#f9f9f9')
    plt.title("Sample of the 3-Minute Intraday Dataset", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('paper_images/data_sample_table.png', dpi=300)
    plt.close()
    print("✅ data_sample_table.png")

    # ── 3. Test Period Price Path ───────────────────────────
    plt.figure(figsize=(14, 6))
    plt.plot(df_test['datetime'], df_test['close'], color='black', linewidth=1)
    plt.title("Test Period Price Path: Reliance Industries (Jan – Mar 2026)", fontsize=14, fontweight='bold')
    plt.xlabel("Timeline (IST)", fontsize=12)
    plt.ylabel("Price (₹)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('paper_images/test_price_path.png', dpi=300)
    plt.close()
    print("✅ test_price_path.png")

    # ── Indicator Visuals (last 80 candles of training) ─────
    df_feat = calculate_features(df_train)
    viz = df_feat.tail(80).copy()
    viz_date = viz['datetime'].iloc[0].strftime('%Y-%m-%d')

    # ── 4. RSI ──────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(viz['datetime'], viz['close'], color='black', linewidth=1.5)
    ax1.set_title(f"Price vs. RSI | {viz_date} (IST)", fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel("Price (₹)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(viz['datetime'], viz['rsi'], color='purple', linewidth=1.5, label='RSI (14)')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.6, label='Overbought (70)')
    ax2.axhline(30, color='green', linestyle='--', alpha=0.6, label='Oversold (30)')
    ax2.fill_between(viz['datetime'], 70, 100, color='red', alpha=0.08)
    ax2.fill_between(viz['datetime'], 0, 30, color='green', alpha=0.08)
    ax2.set_ylabel("RSI", fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', ncol=3)
    plt.gca().xaxis.set_major_formatter(time_fmt)
    plt.xlabel(f"Trading Hours on {viz_date} (IST)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('paper_images/indicator_rsi.png', dpi=300)
    plt.close()
    print("✅ indicator_rsi.png")

    # ── 5. Bollinger Width ──────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(viz['datetime'], viz['close'], color='black', linewidth=1.5)
    ax1.set_title(f"Price vs. Bollinger Band Width | {viz_date} (IST)", fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel("Price (₹)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(viz['datetime'], viz['bb_width'], color='blue', linewidth=1.5, label='Bollinger Width')
    ax2.fill_between(viz['datetime'], 0, viz['bb_width'], color='blue', alpha=0.1)
    ax2.set_ylabel("Width", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    plt.gca().xaxis.set_major_formatter(time_fmt)
    plt.xlabel(f"Trading Hours on {viz_date} (IST)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('paper_images/indicator_bb_width.png', dpi=300)
    plt.close()
    print("✅ indicator_bb_width.png")

    # ── 6. ATR ──────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(viz['datetime'], viz['close'], color='black', linewidth=1.5)
    ax1.set_title(f"Price vs. ATR | {viz_date} (IST)", fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel("Price (₹)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(viz['datetime'], viz['atr'], color='orange', linewidth=1.5, label='ATR (14)')
    ax2.fill_between(viz['datetime'], 0, viz['atr'], color='orange', alpha=0.1)
    ax2.set_ylabel("ATR (Points)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    plt.gca().xaxis.set_major_formatter(time_fmt)
    plt.xlabel(f"Trading Hours on {viz_date} (IST)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('paper_images/indicator_atr.png', dpi=300)
    plt.close()
    print("✅ indicator_atr.png")

    print("\n✅ All paper images saved to paper_images/")


if __name__ == "__main__":
    generate_all()
