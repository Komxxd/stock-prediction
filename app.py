"""
Stock Prediction Terminal — Desktop GUI
========================================
A PySimpleGUI-based analysis terminal for managing multi-stock LSTM
prediction workflows: data viewing, fetching, training, and testing.
"""
import PySimpleGUI as sg
import pandas as pd
import os
import sys
import threading
import queue
import json
import time
import io
from datetime import datetime, timedelta

# ─── Stock Registry ─────────────────────────────────────────────────────────
STOCKS = {
    'Abbottindia': {'dir': 'v3_abbotindia', 'token': '24'},
    'Britannia':  {'dir': 'v3_britannia',  'token': '547'},
    'Nestle':     {'dir': 'v3_nestle',     'token': '17963'},
    'Reliance':   {'dir': 'v3_reliance',   'token': '2885'},
    'Niftybees':  {'dir': 'v3_niftybees',  'token': '10576'},
    'Bankbees':   {'dir': 'v3_bankbees',   'token': '11439'},
}

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── Theme & Colors ─────────────────────────────────────────────────────────
BG_DARK      = '#0d1117'
BG_PANEL     = '#161b22'
BG_CARD      = '#1c2128'
BORDER       = '#30363d'
TEXT_PRIMARY  = '#e6edf3'
TEXT_MUTED    = '#8b949e'
TEXT_SECONDARY = TEXT_MUTED
ACCENT_BLUE  = '#58a6ff'
ACCENT_GREEN  = '#3fb950'
ACCENT_ORANGE = '#d29922'
ACCENT_RED    = '#f85149'
BTN_BG       = '#21262d'
BTN_HOVER     = '#30363d'
TAB_ACTIVE   = '#58a6ff'
TAB_INACTIVE = '#21262d'

FONT_HEADER   = ('Menlo', 14, 'bold')
FONT_BODY     = ('Menlo', 11)
FONT_SMALL    = ('Menlo', 10)
FONT_MONO     = ('Menlo', 10)
FONT_BTN      = ('Menlo', 11, 'bold')
FONT_TAB      = ('Menlo', 12, 'bold')

# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_data_path(stock_key, filename):
    return os.path.join(PROJECT_ROOT, STOCKS[stock_key]['dir'], 'data', filename)

def get_output_dir(stock_key):
    return os.path.join(PROJECT_ROOT, STOCKS[stock_key]['dir'], 'data')

def get_plots_dir(stock_key):
    return os.path.join(PROJECT_ROOT, STOCKS[stock_key]['dir'], 'plots')

def load_csv_preview(filepath, max_rows=200):
    """Load CSV and return header list + data rows for the table."""
    if not os.path.exists(filepath):
        return [], [], 0
    df = pd.read_csv(filepath, nrows=max_rows)
    headers = list(df.columns)
    data = df.values.tolist()
    total = len(pd.read_csv(filepath, usecols=[0]))  # count total rows
    return headers, data, total

def format_row_count(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


# ─── Layout Builders ────────────────────────────────────────────────────────

def make_stock_button(name, key):
    return sg.Button(
        name,
        key=key,
        font=FONT_BTN,
        button_color=(TEXT_MUTED, TAB_INACTIVE),
        border_width=0,
        pad=(0, 0),
        size=(14, 1),
        mouseover_colors=(TEXT_PRIMARY, BTN_HOVER),
    )

def make_section_title(text):
    return sg.Text(text, font=FONT_HEADER, text_color=ACCENT_BLUE, background_color=BG_PANEL, pad=(10, (15, 5)))

def make_subtab_button(text, key, active=False):
    SUBTAB_ACTIVE = ACCENT_GREEN
    SUBTAB_INACTIVE = BG_CARD
    colors = (BG_DARK, SUBTAB_ACTIVE) if active else (TEXT_MUTED, SUBTAB_INACTIVE)
    return sg.Button(
        text, key=key, font=FONT_SMALL, button_color=colors,
        border_width=0, pad=(0, 0), size=(16, 1),
        mouseover_colors=(TEXT_PRIMARY, BTN_HOVER),
    )

def make_inner_tab_button(text, key, active=False):
    """Third-level tab button (inside Run Model)."""
    INNER_ACTIVE = ACCENT_BLUE
    INNER_INACTIVE = BG_CARD
    colors = (BG_DARK, INNER_ACTIVE) if active else (TEXT_MUTED, INNER_INACTIVE)
    return sg.Button(
        text, key=key, font=('Menlo', 9), button_color=colors,
        border_width=0, pad=(0, 0), size=(18, 1),
        mouseover_colors=(TEXT_PRIMARY, BTN_HOVER),
    )

def make_data_section(prefix):
    """Build the full stock panel: Train Dataset, Test Dataset, Run Model."""

    # ── Train Dataset Panel ──
    train_panel = sg.Column(
        [
            [
                sg.Text('Rows: ', font=FONT_SMALL, text_color=TEXT_MUTED, background_color=BG_PANEL, pad=(10, 5)),
                sg.Text('—', key=f'{prefix}_TRAIN_ROWS', font=FONT_SMALL, text_color=ACCENT_GREEN, background_color=BG_PANEL),
                sg.Text('  Range: ', font=FONT_SMALL, text_color=TEXT_MUTED, background_color=BG_PANEL),
                sg.Text('—', key=f'{prefix}_TRAIN_RANGE', font=FONT_SMALL, text_color=TEXT_PRIMARY, background_color=BG_PANEL, expand_x=True),
            ],
            [
                sg.Table(
                    values=[],
                    headings=['datetime', 'open', 'high', 'low', 'close', 'volume'],
                    key=f'{prefix}_TRAIN_TABLE',
                    auto_size_columns=True,
                    display_row_numbers=False,
                    justification='center',
                    num_rows=12,
                    font=FONT_MONO,
                    text_color=TEXT_PRIMARY,
                    background_color=BG_CARD,
                    alternating_row_color='#1a1f27',
                    header_text_color=ACCENT_BLUE,
                    header_background_color=BG_DARK,
                    header_font=('Menlo', 10, 'bold'),
                    row_height=25,
                    pad=(10, 5),
                    expand_x=True,
                    expand_y=True,
                )
            ],
            [
                sg.Text('From:', font=FONT_SMALL, text_color=TEXT_MUTED, background_color=BG_PANEL, pad=(10, 5)),
                sg.Input('2023-01-01', key=f'{prefix}_TRAIN_FROM', size=(12, 1), font=FONT_SMALL,
                         text_color=TEXT_PRIMARY, background_color=BG_CARD),
                sg.Text('To:', font=FONT_SMALL, text_color=TEXT_MUTED, background_color=BG_PANEL),
                sg.Input('2025-06-30', key=f'{prefix}_TRAIN_TO', size=(12, 1), font=FONT_SMALL,
                         text_color=TEXT_PRIMARY, background_color=BG_CARD),
                sg.Push(background_color=BG_PANEL),
                sg.Button('Refetch Train', key=f'{prefix}_REFETCH_TRAIN', font=FONT_BTN,
                          button_color=(BG_DARK, ACCENT_ORANGE), border_width=0, pad=(10, 5)),
            ],
        ],
        key=f'{prefix}_PANEL_TRAIN',
        background_color=BG_PANEL,
        visible=True,
        expand_x=True,
        expand_y=True,
        pad=(0, 0),
    )

    # ── Test Dataset Panel ──
    test_panel = sg.Column(
        [
            [
                sg.Text('Rows: ', font=FONT_SMALL, text_color=TEXT_MUTED, background_color=BG_PANEL, pad=(10, 5)),
                sg.Text('—', key=f'{prefix}_TEST_ROWS', font=FONT_SMALL, text_color=ACCENT_GREEN, background_color=BG_PANEL),
                sg.Text('  Range: ', font=FONT_SMALL, text_color=TEXT_MUTED, background_color=BG_PANEL),
                sg.Text('—', key=f'{prefix}_TEST_RANGE', font=FONT_SMALL, text_color=TEXT_PRIMARY, background_color=BG_PANEL, expand_x=True),
            ],
            [
                sg.Table(
                    values=[],
                    headings=['datetime', 'open', 'high', 'low', 'close', 'volume'],
                    key=f'{prefix}_TEST_TABLE',
                    auto_size_columns=True,
                    display_row_numbers=False,
                    justification='center',
                    num_rows=12,
                    font=FONT_MONO,
                    text_color=TEXT_PRIMARY,
                    background_color=BG_CARD,
                    alternating_row_color='#1a1f27',
                    header_text_color=ACCENT_BLUE,
                    header_background_color=BG_DARK,
                    header_font=('Menlo', 10, 'bold'),
                    row_height=25,
                    pad=(10, 5),
                    expand_x=True,
                    expand_y=True,
                )
            ],
            [
                sg.Text('From:', font=FONT_SMALL, text_color=TEXT_MUTED, background_color=BG_PANEL, pad=(10, 5)),
                sg.Input('2025-07-01', key=f'{prefix}_TEST_FROM', size=(12, 1), font=FONT_SMALL,
                         text_color=TEXT_PRIMARY, background_color=BG_CARD),
                sg.Text('To:', font=FONT_SMALL, text_color=TEXT_MUTED, background_color=BG_PANEL),
                sg.Input('2025-12-31', key=f'{prefix}_TEST_TO', size=(12, 1), font=FONT_SMALL,
                         text_color=TEXT_PRIMARY, background_color=BG_CARD),
                sg.Push(background_color=BG_PANEL),
                sg.Button('Refetch Test', key=f'{prefix}_REFETCH_TEST', font=FONT_BTN,
                          button_color=(BG_DARK, ACCENT_ORANGE), border_width=0, pad=(10, 5)),
            ],
        ],
        key=f'{prefix}_PANEL_TEST',
        background_color=BG_PANEL,
        visible=False,
        expand_x=True,
        expand_y=True,
        pad=(0, 0),
    )

    # ── Run Model Panel ──

    # ── Simplified Run Panel ──
    run_panel = sg.Column(
        [
            [
                sg.Text('Execution Name:', font=FONT_SMALL, text_color=TEXT_MUTED, background_color=BG_PANEL, pad=(10, 8)),
                sg.Input('run_001', key=f'{prefix}_RUN_NAME', size=(20, 1), font=FONT_SMALL,
                         text_color=TEXT_PRIMARY, background_color=BG_CARD),
                sg.Text('Epochs:', font=FONT_SMALL, text_color=TEXT_MUTED, background_color=BG_PANEL),
                sg.Input('50', key=f'{prefix}_RUN_EPOCHS', size=(5, 1), font=FONT_SMALL,
                         text_color=TEXT_PRIMARY, background_color=BG_CARD),
                sg.Push(background_color=BG_PANEL),
                sg.Button('Train', key=f'{prefix}_BTN_TRAIN', font=FONT_BTN,
                          button_color=(BG_DARK, ACCENT_GREEN), border_width=0, pad=(10, 5)),
                sg.Button('Test', key=f'{prefix}_BTN_TEST', font=FONT_BTN, disabled=True,
                          button_color=(BG_DARK, BORDER), border_width=0, pad=(5, 5)),
            ],
            [
                sg.Push(background_color=BG_PANEL),
                sg.Button('Save Execution', key=f'{prefix}_BTN_SAVE', font=FONT_BTN, visible=False,
                          button_color=(BG_DARK, ACCENT_BLUE), border_width=0, pad=(10, 10)),
            ],
            [
                sg.Column([
                    [make_section_title('FEATURE ENGINEERING PREVIEW')],
                    [sg.Column([
                        [sg.Table(
                            values=[],
                            headings=['datetime', 'open', 'high', 'low', 'close', 'volume', 'diff_1', 'diff_5', 'rsi', 'rsi_fast', 'bb_width', 'rolling_std', 'atr', 'ema_ratio', 'price_ema_dist', 'return_1', 'return_5', 'return_10', 'candle_body', 'hl_range', 'vol_ratio', 'raw_body', 'raw_range'],
                            key=f'{prefix}_RUN_FEAT_TABLE',
                            auto_size_columns=False,
                            col_widths=[20, 12, 12, 12, 12, 14, 12, 12, 12, 12, 12, 14, 12, 12, 15, 12, 12, 12, 14, 12, 14, 12, 12],
                            display_row_numbers=False,
                            justification='center',
                            num_rows=8,
                            font=FONT_MONO,
                            text_color=TEXT_PRIMARY,
                            background_color=BG_CARD,
                            alternating_row_color='#1a1f27',
                            header_text_color=ACCENT_ORANGE,
                            header_background_color=BG_DARK,
                            header_font=('Menlo', 10, 'bold'),
                            row_height=25,
                            pad=(0, 0),
                            expand_x=True,
                            expand_y=True,
                        )]
                    ], scrollable=True, vertical_scroll_only=False, background_color=BG_PANEL, expand_x=True, expand_y=True, key=f'{prefix}_FEAT_SCROLL')]
                ], key=f'{prefix}_FEAT_AREA', visible=False, background_color=BG_PANEL, expand_x=True)
            ],
            [
                sg.Column([
                    [make_section_title('TRAINING PROGRESS')],
                    [sg.Text('', key=f'{prefix}_TRAIN_STEP_1', font=FONT_BODY, text_color=TEXT_PRIMARY, background_color=BG_PANEL)],
                    [sg.Text('', key=f'{prefix}_TRAIN_STEP_2', font=FONT_BODY, text_color=TEXT_PRIMARY, background_color=BG_PANEL)],
                    [sg.Text('', key=f'{prefix}_TRAIN_STEP_3', font=FONT_BODY, text_color=TEXT_PRIMARY, background_color=BG_PANEL)],
                ], key=f'{prefix}_TRAIN_AREA', visible=False, background_color=BG_PANEL, expand_x=True)
            ],
            [
                sg.Column([
                    [make_section_title('TEST PERFORMANCE & VISUALS')],
                    [sg.Text('', key=f'{prefix}_TEST_SIZE_INFO', font=FONT_BODY, text_color=ACCENT_BLUE, background_color=BG_PANEL)],
                    [sg.Multiline(
                        '', key=f'{prefix}_RUN_RESULTS', font=FONT_MONO,
                        text_color=TEXT_PRIMARY, background_color=BG_CARD,
                        size=(None, 12), expand_x=True, disabled=True, border_width=0
                    )],
                    [sg.Column([
                        [sg.Text('Confusion Matrix', font=FONT_BODY, background_color=BG_PANEL)],
                        [sg.Image(key=f'{prefix}_IMG_CM', background_color=BG_PANEL, size=(500, 375))],
                        [sg.Text('Equity Curve', font=FONT_BODY, background_color=BG_PANEL, pad=(0, (20, 0)))],
                        [sg.Image(key=f'{prefix}_IMG_EQUITY', background_color=BG_PANEL, size=(500, 375))],
                    ], background_color=BG_PANEL, expand_x=True, scrollable=True, vertical_scroll_only=True, size=(None, 600))]
                ], key=f'{prefix}_RESULTS_AREA', visible=False, expand_x=True, background_color=BG_PANEL)
            ],
            [sg.HSeparator(color=BG_DARK, pad=(0, 15))],
            [
                sg.Text('', key=f'{prefix}_RUN_STATUS', font=FONT_SMALL, text_color=ACCENT_ORANGE,
                        background_color=BG_PANEL, pad=(10, 5), expand_x=True),
            ],
        ],
        key=f'{prefix}_RUN_MAIN',
        background_color=BG_PANEL,
        visible=True,
        expand_x=True,
        expand_y=True,
        pad=(0, 0),
    )

    # Executions History 
    exec_panel = sg.Column(
        [
            [make_section_title('EXECUTION HISTORY')],
            [
                sg.Table(
                    values=[],
                    headings=['Name', 'Status', 'Epochs', 'Duration', 'Started'],
                    key=f'{prefix}_EXEC_TABLE',
                    auto_size_columns=True,
                    display_row_numbers=False,
                    justification='center',
                    num_rows=15,
                    font=FONT_MONO,
                    text_color=TEXT_PRIMARY,
                    background_color=BG_CARD,
                    alternating_row_color='#1a1f27',
                    header_text_color=ACCENT_BLUE,
                    header_background_color=BG_DARK,
                    header_font=('Menlo', 10, 'bold'),
                    row_height=25,
                    pad=(10, 5),
                    expand_x=True,
                    expand_y=True,
                )
            ],
            [
                sg.Push(background_color=BG_PANEL),
                sg.Button('Refresh', key=f'{prefix}_EXEC_REFRESH', font=FONT_BTN,
                          button_color=(BG_DARK, ACCENT_ORANGE), border_width=0, pad=(10, 5)),
            ],
        ],
        key=f'{prefix}_RUN_EXEC',
        background_color=BG_PANEL,
        visible=False,
        expand_x=True,
        expand_y=True,
        pad=(0, 0),
    )

    # Combine into Run Model panel
    run_model_panel = sg.Column(
        [
            [
                make_inner_tab_button('Run', f'{prefix}_INNER_RUN', active=True),
                make_inner_tab_button('Executions', f'{prefix}_INNER_EXEC', active=False),
                sg.Push(background_color=BG_PANEL),
            ],
            [run_panel, exec_panel],
        ],
        key=f'{prefix}_PANEL_RUN',
        background_color=BG_PANEL,
        visible=False,
        expand_x=True,
        expand_y=True,
        pad=(0, 0),
    )

    return [
        # ── Top-level Sub-tabs (Train / Test / Run Model) ──
        [
            make_subtab_button('Train Dataset', f'{prefix}_SUBTAB_TRAIN', active=True),
            make_subtab_button('Test Dataset', f'{prefix}_SUBTAB_TEST', active=False),
            make_subtab_button('Run Model', f'{prefix}_SUBTAB_RUN', active=False),
            sg.Push(background_color=BG_PANEL),
        ],
        # ── Panels (only one visible) ──
        [train_panel, test_panel, run_model_panel],
        # ── Fetch Status ──
        [
            sg.Text('', key=f'{prefix}_FETCH_STATUS', font=FONT_SMALL, text_color=ACCENT_ORANGE,
                    background_color=BG_PANEL, pad=(10, 5), expand_x=True),
        ],
    ]


def build_layout():
    """Construct the full window layout."""
    # Stock tab buttons
    tab_row = [
        make_stock_button(name, f'TAB_{name.upper()}')
        for name in STOCKS.keys()
    ]

    # Build stock panels — each is a non-scrollable Column so children expand properly
    stock_panels = []
    for i, (name, info) in enumerate(STOCKS.items()):
        prefix = name.upper()
        col = sg.Column(
            make_data_section(prefix),
            key=f'COL_{prefix}',
            background_color=BG_PANEL,
            visible=(i == 0),
            expand_x=True,
            expand_y=True,
            pad=(0, 0),
        )
        stock_panels.append(col)

    # All panels in ONE row so they overlap in the same position
    content_row = [sg.Pane(
        [sg.Column(
            [stock_panels],  # Single row with all panels side by side (only one visible)
            background_color=BG_PANEL,
            expand_x=True,
            expand_y=True,
            scrollable=True,
            vertical_scroll_only=True,
            pad=(0, 0),
            key='SCROLL_AREA',
        )],
        orientation='vertical',
        relief=sg.RELIEF_FLAT,
        background_color=BG_PANEL,
        show_handle=False,
        border_width=0,
        pad=(0, 0),
        expand_x=True,
        expand_y=True,
    )]

    layout = [
        tab_row,
        content_row,
    ]

    return layout


# ─── Data Loading Logic ─────────────────────────────────────────────────────

def refresh_data_view(window, stock_key, model_mgr):
    """Load and display train/test CSV data for the given stock."""
    prefix = stock_key.upper()
    train_path = get_data_path(stock_key, 'train_data.csv')
    test_path = get_data_path(stock_key, 'test_data.csv')

    # Train data
    headers, data, total = load_csv_preview(train_path, max_rows=200)
    if data:
        window[f'{prefix}_TRAIN_TABLE'].update(values=data)
        window[f'{prefix}_TRAIN_ROWS'].update(format_row_count(total))
        # Extract date range
        try:
            df_peek = pd.read_csv(train_path, usecols=['datetime'], parse_dates=['datetime'])
            dmin = df_peek['datetime'].min().strftime('%Y-%m-%d')
            dmax = df_peek['datetime'].max().strftime('%Y-%m-%d')
            window[f'{prefix}_TRAIN_RANGE'].update(f'{dmin}  →  {dmax}')
        except Exception:
            window[f'{prefix}_TRAIN_RANGE'].update('—')
    else:
        window[f'{prefix}_TRAIN_TABLE'].update(values=[])
        window[f'{prefix}_TRAIN_ROWS'].update('No data')
        window[f'{prefix}_TRAIN_RANGE'].update('—')

    # Test data
    headers, data, total = load_csv_preview(test_path, max_rows=200)
    if data:
        window[f'{prefix}_TEST_TABLE'].update(values=data)
        window[f'{prefix}_TEST_ROWS'].update(format_row_count(total))
        try:
            df_peek = pd.read_csv(test_path, usecols=['datetime'], parse_dates=['datetime'])
            dmin = df_peek['datetime'].min().strftime('%Y-%m-%d')
            dmax = df_peek['datetime'].max().strftime('%Y-%m-%d')
            window[f'{prefix}_TEST_RANGE'].update(f'{dmin}  →  {dmax}')
        except Exception:
            window[f'{prefix}_TEST_RANGE'].update('—')
    else:
        window[f'{prefix}_TEST_TABLE'].update(values=[])
        window[f'{prefix}_TEST_ROWS'].update('No data')
        window[f'{prefix}_TEST_RANGE'].update('—')

    # Refresh Executions Table
    history = model_mgr.load_history(stock_key)
    # New headings: ['Name', 'Status', 'Epochs', 'Duration', 'Started']
    exec_values = [[ex['Name'], ex['Status'], ex['Epochs'], ex['Duration'], ex['Started']] for ex in history]
    window[f'{prefix}_EXEC_TABLE'].update(values=exec_values)


# ─── Background Fetch Runner ────────────────────────────────────────────────

class FetchRunner:
    """Runs data fetch in a background thread with status updates."""

    def __init__(self):
        self.msg_queue = queue.Queue()
        self.running = False

    def fetch(self, stock_key, dtype, date_from, date_to):
        """Fetch train or test data for the given stock."""
        if self.running:
            return False

        token = STOCKS[stock_key]['token']
        output_dir = get_output_dir(stock_key)
        filename = 'train_data.csv' if dtype == 'train' else 'test_data.csv'
        output_path = os.path.join(output_dir, filename)

        def _worker():
            self.running = True
            prefix = stock_key.upper()
            self.msg_queue.put((prefix, 'STATUS', f'⏳ Fetching {dtype} data for {stock_key}...'))

            try:
                start = datetime.strptime(date_from, '%Y-%m-%d').replace(hour=9, minute=15)
                end = datetime.strptime(date_to, '%Y-%m-%d').replace(hour=15, minute=30)

                if PROJECT_ROOT not in sys.path:
                    sys.path.insert(0, PROJECT_ROOT)
                from scripts.fetch_data import login, get_range_data, to_df

                self.msg_queue.put((prefix, 'STATUS', f'🔑 Logging into Angel One...'))
                api = login()

                self.msg_queue.put((prefix, 'STATUS', f'📡 Fetching {dtype} candles ({date_from} → {date_to})...'))
                raw = get_range_data(api, token, start, end)
                df = to_df(raw)

                os.makedirs(output_dir, exist_ok=True)
                df.to_csv(output_path, index=False)

                self.msg_queue.put((prefix, 'STATUS', f'✅ {dtype.title()} data saved: {len(df)} rows'))
                self.msg_queue.put((prefix, 'REFRESH', stock_key))

            except Exception as e:
                self.msg_queue.put((prefix, 'STATUS', f'❌ Fetch failed: {str(e)[:80]}'))
            finally:
                self.running = False

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return True

    def process_queue(self, window):
        """Drain pending messages and apply to window."""
        refresh_stock = None
        while not self.msg_queue.empty():
            try:
                prefix, msg_type, payload = self.msg_queue.get_nowait()
                if msg_type == 'STATUS':
                    window[f'{prefix}_FETCH_STATUS'].update(payload)
                elif msg_type == 'REFRESH':
                    refresh_stock = payload
            except queue.Empty:
                break
        return refresh_stock


# ─── Execution Manager ───────────────────────────────────────────────────────

class ExecutionManager:
    """Manages model training runs, logging, and history persistence."""

    def __init__(self):
        self.msg_queue = queue.Queue()
        self.running = False
        self.start_ts = 0

    def get_history_path(self, stock_key):
        return os.path.join(get_output_dir(stock_key), 'executions.json')

    def load_history(self, stock_key):
        path = self.get_history_path(stock_key)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_execution(self, stock_key, exec_data):
        history = self.load_history(stock_key)
        history.insert(0, exec_data)  # Newest first
        path = self.get_history_path(stock_key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(history[:100], f, indent=4)  # Keep last 100

    def run_train(self, stock_key, epochs):
        if self.running: return False
        
        prefix = stock_key.upper()
        data_path = get_data_path(stock_key, 'train_data.csv')
        output_dir = get_output_dir(stock_key)
        self.start_ts = time.time()

        def _worker():
            self.running = True
            log_key = f'{prefix}_RUN_LOG'
            status_key = f'{prefix}_RUN_STATUS'
            
            self.msg_queue.put((prefix, 'STATUS', (status_key, 'Loading Dataset...')))
            self.msg_queue.put((prefix, 'LOG', (log_key, f'Loading dataset: {data_path}\n')))
            
            try:
                if PROJECT_ROOT not in sys.path:
                    sys.path.insert(0, PROJECT_ROOT)

                # Step 1: Loading
                df_raw = pd.read_csv(data_path)
                raw_count = len(df_raw)
                self.msg_queue.put((prefix, 'TRAIN_PROGRESS', (1, f"Training Dataset Loaded (Size: {raw_count})")))
                time.sleep(1) # Small pause for narration
                
                # Step 2: Engineering
                df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
                from src.features import calculate_features
                df_feat = calculate_features(df_raw)
                feat_count = len(df_feat)
                
                # Table Preview Data
                cols_to_show = [
                    'datetime', 'open', 'high', 'low', 'close', 'volume', 
                    'diff_1', 'diff_5', 'rsi', 'rsi_fast', 'bb_width', 'rolling_std', 'atr',
                    'ema_ratio', 'price_ema_dist', 'return_1', 'return_5', 'return_10',
                    'candle_body', 'hl_range', 'vol_ratio', 'raw_body', 'raw_range'
                ]
                existing_cols = [c for c in cols_to_show if c in df_feat.columns]
                sample_df = df_feat[existing_cols].head(20).copy()
                numeric_cols = sample_df.select_dtypes(include=['number']).columns
                sample_df[numeric_cols] = sample_df[numeric_cols].round(4)
                sample_data = sample_df.values.tolist()
                
                self.msg_queue.put((prefix, 'FEAT_TABLE', (existing_cols, sample_data)))
                self.msg_queue.put((prefix, 'SHOW_FEAT', None))
                self.msg_queue.put((prefix, 'TRAIN_PROGRESS', (2, f"Feature engineering complete (Size: {feat_count})")))
                time.sleep(1) # Small pause for narration

                # Step 3: Training
                from scripts.train import train_model
                self.msg_queue.put((prefix, 'STATUS', (status_key, 'Training model...')))
                try:
                    train_model(data_path, output_dir=output_dir, epochs=int(epochs))
                    self.msg_queue.put((prefix, 'TRAIN_PROGRESS', (3, "Training phase complete.")))
                    self.msg_queue.put((prefix, 'TRAIN_DONE', None))
                    self.msg_queue.put((prefix, 'STATUS', (status_key, 'Model ready.')))
                finally:
                    pass

            except Exception as e:
                self.msg_queue.put((prefix, 'STATUS', (status_key, f'Error: {str(e)}')))
            finally:
                self.running = False

        threading.Thread(target=_worker, daemon=True).start()
        return True

    def run_test(self, stock_key):
        if self.running: return False
        
        prefix = stock_key.upper()
        data_path = get_data_path(stock_key, 'test_data.csv')
        output_dir = get_output_dir(stock_key)
        model_path = os.path.join(output_dir, 'best_stock_model.pth')
        scaler_path = os.path.join(output_dir, 'x_scaler.pkl')

        def _worker():
            self.running = True
            log_key = f'{prefix}_RUN_LOG'
            status_key = f'{prefix}_RUN_STATUS'
            
            self.msg_queue.put((prefix, 'STATUS', (status_key, 'Evaluating...')))
            time.sleep(0.5)

            try:
                from scripts.test import run_test
                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()
                
                try:
                    res = run_test(data_path, model_path, scaler_path, output_dir)
                    
                    # Report test size first
                    self.msg_queue.put((prefix, 'PROGRESS_TEST', (1, f"Test Dataset Sample Size - {res['test_size']}")))
                    time.sleep(1)
                    
                    # Format results summary
                    summary =  f"BEST MODEL PERFORMANCE\n"
                    summary += f"{'─'*30}\n"
                    summary += f"AUC-ROC: {res['auc']:.4f}\n"
                    summary += f"Win Rate: {res['stats']['win_rate']}\n"
                    summary += f"Net Profit: {res['stats']['net_profit']}\n"
                    summary += f"Sharpe: {res['stats']['sharpe']}\n"
                    summary += f"{'─'*30}\n\n"
                    summary += res['report']
                    
                    # Payload with image paths
                    results_payload = {
                        'summary': summary,
                        'plots': {
                            'cm': os.path.join(output_dir, 'confusion_matrix.png'),
                            'equity': os.path.join(output_dir, 'equity_curve.png')
                        }
                    }
                    
                    self.msg_queue.put((prefix, 'TEST_RESULTS', results_payload))
                    self.msg_queue.put((prefix, 'STATUS', (status_key, 'Evaluation Complete.')))
                finally:
                    pass

            except Exception as e:
                self.msg_queue.put((prefix, 'STATUS', (status_key, f'Error: {str(e)}')))
            finally:
                self.running = False

        threading.Thread(target=_worker, daemon=True).start()
        return True

    def process_queue(self, window):
        refresh_stock = None
        while not self.msg_queue.empty():
            try:
                prefix, msg_type, payload = self.msg_queue.get_nowait()
                if msg_type == 'STATUS':
                    key, text = payload
                    window[key].update(text)
                elif msg_type == 'TRAIN_PROGRESS':
                    step, text = payload
                    window[f'{prefix}_TRAIN_AREA'].update(visible=True)
                    window[f'{prefix}_TRAIN_STEP_{step}'].update(text)
                elif msg_type == 'PROGRESS_TEST':
                    step, text = payload
                    window[f'{prefix}_RESULTS_AREA'].update(visible=True)
                    window[f'{prefix}_TEST_SIZE_INFO'].update(text)
                elif msg_type == 'FEAT_TABLE':
                    cols, data = payload
                    window[f'{prefix}_RUN_FEAT_TABLE'].update(values=data)
                    # Update headings manually if needed, but for minimal UI we assume fixed or just data
                elif msg_type == 'SHOW_FEAT':
                    window[f'{prefix}_FEAT_AREA'].update(visible=True)
                elif msg_type == 'TRAIN_DONE':
                    window[f'{prefix}_BTN_TEST'].update(disabled=False, button_color=(BG_DARK, ACCENT_BLUE))
                elif msg_type == 'TEST_RESULTS':
                    window[f'{prefix}_RUN_RESULTS'].update(payload['summary'], append=True)
                    window[f'{prefix}_RESULTS_AREA'].update(visible=True)
                    window[f'{prefix}_BTN_SAVE'].update(visible=True)
                    
                    # Update Model Test Plots
                    try:
                        from PIL import Image
                        import io
                        def get_img_data(fpath, max_size=(500, 375)):
                            img = Image.open(fpath)
                            img.thumbnail(max_size)
                            bio = io.BytesIO()
                            img.save(bio, format="PNG")
                            return bio.getvalue()

                        window[f'{prefix}_IMG_CM'].update(data=get_img_data(payload['plots']['cm']))
                        window[f'{prefix}_IMG_EQUITY'].update(data=get_img_data(payload['plots']['equity']))
                    except Exception as e:
                        print(f"Test image load error: {e}")
                elif msg_type == 'STATUS_SIM':
                    pass
                elif msg_type == 'SIM_READY':
                    pass
                elif msg_type == 'SIM_TICK':
                    pass
                elif msg_type == 'REFRESH_EXEC':
                    refresh_stock = payload
            except queue.Empty:
                break
        return refresh_stock


# ─── Main Application ───────────────────────────────────────────────────────

def main():
    sg.theme('DarkBlack1')
    sg.set_options(
        font=FONT_BODY,
        text_color=TEXT_PRIMARY,
        background_color=BG_DARK,
        element_background_color=BG_PANEL,
        text_element_background_color=BG_PANEL,
        input_elements_background_color=BG_CARD,
        input_text_color=TEXT_PRIMARY,
        button_color=(TEXT_PRIMARY, BTN_BG),
        border_width=0,
    )

    layout = build_layout()
    window = sg.Window(
        'Stock Prediction Terminal',
        layout,
        size=(1100, 850),
        background_color=BG_DARK,
        finalize=True,
        resizable=True,
        margins=(0, 0),
    )

    fetcher = FetchRunner()
    model_mgr = ExecutionManager()
    active_stock = list(STOCKS.keys())[0]

    # Highlight first tab
    window[f'TAB_{active_stock.upper()}'].update(button_color=(BG_DARK, TAB_ACTIVE))

    # Load initial data
    refresh_data_view(window, active_stock, model_mgr)

    # ── Event Loop ──
    while True:
        event, values = window.read(timeout=300)

        if event in (sg.WIN_CLOSED, 'Exit'):
            break

        # ── Stock Tab Switching ──
        for name in STOCKS.keys():
            tab_key = f'TAB_{name.upper()}'
            if event == tab_key:
                for n in STOCKS.keys():
                    window[f'COL_{n.upper()}'].update(visible=False)
                    window[f'TAB_{n.upper()}'].update(button_color=(TEXT_MUTED, TAB_INACTIVE))

                window[f'COL_{name.upper()}'].update(visible=True)
                window[tab_key].update(button_color=(BG_DARK, TAB_ACTIVE))
                active_stock = name
                refresh_data_view(window, name, model_mgr)
                break

        # ── Level 2 Sub-tab Switching (Train / Test / Run Model) ──
        for name in STOCKS.keys():
            prefix = name.upper()
            if event == f'{prefix}_SUBTAB_TRAIN':
                window[f'{prefix}_PANEL_TRAIN'].update(visible=True)
                window[f'{prefix}_PANEL_TEST'].update(visible=False)
                window[f'{prefix}_PANEL_RUN'].update(visible=False)
                window[f'{prefix}_SUBTAB_TRAIN'].update(button_color=(BG_DARK, ACCENT_GREEN))
                window[f'{prefix}_SUBTAB_TEST'].update(button_color=(TEXT_MUTED, BG_CARD))
                window[f'{prefix}_SUBTAB_RUN'].update(button_color=(TEXT_MUTED, BG_CARD))
                break
            if event == f'{prefix}_SUBTAB_TEST':
                window[f'{prefix}_PANEL_TRAIN'].update(visible=False)
                window[f'{prefix}_PANEL_TEST'].update(visible=True)
                window[f'{prefix}_PANEL_RUN'].update(visible=False)
                window[f'{prefix}_SUBTAB_TRAIN'].update(button_color=(TEXT_MUTED, BG_CARD))
                window[f'{prefix}_SUBTAB_TEST'].update(button_color=(BG_DARK, ACCENT_GREEN))
                window[f'{prefix}_SUBTAB_RUN'].update(button_color=(TEXT_MUTED, BG_CARD))
                break
            if event == f'{prefix}_SUBTAB_RUN':
                window[f'{prefix}_PANEL_TRAIN'].update(visible=False)
                window[f'{prefix}_PANEL_TEST'].update(visible=False)
                window[f'{prefix}_PANEL_RUN'].update(visible=True)
                window[f'{prefix}_SUBTAB_TRAIN'].update(button_color=(TEXT_MUTED, BG_CARD))
                window[f'{prefix}_SUBTAB_TEST'].update(button_color=(TEXT_MUTED, BG_CARD))
                window[f'{prefix}_SUBTAB_RUN'].update(button_color=(BG_DARK, ACCENT_GREEN))
                break

        for name in STOCKS.keys():
            prefix = name.upper()
            if event == f'{prefix}_INNER_RUN':
                window[f'{prefix}_RUN_MAIN'].update(visible=True)
                window[f'{prefix}_RUN_EXEC'].update(visible=False)
                window[f'{prefix}_INNER_RUN'].update(button_color=(BG_DARK, ACCENT_BLUE))
                window[f'{prefix}_INNER_EXEC'].update(button_color=(TEXT_MUTED, BG_CARD))
                break
            if event == f'{prefix}_INNER_EXEC':
                window[f'{prefix}_RUN_MAIN'].update(visible=False)
                window[f'{prefix}_RUN_EXEC'].update(visible=True)
                window[f'{prefix}_INNER_RUN'].update(button_color=(TEXT_MUTED, BG_CARD))
                window[f'{prefix}_INNER_EXEC'].update(button_color=(BG_DARK, ACCENT_BLUE))
                refresh_data_view(window, name, model_mgr)
                break

        # ── Refetch ──
        for name in STOCKS.keys():
            prefix = name.upper()
            if event in (f'{prefix}_REFETCH_TRAIN', f'{prefix}_REFETCH_TEST'):
                if fetcher.running:
                    sg.popup_quick_message('⚠️ A fetch is already running!',
                                           font=FONT_BODY, background_color=ACCENT_RED, text_color=TEXT_PRIMARY)
                    break
                dtype = 'train' if 'TRAIN' in event else 'test'
                date_from = values.get(f'{prefix}_{dtype.upper()}_FROM', '')
                date_to = values.get(f'{prefix}_{dtype.upper()}_TO', '')
                if not date_from or not date_to:
                    sg.popup_quick_message('⚠️ Please fill in the date range!',
                                           font=FONT_BODY, background_color=ACCENT_RED, text_color=TEXT_PRIMARY)
                    break
                fetcher.fetch(name, dtype, date_from, date_to)
                break

        # ── Model Run Actions ──
        for name in STOCKS.keys():
            prefix = name.upper()
            if event == f'{prefix}_BTN_TRAIN':
                epochs = values[f'{prefix}_RUN_EPOCHS']
                # Clear Training Info
                window[f'{prefix}_TRAIN_STEP_1'].update('')
                window[f'{prefix}_TRAIN_STEP_2'].update('')
                window[f'{prefix}_TRAIN_STEP_3'].update('')
                window[f'{prefix}_TRAIN_AREA'].update(visible=False)
                
                window[f'{prefix}_RUN_RESULTS'].update('') # Clear previous results
                window[f'{prefix}_IMG_CM'].update(data=None) # Clear previous images
                window[f'{prefix}_IMG_EQUITY'].update(data=None)
                
                window[f'{prefix}_FEAT_AREA'].update(visible=False)
                window[f'{prefix}_RESULTS_AREA'].update(visible=False)
                window[f'{prefix}_BTN_SAVE'].update(visible=False)
                if not model_mgr.run_train(name, epochs):
                    sg.popup_quick_message('⚠️ Run already in progress')
                break
            if event == f'{prefix}_BTN_TEST':
                if not model_mgr.run_test(name):
                    sg.popup_quick_message('⚠️ Run already in progress')
                break

            if event == f'{prefix}_BTN_SAVE':
                exec_name = values[f'{prefix}_RUN_NAME']
                epochs = values[f'{prefix}_RUN_EPOCHS']
                
                # Fetch dataset info
                train_path = get_data_path(name, 'train_data.csv')
                train_size = os.path.getsize(train_path) // 1024 if os.path.exists(train_path) else 0
                
                duration = f'{int(time.time() - model_mgr.start_ts)}s'
                
                record = {
                    'Name': exec_name,
                    'Status': 'Success',
                    'Epochs': epochs,
                    'Duration': duration,
                    'Started': datetime.now().strftime('%Y-%m-%d %H:%M'),
                }
                model_mgr.save_execution(name, record)
                sg.popup_quick_message('Execution Saved!')
                refresh_data_view(window, name, model_mgr)
                break
            if event == f'{prefix}_EXEC_REFRESH':
                refresh_data_view(window, name, model_mgr)
                break

        # ── Process background updates ──
        refresh_stock = fetcher.process_queue(window)
        if refresh_stock:
            refresh_data_view(window, refresh_stock, model_mgr)
            
        refresh_stock_exec = model_mgr.process_queue(window)
        if refresh_stock_exec:
            refresh_data_view(window, refresh_stock_exec, model_mgr)

    window.close()


if __name__ == '__main__':
    main()
