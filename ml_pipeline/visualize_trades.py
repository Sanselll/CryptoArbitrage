#!/usr/bin/env python3
"""
Trade Visualization Dashboard

Creates an interactive HTML dashboard showing all trades with:
- Price charts for both exchanges (Binance & Bybit)
- Entry/exit markers with annotations
- Spread charts
- Trade details (funding earned, PnL, duration)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Import the price history loader
import sys
sys.path.insert(0, str(Path(__file__).parent))
from common.data.price_history_loader import PriceHistoryLoader


def load_trades(trades_path: str) -> pd.DataFrame:
    """Load trades from CSV and parse datetimes"""
    df = pd.read_csv(trades_path)
    df['entry_datetime'] = pd.to_datetime(df['entry_datetime'])
    df['exit_datetime'] = pd.to_datetime(df['exit_datetime'])
    # Filter to closed trades only
    df = df[df['status'] == 'closed'].copy()
    # Sort by entry time
    df = df.sort_values('entry_datetime').reset_index(drop=True)
    return df


def calculate_spread(entry_long, entry_short, exit_long=None, exit_short=None):
    """Calculate spread percentage between exchanges"""
    if exit_long is not None and exit_short is not None:
        # For exit
        avg_price = (exit_long + exit_short) / 2
        spread = (exit_long - exit_short) / avg_price * 100
    else:
        # For entry
        avg_price = (entry_long + entry_short) / 2
        spread = (entry_long - entry_short) / avg_price * 100
    return spread


def create_trade_chart(trade: pd.Series, price_loader: PriceHistoryLoader, trade_num: int, initial_capital: float = 93.33) -> go.Figure:
    """Create a chart for a single trade"""
    symbol = trade['symbol']
    entry_dt = trade['entry_datetime']
    exit_dt = trade['exit_datetime']

    # Load price history with buffer
    try:
        price_df = price_loader.load_symbol(symbol)
        if price_df is None or len(price_df) == 0:
            return None
    except Exception as e:
        print(f"  Warning: Could not load price data for {symbol}: {e}")
        return None

    # Ensure index is datetime
    if 'timestamp' in price_df.columns:
        price_df = price_df.set_index('timestamp')

    # Make timezone-aware if needed
    if price_df.index.tz is None:
        price_df.index = price_df.index.tz_localize('UTC')

    # Filter to trade window with buffer
    buffer_before = timedelta(hours=2)
    buffer_after = timedelta(hours=1)

    # Convert entry/exit to same timezone
    if entry_dt.tzinfo is None:
        entry_dt = entry_dt.tz_localize('UTC')
    if exit_dt.tzinfo is None:
        exit_dt = exit_dt.tz_localize('UTC')

    mask = (price_df.index >= entry_dt - buffer_before) & (price_df.index <= exit_dt + buffer_after)
    chart_df = price_df.loc[mask].copy()

    if len(chart_df) == 0:
        print(f"  Warning: No price data in range for {symbol}")
        return None

    # Calculate spread
    chart_df['spread_pct'] = (chart_df['bybit_price'] - chart_df['binance_price']) / \
                             ((chart_df['bybit_price'] + chart_df['binance_price']) / 2) * 100

    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[None, 'Spread %']
    )

    # Add Binance price
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df['binance_price'],
            name='Binance',
            line=dict(color='#1f77b4', width=1.5),
            hovertemplate='Binance: $%{y:.6f}<br>%{x}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add Bybit price
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df['bybit_price'],
            name='Bybit',
            line=dict(color='#ff7f0e', width=1.5),
            hovertemplate='Bybit: $%{y:.6f}<br>%{x}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add entry marker
    fig.add_vline(
        x=entry_dt,
        line=dict(color='green', width=2, dash='dash'),
        row=1, col=1
    )
    fig.add_vline(
        x=entry_dt,
        line=dict(color='green', width=1, dash='dash'),
        row=2, col=1
    )

    # Add exit marker
    fig.add_vline(
        x=exit_dt,
        line=dict(color='red', width=2, dash='dash'),
        row=1, col=1
    )
    fig.add_vline(
        x=exit_dt,
        line=dict(color='red', width=1, dash='dash'),
        row=2, col=1
    )

    # Add shaded region for trade duration
    pnl_color = 'rgba(0,255,0,0.1)' if trade['realized_pnl_usd'] > 0 else 'rgba(255,0,0,0.1)'
    fig.add_vrect(
        x0=entry_dt, x1=exit_dt,
        fillcolor=pnl_color,
        line_width=0,
        row=1, col=1
    )

    # Add entry/exit price markers
    fig.add_trace(
        go.Scatter(
            x=[entry_dt, entry_dt],
            y=[trade['entry_long_price'], trade['entry_short_price']],
            mode='markers',
            marker=dict(size=10, color='green', symbol='triangle-up'),
            name='Entry',
            hovertemplate='Entry<br>Long: $%{y:.6f}<br>Short: $%{customdata:.6f}<extra></extra>',
            customdata=[trade['entry_short_price'], trade['entry_long_price']]
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[exit_dt, exit_dt],
            y=[trade['exit_long_price'], trade['exit_short_price']],
            mode='markers',
            marker=dict(size=10, color='red', symbol='triangle-down'),
            name='Exit',
            hovertemplate='Exit<br>Long: $%{y:.6f}<br>Short: $%{customdata:.6f}<extra></extra>',
            customdata=[trade['exit_short_price'], trade['exit_long_price']]
        ),
        row=1, col=1
    )

    # Add spread chart
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df['spread_pct'],
            name='Spread %',
            line=dict(color='purple', width=1),
            fill='tozeroy',
            fillcolor='rgba(128,0,128,0.1)',
            hovertemplate='Spread: %{y:.3f}%<br>%{x}<extra></extra>'
        ),
        row=2, col=1
    )

    # Add zero line on spread chart
    fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dot'), row=2, col=1)

    # Calculate entry/exit spreads
    entry_spread = calculate_spread(trade['entry_long_price'], trade['entry_short_price'])
    exit_spread = calculate_spread(trade['exit_long_price'], trade['exit_short_price'])

    # Calculate position size as % of capital
    position_size = trade['position_size_usd']
    leverage = trade['leverage']
    size_pct_of_capital = (position_size / initial_capital) * 100

    # Get coin_quantity - same quantity for both legs (true delta neutral)
    # If not in CSV (older data), calculate from average entry price
    if 'coin_quantity' in trade and pd.notna(trade['coin_quantity']):
        coin_quantity = trade['coin_quantity']
    else:
        avg_entry_price = (trade['entry_long_price'] + trade['entry_short_price']) / 2
        coin_quantity = (position_size * leverage) / avg_entry_price

    # Calculate long/short PnL from price movements (excluding funding)
    # CRITICAL: Use same coin_quantity for both legs (true delta neutral, matches backend)
    # PnL = coin_quantity * price_change
    long_price_pnl = coin_quantity * (trade['exit_long_price'] - trade['entry_long_price'])
    short_price_pnl = coin_quantity * (trade['entry_short_price'] - trade['exit_short_price'])

    # Get funding for each side
    long_funding = trade.get('long_funding_earned_usd', 0)
    short_funding = trade.get('short_funding_earned_usd', 0)
    total_funding = trade.get('funding_earned_usd', long_funding + short_funding)

    # Total PnL for each side (price + funding)
    long_total_pnl = long_price_pnl + long_funding
    short_total_pnl = short_price_pnl + short_funding

    # Fees
    total_fees = trade.get('total_fees_usd', 0)

    # Update layout
    pnl_sign = '+' if trade['realized_pnl_usd'] >= 0 else ''
    title_text = (
        f"<b>#{trade_num + 1} {symbol}</b> | "
        f"Long: {trade['long_exchange']} / Short: {trade['short_exchange']}<br>"
        f"<span style='font-size:12px'>"
        f"Entry: {entry_dt.strftime('%Y-%m-%d %H:%M')} | "
        f"Exit: {exit_dt.strftime('%Y-%m-%d %H:%M')} | "
        f"Duration: {trade['hours_held']:.1f}h</span>"
    )

    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor='center'),
        height=450,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=220, t=80, b=30),
        hovermode='x unified'
    )

    # Format signs for display
    def fmt_sign(val):
        return '+' if val >= 0 else ''

    # Add detailed annotation box with trade details
    annotation_text = (
        f"<b>═══ POSITION ═══</b><br>"
        f"Size: ${position_size:.2f} ({size_pct_of_capital:.1f}% of capital)<br>"
        f"Leverage: {trade['leverage']:.0f}x<br>"
        f"<br>"
        f"<b>═══ SPREADS ═══</b><br>"
        f"Entry: {entry_spread:.3f}%<br>"
        f"Exit: {exit_spread:.3f}%<br>"
        f"<br>"
        f"<b>═══ LONG ({trade['long_exchange']}) ═══</b><br>"
        f"Price PnL: {fmt_sign(long_price_pnl)}${long_price_pnl:.3f}<br>"
        f"Funding: {fmt_sign(long_funding)}${long_funding:.3f}<br>"
        f"Total: {fmt_sign(long_total_pnl)}${long_total_pnl:.3f}<br>"
        f"<br>"
        f"<b>═══ SHORT ({trade['short_exchange']}) ═══</b><br>"
        f"Price PnL: {fmt_sign(short_price_pnl)}${short_price_pnl:.3f}<br>"
        f"Funding: {fmt_sign(short_funding)}${short_funding:.3f}<br>"
        f"Total: {fmt_sign(short_total_pnl)}${short_total_pnl:.3f}<br>"
        f"<br>"
        f"<b>═══ TOTALS ═══</b><br>"
        f"Price PnL: {fmt_sign(long_price_pnl + short_price_pnl)}${long_price_pnl + short_price_pnl:.3f}<br>"
        f"Funding: {fmt_sign(total_funding)}${total_funding:.3f}<br>"
        f"Fees: -${total_fees:.3f}<br>"
        f"<b>Net PnL: {pnl_sign}${trade['realized_pnl_usd']:.3f} ({pnl_sign}{trade['realized_pnl_pct']:.2f}%)</b>"
    )

    fig.add_annotation(
        xref='paper', yref='paper',
        x=1.02, y=0.5,
        xanchor='left',
        text=annotation_text,
        showarrow=False,
        font=dict(size=10, family='monospace'),
        align='left',
        bgcolor='rgba(255,255,255,0.95)',
        bordercolor='gray',
        borderwidth=1,
        borderpad=8
    )

    fig.update_xaxes(title_text='', row=2, col=1)
    fig.update_yaxes(title_text='Price ($)', row=1, col=1)
    fig.update_yaxes(title_text='Spread %', row=2, col=1)

    return fig


def create_summary_section(trades: pd.DataFrame) -> go.Figure:
    """Create summary charts for all trades"""

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Cumulative P&L Over Time',
            'P&L by Symbol',
            'Trade Duration Distribution',
            'Win Rate by Symbol'
        ],
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "histogram"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 1. Cumulative PnL timeline
    trades_sorted = trades.sort_values('exit_datetime').copy()
    trades_sorted['cumulative_pnl'] = trades_sorted['realized_pnl_usd'].cumsum()

    colors = ['green' if x >= 0 else 'red' for x in trades_sorted['realized_pnl_usd']]

    fig.add_trace(
        go.Scatter(
            x=trades_sorted['exit_datetime'],
            y=trades_sorted['cumulative_pnl'],
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(color='blue', width=2),
            marker=dict(size=6, color=colors),
            hovertemplate='%{x}<br>Cumulative: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_hline(y=0, line=dict(color='gray', dash='dash'), row=1, col=1)

    # 2. PnL by symbol
    symbol_pnl = trades.groupby('symbol')['realized_pnl_usd'].sum().sort_values(ascending=True)
    colors = ['green' if x >= 0 else 'red' for x in symbol_pnl.values]

    fig.add_trace(
        go.Bar(
            x=symbol_pnl.values,
            y=symbol_pnl.index,
            orientation='h',
            marker_color=colors,
            name='P&L by Symbol',
            hovertemplate='%{y}: $%{x:.2f}<extra></extra>'
        ),
        row=1, col=2
    )

    # 3. Trade duration distribution
    fig.add_trace(
        go.Histogram(
            x=trades['hours_held'],
            nbinsx=20,
            name='Duration',
            marker_color='steelblue',
            hovertemplate='%{x:.1f}h: %{y} trades<extra></extra>'
        ),
        row=2, col=1
    )

    # 4. Win rate by symbol
    symbol_stats = trades.groupby('symbol').agg({
        'realized_pnl_usd': ['count', lambda x: (x > 0).sum()]
    }).reset_index()
    symbol_stats.columns = ['symbol', 'total', 'wins']
    symbol_stats['win_rate'] = symbol_stats['wins'] / symbol_stats['total'] * 100
    symbol_stats = symbol_stats.sort_values('win_rate', ascending=True)

    colors = ['green' if x >= 50 else 'red' for x in symbol_stats['win_rate']]

    fig.add_trace(
        go.Bar(
            x=symbol_stats['win_rate'],
            y=symbol_stats['symbol'],
            orientation='h',
            marker_color=colors,
            name='Win Rate',
            hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
        ),
        row=2, col=2
    )
    fig.add_vline(x=50, line=dict(color='gray', dash='dash'), row=2, col=2)

    # Update layout with detailed breakdown
    total_pnl = trades['realized_pnl_usd'].sum()
    total_funding = trades['funding_earned_usd'].sum()
    long_funding_total = trades['long_funding_earned_usd'].sum()
    short_funding_total = trades['short_funding_earned_usd'].sum()
    total_fees = trades['total_fees_usd'].sum()
    win_rate = (trades['realized_pnl_usd'] > 0).mean() * 100
    total_position_size = trades['position_size_usd'].sum()
    avg_position_size = trades['position_size_usd'].mean()

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Trading Summary</b> | "
                f"Trades: {len(trades)} | "
                f"Win Rate: {win_rate:.1f}%<br>"
                f"<span style='font-size:12px'>"
                f"Total P&L: <b>${total_pnl:.2f}</b> | "
                f"Funding: ${total_funding:.2f} (Long: ${long_funding_total:.2f} / Short: ${short_funding_total:.2f}) | "
                f"Fees: ${total_fees:.2f} | "
                f"Avg Size: ${avg_position_size:.2f}"
                f"</span>"
            ),
            x=0.5,
            xanchor='center'
        ),
        height=600,
        showlegend=False,
        margin=dict(l=50, r=50, t=100, b=50)
    )

    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative P&L ($)', row=1, col=1)
    fig.update_xaxes(title_text='P&L ($)', row=1, col=2)
    fig.update_xaxes(title_text='Duration (hours)', row=2, col=1)
    fig.update_yaxes(title_text='Count', row=2, col=1)
    fig.update_xaxes(title_text='Win Rate (%)', row=2, col=2)

    return fig


def create_dashboard(trades: pd.DataFrame, price_loader: PriceHistoryLoader, output_path: str, initial_capital: float = 93.33):
    """Create the full dashboard HTML"""

    print(f"\nCreating dashboard with {len(trades)} trades...")

    # Start HTML
    html_parts = ['''
<!DOCTYPE html>
<html>
<head>
    <title>Funding Rate Arbitrage - Trade Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .chart-container {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            padding: 10px;
        }
        .trade-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .trade-table th, .trade-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .trade-table th {
            background-color: #4CAF50;
            color: white;
        }
        .trade-table tr:hover {
            background-color: #f5f5f5;
        }
        .positive { color: green; font-weight: bold; }
        .negative { color: red; font-weight: bold; }
    </style>
</head>
<body>
<div class="container">
    <h1>Funding Rate Arbitrage - Trade Dashboard</h1>
''']

    # Create summary section
    print("  Creating summary charts...")
    summary_fig = create_summary_section(trades)
    html_parts.append('<div class="chart-container" id="summary">')
    html_parts.append(summary_fig.to_html(full_html=False, include_plotlyjs=False))
    html_parts.append('</div>')

    # Create trades table
    html_parts.append('<h2>All Trades</h2>')
    html_parts.append('<div style="overflow-x: auto;">')
    html_parts.append('<table class="trade-table" style="font-size: 12px;">')
    html_parts.append('''
        <tr>
            <th>#</th>
            <th>Symbol</th>
            <th>Entry</th>
            <th>Exit</th>
            <th>Hours</th>
            <th>Size ($)</th>
            <th>Size %</th>
            <th>Long Exch</th>
            <th>Short Exch</th>
            <th>Long Fund</th>
            <th>Short Fund</th>
            <th>Total Fund</th>
            <th>Long PnL</th>
            <th>Short PnL</th>
            <th>Total P&L</th>
            <th>P&L %</th>
        </tr>
    ''')

    for idx, trade in trades.iterrows():
        pnl_class = 'positive' if trade['realized_pnl_usd'] >= 0 else 'negative'
        pnl_sign = '+' if trade['realized_pnl_usd'] >= 0 else ''
        long_fund_class = 'positive' if trade['long_funding_earned_usd'] >= 0 else 'negative'
        short_fund_class = 'positive' if trade['short_funding_earned_usd'] >= 0 else 'negative'
        total_fund_class = 'positive' if trade['funding_earned_usd'] >= 0 else 'negative'
        size_pct = (trade['position_size_usd'] / initial_capital) * 100

        # Calculate long/short PnL from price movements + funding
        # CRITICAL: Use same coin_quantity for both legs (true delta neutral, matches backend)
        position_size = trade['position_size_usd']
        leverage = trade['leverage']
        # Get coin_quantity - same quantity for both legs
        if 'coin_quantity' in trade.index and pd.notna(trade['coin_quantity']):
            coin_quantity = trade['coin_quantity']
        else:
            avg_entry_price = (trade['entry_long_price'] + trade['entry_short_price']) / 2
            coin_quantity = (position_size * leverage) / avg_entry_price
        long_price_pnl = coin_quantity * (trade['exit_long_price'] - trade['entry_long_price'])
        short_price_pnl = coin_quantity * (trade['entry_short_price'] - trade['exit_short_price'])
        long_total_pnl = long_price_pnl + trade['long_funding_earned_usd']
        short_total_pnl = short_price_pnl + trade['short_funding_earned_usd']
        long_pnl_class = 'positive' if long_total_pnl >= 0 else 'negative'
        short_pnl_class = 'positive' if short_total_pnl >= 0 else 'negative'
        long_pnl_sign = '+' if long_total_pnl >= 0 else ''
        short_pnl_sign = '+' if short_total_pnl >= 0 else ''

        html_parts.append(f'''
        <tr>
            <td><a href="#trade-{idx}">{idx + 1}</a></td>
            <td><b>{trade['symbol']}</b></td>
            <td>{trade['entry_datetime'].strftime('%m-%d %H:%M')}</td>
            <td>{trade['exit_datetime'].strftime('%m-%d %H:%M')}</td>
            <td>{trade['hours_held']:.1f}</td>
            <td>${trade['position_size_usd']:.2f}</td>
            <td>{size_pct:.1f}%</td>
            <td>{trade['long_exchange']}</td>
            <td>{trade['short_exchange']}</td>
            <td class="{long_fund_class}">${trade['long_funding_earned_usd']:.3f}</td>
            <td class="{short_fund_class}">${trade['short_funding_earned_usd']:.3f}</td>
            <td class="{total_fund_class}">${trade['funding_earned_usd']:.3f}</td>
            <td class="{long_pnl_class}">{long_pnl_sign}${long_total_pnl:.3f}</td>
            <td class="{short_pnl_class}">{short_pnl_sign}${short_total_pnl:.3f}</td>
            <td class="{pnl_class}">{pnl_sign}${trade['realized_pnl_usd']:.3f}</td>
            <td class="{pnl_class}">{pnl_sign}{trade['realized_pnl_pct']:.2f}%</td>
        </tr>
        ''')

    html_parts.append('</table>')
    html_parts.append('</div>')


    # Create individual trade charts
    html_parts.append('<h2>Individual Trade Charts</h2>')

    for idx, trade in trades.iterrows():
        print(f"  Creating chart {idx + 1}/{len(trades)}: {trade['symbol']}...")

        fig = create_trade_chart(trade, price_loader, idx, initial_capital)
        if fig is not None:
            html_parts.append(f'<div class="chart-container" id="trade-{idx}">')
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
            html_parts.append('</div>')
        else:
            html_parts.append(f'<div class="chart-container" id="trade-{idx}">')
            html_parts.append(f'<p>Could not load price data for trade #{idx + 1} ({trade["symbol"]})</p>')
            html_parts.append('</div>')

    # Close HTML
    html_parts.append('''
</div>
</body>
</html>
''')

    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write('\n'.join(html_parts))

    print(f"\nDashboard saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create trade visualization dashboard')
    parser.add_argument('--trades-path', default='trades_inference.csv',
                        help='Path to trades CSV file')
    parser.add_argument('--price-history-path', default='data/production/price_history',
                        help='Path to price history directory')
    parser.add_argument('--output', default='visualizations/trades_dashboard.html',
                        help='Output HTML file path')
    parser.add_argument('--initial-capital', type=float, default=93.33,
                        help='Initial capital for calculating position size percentage')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of trades to visualize')

    args = parser.parse_args()

    # Load trades
    print(f"Loading trades from {args.trades_path}...")
    trades = load_trades(args.trades_path)
    print(f"  Loaded {len(trades)} closed trades")
    print(f"  Initial capital: ${args.initial_capital:.2f}")

    if args.limit:
        trades = trades.head(args.limit)
        print(f"  Limited to {len(trades)} trades")

    # Initialize price loader
    print(f"Initializing price loader from {args.price_history_path}...")
    price_loader = PriceHistoryLoader(args.price_history_path)

    # Create dashboard
    output_path = create_dashboard(trades, price_loader, args.output, args.initial_capital)

    print(f"\nDone! Open {output_path} in your browser to view the dashboard.")


if __name__ == '__main__':
    main()
