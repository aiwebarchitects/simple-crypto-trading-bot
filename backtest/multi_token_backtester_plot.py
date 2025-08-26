# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# VERSION: Optimized version - Run backtest first, then plot only top 3 results

import json
import os
import sys
import logging
from datetime import datetime, timedelta
import threading
import time
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for threading
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# --- Configuration ---
RESULTS_FILE = 'backtest_results.json'
LOG_FILE = 'logs/multi_backtester.log'
DATA_CACHE_DIR = 'data_cache'
PLOTS_DIR = 'plots'
BACKTEST_DAYS = 4 # Using 4-day for recent analysis before trading day
INITIAL_CAPITAL = 100.0
MAX_CONCURRENT_BACKTESTS = 3

# Optimization Ranges (Must be defined here)
OPT_SHORT_RANGE = (5, 25, 2)    # Start, Stop (exclusive), Step
OPT_LONG_RANGE = (20, 50, 3)
OPT_SIGNAL_RANGE = (4, 18, 2)

# --- Setup Logging ---
os.makedirs('logs', exist_ok=True)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MultiTokenBacktester')

# --- Helper Functions ---
def load_json(filepath, default=None):
    """Loads JSON data from a file."""
    default_val = default if default is not None else {}
    if not os.path.exists(filepath):
        logger.debug(f"File not found: {filepath}. Returning default.")
        return default_val
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
            if not content:
                logger.warning(f"File is empty: {filepath}. Returning default.")
                return default_val
            return json.loads(content)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {filepath}. Returning default.")
        return default_val
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}. Returning default.")
        return default_val

def save_json(data, filepath):
    """Saves data to a JSON file."""
    try:
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, default=str)  # Convert non-serializable objects to strings
        logger.debug(f"Successfully saved data to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {e}")
        return False

# --- Binance API Functions ---
def get_top_100_coins():
    """Fetch top 100 coins by market cap from CoinGecko and get their Binance symbols."""
    logger.info("Fetching top 100 coins by market cap...")
    try:
        # Get top 100 coins from CoinGecko
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 100,
            'page': 1,
            'sparkline': False
        }
        headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        coins_data = response.json()
        
        # Get available trading pairs from Binance
        binance_url = "https://api.binance.com/api/v3/exchangeInfo"
        binance_response = requests.get(binance_url, timeout=30)
        binance_response.raise_for_status()
        binance_data = binance_response.json()
        
        # Extract USDT trading pairs
        usdt_symbols = set()
        for symbol_info in binance_data['symbols']:
            if (symbol_info['status'] == 'TRADING' and 
                symbol_info['quoteAsset'] == 'USDT' and
                symbol_info['baseAsset'] != 'USDT'):
                usdt_symbols.add(symbol_info['baseAsset'])
        
        # Match CoinGecko coins with Binance symbols
        matched_tokens = []
        for coin in coins_data:
            symbol = coin['symbol'].upper()
            # Handle special cases for symbol mapping
            if symbol == 'BTC':
                binance_symbol = 'BTC'
            elif symbol == 'ETH':
                binance_symbol = 'ETH'
            elif symbol == 'BNB':
                binance_symbol = 'BNB'
            else:
                binance_symbol = symbol
            
            if binance_symbol in usdt_symbols:
                matched_tokens.append({
                    'symbol': binance_symbol,
                    'name': coin['name'],
                    'binance_pair': f"{binance_symbol}USDT",
                    'market_cap_rank': coin['market_cap_rank']
                })
        
        logger.info(f"Found {len(matched_tokens)} coins available on Binance from top 100")
        return matched_tokens
        
    except Exception as e:
        logger.error(f"Error fetching top 100 coins: {e}")
        return []

def fetch_historical_data(token_symbol, binance_pair, days):
    """Fetch or load cached historical price data using Binance API."""
    cache_filename = os.path.join(DATA_CACHE_DIR, f"{token_symbol}_{days}d.csv")
    logger.info(f"[{token_symbol}] Attempting fetch/load ({days} days)")

    refresh_interval = 3600 # 1 hour cache validity
    if os.path.exists(cache_filename):
        try:
            cache_age = time.time() - os.path.getmtime(cache_filename)
            if cache_age < refresh_interval:
                logger.info(f"[{token_symbol}] Loading fresh data from cache")
                return pd.read_csv(cache_filename, index_col='timestamp', parse_dates=True)
            else:
                 logger.info(f"[{token_symbol}] Cache older than {refresh_interval/3600:.1f}h. Refetching.")
        except Exception as e:
            logger.warning(f"[{token_symbol}] Error reading cache: {e}. Refetching.")

    logger.info(f"[{token_symbol}] Fetching fresh data from Binance...")
    
    # Add delay to respect API rate limits
    time.sleep(0.5)  # 500ms delay between requests
    
    try:
        # Calculate start time for the required days plus buffer
        end_time = int(time.time() * 1000)  # Binance uses milliseconds
        start_time = end_time - (days + 3) * 24 * 60 * 60 * 1000  # Buffer for calculations

        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': binance_pair,
            'interval': '1h',  # 1-hour intervals
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        klines_data = response.json()

        if not klines_data:
             logger.warning(f"[{token_symbol}] No price data from Binance.")
             return pd.DataFrame()

        # Convert klines data to DataFrame
        df = pd.DataFrame(klines_data, columns=[
            'timestamp_ms', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert to proper data types
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
        df['price'] = df['close'].astype(float)
        df.set_index('timestamp', inplace=True)
        df = df[['price']]

        # Filter to the exact period only AFTER ensuring enough data for calculation
        cutoff_time_dt = datetime.fromtimestamp(end_time / 1000) - timedelta(days=days)
        df = df[df.index >= cutoff_time_dt]

        if df.empty:
            logger.warning(f"[{token_symbol}] Dataframe empty after time filter.")
            return df

        df.to_csv(cache_filename)
        logger.info(f"[{token_symbol}] Fetched & cached {len(df)} points")
        return df

    except requests.exceptions.HTTPError as e:
        logger.error(f"[{token_symbol}] HTTP Error fetching: {e.response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"[{token_symbol}] Network Error fetching: {e}")
    except Exception as e:
        logger.error(f"[{token_symbol}] Unexpected error fetching data: {e}", exc_info=False)

    if os.path.exists(cache_filename):
         logger.warning(f"[{token_symbol}] Fetch failed, using potentially stale cache.")
         try: return pd.read_csv(cache_filename, index_col='timestamp', parse_dates=True)
         except Exception as e: logger.error(f"[{token_symbol}] Error reading stale cache: {e}")

    return pd.DataFrame()

# --- Indicator Calculation (Internal Function) ---
def _calculate_macd(df, short_ema, long_ema, signal_ema):
    """Internal function to calculate MACD for a given DataFrame and params."""
    if df.empty or 'price' not in df.columns:
        logger.debug("MACD calc skipped: df empty/no price col")
        return pd.DataFrame() # Return empty if cannot calculate

    # Check minimum required points more accurately based on longest period
    required_points = max(short_ema, long_ema, signal_ema) * 2 # Still an estimate, may need adjustment
    if len(df) < required_points:
        logger.warning(f"Not enough data points ({len(df)}) for MACD({short_ema},{long_ema},{signal_ema}). Req ~{required_points}. Skipping combination.")
        return pd.DataFrame() # Return empty if not enough data

    try:
        df = df.copy()
        df['ema_short'] = df['price'].ewm(span=short_ema, adjust=False).mean()
        df['ema_long'] = df['price'].ewm(span=long_ema, adjust=False).mean()
        df['macd_line'] = df['ema_short'] - df['ema_long']
        df['signal_line'] = df['macd_line'].ewm(span=signal_ema, adjust=False).mean()

        prev_macd = df['macd_line'].shift(1)
        prev_signal = df['signal_line'].shift(1)
        df['signal'] = np.where((df['macd_line'] > df['signal_line']) & (prev_macd <= prev_signal), 1, 0)
        df['signal'] = np.where((df['macd_line'] < df['signal_line']) & (prev_macd >= prev_signal), -1, df['signal'])

        # Drop initial rows where calculations might be NaN
        df.dropna(subset=['price', 'macd_line', 'signal_line', 'signal'], inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error in _calculate_macd: {e}", exc_info=False)
        return pd.DataFrame()


# --- Simplified Simulation (Internal Function) ---
def _run_simulation(df_with_signals):
    """Internal function to run simulation for a DataFrame with signals."""
    # Ensure required columns exist after potential dropna in MACD calc
    if df_with_signals.empty or not all(col in df_with_signals.columns for col in ['signal', 'price']):
        logger.debug("Simulation skipped: df empty or missing signal/price after MACD calc.")
        return INITIAL_CAPITAL, 0.0, []

    capital = INITIAL_CAPITAL
    position = 0.0
    trades = 0
    trade_log = []
    
    for index, row in df_with_signals.iterrows():
        signal = row['signal']
        price = row['price']
        # Basic validity check for price
        if not isinstance(price, (int, float)) or price <= 0:
            logger.warning(f"Skipping row due to invalid price: {price} at {index}")
            continue

        if signal == 1 and position == 0 and capital > 0:
            position = capital / price
            capital = 0.0
            trades += 1
            trade_log.append({'timestamp': index, 'action': 'BUY', 'price': price, 'position': position})
        elif signal == -1 and position > 0:
            capital = position * price
            position = 0.0
            trades += 1
            trade_log.append({'timestamp': index, 'action': 'SELL', 'price': price, 'capital': capital})

    if position > 0 and not df_with_signals.empty:
        last_price = df_with_signals['price'].iloc[-1]
        # Check last price validity
        if isinstance(last_price, (int, float)) and last_price > 0:
            capital = position * last_price
            trade_log.append({'timestamp': df_with_signals.index[-1], 'action': 'FINAL_SELL', 'price': last_price, 'capital': capital})
        else:
            logger.warning(f"Could not liquidate position at end, invalid last price: {last_price}. Final capital might be inaccurate.")
            capital = 0.0

    profit_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100 if INITIAL_CAPITAL > 0 else 0
    return capital, profit_pct, trade_log

# --- Plotting Functions ---
def create_backtest_plot(symbol, df_with_signals, trade_log, best_params, profit_pct):
    """Create a comprehensive backtest plot for a token."""
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
        plt.subplots_adjust(hspace=0.3)  # Add space between subplots
        
        # Plot 1: Price and EMAs with trade signals
        ax1.plot(df_with_signals.index, df_with_signals['price'], label='Price', color='black', linewidth=1.5)
        ax1.plot(df_with_signals.index, df_with_signals['ema_short'], label=f'EMA {best_params["short_ema"]}', color='blue', alpha=0.7)
        ax1.plot(df_with_signals.index, df_with_signals['ema_long'], label=f'EMA {best_params["long_ema"]}', color='red', alpha=0.7)
        
        # Mark buy/sell signals
        buy_labeled = False
        sell_labeled = False
        for trade in trade_log:
            if trade['action'] == 'BUY':
                label = 'Buy' if not buy_labeled else ""
                ax1.scatter(trade['timestamp'], trade['price'], color='green', marker='^', s=100, zorder=5, label=label)
                buy_labeled = True
            elif trade['action'] == 'SELL':
                label = 'Sell' if not sell_labeled else ""
                ax1.scatter(trade['timestamp'], trade['price'], color='red', marker='v', s=100, zorder=5, label=label)
                sell_labeled = True
        
        ax1.set_title(f'{symbol} - Price Action & Trading Signals (Profit: {profit_pct:.2f}%)')
        ax1.set_ylabel('Price (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: MACD
        ax2.plot(df_with_signals.index, df_with_signals['macd_line'], label='MACD Line', color='blue')
        ax2.plot(df_with_signals.index, df_with_signals['signal_line'], label='Signal Line', color='red')
        ax2.fill_between(df_with_signals.index, df_with_signals['macd_line'] - df_with_signals['signal_line'], 
                        0, alpha=0.3, color='green', where=(df_with_signals['macd_line'] > df_with_signals['signal_line']))
        ax2.fill_between(df_with_signals.index, df_with_signals['macd_line'] - df_with_signals['signal_line'], 
                        0, alpha=0.3, color='red', where=(df_with_signals['macd_line'] <= df_with_signals['signal_line']))
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title(f'MACD ({best_params["short_ema"]}, {best_params["long_ema"]}, {best_params["signal_ema"]})')
        ax2.set_ylabel('MACD')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Portfolio value over time
        portfolio_values = []
        current_capital = INITIAL_CAPITAL
        current_position = 0
        
        for index, row in df_with_signals.iterrows():
            # Check if there's a trade at this timestamp
            trade_at_time = next((t for t in trade_log if t['timestamp'] == index), None)
            if trade_at_time:
                if trade_at_time['action'] == 'BUY':
                    current_position = trade_at_time['position']
                    current_capital = 0
                elif trade_at_time['action'] in ['SELL', 'FINAL_SELL']:
                    current_capital = trade_at_time['capital']
                    current_position = 0
            
            # Calculate current portfolio value
            if current_position > 0:
                portfolio_value = current_position * row['price']
            else:
                portfolio_value = current_capital
            
            portfolio_values.append(portfolio_value)
        
        ax3.plot(df_with_signals.index, portfolio_values, label='Portfolio Value', color='green', linewidth=2)
        ax3.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
        ax3.set_title('Portfolio Value Over Time')
        ax3.set_ylabel('Value (USDT)')
        ax3.set_xlabel('Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax3.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(PLOTS_DIR, f"best_backtest_result_{symbol}_{timestamp}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[{symbol}] Plot saved to {plot_filename}")
        return plot_filename
        
    except Exception as e:
        logger.error(f"[{symbol}] Error creating plot: {e}")
        return None

def create_summary_plot(results_data):
    """Create a summary plot showing all tokens' performance."""
    try:
        # Prepare data for plotting
        symbols = []
        profits = []
        market_ranks = []
        
        for symbol, data in results_data.items():
            symbols.append(symbol)
            profits.append(data.get('backtest_profit_pct', 0))
            market_ranks.append(data.get('market_cap_rank', 999))
        
        # Sort by profit
        sorted_data = sorted(zip(symbols, profits, market_ranks), key=lambda x: x[1], reverse=True)
        symbols, profits, market_ranks = zip(*sorted_data)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Profit by token
        colors = ['green' if p > 0 else 'red' for p in profits]
        bars = ax1.bar(range(len(symbols)), profits, color=colors, alpha=0.7)
        ax1.set_title('Backtest Results - Profit/Loss by Token')
        ax1.set_ylabel('Profit/Loss (%)')
        ax1.set_xticks(range(len(symbols)))
        ax1.set_xticklabels(symbols, rotation=45, ha='right')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, profit) in enumerate(zip(bars, profits)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                    f'{profit:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        
        # Plot 2: Profit vs Market Cap Rank
        scatter = ax2.scatter(market_ranks, profits, c=profits, cmap='RdYlGn', alpha=0.7, s=60)
        ax2.set_title('Profit vs Market Cap Rank')
        ax2.set_xlabel('Market Cap Rank')
        ax2.set_ylabel('Profit/Loss (%)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax2, label='Profit/Loss (%)')
        
        # Add annotations for top performers
        for i, (symbol, profit, rank) in enumerate(zip(symbols[:5], profits[:5], market_ranks[:5])):
            ax2.annotate(symbol, (rank, profit), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(PLOTS_DIR, f"summary_results_{timestamp}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary plot saved to {plot_filename}")
        return plot_filename
        
    except Exception as e:
        logger.error(f"Error creating summary plot: {e}")
        return None

# --- Optimization and Backtesting Core ---
results_lock = threading.Lock()

def optimize_and_backtest_token(token, all_results):
    """Fetches data, optimizes MACD params, runs simulation for one token."""
    symbol = token.get('symbol')
    binance_pair = token.get('binance_pair')
    if not symbol or not binance_pair:
        logger.warning(f"Skipping token due to missing symbol/binance_pair: {token}")
        return

    thread_start_time = time.time()
    logger.info(f"--- Starting Opt/Backtest for {symbol} ---")
    df_base = fetch_historical_data(symbol, binance_pair, BACKTEST_DAYS)
    if df_base.empty:
        logger.error(f"[{symbol}] Failed to get data. Skipping.")
        return

    optimization_results = []
    tested_count = 0
    # --- Optimization Loop ---
    logger.info(f"[{symbol}] Optimizing parameters...")
    for short_ema in range(*OPT_SHORT_RANGE):
        for long_ema in range(*OPT_LONG_RANGE):
            if short_ema >= long_ema: continue
            for signal_ema in range(*OPT_SIGNAL_RANGE):
                tested_count += 1
                # Calculate indicators for this specific combination
                df_indicators = _calculate_macd(df_base, short_ema, long_ema, signal_ema)
                # Run simulation only if indicators were calculated successfully
                if not df_indicators.empty and 'signal' in df_indicators.columns:
                    final_capital, profit_pct, trade_log = _run_simulation(df_indicators)
                    optimization_results.append({
                        'short_ema': short_ema, 'long_ema': long_ema, 'signal_ema': signal_ema,
                        'final_capital': final_capital, 'profit_percent': profit_pct,
                        'trade_log': trade_log, 'df_indicators': df_indicators
                    })

    if not optimization_results:
        logger.error(f"[{symbol}] Optimization failed: No valid simulation results found.")
        return

    # --- Find Best Parameters ---
    results_df = pd.DataFrame([{k: v for k, v in result.items() if k not in ['trade_log', 'df_indicators']} 
                              for result in optimization_results])
    results_df.sort_values(by='final_capital', ascending=False, inplace=True)
    best_result_idx = results_df.index[0]
    best_result = optimization_results[best_result_idx]

    best_params = {
        'short_ema': int(best_result['short_ema']),
        'long_ema': int(best_result['long_ema']),
        'signal_ema': int(best_result['signal_ema'])
    }
    best_profit_pct = best_result['profit_percent']
    thread_end_time = time.time()

    logger.info(f"[{symbol}] Optimization Complete ({tested_count} combos in {thread_end_time - thread_start_time:.2f}s). Best Params: {best_params}, Profit: {best_profit_pct:.2f}%")

    # --- Store Final Result (WITHOUT creating plot yet) ---
    token_result_entry = {
        **best_params,
        "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backtest_profit_pct": round(best_profit_pct, 2),
        "binance_pair": binance_pair,
        "market_cap_rank": token.get('market_cap_rank', 'N/A'),
        "best_result_data": best_result  # Store the complete result for plotting later
    }

    with results_lock:
        all_results[symbol] = token_result_entry
    logger.info(f"+++ Saved Best Result for {symbol} +++")

def create_plots_for_top_performers(results_data, top_n=3):
    """Create plots only for the top N performing tokens."""
    logger.info(f"Creating plots for top {top_n} performers...")
    
    # Sort results by profit percentage
    sorted_results = sorted(results_data.items(), key=lambda x: x[1].get('backtest_profit_pct', 0), reverse=True)
    top_performers = sorted_results[:top_n]
    
    plot_files = []
    for symbol, data in top_performers:
        logger.info(f"Creating plot for {symbol} (Profit: {data.get('backtest_profit_pct', 0):.2f}%)")
        
        # Get the stored result data
        best_result = data.get('best_result_data')
        if not best_result:
            logger.warning(f"No result data found for {symbol}, skipping plot")
            continue
            
        best_params = {
            'short_ema': data['short_ema'],
            'long_ema': data['long_ema'],
            'signal_ema': data['signal_ema']
        }
        
        plot_filename = create_backtest_plot(
            symbol, 
            best_result['df_indicators'], 
            best_result['trade_log'], 
            best_params, 
            best_result['profit_percent']
        )
        
        if plot_filename:
            plot_files.append(plot_filename)
            # Update the result with plot filename
            data['plot_file'] = plot_filename
    
    logger.info(f"Created {len(plot_files)} plots for top performers")
    return plot_files

# --- Main Execution ---
def main():
    logger.info(f"===== Starting Multi-Token Backtester - Optimized Version (Binance API | {BACKTEST_DAYS}-Day Data) =====")
    
    # Fetch top 100 coins directly (no tokens.json dependency)
    logger.info("Fetching top 100 coins...")
    tokens = get_top_100_coins()
    if not tokens:
        logger.error("Failed to fetch top 100 coins. Exiting.")
        return
    logger.info(f"Found {len(tokens)} tokens to process.")

    current_results = {}
    threads = []
    active_thread_objects = []
    thread_limiter = threading.Semaphore(MAX_CONCURRENT_BACKTESTS)
    start_time_main = time.time()

    # Process tokens in batches to avoid hanging
    processed_count = 0
    batch_size = MAX_CONCURRENT_BACKTESTS
    
    logger.info("=== PHASE 1: Running backtests for all tokens ===")
    while processed_count < len(tokens):
        batch_tokens = tokens[processed_count:processed_count + batch_size]
        batch_threads = []
        
        logger.info(f"Processing batch {processed_count//batch_size + 1}: tokens {processed_count + 1}-{min(processed_count + len(batch_tokens), len(tokens))}")
        
        # Start threads for this batch
        for token in batch_tokens:
            symbol = token.get('symbol', 'UNKNOWN')
            logger.info(f"Starting thread for {symbol}...")
            thread = threading.Thread(
                target=optimize_and_backtest_token,
                args=(token, current_results),
                daemon=True, name=f"OptBacktest-{symbol}"
            )
            batch_threads.append((thread, symbol))
            thread.start()
        
        # Wait for all threads in this batch to complete
        for thread, symbol in batch_threads:
            try:
                thread.join(timeout=1800)  # 30 minutes timeout per token
                if thread.is_alive():
                    logger.warning(f"Thread for {symbol} timed out (30 min).")
                else:
                    logger.info(f"Thread for {symbol} completed.")
            except Exception as e:
                logger.error(f"Error joining thread for {symbol}: {e}")
        
        processed_count += len(batch_tokens)
        
        # Small delay between batches to avoid overwhelming APIs
        if processed_count < len(tokens):
            logger.info(f"Batch completed. Waiting 2 seconds before next batch...")
            time.sleep(2)

    end_time_main = time.time()
    logger.info(f"All threads completed processing (Total time: {end_time_main - start_time_main:.2f}s).")

    if not current_results: 
        logger.warning(f"No results were generated. Not saving {RESULTS_FILE}.")
        return
    
    # Save results
    if save_json(current_results, RESULTS_FILE): 
        logger.info(f"Saved {len(current_results)} results to {RESULTS_FILE}")
    else: 
        logger.error(f"Failed to save results to {RESULTS_FILE}")
        return

    logger.info("=== PHASE 2: Creating plots for top 3 performers ===")
    
    # Create plots only for top 3 performers
    plot_files = create_plots_for_top_performers(current_results, top_n=3)
    
    # Create summary plot
    summary_plot_file = create_summary_plot(current_results)
    if summary_plot_file:
        logger.info(f"Summary plot created: {summary_plot_file}")

    # Clean up stored result data to save space
    for symbol, data in current_results.items():
        if 'best_result_data' in data:
            del data['best_result_data']
    
    # Save final results without the large data objects
    if save_json(current_results, RESULTS_FILE): 
        logger.info(f"Final results saved to {RESULTS_FILE}")

    logger.info("===== Multi-Token Backtester with Optimized Plotting Finished =====")
    logger.info(f"Created {len(plot_files)} plots for top performers")

if __name__ == "__main__":
    main()
