# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# VERSION: Binance API version for top 100 coins by market cap

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

# --- Configuration ---
TOKENS_FILE = 'tokens.json'
RESULTS_FILE = 'backtest_results.json'
LOG_FILE = 'logs/multi_backtester.log'
DATA_CACHE_DIR = 'data_cache'
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
            json.dump(data, f, indent=4)
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
        if e.response.status_code == 429:  # Rate limit exceeded
            logger.warning(f"[{token_symbol}] Rate limit hit, waiting 5 seconds...")
            time.sleep(5)
            # Try once more after delay
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                klines_data = response.json()
                if klines_data:
                    df = pd.DataFrame(klines_data, columns=[
                        'timestamp_ms', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
                    df['price'] = df['close'].astype(float)
                    df.set_index('timestamp', inplace=True)
                    df = df[['price']]
                    cutoff_time_dt = datetime.fromtimestamp(end_time / 1000) - timedelta(days=days)
                    df = df[df.index >= cutoff_time_dt]
                    if not df.empty:
                        df.to_csv(cache_filename)
                        logger.info(f"[{token_symbol}] Fetched & cached {len(df)} points (retry)")
                        return df
            except Exception as retry_e:
                logger.error(f"[{token_symbol}] Retry failed: {retry_e}")
        else:
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
        return INITIAL_CAPITAL, 0.0

    capital = INITIAL_CAPITAL
    position = 0.0
    trades = 0
    for index, row in df_with_signals.iterrows():
        signal = row['signal']
        price = row['price']
        # Basic validity check for price
        if not isinstance(price, (int, float)) or price <= 0:
            logger.warning(f"Skipping row due to invalid price: {price} at {index}")
            continue

        if signal == 1 and position == 0 and capital > 0:
            position = capital / price; capital = 0.0; trades += 1
        elif signal == -1 and position > 0:
            capital = position * price; position = 0.0; trades += 1

    if position > 0 and not df_with_signals.empty:
        last_price = df_with_signals['price'].iloc[-1]
        # Check last price validity
        if isinstance(last_price, (int, float)) and last_price > 0:
            capital = position * last_price
        else:
            logger.warning(f"Could not liquidate position at end, invalid last price: {last_price}. Final capital might be inaccurate.")
            # Decide how to handle this: keep capital=0 or use last valid price? For now, keep capital=0
            capital = 0.0


    profit_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100 if INITIAL_CAPITAL > 0 else 0
    return capital, profit_pct

# --- Optimization and Backtesting Core (Now within this script) ---
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
                    final_capital, profit_pct = _run_simulation(df_indicators)
                    optimization_results.append({
                        'short_ema': short_ema, 'long_ema': long_ema, 'signal_ema': signal_ema,
                        'final_capital': final_capital, 'profit_percent': profit_pct
                    })
                # else: # Optional: log if a combination was skipped due to calc error
                #    logger.debug(f"[{symbol}] Skipped combo ({short_ema},{long_ema},{signal_ema}) due to indicator calc issue.")


    if not optimization_results:
        logger.error(f"[{symbol}] Optimization failed: No valid simulation results found.")
        return

    # --- Find Best Parameters ---
    results_df = pd.DataFrame(optimization_results)
    # Sort by final capital primarily, maybe add profit_percent as secondary?
    results_df.sort_values(by='final_capital', ascending=False, inplace=True)
    best_result = results_df.iloc[0]

    best_params = {
        'short_ema': int(best_result['short_ema']),
        'long_ema': int(best_result['long_ema']),
        'signal_ema': int(best_result['signal_ema'])
    }
    best_profit_pct = best_result['profit_percent']
    thread_end_time = time.time()

    logger.info(f"[{symbol}] Optimization Complete ({tested_count} combos in {thread_end_time - thread_start_time:.2f}s). Best Params: {best_params}, Profit: {best_profit_pct:.2f}%")

    # --- Store Final Result ---
    token_result_entry = {
        **best_params,
        "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backtest_profit_pct": round(best_profit_pct, 2),
        "binance_pair": binance_pair,
        "market_cap_rank": token.get('market_cap_rank', 'N/A')
    }

    with results_lock:
        all_results[symbol] = token_result_entry
    logger.info(f"+++ Saved Best Result for {symbol} +++")


# --- Main Execution ---
def main():
    logger.info(f"===== Starting Multi-Token Backtester (Binance API | {BACKTEST_DAYS}-Day Data) =====")
    
    # Check if tokens.json exists, if not create it with top 100 coins
    if not os.path.exists(TOKENS_FILE):
        logger.info("tokens.json not found. Fetching top 100 coins...")
        top_coins = get_top_100_coins()
        if top_coins:
            save_json(top_coins, TOKENS_FILE)
            logger.info(f"Created {TOKENS_FILE} with {len(top_coins)} coins")
        else:
            logger.error("Failed to fetch top 100 coins. Exiting.")
            return
    
    tokens = load_json(TOKENS_FILE, [])
    if not tokens:
        logger.error(f"No tokens found in {TOKENS_FILE}. Exiting.")
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

    if not current_results: logger.warning(f"No results were generated. Not saving {RESULTS_FILE}.")
    elif save_json(current_results, RESULTS_FILE): logger.info(f"Saved {len(current_results)} results to {RESULTS_FILE}")
    else: logger.error(f"Failed to save results to {RESULTS_FILE}")

    logger.info("===== Multi-Token Backtester Finished =====")

if __name__ == "__main__":
    main()
