#!/usr/bin/env python3
"""
Multi-Token Backtester for Low/High Reversal Strategy
Backtests the exact algorithm used in the trading bot:
- LONG: Buy 0.15% above significant lows
- SHORT: Sell 0.15% below daily highs
- Risk management with stop loss and trailing stops
"""

import json
import os
import sys
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
RESULTS_FILE = 'results.json'
LOG_DIR = 'logs'
LOG_FILE = os.path.join(LOG_DIR, 'low_high_reversal_backtester.log')
DATA_CACHE_DIR = 'data_cache'
BACKTEST_DAYS = 7
INITIAL_CAPITAL = 100.0
MAX_CONCURRENT_BACKTESTS = 5

# Parameter optimization ranges
LONG_OFFSET_RANGE = (0.05, 0.30, 0.05)  # 0.05% to 0.30% in 0.05% steps
SHORT_OFFSET_RANGE = (0.05, 0.30, 0.05)  # 0.05% to 0.30% in 0.05% steps
STOP_LOSS_RANGE = (3.0, 8.0, 1.0)  # 3% to 8% in 1% steps
TRAILING_STOP_RANGE = (1.0, 4.0, 0.5)  # 1% to 4% in 0.5% steps

# Setup logging - create directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('LowHighReversalBacktester')

class LowHighReversalBacktester:
    def __init__(self):
        self.stable_coins = {
            'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'USDD', 'FRAX', 
            'PYUSD', 'FDUSD', 'USDE', 'LUSD', 'GUSD', 'SUSD', 'USTC'
        }
        
    def get_top_coins(self, limit: int = 50) -> List[Dict]:
        """Get top coins from CoinGecko"""
        try:
            logger.info(f"Fetching top {limit} coins from CoinGecko...")
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': limit,
                'page': 1,
                'sparkline': False
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            coins = response.json()
            
            # Filter out stablecoins and get Binance symbols
            binance_symbols = self.get_binance_symbol_mapping()
            filtered_coins = []
            
            for coin in coins:
                symbol = coin['symbol'].upper()
                if symbol not in self.stable_coins and symbol in binance_symbols:
                    filtered_coins.append({
                        'symbol': symbol,
                        'name': coin['name'],
                        'binance_pair': binance_symbols[symbol],
                        'market_cap_rank': coin['market_cap_rank']
                    })
            
            logger.info(f"Found {len(filtered_coins)} tradeable coins")
            return filtered_coins
            
        except Exception as e:
            logger.error(f"Error fetching coins: {e}")
            return []

    def get_binance_symbol_mapping(self) -> Dict[str, str]:
        """Get Binance symbol mapping"""
        try:
            url = "https://api.binance.com/api/v3/exchangeInfo"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            symbol_map = {}
            for symbol_info in data['symbols']:
                if symbol_info['quoteAsset'] == 'USDT' and symbol_info['status'] == 'TRADING':
                    base_asset = symbol_info['baseAsset']
                    symbol_map[base_asset] = symbol_info['symbol']
            
            return symbol_map
            
        except Exception as e:
            logger.error(f"Error getting Binance symbols: {e}")
            return {}

    def fetch_historical_data(self, symbol: str, binance_pair: str, days: int) -> pd.DataFrame:
        """Fetch historical data from Binance"""
        cache_filename = os.path.join(DATA_CACHE_DIR, f"{symbol}_{days}d.csv")
        
        # Check cache
        if os.path.exists(cache_filename):
            cache_age = time.time() - os.path.getmtime(cache_filename)
            if cache_age < 3600:  # 1 hour cache
                try:
                    df = pd.read_csv(cache_filename, index_col='timestamp', parse_dates=True)
                    if not df.empty:
                        logger.info(f"Loaded {len(df)} cached data points for {symbol}")
                        return df
                except Exception as e:
                    logger.warning(f"Error reading cache for {symbol}: {e}")
        
        try:
            logger.info(f"Fetching {days} days of data for {symbol} ({binance_pair})")
            
            # Use hourly data instead of minute data for better reliability
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': binance_pair,
                'interval': '1h',  # Use hourly data
                'limit': days * 24 + 24  # Get enough data points
            }
            
            # Add delay to avoid rate limiting
            time.sleep(0.2)
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            logger.info(f"Received {len(data)} data points for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp_ms', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert data types
            df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Remove any rows with NaN values
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            
            if df.empty:
                logger.warning(f"No valid data after cleaning for {symbol}")
                return pd.DataFrame()
            
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']].sort_index()
            
            # Keep only the last N days
            if len(df) > days * 24:
                df = df.tail(days * 24)
            
            # Cache the data
            try:
                df.to_csv(cache_filename)
                logger.info(f"Cached {len(df)} data points for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to cache data for {symbol}: {e}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching data for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def find_significant_lows(self, df: pd.DataFrame, min_drop_percent: float = 1.0) -> List[Dict]:
        """Find significant price lows using the same algorithm as the scanner"""
        if len(df) < 40:
            return []
        
        opportunities = []
        
        # Look for significant drops
        for i in range(20, len(df) - 20):
            current_low = df.iloc[i]['low']
            
            # Check larger window for significant drops
            before_high = df.iloc[i-20:i]['high'].max()
            after_high = df.iloc[i+1:i+21]['high'].max()
            
            # Calculate drop percentages
            drop_before = ((before_high - current_low) / before_high) * 100
            recovery_after = ((after_high - current_low) / current_low) * 100
            
            # Check for significance
            if drop_before >= min_drop_percent and recovery_after >= min_drop_percent:
                # Check if this is a local minimum
                window_low = df.iloc[i-10:i+11]['low'].min()
                if current_low == window_low:
                    significance_score = drop_before + recovery_after
                    
                    opportunities.append({
                        'timestamp': df.index[i],
                        'price': current_low,
                        'drop_before': drop_before,
                        'recovery_after': recovery_after,
                        'significance_score': significance_score,
                        'type': 'low'
                    })
        
        # Sort by significance score
        opportunities.sort(key=lambda x: x['significance_score'], reverse=True)
        return opportunities

    def find_daily_highs(self, df: pd.DataFrame) -> List[Dict]:
        """Find daily highs using the same algorithm as the scanner"""
        if len(df) < 1:
            return []
        
        opportunities = []
        
        # Group by day and find daily highs
        df_daily = df.groupby(df.index.date).agg({
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        for date, row in df_daily.iterrows():
            daily_high = row['high']
            daily_low = row['low']
            
            # Calculate daily range
            daily_range = ((daily_high - daily_low) / daily_low) * 100
            
            if daily_range > 1.0:  # At least 1% daily range
                # Find the exact timestamp of the high
                day_data = df[df.index.date == date]
                high_idx = day_data['high'].idxmax()
                
                opportunities.append({
                    'timestamp': high_idx,
                    'price': daily_high,
                    'daily_range': daily_range,
                    'significance_score': daily_range,
                    'type': 'high'
                })
        
        # Sort by significance score
        opportunities.sort(key=lambda x: x['significance_score'], reverse=True)
        return opportunities

    def simulate_strategy(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Simulate the low/high reversal strategy"""
        if df.empty:
            return self.get_empty_result()
        
        # Extract parameters
        long_offset = params['long_offset_percent']
        short_offset = params['short_offset_percent']
        stop_loss = params['stop_loss_percent']
        trailing_stop = params['trailing_stop_percent']
        
        # Find opportunities
        low_opportunities = self.find_significant_lows(df)
        high_opportunities = self.find_daily_highs(df)
        
        # Combine and sort by timestamp
        all_opportunities = low_opportunities + high_opportunities
        all_opportunities.sort(key=lambda x: x['timestamp'])
        
        # Simulation variables
        capital = INITIAL_CAPITAL
        position = 0.0
        position_type = None  # 'long' or 'short'
        entry_price = 0.0
        max_profit = 0.0
        trades = []
        
        # Process each opportunity
        for opp in all_opportunities:
            opp_time = opp['timestamp']
            opp_price = opp['price']
            opp_type = opp['type']
            
            # Get current market data at this time
            try:
                current_data = df.loc[opp_time:]
                if current_data.empty:
                    continue
            except:
                continue
            
            # If we have a position, check for exit conditions first
            if position != 0:
                for timestamp, row in current_data.iterrows():
                    current_price = row['close']
                    
                    if position_type == 'long':
                        profit_pct = ((current_price - entry_price) / entry_price) * 100
                    else:  # short
                        profit_pct = ((entry_price - current_price) / entry_price) * 100
                    
                    # Update max profit for trailing stop
                    if profit_pct > max_profit:
                        max_profit = profit_pct
                    
                    # Check stop loss
                    if profit_pct <= -stop_loss:
                        # Close position - stop loss
                        if position_type == 'long':
                            capital = position * current_price
                        else:
                            capital = position + (entry_price - current_price) * (position / entry_price)
                        
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': timestamp,
                            'type': position_type,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'profit_pct': profit_pct,
                            'exit_reason': 'stop_loss'
                        })
                        
                        position = 0.0
                        position_type = None
                        max_profit = 0.0
                        break
                    
                    # Check trailing stop
                    if max_profit > 2.0 and profit_pct < (max_profit - trailing_stop):
                        # Close position - trailing stop
                        if position_type == 'long':
                            capital = position * current_price
                        else:
                            capital = position + (entry_price - current_price) * (position / entry_price)
                        
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': timestamp,
                            'type': position_type,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'profit_pct': profit_pct,
                            'exit_reason': 'trailing_stop'
                        })
                        
                        position = 0.0
                        position_type = None
                        max_profit = 0.0
                        break
            
            # If no position, look for entry
            if position == 0 and capital > 0:
                if opp_type == 'low':
                    # LONG entry: buy 0.15% above the low
                    target_price = opp_price * (1 + long_offset / 100)
                    
                    # Check if price reaches our target
                    future_data = df.loc[opp_time:]
                    for timestamp, row in future_data.iterrows():
                        if row['low'] <= target_price <= row['high']:
                            # Enter long position
                            position = capital / target_price
                            capital = 0.0
                            position_type = 'long'
                            entry_price = target_price
                            entry_time = timestamp
                            max_profit = 0.0
                            break
                
                elif opp_type == 'high':
                    # SHORT entry: sell 0.15% below the high
                    target_price = opp_price * (1 - short_offset / 100)
                    
                    # Check if price reaches our target
                    future_data = df.loc[opp_time:]
                    for timestamp, row in future_data.iterrows():
                        if row['low'] <= target_price <= row['high']:
                            # Enter short position
                            position = capital  # Capital amount for short
                            capital = 0.0
                            position_type = 'short'
                            entry_price = target_price
                            entry_time = timestamp
                            max_profit = 0.0
                            break
        
        # Close any remaining position at the end
        if position != 0:
            final_price = df.iloc[-1]['close']
            if position_type == 'long':
                capital = position * final_price
                profit_pct = ((final_price - entry_price) / entry_price) * 100
            else:
                capital = position + (entry_price - final_price) * (position / entry_price)
                profit_pct = ((entry_price - final_price) / entry_price) * 100
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': df.index[-1],
                'type': position_type,
                'entry_price': entry_price,
                'exit_price': final_price,
                'profit_pct': profit_pct,
                'exit_reason': 'end_of_period'
            })
        
        # Calculate results
        total_return = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        winning_trades = [t for t in trades if t['profit_pct'] > 0]
        losing_trades = [t for t in trades if t['profit_pct'] <= 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t['profit_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit_pct'] for t in losing_trades]) if losing_trades else 0
        
        return {
            'final_capital': capital,
            'total_return_pct': total_return,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades and avg_loss != 0 else float('inf'),
            'max_drawdown': self.calculate_max_drawdown(trades),
            'trades': trades
        }

    def calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown"""
        if not trades:
            return 0.0
        
        capital_curve = [INITIAL_CAPITAL]
        running_capital = INITIAL_CAPITAL
        
        for trade in trades:
            running_capital *= (1 + trade['profit_pct'] / 100)
            capital_curve.append(running_capital)
        
        peak = capital_curve[0]
        max_dd = 0.0
        
        for capital in capital_curve:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak * 100
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd

    def get_empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'final_capital': INITIAL_CAPITAL,
            'total_return_pct': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win_pct': 0.0,
            'avg_loss_pct': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'trades': []
        }

    def optimize_parameters(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Optimize parameters for a single coin"""
        logger.info(f"Optimizing parameters for {symbol}")
        
        best_result = None
        best_score = -float('inf')
        best_params = None
        
        param_combinations = []
        
        # Generate all parameter combinations
        long_offsets = np.arange(*LONG_OFFSET_RANGE)
        short_offsets = np.arange(*SHORT_OFFSET_RANGE)
        stop_losses = np.arange(*STOP_LOSS_RANGE)
        trailing_stops = np.arange(*TRAILING_STOP_RANGE)
        
        for long_offset in long_offsets:
            for short_offset in short_offsets:
                for stop_loss in stop_losses:
                    for trailing_stop in trailing_stops:
                        param_combinations.append({
                            'long_offset_percent': round(long_offset, 2),
                            'short_offset_percent': round(short_offset, 2),
                            'stop_loss_percent': round(stop_loss, 1),
                            'trailing_stop_percent': round(trailing_stop, 1)
                        })
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations for {symbol}")
        
        # Test each combination
        for i, params in enumerate(param_combinations):
            if i % 50 == 0:
                logger.info(f"{symbol}: Testing combination {i+1}/{len(param_combinations)}")
            
            result = self.simulate_strategy(df, params)
            
            # Score based on total return, win rate, and profit factor
            score = (
                result['total_return_pct'] * 0.4 +
                result['win_rate'] * 0.3 +
                min(result['profit_factor'], 10) * 0.2 +
                max(0, 10 - result['max_drawdown']) * 0.1
            )
            
            if score > best_score:
                best_score = score
                best_result = result.copy()
                best_params = params.copy()
        
        if best_result:
            best_result['best_params'] = best_params
            best_result['optimization_score'] = best_score
            logger.info(f"{symbol}: Best score {best_score:.2f}, Return {best_result['total_return_pct']:.2f}%, Win rate {best_result['win_rate']:.1f}%")
        
        return best_result or self.get_empty_result()

    def backtest_coin(self, coin: Dict) -> Dict:
        """Backtest a single coin"""
        symbol = coin['symbol']
        binance_pair = coin['binance_pair']
        
        try:
            # Fetch data
            df = self.fetch_historical_data(symbol, binance_pair, BACKTEST_DAYS)
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Optimize parameters
            result = self.optimize_parameters(symbol, df)
            
            # Add coin info
            result.update({
                'symbol': symbol,
                'name': coin['name'],
                'binance_pair': binance_pair,
                'market_cap_rank': coin['market_cap_rank'],
                'backtest_period_days': BACKTEST_DAYS,
                'last_run': datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error backtesting {symbol}: {e}")
            return None

def main():
    """Main execution function"""
    logger.info("=" * 80)
    logger.info("Starting Low/High Reversal Strategy Backtester")
    logger.info("=" * 80)
    
    backtester = LowHighReversalBacktester()
    
    # Get coins to test
    coins = backtester.get_top_coins(30)  # Test top 30 coins
    if not coins:
        logger.error("No coins found for backtesting")
        return
    
    logger.info(f"Backtesting {len(coins)} coins over {BACKTEST_DAYS} days")
    
    # Run backtests
    results = []
    completed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_BACKTESTS) as executor:
        # Submit all tasks
        future_to_coin = {
            executor.submit(backtester.backtest_coin, coin): coin 
            for coin in coins
        }
        
        # Process completed tasks
        for future in as_completed(future_to_coin):
            completed += 1
            coin = future_to_coin[future]
            
            try:
                result = future.result()
                if result:
                    results.append(result)
                    logger.info(f"✅ {completed}/{len(coins)} - {coin['symbol']}: "
                              f"Return {result['total_return_pct']:.2f}%, "
                              f"Win Rate {result['win_rate']:.1f}%, "
                              f"Trades {result['total_trades']}")
                else:
                    logger.warning(f"⚠️ {completed}/{len(coins)} - {coin['symbol']}: No result")
                    
            except Exception as e:
                logger.error(f"❌ {completed}/{len(coins)} - {coin['symbol']}: Error - {e}")
    
    if not results:
        logger.error("No results generated")
        return
    
    # Sort results by multiple criteria
    def sort_key(r):
        return (
            r['win_rate'],  # Primary: win rate
            r['total_return_pct'],  # Secondary: total return
            -r['max_drawdown']  # Tertiary: lower drawdown is better
        )
    
    results.sort(key=sort_key, reverse=True)
    
    # Get top 10 results
    top_10 = results[:10]
    
    # Save only top 10 results with essential info
    top_10_clean = []
    for result in top_10:
        clean_result = {
            'symbol': result['symbol'],
            'name': result['name'],
            'total_return_pct': result['total_return_pct'],
            'win_rate': result['win_rate'],
            'total_trades': result['total_trades'],
            'profit_factor': result['profit_factor'],
            'max_drawdown': result['max_drawdown'],
            'best_params': result['best_params']
        }
        top_10_clean.append(clean_result)
    
    output_data = {
        'backtest_info': {
            'strategy': 'Low/High Reversal',
            'backtest_period_days': BACKTEST_DAYS,
            'total_coins_tested': len(coins),
            'successful_backtests': len(results),
            'timestamp': datetime.now().isoformat()
        },
        'top_10_results': top_10_clean
    }
    
    # Ensure backtest directory exists
    os.makedirs('backtest', exist_ok=True)
    
    try:
        with open(RESULTS_FILE, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        logger.info(f"Results saved to {RESULTS_FILE}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    # Display summary
    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST SUMMARY - TOP 10 RESULTS")
    logger.info("=" * 80)
    logger.info(f"{'Rank':<4} {'Symbol':<8} {'Return%':<8} {'WinRate%':<9} {'Trades':<7} {'PF':<6} {'MaxDD%':<7}")
    logger.info("-" * 80)
    
    for i, result in enumerate(top_10, 1):
        logger.info(f"{i:<4} {result['symbol']:<8} {result['total_return_pct']:<7.1f} "
                   f"{result['win_rate']:<8.1f} {result['total_trades']:<7} "
                   f"{result['profit_factor']:<5.1f} {result['max_drawdown']:<7.1f}")
    
    if top_10:
        best = top_10[0]
        logger.info(f"\n🏆 BEST PERFORMER: {best['symbol']}")
        logger.info(f"   Return: {best['total_return_pct']:.2f}%")
        logger.info(f"   Win Rate: {best['win_rate']:.1f}%")
        logger.info(f"   Total Trades: {best['total_trades']}")
        logger.info(f"   Profit Factor: {best['profit_factor']:.2f}")
        logger.info(f"   Max Drawdown: {best['max_drawdown']:.2f}%")
        logger.info(f"   Best Parameters: {best['best_params']}")
    
    logger.info("\n✅ Backtest completed successfully!")

if __name__ == "__main__":
    main()
