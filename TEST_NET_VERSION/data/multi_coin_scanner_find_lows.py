#!/usr/bin/env python3
"""
Multi-Coin Scanner - Top 100 Coins Significant Lows
Scans top 100 cryptocurrencies for best opportunities and saves to CSV
use this code to place limit order with the prices from the results.
buy orders. with 15 usd each
max 10 positions.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import csv
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class MultiCoinScanner:
    def __init__(self):
        self.results = []
        self.lock = threading.Lock()
        self.stable_coins = {
            'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'USDD', 'FRAX', 
            'PYUSD', 'FDUSD', 'USDE', 'LUSD', 'GUSD', 'SUSD', 'USTC'
        }
        
    def get_top_100_coins(self) -> List[Dict]:
        """Get top 100 coins from CoinGecko"""
        try:
            print("📊 Fetching top 100 coins from CoinGecko...")
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 100,
                'page': 1,
                'sparkline': False
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            coins = response.json()
            
            # Filter out stablecoins
            filtered_coins = []
            for coin in coins:
                symbol = coin['symbol'].upper()
                if symbol not in self.stable_coins:
                    filtered_coins.append({
                        'id': coin['id'],
                        'symbol': symbol,
                        'name': coin['name'],
                        'market_cap_rank': coin['market_cap_rank']
                    })
            
            print(f"✅ Found {len(filtered_coins)} coins (excluding stablecoins)")
            return filtered_coins
            
        except Exception as e:
            print(f"❌ Error fetching coins: {e}")
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
            print(f"⚠️ Error getting Binance symbols: {e}")
            return {}

    def get_binance_minute_data(self, symbol: str, limit: int = 1440) -> Optional[pd.DataFrame]:
        """Get minute-by-minute data from Binance"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '1m',
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return None
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col])
            df['volume'] = pd.to_numeric(df['volume'])
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            return None

    def find_best_opportunity(self, df: pd.DataFrame, min_drop_percent: float = 1.0) -> Optional[Dict]:
        """Find the single best opportunity for a coin"""
        if df is None or len(df) < 40:
            return None
        
        best_opportunity = None
        best_score = 0
        
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
                    
                    if significance_score > best_score:
                        best_score = significance_score
                        best_opportunity = {
                            'price': current_low,
                            'timestamp': df.iloc[i]['timestamp'],
                            'drop_before': drop_before,
                            'recovery_after': recovery_after,
                            'significance_score': significance_score,
                            'volume': df.iloc[i]['volume']
                        }
        
        return best_opportunity

    def scan_coin(self, coin: Dict, binance_symbols: Dict) -> Optional[Dict]:
        """Scan a single coin for opportunities"""
        symbol = coin['symbol']
        
        # Check if coin is available on Binance
        if symbol not in binance_symbols:
            return None
        
        binance_symbol = binance_symbols[symbol]
        
        try:
            # Get data
            df = self.get_binance_minute_data(binance_symbol, 1440)
            if df is None:
                return None
            
            # Find best opportunity
            opportunity = self.find_best_opportunity(df, min_drop_percent=0.8)
            if opportunity is None:
                return None
            
            current_price = df.iloc[-1]['close']
            
            result = {
                'coin_name': coin['name'],
                'symbol': symbol,
                'binance_symbol': binance_symbol,
                'market_cap_rank': coin['market_cap_rank'],
                'current_price': current_price,
                'best_opportunity_price': opportunity['price'],
                'opportunity_timestamp': opportunity['timestamp'],
                'drop_percent': opportunity['drop_before'],
                'recovery_percent': opportunity['recovery_after'],
                'significance_score': opportunity['significance_score'],
                'volume': opportunity['volume'],
                'price_recovery': ((current_price - opportunity['price']) / opportunity['price']) * 100
            }
            
            return result
            
        except Exception as e:
            return None

    def scan_all_coins(self, max_workers: int = 10) -> List[Dict]:
        """Scan all coins using threading"""
        print("🔍 Starting multi-coin scan...")
        
        # Get top 100 coins
        coins = self.get_top_100_coins()
        if not coins:
            return []
        
        # Get Binance symbols
        binance_symbols = self.get_binance_symbol_mapping()
        if not binance_symbols:
            print("❌ Failed to get Binance symbols")
            return []
        
        results = []
        completed = 0
        
        print(f"📊 Scanning {len(coins)} coins with {max_workers} threads...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_coin = {
                executor.submit(self.scan_coin, coin, binance_symbols): coin 
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
                        print(f"✅ {completed}/{len(coins)} - {coin['symbol']}: Score {result['significance_score']:.1f}")
                    else:
                        print(f"⚠️ {completed}/{len(coins)} - {coin['symbol']}: No opportunity found")
                        
                except Exception as e:
                    print(f"❌ {completed}/{len(coins)} - {coin['symbol']}: Error - {e}")
                
                # Rate limiting
                time.sleep(0.1)
        
        # Sort by significance score
        results.sort(key=lambda x: x['significance_score'], reverse=True)
        
        return results

    def save_to_csv(self, results: List[Dict], filename: str = None) -> str:
        """Save results to CSV"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reports/crypto_opportunities_{timestamp}.csv"
        
        if not results:
            print("❌ No results to save")
            return filename
        
        # Define CSV columns
        fieldnames = [
            'rank', 'coin_name', 'symbol', 'binance_symbol', 'market_cap_rank',
            'current_price', 'best_opportunity_price', 'opportunity_timestamp',
            'drop_percent', 'recovery_percent', 'significance_score', 
            'volume', 'price_recovery'
        ]
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for i, result in enumerate(results, 1):
                    row = result.copy()
                    row['rank'] = i
                    row['opportunity_timestamp'] = result['opportunity_timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow(row)
            
            print(f"💾 Results saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error saving CSV: {e}")
            return filename

    def display_top_opportunities(self, results: List[Dict], top_n: int = 20):
        """Display top opportunities"""
        if not results:
            print("❌ No opportunities found")
            return
        
        print(f"\n" + "=" * 100)
        print(f"🎯 TOP {min(top_n, len(results))} CRYPTOCURRENCY OPPORTUNITIES")
        print("=" * 100)
        
        print(f"{'#':<3} {'SYMBOL':<8} {'NAME':<20} {'PRICE':<12} {'SCORE':<8} {'DROP%':<8} {'TIMESTAMP':<16}")
        print("-" * 100)
        
        for i, result in enumerate(results[:top_n], 1):
            timestamp_str = result['opportunity_timestamp'].strftime('%m-%d %H:%M')
            print(f"{i:<3} {result['symbol']:<8} {result['coin_name'][:18]:<20} "
                  f"${result['best_opportunity_price']:<11.6f} {result['significance_score']:<7.1f} "
                  f"{result['drop_percent']:<7.1f} {timestamp_str:<16}")
        
        # Show best opportunity details
        if results:
            best = results[0]
            print(f"\n🎯 Best Opportunity: ${best['best_opportunity_price']:.6f} (Score: {best['significance_score']:.1f})")
            print(f"   Token: {best['coin_name']} ({best['symbol']})")
            print(f"   Timestamp: {best['opportunity_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Drop: {best['drop_percent']:.1f}% | Recovery: {best['recovery_percent']:.1f}%")

def main():
    scanner = MultiCoinScanner()
    
    print("🚀 Multi-Coin Opportunity Scanner Starting...")
    print("=" * 70)
    
    # Scan all coins
    results = scanner.scan_all_coins(max_workers=8)
    
    if results:
        # Display results
        scanner.display_top_opportunities(results, top_n=20)
        
        # Save to CSV
        filename = scanner.save_to_csv(results)
        
        print(f"\n✅ Scan completed!")
        print(f"📊 Found {len(results)} opportunities")
        print(f"💾 Data saved to: {filename}")
        
        if results:
            best = results[0]
            print(f"\n🎯 BEST OPPORTUNITY:")
            print(f"   {best['coin_name']} ({best['symbol']})")
            print(f"   Price: ${best['best_opportunity_price']:.6f}")
            print(f"   Score: {best['significance_score']:.1f}")
            print(f"   Time: {best['opportunity_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("❌ No opportunities found")

if __name__ == "__main__":
    main()
