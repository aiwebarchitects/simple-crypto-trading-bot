#!/usr/bin/env python3
"""
Multi-Coin Tops Scanner - Top 100 Coins Significant Highs
Scans top 100 cryptocurrencies for best selling opportunities (tops) and saves to CSV
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

class MultiCoinTopsScanner:
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

    def find_highest_price_of_day(self, df: pd.DataFrame) -> Optional[Dict]:
        """Find the highest price in the last 24 hours"""
        if df is None or len(df) < 1:
            return None
        
        # Find the row with the highest price
        max_high_idx = df['high'].idxmax()
        highest_row = df.iloc[max_high_idx]
        
        # Calculate some basic metrics
        current_price = df.iloc[-1]['close']
        lowest_price = df['low'].min()
        
        # Calculate rise from daily low to daily high
        daily_range = ((highest_row['high'] - lowest_price) / lowest_price) * 100
        
        # Calculate decline from high to current
        decline_from_high = ((highest_row['high'] - current_price) / highest_row['high']) * 100
        
        opportunity = {
            'price': highest_row['high'],
            'timestamp': highest_row['timestamp'],
            'daily_range': daily_range,
            'decline_from_high': decline_from_high,
            'significance_score': daily_range,  # Use daily range as significance
            'volume': highest_row['volume']
        }
        
        return opportunity

    def scan_coin(self, coin: Dict, binance_symbols: Dict) -> Optional[Dict]:
        """Scan a single coin for top opportunities"""
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
            
            # Find highest price of the day
            opportunity = self.find_highest_price_of_day(df)
            if opportunity is None:
                return None
            
            current_price = df.iloc[-1]['close']
            
            result = {
                'coin_name': coin['name'],
                'symbol': symbol,
                'binance_symbol': binance_symbol,
                'market_cap_rank': coin['market_cap_rank'],
                'current_price': current_price,
                'best_top_price': opportunity['price'],
                'top_timestamp': opportunity['timestamp'],
                'daily_range_percent': opportunity['daily_range'],
                'decline_from_high_percent': opportunity['decline_from_high'],
                'significance_score': opportunity['significance_score'],
                'volume': opportunity['volume'],
                'price_decline': ((opportunity['price'] - current_price) / opportunity['price']) * 100
            }
            
            return result
            
        except Exception as e:
            return None

    def scan_all_coins(self, max_workers: int = 10) -> List[Dict]:
        """Scan all coins using threading"""
        print("🔍 Starting multi-coin tops scan...")
        
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
        
        print(f"📊 Scanning {len(coins)} coins for daily highs with {max_workers} threads...")
        
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
                        print(f"⚠️ {completed}/{len(coins)} - {coin['symbol']}: No data available")
                        
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
            filename = f"reports/crypto_daily_highs_{timestamp}.csv"
        
        if not results:
            print("❌ No results to save")
            return filename
        
        # Define CSV columns
        fieldnames = [
            'rank', 'coin_name', 'symbol', 'binance_symbol', 'market_cap_rank',
            'current_price', 'daily_high_price', 'high_timestamp',
            'daily_range_percent', 'decline_from_high_percent', 'significance_score', 
            'volume', 'price_decline'
        ]
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for i, result in enumerate(results, 1):
                    row = {
                        'rank': i,
                        'coin_name': result['coin_name'],
                        'symbol': result['symbol'],
                        'binance_symbol': result['binance_symbol'],
                        'market_cap_rank': result['market_cap_rank'],
                        'current_price': result['current_price'],
                        'daily_high_price': result['best_top_price'],
                        'high_timestamp': result['top_timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                        'daily_range_percent': result['daily_range_percent'],
                        'decline_from_high_percent': result['decline_from_high_percent'],
                        'significance_score': result['significance_score'],
                        'volume': result['volume'],
                        'price_decline': result['price_decline']
                    }
                    writer.writerow(row)
            
            print(f"💾 Results saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error saving CSV: {e}")
            return filename

    def display_top_opportunities(self, results: List[Dict], top_n: int = 20):
        """Display top selling opportunities"""
        if not results:
            print("❌ No daily highs found")
            return
        
        print(f"\n" + "=" * 100)
        print(f"📈 TOP {min(top_n, len(results))} CRYPTOCURRENCY DAILY HIGHS")
        print("=" * 100)
        
        print(f"{'#':<3} {'SYMBOL':<8} {'NAME':<20} {'DAILY HIGH':<12} {'RANGE%':<8} {'DECLINE%':<8} {'TIMESTAMP':<16}")
        print("-" * 100)
        
        for i, result in enumerate(results[:top_n], 1):
            timestamp_str = result['top_timestamp'].strftime('%m-%d %H:%M')
            print(f"{i:<3} {result['symbol']:<8} {result['coin_name'][:18]:<20} "
                  f"${result['best_top_price']:<11.6f} {result['daily_range_percent']:<7.1f} "
                  f"{result['decline_from_high_percent']:<7.1f} {timestamp_str:<16}")
        
        # Show best daily high details
        if results:
            best = results[0]
            print(f"\n📈 Highest Price of Day: ${best['best_top_price']:.6f} (Range: {best['daily_range_percent']:.1f}%)")
            print(f"   Token: {best['coin_name']} ({best['symbol']})")
            print(f"   Timestamp: {best['top_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Daily Range: {best['daily_range_percent']:.1f}% | Current Decline: {best['decline_from_high_percent']:.1f}%")

    def filter_noon_highs(self, results: List[Dict]) -> List[Dict]:
        """Filter daily highs that occur around 12 o'clock (11-13 range)"""
        noon_highs = []
        for result in results:
            hour = result['top_timestamp'].hour
            if 11 <= hour <= 13:  # 11 AM to 1 PM range
                noon_highs.append(result)
        
        return noon_highs

    def display_trading_insights(self, results: List[Dict]):
        """Display trading insights for tops"""
        if not results:
            return
        
        print(f"\n" + "=" * 100)
        print("📊 DAILY HIGHS ANALYSIS")
        print("=" * 100)
        
        # Noon highs analysis
        noon_highs = self.filter_noon_highs(results)
        if noon_highs:
            print(f"🕐 Noon Highs (11-13h): {len(noon_highs)} found")
            print("   Pattern: Coins hitting daily highs around lunch time")
        
        # High daily range
        high_range = [r for r in results if r['daily_range_percent'] > 10.0]
        if high_range:
            print(f"🎯 High Daily Range (>10%): {len(high_range)}")
            print("   These coins had significant daily movements")
        
        # Recent highs (last 6 hours)
        recent_highs = [r for r in results if r['top_timestamp'] > datetime.now() - timedelta(hours=6)]
        if recent_highs:
            print(f"🕕 Recent Highs (6h): {len(recent_highs)}")
            print("   Coins that hit their daily high recently")
        
        # Still near highs (less than 2% decline)
        near_highs = [r for r in results if r['decline_from_high_percent'] < 2.0]
        if near_highs:
            print(f"📈 Still Near Highs (<2% decline): {len(near_highs)}")
            print("   Coins still trading close to their daily high")

def main():
    scanner = MultiCoinTopsScanner()
    
    print("🚀 Multi-Coin Daily Highs Scanner Starting...")
    print("📈 Finding the highest price of the day for each coin")
    print("=" * 70)
    
    # Scan all coins
    results = scanner.scan_all_coins(max_workers=8)
    
    if results:
        # Display results
        scanner.display_top_opportunities(results, top_n=20)
        
        # Display trading insights
        scanner.display_trading_insights(results)
        
        # Save to CSV
        filename = scanner.save_to_csv(results)
        
        print(f"\n✅ Daily highs scan completed!")
        print(f"📊 Found {len(results)} daily highs")
        print(f"💾 Data saved to: {filename}")
        
        if results:
            best = results[0]
            print(f"\n📈 HIGHEST DAILY PRICE:")
            print(f"   {best['coin_name']} ({best['symbol']})")
            print(f"   Daily High: ${best['best_top_price']:.6f}")
            print(f"   Daily Range: {best['daily_range_percent']:.1f}%")
            print(f"   Time: {best['top_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Current Decline: {best['decline_from_high_percent']:.1f}% from high")
    else:
        print("❌ No daily highs found")

if __name__ == "__main__":
    main()
