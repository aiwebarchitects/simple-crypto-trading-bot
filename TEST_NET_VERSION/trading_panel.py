#!/usr/bin/env python3
"""
Trading Panel - Terminal-based Interface for Simple Buy 24h Low Bot
Python interface with tabs for monitoring and controlling the system.

"""

import os
import sys
import time
import threading
import subprocess
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import csv

import example_utils
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from trading_scanner import TradingScanner
from fetch_available_coins import fetch_available_coins
from default_config import Config, get_trading_params, get_risk_params, get_timing_params

class TradingPanel:
    def __init__(self):
        # Load configuration from default_config
        self.config = Config()
        
        # System settings from config
        self.debug_mode = self.config.DEBUG_MODE
        self.max_positions = self.config.MAX_POSITIONS
        self.position_size_usd = self.config.POSITION_SIZE_USD
        self.long_offset_percent = self.config.LONG_OFFSET_PERCENT
        self.short_offset_percent = self.config.SHORT_OFFSET_PERCENT
        
        # Risk management settings from config
        self.stop_loss_percent = self.config.STOP_LOSS_PERCENT
        self.trailing_stop_percent = self.config.TRAILING_STOP_PERCENT
        self.trailing_stop_min_profit = self.config.TRAILING_STOP_MIN_PROFIT
        self.profit_take_threshold = self.config.PROFIT_TAKE_THRESHOLD
        self.profit_take_drop = self.config.PROFIT_TAKE_DROP
        
        # Backtest results and optimized parameters
        self.backtest_results = {}
        self.optimized_params = {}
        
        # Initialize system state early (needed for logging)
        self.system_running = False
        self.last_trading_cycle = None
        self.last_position_check = None
        self.system_logs = []
        self.scanner_output = {}
        self.position_history = {}
        self.current_tab = "home"
        
        # Check and load backtest results
        self.check_and_load_backtest_results()
        
        # Timing settings from config
        self.trading_cycle_minutes = self.config.TRADING_CYCLE_MINUTES
        self.position_check_seconds = self.config.POSITION_CHECK_SECONDS
        self.order_max_age_hours = self.config.ORDER_MAX_AGE_HOURS
        
        # Setup Hyperliquid connection using config
        self.address, self.info, self.exchange = example_utils.setup(self.config.API_URL, skip_ws=True)
        
        # Available coins - dynamically fetched from Hyperliquid
        self.log("🔄 Fetching available coins from Hyperliquid...")
        self.available_coins = fetch_available_coins()
        
        # Fallback to configured coins if fetch fails
        if not self.available_coins:
            self.log("⚠️ Failed to fetch coins from Hyperliquid, using fallback list")
            self.available_coins = self.config.FALLBACK_COINS
        else:
            self.log(f"✅ Successfully fetched {len(self.available_coins)} coins from Hyperliquid")
        
        # Initialize trading scanner
        scanner_config = {
            'max_positions': self.max_positions,
            'position_size_usd': self.position_size_usd,
            'long_offset_percent': self.long_offset_percent,
            'short_offset_percent': self.short_offset_percent
        }
        self.scanner = TradingScanner(self.exchange, self.info, self.available_coins, scanner_config)
        self.scanner.set_log_callback(self.log)
        
        self.log("🚀 Trading Panel Initialized")
        self.log(f"📊 Max Positions: {self.max_positions}")
        self.log(f"💰 Position Size: ${self.position_size_usd} USDC each")
        self.log(f"🔗 Account: {self.address}")

    def log(self, message: str):
        """Add message to system logs"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.system_logs.append(log_entry)
        if len(self.system_logs) > self.config.MAX_LOG_ENTRIES:
            self.system_logs.pop(0)

    def check_and_load_backtest_results(self):
        """Check backtest results age and load optimized parameters"""
        backtest_file = 'backtest/results.json'
        
        try:
            # Check if backtest results file exists
            if not os.path.exists(backtest_file):
                self.log("⚠️ No backtest results found, running fresh backtest...")
                self.run_backtest()
                return
            
            # Check file age
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(backtest_file))
            age_hours = (datetime.now() - file_mod_time).total_seconds() / 3600
            
            if age_hours > 24:
                self.log(f"⚠️ Backtest results are {age_hours:.1f} hours old (>24h), running fresh backtest...")
                self.run_backtest()
                return
            
            # Load fresh results
            self.load_backtest_results()
            self.log(f"✅ Using fresh backtest results ({age_hours:.1f} hours old)")
            
        except Exception as e:
            self.log(f"❌ Error checking backtest results: {e}")
            self.log("⚠️ Using default parameters")

    def run_backtest(self):
        """Run the backtest script to generate fresh results"""
        try:
            self.log("🔄 Running backtest to generate fresh results...")
            
            # Change to backtest directory and run the script
            result = subprocess.run(
                ['python3', 'multi_token_backtester_low_high_reversal.py'],
                cwd='backtest',
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            if result.returncode == 0:
                self.log("✅ Backtest completed successfully")
                self.load_backtest_results()
            else:
                self.log(f"❌ Backtest failed: {result.stderr}")
                self.log("⚠️ Using default parameters")
                
        except subprocess.TimeoutExpired:
            self.log("❌ Backtest timed out after 30 minutes")
            self.log("⚠️ Using default parameters")
        except Exception as e:
            self.log(f"❌ Error running backtest: {e}")
            self.log("⚠️ Using default parameters")

    def load_backtest_results(self):
        """Load backtest results and extract optimized parameters"""
        try:
            with open('backtest/results.json', 'r') as f:
                self.backtest_results = json.load(f)
            
            # Extract optimized parameters for each coin
            self.optimized_params = {}
            
            if 'top_10_results' in self.backtest_results:
                for result in self.backtest_results['top_10_results']:
                    symbol = result['symbol']
                    best_params = result.get('best_params', {})
                    
                    self.optimized_params[symbol] = {
                        'long_offset_percent': best_params.get('long_offset_percent', self.config.LONG_OFFSET_PERCENT),
                        'short_offset_percent': best_params.get('short_offset_percent', self.config.SHORT_OFFSET_PERCENT),
                        'stop_loss_percent': best_params.get('stop_loss_percent', self.config.STOP_LOSS_PERCENT),
                        'trailing_stop_percent': best_params.get('trailing_stop_percent', self.config.TRAILING_STOP_PERCENT),
                        'total_return_pct': result.get('total_return_pct', 0),
                        'win_rate': result.get('win_rate', 0)
                    }
                
                self.log(f"📊 Loaded optimized parameters for {len(self.optimized_params)} coins")
                
                # Log top performers
                top_3 = self.backtest_results['top_10_results'][:3]
                for i, result in enumerate(top_3, 1):
                    self.log(f"🏆 #{i}: {result['symbol']} - Return: {result['total_return_pct']:.1f}%, Win Rate: {result['win_rate']:.1f}%")
            
        except Exception as e:
            self.log(f"❌ Error loading backtest results: {e}")
            self.optimized_params = {}

    def get_optimized_params_for_coin(self, coin: str) -> Dict:
        """Get optimized parameters for a specific coin, fallback to defaults"""
        if coin in self.optimized_params:
            params = self.optimized_params[coin]
            self.log(f"📈 Using optimized params for {coin}: SL={params['stop_loss_percent']:.1f}%, TS={params['trailing_stop_percent']:.1f}%")
            return params
        else:
            # Return default parameters
            default_params = {
                'long_offset_percent': self.config.LONG_OFFSET_PERCENT,
                'short_offset_percent': self.config.SHORT_OFFSET_PERCENT,
                'stop_loss_percent': self.config.STOP_LOSS_PERCENT,
                'trailing_stop_percent': self.config.TRAILING_STOP_PERCENT
            }
            self.log(f"⚙️ Using default params for {coin} (no backtest data)")
            return default_params

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')

    def get_account_info(self) -> Dict:
        """Get current account information"""
        try:
            user_state = self.info.user_state(self.address)
            return user_state
        except Exception as e:
            self.log(f"❌ Error getting account info: {e}")
            return {}

    def get_open_orders(self) -> List[Dict]:
        """Get all open orders"""
        try:
            open_orders = self.info.open_orders(self.address)
            return open_orders
        except Exception as e:
            self.log(f"❌ Error getting open orders: {e}")
            return []

    def get_active_positions(self) -> List[Dict]:
        """Get all active positions"""
        try:
            user_state = self.get_account_info()
            active_positions = []
            
            for position in user_state.get("assetPositions", []):
                pos_data = position.get("position", {})
                size = float(pos_data.get("szi", 0))
                
                if size != 0:  # Only active positions
                    coin = pos_data.get('coin')
                    entry_px = float(pos_data.get('entryPx', 0))
                    unrealized_pnl = float(pos_data.get('unrealizedPnl', 0))
                    leverage = float(pos_data.get('leverage', {}).get('value', 1))
                    
                    # Calculate current price and profit percentage
                    current_price = self.get_current_price(coin)
                    if current_price and entry_px:
                        # Base price change percentage
                        if size > 0:  # Long position
                            price_change_percent = ((current_price - entry_px) / entry_px) * 100
                        else:  # Short position
                            price_change_percent = ((entry_px - current_price) / entry_px) * 100
                        
                        # Apply leverage to get actual profit percentage
                        profit_percent = price_change_percent * leverage
                    else:
                        profit_percent = 0
                    
                    position_info = {
                        'coin': coin,
                        'size': size,
                        'entry_px': entry_px,
                        'current_price': current_price,
                        'unrealized_pnl': unrealized_pnl,
                        'profit_percent': profit_percent,
                        'leverage': leverage,
                        'side': 'LONG' if size > 0 else 'SHORT'
                    }
                    
                    active_positions.append(position_info)
                    
                    # Update position history for trailing stop
                    self.update_position_history(coin, profit_percent)
            
            return active_positions
        except Exception as e:
            self.log(f"❌ Error getting active positions: {e}")
            return []

    def get_current_price(self, coin: str) -> float:
        """Get current price for a coin"""
        try:
            all_mids = self.info.all_mids()
            
            # Try different response formats
            if isinstance(all_mids, dict):
                # Format: {"BTC": "43000.0", "ETH": "2500.0"}
                if coin in all_mids:
                    return float(all_mids[coin])
                    
            elif isinstance(all_mids, list):
                # Format: [{"coin": "BTC", "px": "43000.0"}, ...]
                for mid in all_mids:
                    if isinstance(mid, dict) and mid.get('coin') == coin:
                        return float(mid.get('px', 0))
            
            # If not found, try meta info
            meta = self.info.meta()
            if meta and 'universe' in meta:
                for asset in meta['universe']:
                    if asset.get('name') == coin:
                        # Get mark price from meta
                        mark_px = asset.get('markPx')
                        if mark_px:
                            return float(mark_px)
            
            return 0
        except Exception as e:
            self.log(f"❌ Error getting price for {coin}: {e}")
            return 0

    def update_position_history(self, coin: str, profit_percent: float):
        """Update position history for trailing stop loss"""
        if coin not in self.position_history:
            self.position_history[coin] = {
                'max_profit': profit_percent,
                'last_update': datetime.now()
            }
        else:
            # Update max profit if current profit is higher
            if profit_percent > self.position_history[coin]['max_profit']:
                self.position_history[coin]['max_profit'] = profit_percent
            self.position_history[coin]['last_update'] = datetime.now()

    def check_position_risk_management(self):
        """Check positions for stop loss and take profit using optimized parameters"""
        try:
            positions = self.get_active_positions()
            
            for position in positions:
                coin = position['coin']
                profit_percent = position['profit_percent']
                size = position['size']
                
                # Get optimized parameters for this coin
                params = self.get_optimized_params_for_coin(coin)
                stop_loss_percent = params['stop_loss_percent']
                trailing_stop_percent = params['trailing_stop_percent']
                
                # Debug logging
                self.log(f"🔍 Checking {coin}: Current {profit_percent:.2f}% (SL: {stop_loss_percent:.1f}%, TS: {trailing_stop_percent:.1f}%)")
                
                # Check stop loss using optimized parameter
                if profit_percent <= -stop_loss_percent:
                    self.log(f"🛑 Stop Loss triggered for {coin}: {profit_percent:.2f}% (threshold: {stop_loss_percent:.1f}%)")
                    self.close_position(coin, "Stop Loss")
                    continue
                
                # Check trailing stop loss using optimized parameter
                if coin in self.position_history:
                    max_profit = self.position_history[coin]['max_profit']
                    trailing_stop_level = max_profit - trailing_stop_percent
                    
                    self.log(f"🔍 {coin} - Max: {max_profit:.2f}%, Current: {profit_percent:.2f}%, Stop Level: {trailing_stop_level:.2f}%")
                    
                    # Trailing stop only triggers if:
                    # 1. We had some profit (max_profit > trailing_stop_min_profit)
                    # 2. Current profit drops below trailing stop level
                    self.log(f"🔍 {coin} Trigger Check: Max>{self.trailing_stop_min_profit}? {max_profit > self.trailing_stop_min_profit}, Current<Stop? {profit_percent}<{trailing_stop_level} = {profit_percent < trailing_stop_level}")
                    
                    if max_profit > self.trailing_stop_min_profit and profit_percent < trailing_stop_level:
                        self.log(f"📉 TRAILING STOP TRIGGERED for {coin}: Max {max_profit:.2f}% -> Current {profit_percent:.2f}% (Stop at {trailing_stop_level:.2f}%)")
                        self.close_position(coin, "Trailing Stop")
                        continue
                    
                    # Take profit if reached threshold and dropped
                    if max_profit >= self.profit_take_threshold and profit_percent <= self.profit_take_drop:
                        self.log(f"💰 Take Profit triggered for {coin}: Max {max_profit:.2f}% -> Current {profit_percent:.2f}%")
                        self.close_position(coin, "Take Profit")
                        continue
                        
        except Exception as e:
            self.log(f"❌ Error in risk management: {e}")

    def close_position(self, coin: str, reason: str):
        """Close a position"""
        try:
            self.log(f"🔄 Closing {coin} position: {reason}")
            
            # Use the simple market_close method like in executer/close_position.py
            result = self.exchange.market_close(coin)
            
            if result.get("status") == "ok":
                self.log(f"✅ {coin} position closed successfully: {reason}")
                # Remove from position history
                if coin in self.position_history:
                    del self.position_history[coin]
            else:
                self.log(f"❌ Failed to close {coin} position: {result}")
                    
        except Exception as e:
            self.log(f"❌ Error closing {coin} position: {e}")

    def run_scanner(self, scanner_type: str) -> List[Dict]:
        """Run scanner and return results"""
        try:
            if scanner_type == "low":
                script = 'data/multi_coin_scanner_find_lows.py'
                csv_prefix = 'crypto_opportunities_'
            else:
                script = 'data/multi_coin_tops_scanner.py'
                csv_prefix = 'crypto_daily_highs_'
            
            result = subprocess.run(['python3', script], capture_output=True, text=True)
            
            if result.returncode != 0:
                self.log(f"❌ {scanner_type} scanner failed: {result.stderr}")
                return []
            
            # Find the most recent CSV file
            csv_files = [f for f in os.listdir('reports') if f.startswith(csv_prefix) and f.endswith('.csv')]
            if not csv_files:
                return []
            
            latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join('reports', x)))
            latest_file = os.path.join('reports', latest_file)
            
            # Read results
            opportunities = []
            with open(latest_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    opportunities.append(dict(row))
            
            return opportunities[:10]  # Return top 10
            
        except Exception as e:
            self.log(f"❌ Error running {scanner_type} scanner: {e}")
            return []

    def run_trading_cycle(self):
        """Run one complete trading cycle using the scanner module"""
        try:
            # Get current positions and orders
            positions = self.get_active_positions()
            orders = self.get_open_orders()
            
            # Use the scanner module to check capacity and scan
            result = self.scanner.check_capacity_and_scan(positions, orders)
            
            # Store scanner output for display
            self.scanner_output = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'low_opportunities': result.get('low_opportunities', []),
                'high_opportunities': result.get('high_opportunities', []),
                'capacity_info': {
                    'current_count': result.get('current_count', 0),
                    'available_slots': result.get('available_slots', 0),
                    'capacity_full': result.get('capacity_full', False)
                }
            }
            
            self.last_trading_cycle = datetime.now()
            
        except Exception as e:
            self.log(f"❌ Error in trading cycle: {e}")

    def start_background_tasks(self):
        """Start background monitoring tasks"""
        def position_monitor():
            while self.system_running:
                try:
                    self.check_position_risk_management()
                    self.last_position_check = datetime.now()
                    time.sleep(self.position_check_seconds)
                except Exception as e:
                    self.log(f"❌ Position monitor error: {e}")
                    time.sleep(self.position_check_seconds)

        def trading_cycle_scheduler():
            while self.system_running:
                try:
                    self.run_trading_cycle()
                    time.sleep(self.trading_cycle_minutes * 60)
                except Exception as e:
                    self.log(f"❌ Trading cycle error: {e}")
                    time.sleep(60)  # Wait 1 minute before retry

        # Start background threads
        threading.Thread(target=position_monitor, daemon=True).start()
        threading.Thread(target=trading_cycle_scheduler, daemon=True).start()
        
        self.log("🚀 Background tasks started")

    def start_system(self):
        """Start the trading system"""
        self.system_running = True
        
        # Immediately run capacity check and scan on start
        self.log("🚀 Starting system - running immediate capacity check and scan...")
        self.run_trading_cycle()
        
        # Start background tasks
        self.start_background_tasks()
        self.log("✅ Trading system started")

    def stop_system(self):
        """Stop the trading system"""
        self.system_running = False
        self.log("⏹️ Trading system stopped")

    def display_home(self):
        """Display home tab"""
        print("=" * 80)
        print("🏠 HOME - TRADING SYSTEM STATUS")
        print("=" * 80)
        
        # System status
        status = "🟢 RUNNING" if self.system_running else "🔴 STOPPED"
        print(f"System Status: {status}")
        
        # Account info
        account_info = self.get_account_info()
        account_value = account_info.get('marginSummary', {}).get('accountValue', 0)
        print(f"Account Value: ${float(account_value):.2f}")
        
        # Positions and orders count
        positions = self.get_active_positions()
        orders = self.get_open_orders()
        print(f"Active Positions: {len(positions)}")
        print(f"Open Orders: {len(orders)}")
        
        # Last updates
        last_cycle = self.last_trading_cycle.strftime('%H:%M:%S') if self.last_trading_cycle else 'Never'
        last_check = self.last_position_check.strftime('%H:%M:%S') if self.last_position_check else 'Never'
        print(f"Last Trading Cycle: {last_cycle}")
        print(f"Last Position Check: {last_check}")
        
        # Backtest information
        print(f"\n📊 BACKTEST STATUS:")
        print("-" * 40)
        if self.backtest_results:
            backtest_info = self.backtest_results.get('backtest_info', {})
            timestamp = backtest_info.get('timestamp', 'Unknown')
            if timestamp != 'Unknown':
                try:
                    backtest_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    age_hours = (datetime.now() - backtest_time.replace(tzinfo=None)).total_seconds() / 3600
                    print(f"Backtest Age: {age_hours:.1f} hours")
                except:
                    print(f"Backtest Time: {timestamp}")
            
            coins_tested = backtest_info.get('total_coins_tested', 0)
            successful = backtest_info.get('successful_backtests', 0)
            print(f"Coins Tested: {coins_tested} | Successful: {successful}")
            print(f"Optimized Parameters: {len(self.optimized_params)} coins")
            
            # Show top 3 performers
            if 'top_10_results' in self.backtest_results:
                top_3 = self.backtest_results['top_10_results'][:3]
                print("Top Performers:")
                for i, result in enumerate(top_3, 1):
                    print(f"  {i}. {result['symbol']}: {result['total_return_pct']:.1f}% return, {result['win_rate']:.1f}% win rate")
        else:
            print("No backtest data available - using default parameters")
        
        print("\n📋 RECENT LOGS:")
        print("-" * 40)
        for log in self.system_logs[-8:]:  # Show fewer logs to make room for backtest info
            print(log)

    def display_positions(self):
        """Display positions tab"""
        print("=" * 80)
        print("📊 POSITIONS - ACTIVE POSITIONS & RISK MANAGEMENT")
        print("=" * 80)
        
        positions = self.get_active_positions()
        
        if not positions:
            print("No active positions")
            return
        
        print(f"{'Coin':<8} {'Side':<5} {'Lev':<4} {'Size':<12} {'Entry':<10} {'Current':<10} {'P&L%':<8} {'P&L$':<10}")
        print("-" * 85)
        
        for pos in positions:
            coin = pos['coin']
            side = pos['side']
            leverage = f"{pos.get('leverage', 1):.0f}x"
            
            # Clean size formatting - remove trailing zeros
            size_val = pos['size']
            if size_val == int(size_val):
                size = f"{int(size_val)}"
            else:
                size = f"{size_val:.4f}".rstrip('0').rstrip('.')
            
            # Clean price formatting - remove trailing zeros
            entry_val = pos['entry_px']
            if entry_val >= 1:
                entry = f"${entry_val:.2f}".rstrip('0').rstrip('.')
            else:
                entry = f"${entry_val:.6f}".rstrip('0').rstrip('.')
            
            current_val = pos['current_price']
            if current_val >= 1:
                current = f"${current_val:.2f}".rstrip('0').rstrip('.')
            else:
                current = f"${current_val:.6f}".rstrip('0').rstrip('.')
            
            pnl_percent = f"{pos['profit_percent']:.2f}%"
            pnl_usd = f"${pos['unrealized_pnl']:.2f}"
            
            # Color coding for profit/loss
            if pos['profit_percent'] > 0:
                pnl_percent = f"🟢{pnl_percent}"
            else:
                pnl_percent = f"🔴{pnl_percent}"
            
            print(f"{coin:<8} {side:<5} {leverage:<4} {size:<12} {entry:<10} {current:<10} {pnl_percent:<8} {pnl_usd:<10}")
            
            # Get optimized parameters for this coin
            params = self.get_optimized_params_for_coin(coin)
            stop_loss_percent = params['stop_loss_percent']
            trailing_stop_percent = params['trailing_stop_percent']
            
            # Show optimized parameters and trailing stop info
            param_source = "📈 Optimized" if coin in self.optimized_params else "⚙️ Default"
            print(f"         {param_source} | SL: {stop_loss_percent:.1f}% | TS: {trailing_stop_percent:.1f}%")
            
            if coin in self.position_history:
                max_profit = self.position_history[coin]['max_profit']
                trailing_stop_level = max_profit - trailing_stop_percent
                print(f"         Max Profit: {max_profit:.2f}% | Trailing Stop Level: {trailing_stop_level:.2f}%")
            
            # Show backtest performance if available
            if coin in self.optimized_params:
                backtest_return = self.optimized_params[coin]['total_return_pct']
                backtest_winrate = self.optimized_params[coin]['win_rate']
                print(f"         Backtest: {backtest_return:.1f}% return, {backtest_winrate:.1f}% win rate")

    def display_orders(self):
        """Display orders tab"""
        print("=" * 80)
        print("📋 ORDERS - OPEN ORDERS")
        print("=" * 80)
        
        orders = self.get_open_orders()
        
        if not orders:
            print("No open orders")
            return
        
        print(f"{'Coin':<8} {'Side':<5} {'Size':<12} {'Price':<10} {'Value':<10}")
        print("-" * 60)
        
        for order in orders:
            coin = order.get('coin', 'N/A')
            side = 'BUY' if order.get('side') == 'B' else 'SELL'
            size = f"{float(order.get('sz', 0)):.4f}"
            price = f"${float(order.get('limitPx', 0)):.4f}"
            value = f"${float(order.get('sz', 0)) * float(order.get('limitPx', 0)):.2f}"
            
            print(f"{coin:<8} {side:<5} {size:<12} {price:<10} {value:<10}")

    def display_scanner(self):
        """Display scanner tab"""
        print("=" * 80)
        print("🔍 SCANNER - MARKET OPPORTUNITIES")
        print("=" * 80)
        
        if not self.scanner_output:
            print("No scanner data available")
            return
        
        timestamp = self.scanner_output.get('timestamp', 'Unknown')
        print(f"Last Update: {timestamp}")
        
        # Low opportunities
        print("\n📈 LOW PRICE OPPORTUNITIES (LONG):")
        print("-" * 50)
        low_opps = self.scanner_output.get('low_opportunities', [])
        if low_opps:
            print(f"{'Coin':<8} {'Low Price':<12} {'Current':<12} {'Score':<8}")
            print("-" * 40)
            for opp in low_opps[:5]:
                coin = opp.get('symbol', 'N/A')
                low_price = f"${float(opp.get('best_opportunity_price', 0)):.6f}"
                current = f"${float(opp.get('current_price', 0)):.6f}"
                score = f"{float(opp.get('significance_score', 0)):.2f}"
                print(f"{coin:<8} {low_price:<12} {current:<12} {score:<8}")
        else:
            print("No low opportunities found")
        
        # High opportunities
        print("\n📉 HIGH PRICE OPPORTUNITIES (SHORT):")
        print("-" * 50)
        high_opps = self.scanner_output.get('high_opportunities', [])
        if high_opps:
            print(f"{'Coin':<8} {'High Price':<12} {'Current':<12} {'Score':<8}")
            print("-" * 40)
            for opp in high_opps[:5]:
                coin = opp.get('symbol', 'N/A')
                high_price = f"${float(opp.get('daily_high_price', 0)):.6f}"
                current = f"${float(opp.get('current_price', 0)):.6f}"
                score = f"{float(opp.get('significance_score', 0)):.2f}"
                print(f"{coin:<8} {high_price:<12} {current:<12} {score:<8}")
        else:
            print("No high opportunities found")

    def display_menu(self):
        """Display navigation menu"""
        print("\n" + "=" * 80)
        print("NAVIGATION MENU")
        print("=" * 80)
        print("1. 🏠 Home        2. 📊 Positions    3. 📋 Orders      4. 🔍 Scanner")
        print("5. ▶️  Start       6. ⏹️  Stop        7. 🔄 Refresh     8. ❌ Exit")
        print("9. 💰 Close Position (enter coin)")
        print("=" * 80)

    def run_interface(self):
        """Run the main interface loop"""
        import select
        import sys
        
        while True:
            self.clear_screen()
            
            # Display current tab
            if self.current_tab == "home":
                self.display_home()
            elif self.current_tab == "positions":
                self.display_positions()
            elif self.current_tab == "orders":
                self.display_orders()
            elif self.current_tab == "scanner":
                self.display_scanner()
            
            self.display_menu()
            
            # Show auto-refresh status
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"\n🔄 Auto-refresh: ON | Current Time: {current_time}")
            print("Press any key + ENTER for menu, or wait 5 seconds for auto-refresh...")
            
            try:
                # Check for input with timeout
                if sys.stdin in select.select([sys.stdin], [], [], 5)[0]:
                    choice = input().strip()
                    
                    if choice == "1":
                        self.current_tab = "home"
                    elif choice == "2":
                        self.current_tab = "positions"
                    elif choice == "3":
                        self.current_tab = "orders"
                    elif choice == "4":
                        self.current_tab = "scanner"
                    elif choice == "5":
                        self.start_system()
                    elif choice == "6":
                        self.stop_system()
                    elif choice == "7":
                        continue  # Just refresh
                    elif choice == "8":
                        print("Exiting...")
                        self.stop_system()
                        break
                    elif choice == "9":
                        coin = input("Enter coin to close: ").strip().upper()
                        if coin:
                            self.close_position(coin, "Manual Close")
                else:
                    # Auto-refresh after 5 seconds
                    continue
                
            except KeyboardInterrupt:
                print("\nExiting...")
                self.stop_system()
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(2)

def main():
    """Main function"""
    print("🚀 Starting Trading Panel...")
    panel = TradingPanel()
    panel.run_interface()

if __name__ == "__main__":
    main()
