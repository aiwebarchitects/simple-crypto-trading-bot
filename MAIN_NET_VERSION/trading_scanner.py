#!/usr/bin/env python3
"""
Trading Scanner Module - Handles market scanning and order placement logic
Separated from the main trading panel for better organization.
"""

import os
import subprocess
import csv
from datetime import datetime
from typing import List, Dict, Set
from default_config import Config

class TradingScanner:
    def __init__(self, exchange, info, available_coins: Set[str], config: Dict = None):
        self.exchange = exchange
        self.info = info
        self.available_coins = available_coins
        
        # Use centralized configuration from default_config.py
        # Allow override via config parameter for backward compatibility
        if config:
            self.max_positions = config.get('max_positions', Config.MAX_POSITIONS)
            self.position_size_usd = config.get('position_size_usd', Config.POSITION_SIZE_USD)
            self.long_offset_percent = config.get('long_offset_percent', Config.LONG_OFFSET_PERCENT)
            self.short_offset_percent = config.get('short_offset_percent', Config.SHORT_OFFSET_PERCENT)
        else:
            # Use default configuration
            self.max_positions = Config.MAX_POSITIONS
            self.position_size_usd = Config.POSITION_SIZE_USD
            self.long_offset_percent = Config.LONG_OFFSET_PERCENT
            self.short_offset_percent = Config.SHORT_OFFSET_PERCENT
        
        self.log_callback = None

    def set_log_callback(self, callback):
        """Set callback function for logging"""
        self.log_callback = callback

    def log(self, message: str):
        """Log message using callback if available"""
        if self.log_callback:
            self.log_callback(message)

    def run_scanner(self, scanner_type: str) -> List[Dict]:
        """Run scanner and return results"""
        try:
            # Use configuration from default_config.py
            if scanner_type == "low":
                script = Config.LOW_SCANNER_SCRIPT
                csv_prefix = Config.LOW_OPPORTUNITIES_PREFIX
            else:
                script = Config.HIGH_SCANNER_SCRIPT
                csv_prefix = Config.HIGH_OPPORTUNITIES_PREFIX
            
            self.log(f"🔍 Running {scanner_type} scanner...")
            result = subprocess.run(['python3', script], capture_output=True, text=True)
            
            if result.returncode != 0:
                self.log(f"❌ {scanner_type} scanner failed: {result.stderr}")
                return []
            
            # Find the most recent CSV file
            csv_files = [f for f in os.listdir(Config.REPORTS_DIRECTORY) if f.startswith(csv_prefix) and f.endswith('.csv')]
            if not csv_files:
                self.log(f"❌ No {scanner_type} CSV files found")
                return []
            
            latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(Config.REPORTS_DIRECTORY, x)))
            latest_file = os.path.join(Config.REPORTS_DIRECTORY, latest_file)
            
            # Read results
            opportunities = []
            with open(latest_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    opportunities.append(dict(row))
            
            self.log(f"✅ Found {len(opportunities)} {scanner_type} opportunities")
            return opportunities[:Config.SCANNER_TOP_RESULTS]  # Return top results from config
            
        except Exception as e:
            self.log(f"❌ Error running {scanner_type} scanner: {e}")
            return []

    def check_capacity_and_scan(self, positions: List[Dict], orders: List[Dict]) -> Dict:
        """Check current capacity and run scans if needed"""
        try:
            # Calculate existing coins and capacity
            existing_coins = set([p['coin'] for p in positions] + [o.get('coin') for o in orders])
            current_count = len(existing_coins)
            available_slots = self.max_positions - current_count
            
            self.log(f"📊 Capacity Check: {current_count}/{self.max_positions} positions used")
            self.log(f"💡 Available slots: {available_slots}")
            self.log(f"🪙 Existing coins: {', '.join(sorted(existing_coins)) if existing_coins else 'None'}")
            
            # If at capacity, no need to scan
            if available_slots <= 0:
                self.log("⚠️ Maximum positions reached - no scanning needed")
                return {
                    'capacity_full': True,
                    'current_count': current_count,
                    'available_slots': 0,
                    'existing_coins': existing_coins,
                    'low_opportunities': [],
                    'high_opportunities': [],
                    'orders_placed': 0
                }
            
            # Run scanners
            self.log(f"🚀 Starting market scan for {available_slots} available slots...")
            low_opportunities = self.run_scanner("low")
            high_opportunities = self.run_scanner("high")
            
            # Process opportunities and place orders
            orders_placed = self.process_opportunities(
                low_opportunities, 
                high_opportunities, 
                existing_coins, 
                available_slots
            )
            
            return {
                'capacity_full': False,
                'current_count': current_count,
                'available_slots': available_slots,
                'existing_coins': existing_coins,
                'low_opportunities': low_opportunities,
                'high_opportunities': high_opportunities,
                'orders_placed': orders_placed
            }
            
        except Exception as e:
            self.log(f"❌ Error in capacity check and scan: {e}")
            return {
                'capacity_full': False,
                'current_count': 0,
                'available_slots': self.max_positions,
                'existing_coins': set(),
                'low_opportunities': [],
                'high_opportunities': [],
                'orders_placed': 0
            }

    def process_opportunities(self, low_opportunities: List[Dict], high_opportunities: List[Dict], 
                            existing_coins: Set[str], available_slots: int) -> int:
        """Process opportunities and place orders"""
        orders_placed = 0
        
        # Process LONG opportunities
        self.log("📈 Processing LONG opportunities...")
        for opp in low_opportunities:
            if orders_placed >= available_slots:
                self.log(f"🛑 Reached available slots limit ({available_slots})")
                break
                
            coin = opp.get('symbol', '').upper()
            
            # Skip if coin not available or already exists
            if coin not in self.available_coins:
                self.log(f"⏭️ Skipping {coin}: Not in available coins list")
                continue
                
            if coin in existing_coins:
                self.log(f"⏭️ Skipping {coin}: Already have position/order")
                continue
            
            try:
                low_price = float(opp.get('best_opportunity_price', 0))
                current_price = float(opp.get('current_price', 0))
                
                if low_price <= 0 or current_price <= 0:
                    self.log(f"⏭️ Skipping {coin}: Invalid prices")
                    continue
                
                # Calculate target price (0.15% above low)
                target_price = low_price * (1 + self.long_offset_percent / 100)
                
                # Only place order if current price is above target (opportunity exists)
                if current_price > target_price:
                    # Calculate position size with proper rounding
                    position_size = self.position_size_usd / target_price
                    position_size = round(position_size, 4)  # Round to 4 decimal places
                    target_price = round(target_price, 6)    # Round price to 6 decimal places
                    
                    self.log(f"🎯 {coin} LONG Opportunity:")
                    self.log(f"   Lowest Price: ${low_price:.6f}")
                    self.log(f"   Target Price: ${target_price:.6f} (+{self.long_offset_percent}%)")
                    self.log(f"   Current Price: ${current_price:.6f}")
                    self.log(f"   Position Size: {position_size}")
                    
                    # Place limit order
                    if self.place_long_order(coin, position_size, target_price):
                        orders_placed += 1
                        existing_coins.add(coin)
                else:
                    self.log(f"⏭️ Skipping {coin} LONG: Current price ${current_price:.6f} <= Target ${target_price:.6f}")
                        
            except Exception as e:
                self.log(f"❌ Error processing {coin} long opportunity: {e}")
        
        # Process SHORT opportunities
        self.log("📉 Processing SHORT opportunities...")
        for opp in high_opportunities:
            if orders_placed >= available_slots:
                self.log(f"🛑 Reached available slots limit ({available_slots})")
                break
                
            coin = opp.get('symbol', '').upper()
            
            # Skip if coin not available or already exists
            if coin not in self.available_coins:
                self.log(f"⏭️ Skipping {coin}: Not in available coins list")
                continue
                
            if coin in existing_coins:
                self.log(f"⏭️ Skipping {coin}: Already have position/order")
                continue
            
            try:
                high_price = float(opp.get('daily_high_price', 0))
                current_price = float(opp.get('current_price', 0))
                
                if high_price <= 0 or current_price <= 0:
                    self.log(f"⏭️ Skipping {coin}: Invalid prices")
                    continue
                
                # Calculate target price (0.15% below high)
                target_price = high_price * (1 - self.short_offset_percent / 100)
                
                # Only place order if current price is below target (opportunity exists)
                if current_price < target_price:
                    # Calculate position size with proper rounding
                    position_size = self.position_size_usd / target_price
                    position_size = round(position_size, 4)  # Round to 4 decimal places
                    target_price = round(target_price, 6)    # Round price to 6 decimal places
                    
                    self.log(f"🎯 {coin} SHORT Opportunity:")
                    self.log(f"   Highest Price: ${high_price:.6f}")
                    self.log(f"   Target Price: ${target_price:.6f} (-{self.short_offset_percent}%)")
                    self.log(f"   Current Price: ${current_price:.6f}")
                    self.log(f"   Position Size: {position_size}")
                    
                    # Place limit order
                    if self.place_short_order(coin, position_size, target_price):
                        orders_placed += 1
                        existing_coins.add(coin)
                else:
                    self.log(f"⏭️ Skipping {coin} SHORT: Current price ${current_price:.6f} >= Target ${target_price:.6f}")
                        
            except Exception as e:
                self.log(f"❌ Error processing {coin} short opportunity: {e}")
        
        if orders_placed > 0:
            self.log(f"✅ Successfully placed {orders_placed} new orders")
        else:
            self.log("ℹ️ No new orders placed this cycle")
            
        return orders_placed

    def get_coin_size_decimals(self, coin: str) -> int:
        """Get the size decimals for a coin from Hyperliquid metadata"""
        try:
            meta = self.info.meta()
            if meta and 'universe' in meta:
                for asset in meta['universe']:
                    if asset.get('name') == coin:
                        return asset.get('szDecimals', 0)
            return 0  # Default to whole numbers if not found
        except:
            return 0

    def calculate_position_size(self, coin: str, price: float) -> float:
        """Calculate position size based on USD amount with proper rounding respecting coin decimals"""
        try:
            raw_size = self.position_size_usd / price
            
            # Ensure minimum $10 order value
            min_usd_value = 10.0
            min_size = min_usd_value / price
            
            # Use the larger of calculated size or minimum size
            target_size = max(raw_size, min_size)
            
            # Get the proper decimal places from Hyperliquid metadata
            size_decimals = self.get_coin_size_decimals(coin)
            
            # Round to the correct number of decimal places
            rounded_size = round(target_size, size_decimals)
            
            self.log(f"📏 {coin} Size Calculation: Raw={raw_size:.6f}, Target={target_size:.6f}, Decimals={size_decimals}, Final={rounded_size}")
            
            return rounded_size
        except Exception as e:
            self.log(f"❌ Error calculating size for {coin}: {e}")
            return 3.0  # Fallback minimum size that ensures $10+ value

    def get_tick_size_price(self, coin: str, target_price: float) -> float:
        """Round price to valid tick size using Hyperliquid's proper rounding rules"""
        try:
            max_decimals = 6  # For perps
            sz_decimals = self.get_coin_size_decimals(coin)
            
            # If price is greater than 100k, round to integer
            if target_price > 100_000:
                return round(target_price)
            
            # Round to 5 significant figures and max_decimals - szDecimals decimal places
            # First round to 5 significant figures
            rounded_price = round(float(f"{target_price:.5g}"), max_decimals - sz_decimals)
            
            return rounded_price
        except Exception as e:
            self.log(f"❌ Error rounding price for {coin}: {e}")
            # Fallback to simple rounding
            return round(target_price, 2)

    def place_long_order(self, coin: str, size: float, price: float) -> bool:
        """Place a long limit order using the working method from start_system.py"""
        try:
            # Recalculate size using the working method
            proper_size = self.calculate_position_size(coin, price)
            adjusted_price = self.get_tick_size_price(coin, price)
            
            self.log(f"🔄 Placing LONG order: {coin} {proper_size} @ ${adjusted_price}")
            
            # Use the working exchange.order method instead of bulk_orders
            is_buy = True
            result = self.exchange.order(coin, is_buy, proper_size, adjusted_price, {"limit": {"tif": "Gtc"}})
            
            self.log(f"📋 API Response: {result}")
            
            # Check for errors in the response
            if result.get("status") == "ok":
                # Check if there are error details in the response
                response_data = result.get("response", {})
                if isinstance(response_data, dict):
                    data = response_data.get("data", {})
                    if isinstance(data, dict):
                        statuses = data.get("statuses", [])
                        if statuses and isinstance(statuses[0], dict) and "error" in statuses[0]:
                            error_msg = statuses[0]["error"]
                            self.log(f"❌ LONG order rejected for {coin}: {error_msg}")
                            return False
                
                self.log(f"✅ LONG order placed successfully for {coin}")
                return True
            else:
                self.log(f"❌ Failed to place LONG order for {coin}: {result}")
                return False
                
        except Exception as e:
            self.log(f"❌ Error placing LONG order for {coin}: {e}")
            return False

    def place_short_order(self, coin: str, size: float, price: float) -> bool:
        """Place a short limit order using the working method from start_system.py"""
        try:
            # Recalculate size using the working method
            proper_size = self.calculate_position_size(coin, price)
            adjusted_price = self.get_tick_size_price(coin, price)
            
            self.log(f"🔄 Placing SHORT order: {coin} {proper_size} @ ${adjusted_price}")
            
            # Use the working exchange.order method instead of bulk_orders
            is_buy = False
            result = self.exchange.order(coin, is_buy, proper_size, adjusted_price, {"limit": {"tif": "Gtc"}})
            
            self.log(f"📋 API Response: {result}")
            
            # Check for errors in the response
            if result.get("status") == "ok":
                # Check if there are error details in the response
                response_data = result.get("response", {})
                if isinstance(response_data, dict):
                    data = response_data.get("data", {})
                    if isinstance(data, dict):
                        statuses = data.get("statuses", [])
                        if statuses and isinstance(statuses[0], dict) and "error" in statuses[0]:
                            error_msg = statuses[0]["error"]
                            self.log(f"❌ SHORT order rejected for {coin}: {error_msg}")
                            return False
                
                self.log(f"✅ SHORT order placed successfully for {coin}")
                return True
            else:
                self.log(f"❌ Failed to place SHORT order for {coin}: {result}")
                return False
                
        except Exception as e:
            self.log(f"❌ Error placing SHORT order for {coin}: {e}")
            return False
