#!/usr/bin/env python3
"""
Order Manager - Helper functions for managing orders and positions
- need to check  order time, for age check if older then 24 hours then cancel it. cancel open order.
- 
"""

import sys
import os
from typing import List, Dict, Optional

import example_utils
from default_config import Config
from hyperliquid.utils import constants
from hyperliquid.info import Info


class OrderManager:
    def __init__(self):
        config = Config()
        self.address, self.info, self.exchange = example_utils.setup(config.API_URL, skip_ws=True)
    
    def get_user_state(self) -> Dict:
        """Get complete user state"""
        try:
            return self.info.user_state(self.address)
        except Exception as e:
            print(f"❌ Error getting user state: {e}")
            return {}
    
    def get_open_orders(self) -> List[Dict]:
        """Get all open orders"""
        try:
            open_orders = self.info.open_orders(self.address)
            return open_orders
        except Exception as e:
            print(f"❌ Error getting open orders: {e}")
            return []
    
    def get_positions(self) -> List[Dict]:
        """Get all current positions"""
        try:
            user_state = self.get_user_state()
            positions = []
            
            for asset_position in user_state.get("assetPositions", []):
                position = asset_position.get("position", {})
                if position.get("szi") != "0":  # Non-zero position
                    positions.append({
                        'coin': position.get('coin'),
                        'size': float(position.get('szi', 0)),
                        'entry_price': float(position.get('entryPx', 0)),
                        'unrealized_pnl': float(position.get('unrealizedPnl', 0)),
                        'return_on_equity': float(position.get('returnOnEquity', 0))
                    })
            
            return positions
        except Exception as e:
            print(f"❌ Error getting positions: {e}")
            return []
    
    def has_open_order_for_coin(self, coin: str) -> bool:
        """Check if there's an open order for a specific coin"""
        open_orders = self.get_open_orders()
        for order in open_orders:
            if order.get('coin') == coin:
                return True
        return False
    
    def has_position_for_coin(self, coin: str) -> bool:
        """Check if there's an open position for a specific coin"""
        positions = self.get_positions()
        for position in positions:
            if position['coin'] == coin:
                return True
        return False
    
    def get_account_value(self) -> float:
        """Get total account value"""
        try:
            user_state = self.get_user_state()
            margin_summary = user_state.get("marginSummary", {})
            return float(margin_summary.get("accountValue", 0))
        except Exception as e:
            print(f"❌ Error getting account value: {e}")
            return 0.0
    
    def display_account_summary(self):
        """Display account summary"""
        try:
            user_state = self.get_user_state()
            margin_summary = user_state.get("marginSummary", {})
            
            print(f"\n📊 Account Summary:")
            print(f"   Account Value: ${float(margin_summary.get('accountValue', 0)):.2f}")
            print(f"   Total Margin Used: ${float(margin_summary.get('totalMarginUsed', 0)):.2f}")
            print(f"   Total Unrealized PnL: ${float(margin_summary.get('totalNtlPos', 0)):.2f}")
            
            positions = self.get_positions()
            open_orders = self.get_open_orders()
            
            print(f"   Open Positions: {len(positions)}")
            print(f"   Open Orders: {len(open_orders)}")
            
            if positions:
                print(f"\n📈 Current Positions:")
                for pos in positions:
                    side = "LONG" if pos['size'] > 0 else "SHORT"
                    print(f"   {pos['coin']}: {side} {abs(pos['size']):.4f} @ ${pos['entry_price']:.4f} | PnL: ${pos['unrealized_pnl']:.2f}")
            
            if open_orders:
                print(f"\n📋 Open Orders:")
                for order in open_orders:
                    side = "BUY" if order.get('side') == 'B' else "SELL"
                    print(f"   {order.get('coin')}: {side} {order.get('sz')} @ ${order.get('limitPx')}")
                    
        except Exception as e:
            print(f"❌ Error displaying account summary: {e}")
    
    def cancel_all_orders(self):
        """Cancel all open orders"""
        try:
            open_orders = self.get_open_orders()
            if not open_orders:
                print("No open orders to cancel")
                return
            
            print(f"Cancelling {len(open_orders)} open orders...")
            
            for order in open_orders:
                try:
                    coin = order.get('coin')
                    oid = order.get('oid')
                    
                    result = self.exchange.cancel(coin, oid)
                    if result.get('status') == 'ok':
                        print(f"✅ Cancelled order {oid} for {coin}")
                    else:
                        print(f"❌ Failed to cancel order {oid} for {coin}: {result}")
                        
                except Exception as e:
                    print(f"❌ Error cancelling order: {e}")
                    
        except Exception as e:
            print(f"❌ Error in cancel_all_orders: {e}")


def main():
    """Test the order manager"""
    manager = OrderManager()
    manager.display_account_summary()


if __name__ == "__main__":
    main()
