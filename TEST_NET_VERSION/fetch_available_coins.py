#!/usr/bin/env python3
"""
Simple script to fetch available coins from Hyperliquid
"""

import sys
import os

# Add the current directory to Python path to import example_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import example_utils
from hyperliquid.utils import constants
from hyperliquid.info import Info

def fetch_available_coins():
    """
    Fetch available coins from Hyperliquid API
    Returns a set of coin symbols
    """
    try:
        # Setup Hyperliquid connection
        address, info, exchange = example_utils.setup(constants.TESTNET_API_URL, skip_ws=True)
        
        print("🔗 Connected to Hyperliquid API")
        print(f"📍 Address: {address}")
        
        # Get meta information which contains universe of available assets
        meta = info.meta()
        
        if not meta or 'universe' not in meta:
            print("❌ No meta information or universe found")
            return set()
        
        # Extract coin symbols from universe
        available_coins = set()
        
        print(f"\n📊 Found {len(meta['universe'])} assets in universe:")
        print("-" * 50)
        
        for asset in meta['universe']:
            coin_name = asset.get('name')
            if coin_name:
                available_coins.add(coin_name)
                
                # Display some info about each coin
                max_leverage = asset.get('maxLeverage', 'N/A')
                only_isolated = asset.get('onlyIsolated', False)
                
                print(f"🪙 {coin_name:<8} | Max Leverage: {max_leverage:<3} | Isolated Only: {only_isolated}")
        
        print(f"\n✅ Successfully fetched {len(available_coins)} available coins")
        print(f"🎯 Available coins: {sorted(available_coins)}")
        
        return available_coins
        
    except Exception as e:
        print(f"❌ Error fetching available coins: {e}")
        return set()

def main():
    """Main function to test coin fetching"""
    print("🚀 Fetching available coins from Hyperliquid...")
    
    coins = fetch_available_coins()
    
    if coins:
        print(f"\n🎉 Success! Found {len(coins)} coins")
        print("📋 Coin list as Python set:")
        print(f"available_coins = {repr(coins)}")
    else:
        print("❌ Failed to fetch coins")

if __name__ == "__main__":
    main()

