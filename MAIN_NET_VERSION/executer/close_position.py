import json
import example_utils
from hyperliquid.utils import constants


def main():
    """
    Close existing positions for a specified coin.
    This will close all open positions for the given asset.
    Tested: Working
    """
    from default_config import Config
    config = Config()
    address, info, exchange = example_utils.setup(config.API_URL, skip_ws=True)

    # Configuration
    coin = "ETH"
    
    print(f"=== Closing position for {coin} ===")
    
    try:
        # First, check current positions
        user_state = info.user_state(address)
        positions = []
        target_position = None
        
        for position in user_state["assetPositions"]:
            if position["position"]["coin"] == coin:
                target_position = position["position"]
                positions.append(position["position"])
        
        if not target_position:
            print(f"❌ No open position found for {coin}")
            return
            
        # Display current position info
        position_size = float(target_position["szi"])
        position_side = "LONG" if position_size > 0 else "SHORT"
        unrealized_pnl = target_position["unrealizedPnl"]
        
        print(f"📊 Current Position:")
        print(f"   Coin: {coin}")
        print(f"   Size: {abs(position_size)}")
        print(f"   Side: {position_side}")
        print(f"   Unrealized PnL: ${unrealized_pnl}")
        
        # Close the position using market_close
        print(f"🔄 Closing {position_side} position...")
        
        order_result = exchange.market_close(coin)
        
        if order_result["status"] == "ok":
            print(f"✅ Position closed successfully!")
            
            for status in order_result["response"]["data"]["statuses"]:
                try:
                    filled = status["filled"]
                    order_id = filled["oid"]
                    total_size = filled["totalSz"]
                    avg_price = filled["avgPx"]
                    
                    print(f"📊 Close Order Details:")
                    print(f"   Order ID: {order_id}")
                    print(f"   Closed Size: {total_size} {coin}")
                    print(f"   Close Price: ${avg_price}")
                    
                except KeyError:
                    print(f"❌ Close Error: {status.get('error', 'Unknown error')}")
        else:
            print(f"❌ Position close failed: {order_result}")
            
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        
    print("=== Position close process completed ===")


if __name__ == "__main__":
    main()
