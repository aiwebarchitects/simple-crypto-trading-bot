import example_utils
from hyperliquid.utils import constants


def get_tick_size_price(info, coin, target_price):
    """
    Round price to valid tick size - use whole numbers for ETH.
    Tested: Working
    """
    # For ETH, round to nearest whole number
    if coin == "ETH":
        return int(round(target_price))
    else:
        # For other coins, round to 2 decimal places
        return round(target_price, 2)


def main():
    """
    Create a short limit order above current market price.
    This places a sell order at a price higher than current market (e.g., 1% above).
    """
    address, info, exchange = example_utils.setup(constants.TESTNET_API_URL, skip_ws=True)

    # Configuration
    coin = "ETH"
    size = 0.05  # Amount to sell
    percentage_above = 1.0  # 1% above market price
    
    print(f"=== Creating SHORT LIMIT order (above market) ===")
    print(f"Coin: {coin}")
    print(f"Size: {size}")
    print(f"Target: {percentage_above}% above market price")
    
    try:
        # Get current market price
        all_mids = info.all_mids()
        if coin not in all_mids:
            print(f"❌ Could not get market price for {coin}")
            return
            
        current_price = float(all_mids[coin])
        raw_target_price = current_price * (1 + percentage_above / 100)
        target_price = get_tick_size_price(info, coin, raw_target_price)
        
        print(f"📊 Price Information:")
        print(f"   Current Market Price: ${current_price:.2f}")
        print(f"   Target Limit Price: ${target_price:.2f}")
        print(f"   Premium: ${target_price - current_price:.2f} ({percentage_above}%)")
        
        # Place limit sell order above market
        is_buy = False
        order_result = exchange.order(coin, is_buy, size, target_price, {"limit": {"tif": "Gtc"}})
        
        if order_result["status"] == "ok":
            print(f"✅ SHORT LIMIT order placed successfully!")
            
            for status in order_result["response"]["data"]["statuses"]:
                if "resting" in status:
                    order_info = status["resting"]
                    order_id = order_info["oid"]
                    
                    print(f"📊 Order Details:")
                    print(f"   Order ID: {order_id}")
                    print(f"   Type: SHORT LIMIT (Sell)")
                    print(f"   Size: {size} {coin}")
                    print(f"   Limit Price: ${target_price:.2f}")
                    print(f"   Status: Waiting for fill")
                    
                elif "filled" in status:
                    filled = status["filled"]
                    print(f"🎉 Order filled immediately!")
                    print(f"   Order ID: {filled['oid']}")
                    print(f"   Filled Size: {filled['totalSz']} {coin}")
                    print(f"   Fill Price: ${filled['avgPx']}")
                    
                else:
                    print(f"❌ Order Error: {status.get('error', 'Unknown error')}")
        else:
            print(f"❌ SHORT LIMIT order failed: {order_result}")
            
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        
    print("=== SHORT LIMIT order process completed ===")


if __name__ == "__main__":
    main()
