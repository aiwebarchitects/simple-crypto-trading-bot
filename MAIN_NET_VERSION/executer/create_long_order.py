import example_utils
from hyperliquid.utils import constants

def main():
    """
    Create a long order (buy) for a specified coin and size.
    This opens a long position by buying the asset Instanstly at Market Price.
    Tested: Working
    """
    from default_config import Config
    config = Config()
    address, info, exchange = example_utils.setup(config.API_URL, skip_ws=True)

    # Configuration
    coin = "ETH"
    size = 0.05  # Amount to buy
    is_buy = True  # True = Buy/Long
    
    print(f"=== Creating LONG order ===")
    print(f"Coin: {coin}")
    print(f"Size: {size}")
    print(f"Action: LONG (Buy)")
    
    try:
        # Execute market long order
        order_result = exchange.market_open(coin, is_buy, size, None, 0.01)
        
        if order_result["status"] == "ok":
            print(f"✅ LONG order executed successfully!")
            
            for status in order_result["response"]["data"]["statuses"]:
                try:
                    filled = status["filled"]
                    order_id = filled["oid"]
                    total_size = filled["totalSz"]
                    avg_price = filled["avgPx"]
                    
                    print(f"📊 Order Details:")
                    print(f"   Order ID: {order_id}")
                    print(f"   Filled Size: {total_size} {coin}")
                    print(f"   Average Price: ${avg_price}")
                    print(f"   Position: LONG {total_size} {coin} @ ${avg_price}")
                    
                except KeyError:
                    print(f"❌ Order Error: {status.get('error', 'Unknown error')}")
        else:
            print(f"❌ LONG order failed: {order_result}")
            
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        
    print("=== LONG order process completed ===")


if __name__ == "__main__":
    main()
