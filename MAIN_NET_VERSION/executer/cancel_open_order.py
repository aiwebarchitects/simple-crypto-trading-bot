import example_utils
from hyperliquid.utils import constants


def main():
    """
    Cancel a specific open order by order ID.
    This will cancel a single order for a given coin and order ID.
    Tested: Working
    """
    from default_config import Config
    config = Config()
    address, info, exchange = example_utils.setup(config.API_URL, skip_ws=True)

    # Configuration - You need to specify the coin and order ID
    coin = "ETH"
    order_id = None  # Set this to the specific order ID you want to cancel
    
    print(f"=== Canceling specific order ===")
    print(f"Coin: {coin}")
    
    try:
        # Get all open orders to find the target order
        open_orders = info.open_orders(address)
        
        if not open_orders:
            print("❌ No open orders found")
            return
            
        print(f"📊 Found {len(open_orders)} open order(s):")
        
        target_order = None
        for i, order in enumerate(open_orders):
            order_coin = order["coin"]
            order_oid = order["oid"]
            order_side = order["side"]
            order_sz = order["sz"]
            order_px = order["limitPx"]
            
            print(f"   {i+1}. Order ID: {order_oid}")
            print(f"      Coin: {order_coin}")
            print(f"      Side: {order_side}")
            print(f"      Size: {order_sz}")
            print(f"      Price: ${order_px}")
            print()
            
            # If specific order_id is provided, find it
            if order_id and order_oid == order_id:
                target_order = order
            # If no specific order_id, use the first order for the specified coin
            elif not order_id and order_coin == coin and not target_order:
                target_order = order
        
        if not target_order:
            if order_id:
                print(f"❌ Order ID {order_id} not found")
            else:
                print(f"❌ No open orders found for {coin}")
            return
            
        # Cancel the specific order
        cancel_coin = target_order["coin"]
        cancel_oid = target_order["oid"]
        
        print(f"🔄 Canceling order {cancel_oid} for {cancel_coin}...")
        
        cancel_result = exchange.cancel(cancel_coin, cancel_oid)
        
        if cancel_result["status"] == "ok":
            print(f"✅ Order canceled successfully!")
            print(f"📊 Canceled Order Details:")
            print(f"   Order ID: {cancel_oid}")
            print(f"   Coin: {cancel_coin}")
            print(f"   Side: {target_order['side']}")
            print(f"   Size: {target_order['sz']}")
        else:
            print(f"❌ Order cancellation failed: {cancel_result}")
            
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        
    print("=== Order cancellation process completed ===")


if __name__ == "__main__":
    main()
