import example_utils
from hyperliquid.utils import constants


def main():
    """
    Cancel all open orders for the account.
    This will cancel every open order across all coins.
    Tested: NO
    """
    address, info, exchange = example_utils.setup(constants.TESTNET_API_URL, skip_ws=True)

    print(f"=== Canceling ALL open orders ===")
    
    try:
        # Get all open orders
        open_orders = info.open_orders(address)
        
        if not open_orders:
            print("✅ No open orders found - nothing to cancel")
            return
            
        print(f"📊 Found {len(open_orders)} open order(s) to cancel:")
        
        # Display all orders before canceling
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
        
        print(f"\n🔄 Canceling all {len(open_orders)} orders...")
        
        # Cancel each order
        canceled_count = 0
        failed_count = 0
        
        for order in open_orders:
            order_coin = order["coin"]
            order_oid = order["oid"]
            
            try:
                print(f"   Canceling {order_oid} ({order_coin})...")
                cancel_result = exchange.cancel(order_coin, order_oid)
                
                if cancel_result["status"] == "ok":
                    print(f"   ✅ {order_oid} canceled successfully")
                    canceled_count += 1
                else:
                    print(f"   ❌ {order_oid} cancellation failed: {cancel_result}")
                    failed_count += 1
                    
            except Exception as e:
                print(f"   ❌ Error canceling {order_oid}: {str(e)}")
                failed_count += 1
        
        # Summary
        print(f"\n📊 Cancellation Summary:")
        print(f"   Total Orders: {len(open_orders)}")
        print(f"   Successfully Canceled: {canceled_count}")
        print(f"   Failed: {failed_count}")
        
        if canceled_count == len(open_orders):
            print("✅ All orders canceled successfully!")
        elif canceled_count > 0:
            print("⚠️ Some orders canceled, but some failed")
        else:
            print("❌ No orders were canceled")
            
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        
    print("=== Cancel all orders process completed ===")


if __name__ == "__main__":
    main()
