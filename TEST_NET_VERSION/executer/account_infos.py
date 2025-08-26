# this code gives back our account infos

from hyperliquid.info import Info
from hyperliquid.utils import constants

info = Info(constants.TESTNET_API_URL, skip_ws=True)
user_state = info.user_state("YOUR_WALLET_ADDRESS_HERE")
print(user_state)
