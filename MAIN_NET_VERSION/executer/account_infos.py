# this code gives back our account infos

from hyperliquid.info import Info
from hyperliquid.utils import constants

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from default_config import Config

config = Config()
info = Info(config.API_URL, skip_ws=True)
user_state = info.user_state("0x0c123456789")
print(user_state)
