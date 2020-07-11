from gym.envs.registration import register
from gym_trading.config import PACKAGE_DIR

register(
    id='StockTrading-v0',
    entry_point='gym_trading.envs:StockTradingEnv',
    kwargs={},
)
