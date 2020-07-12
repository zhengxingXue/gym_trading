from gym.envs.registration import register
from gym_trading.config import PACKAGE_DIR

register(
    id='StockTrading-v0',
    entry_point='gym_trading.envs:StockTradingEnvV0',
    kwargs={},
)

register(
    id='StockTrading-v1',
    entry_point='gym_trading.envs:StockTradingEnvV1',
    kwargs={},
)
