from gym_trading.config import PACKAGE_DIR
from gym.envs.registration import register

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

register(
    id='StockTrading-IBM-MSFT-v1',
    entry_point='gym_trading.envs:StockTradingEnvV1',
    kwargs={'file_array':['data/daily_IBM.csv', 'data/daily_MSFT.csv']},
)
