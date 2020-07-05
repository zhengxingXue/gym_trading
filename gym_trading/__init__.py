from gym.envs.registration import register

register(
    id='StockTrading-v0',
    entry_point='gym_trading.envs:StockTradingEnv',
    kwargs={},
)
