from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from gym_trading.envs.stock_env_v0 import StockTradingEnv

import pandas as pd


def main():
    df = pd.read_csv('./gym_trading/data/AAPL.csv')
    df = df.sort_values('Date')

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(df)])

    model = PPO2(MlpPolicy, env, verbose=1)

    model.learn(10000)


if __name__ == "__main__":
    main()

