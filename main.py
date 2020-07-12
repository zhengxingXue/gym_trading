import os
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import gym
from gym_trading.envs.stock_env_v0 import StockTradingEnvV0
from gym_trading.config import PACKAGE_DIR
import pandas as pd

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def main():

    env = gym.make('StockTrading-v0')
    env.render()

    file = 'data/AAPL.csv'
    file_path = os.path.join(PACKAGE_DIR, file)
    df = pd.read_csv(file_path)
    df = df.sort_values('Date')

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnvV0(df)])

    model = PPO2(MlpPolicy, env, verbose=1)

    model.learn(1000)


if __name__ == "__main__":
    main()

