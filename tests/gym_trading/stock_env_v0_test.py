import os

import gym
import pandas as pd
import pytest

from gym_trading.config import PACKAGE_DIR
from gym_trading.envs.stock_env_v0 import StockTradingEnvV0


@pytest.fixture
def stock_trading_env():
    file = 'data/AAPL.csv'
    file_path = os.path.join(PACKAGE_DIR, file)
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()
    df = df.sort_values('date')
    return StockTradingEnvV0(df)


@pytest.fixture
def stock_trading_v0():
    return gym.make('StockTrading-v0')


def test_env_action_space(stock_trading_env, stock_trading_v0):
    assert stock_trading_v0.action_space == stock_trading_env.action_space


def test_env_obs_space(stock_trading_env, stock_trading_v0):
    assert stock_trading_v0.observation_space == stock_trading_env.observation_space


def test_env_reset(stock_trading_env, stock_trading_v0):
    stock_trading_v0.reset()
    stock_trading_env.reset()


def test_env_step(stock_trading_env, stock_trading_v0):
    stock_trading_v0.step([0, 0])
    stock_trading_env.step([0, 0])
