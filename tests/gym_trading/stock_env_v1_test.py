import os
import copy
import pytest
import pandas as pd
import numpy as np
import gym
from gym import spaces
from gym_trading.config import PACKAGE_DIR
from gym_trading.envs.stock_env_v1 import StockTradingEnvV1


@pytest.fixture
def stock_trading_env_one_stock():
    return StockTradingEnvV1(df=None, file_array=['data/daily_IBM.csv'])


@pytest.fixture
def stock_trading_env_two_stock():
    return StockTradingEnvV1(df=None, file_array=['data/daily_IBM.csv', 'data/daily_MSFT.csv'])


@pytest.fixture
def stock_trading_v1():
    return gym.make('StockTrading-v1')


@pytest.mark.parametrize("file_array, stock_number, action_space", [
    (['data/daily_IBM.csv'], 1, [3]),
    (['data/daily_IBM.csv', 'data/daily_MSFT.csv'], 2, [3, 3]),
    (['data/daily_IBM.csv', 'data/daily_MSFT.csv', 'data/daily_QCOM.csv'], 3, [3, 3, 3]),
])
def test_env_init(file_array, stock_number, action_space):
    env = StockTradingEnvV1(df=None, file_array=file_array)
    assert env.action_space == spaces.MultiDiscrete(action_space)
    assert env.stock_number == stock_number
    assert env.reset().shape == env.observation_space.shape
    assert env.column_number == 5 * stock_number


def test_env_make(stock_trading_env_one_stock, stock_trading_v1):
    assert stock_trading_v1.action_space == stock_trading_env_one_stock.action_space
    assert stock_trading_env_one_stock.stock_number == stock_trading_v1.stock_number


def test_env_two_stock(stock_trading_env_two_stock):
    env = stock_trading_env_two_stock
    assert env.stock_number == 2
    assert env.df.shape[0] == env.normalized_df.shape[0]

    file_array = ['data/daily_IBM.csv', 'data/daily_MSFT.csv']

    for i, file in zip(range(2), file_array):
        file_path = os.path.join(PACKAGE_DIR, file)
        df = pd.read_csv(file_path)
        df = df.sort_values('date')
        df = df.reset_index(drop=True)

        for column in df.columns:
            if column != 'date':
                assert (df[column] == env.df[env.stock_name_array[i] + '_' + column]).all()

        df.pop('date')
        normalized_df = (df - df.min()) / (df.max() - df.min())
        for column in normalized_df.columns:
            assert (normalized_df[column] == env.normalized_df[env.stock_name_array[i] + '_' + column]).all()

        max_share_value = df[['open', 'high', 'low', 'close']].max().max()
        assert max_share_value == env.max_share_value_array[i]


def test_start_point(stock_trading_env_two_stock):
    env = stock_trading_env_two_stock
    obs = env.reset()
    stacked_obs_part = np.reshape(obs[:env.column_number*env.observation_frame], (-1, 10))
    index_end = env.current_step + env.start_point - 1
    index_start = env.current_step + env.start_point - env.observation_frame
    ref_obs_norm = env.normalized_df.loc[index_start:index_end].values
    assert ((ref_obs_norm == stacked_obs_part).all()).all()

    if env.start_point == 10:
        assert env.df['date'].loc[env.start_point] == '2000-01-18'
    # print(env.df['date'].loc[env.start_point])


def test_print(stock_trading_env_two_stock):
    env = stock_trading_env_two_stock
    # print(env.reset())
    # print(env.normalized_df.head())
    # print(env.observation_space.shape)
