import os

import gym
import numpy as np
import pandas as pd
import pytest
from gym import spaces

from gym_trading.config import PACKAGE_DIR
from gym_trading.envs.helper import action_to_str
from gym_trading.envs.stock_env_v1 import StockTradingEnvV1


@pytest.fixture
def stock_trading_env_one_stock():
    return StockTradingEnvV1(df=None,
                             debug=True,
                             file_array=['data/daily_IBM.csv'])


@pytest.fixture
def stock_trading_env_two_stock():
    return StockTradingEnvV1(df=None,
                             debug=True,
                             file_array=['data/daily_IBM.csv', 'data/daily_MSFT.csv'])


@pytest.fixture
def stock_trading_v1():
    return gym.make('StockTrading-norm-v1')


@pytest.fixture
def stock_trading_v1_make():
    return gym.make('StockTrading-IBM-MSFT-norm-v1')


@pytest.mark.parametrize("file_array, stock_number, action_space", [
    (['data/daily_IBM.csv'], 1, [3]),
    (['data/daily_IBM.csv', 'data/daily_MSFT.csv'], 2, [3, 3]),
    (['data/daily_IBM.csv', 'data/daily_MSFT.csv', 'data/daily_QCOM.csv'], 3, [3, 3, 3]),
])
def test_env_init(file_array, stock_number, action_space):
    from stable_baselines.common.env_checker import check_env
    env = StockTradingEnvV1(df=None, file_array=file_array)
    check_env(env)
    assert env.action_space == spaces.MultiDiscrete(action_space)
    assert env.stock_number == stock_number
    assert env.reset().shape == env.observation_space.shape
    assert env.obs_column_number == 1 * stock_number


def test_env_make(stock_trading_env_one_stock, stock_trading_v1, stock_trading_v1_make, stock_trading_env_two_stock):
    assert stock_trading_v1.action_space == stock_trading_env_one_stock.action_space
    assert stock_trading_env_one_stock.stock_number == stock_trading_v1.stock_number
    env = stock_trading_v1_make
    assert env.action_space == stock_trading_env_two_stock.action_space
    assert env.observation_space == stock_trading_env_two_stock.observation_space
    assert env.stock_number == stock_trading_env_two_stock.stock_number


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
    stacked_obs_part = np.reshape(obs[:env.obs_column_number * env.observation_frame], (-1, 2))
    index_end = env.current_step + env.start_point
    index_start = env.current_step + env.start_point - env.observation_frame + 1
    ref_obs_norm = env.normalized_df[env.observation_column_name_array].loc[index_start:index_end].values
    assert ((ref_obs_norm == stacked_obs_part).all()).all()

    assert env.df['date'].loc[env.start_point] == '2000-01-18'
    # print(env.df['date'].loc[env.start_point])


def check_action(env, info, actions, reward):
    init_balance = info['balance_before_step']
    stock_close_value_array = np.zeros(env.stock_number)

    for i, stock_name, action in zip(range(env.stock_number), env.stock_name_array, actions):
        assert info[stock_name]['action'] == action_to_str[action]

        share_value_column = stock_name + '_' + 'close'
        share_value_row = env.current_step + env.start_point - 1
        share_value = env.df[share_value_column].loc[share_value_row]
        stock_close_value_array[i] = share_value * env.stack

        if action_to_str[action] == 'buy':
            price = info[stock_name]['price']
            if isinstance(price, float):
                column_name = stock_name + '_' + 'close'
                row_number = env.current_step + env.start_point - 1
                next_day_open = env.df[column_name].loc[row_number]
                assert price == next_day_open
                init_balance -= price * env.stack
            else:
                # buy action failed as already hold shares
                assert env.hold_share_array[i] == 1
        elif action_to_str[action] == 'sell':
            price = info[stock_name]['price']
            if isinstance(price, float):
                column_name = stock_name + '_' + 'close'
                row_number = env.current_step + env.start_point - 1  # already stepped
                next_day_open = env.df[column_name].loc[row_number]
                assert price == next_day_open
                init_balance += price * env.stack
            else:
                # sell action failed as no share hold
                assert env.hold_share_array[i] == 0
        else:
            pass

    assert init_balance == info['balance_after_step']
    net_worth = env.balance + np.sum(stock_close_value_array * env.hold_share_array)
    assert net_worth == info['net_worth_after_step']
    assert reward == info['net_worth_after_step'] - info['net_worth_before_step']


def test_step_hold(stock_trading_env_two_stock):
    env = stock_trading_env_two_stock
    env.reset()
    init_balance = env.balance
    init_net_worth = env.net_worth
    actions = [2, 2]
    obs, reward, done, info = env.step(actions)
    check_action(env, info, actions, reward)
    # no change as no share hold
    assert env.balance == init_balance
    assert env.net_worth == init_net_worth
    assert env.current_step == 1


def test_step_buy(stock_trading_env_two_stock):
    env = stock_trading_env_two_stock
    env.reset()
    actions = [0, 0]
    obs, reward, done, info = env.step(actions)
    check_action(env, info, actions, reward)
    actions = [0, 2]
    obs, reward, done, info = env.step(actions)
    check_action(env, info, actions, reward)
    actions = [2, 0]
    obs, reward, done, info = env.step(actions)
    check_action(env, info, actions, reward)


def test_step_sell(stock_trading_env_two_stock):
    env = stock_trading_env_two_stock
    env.reset()
    actions = [1, 2]
    obs, reward, done, info = env.step(actions)
    check_action(env, info, actions, reward)
    actions = [2, 1]
    obs, reward, done, info = env.step(actions)
    check_action(env, info, actions, reward)
    actions = [1, 1]
    obs, reward, done, info = env.step(actions)
    check_action(env, info, actions, reward)


def test_step_loop(stock_trading_env_two_stock):
    env = stock_trading_env_two_stock
    env.reset()
    done = False
    while not done:
        actions = env.action_space.sample()
        obs, reward, done, info = env.step(actions)
        check_action(env, info, actions, reward)
    assert env.current_step == env.total_time_step


def test_observe_future_frame():
    env = StockTradingEnvV1(debug=True,
                            observation_frame=10,
                            observe_future_frame=10,
                            file_array=['data/daily_IBM.csv', 'data/daily_MSFT.csv'])
    assert env.observe_future_frame == 10
    env.reset()
    done = False
    while not done:
        actions = env.action_space.sample()
        obs, reward, done, info = env.step(actions)
        check_action(env, info, actions, reward)
    assert env.current_step == env.total_time_step
    # env.render()


def test_normalize_false():
    from stable_baselines.common.env_checker import check_env
    env = StockTradingEnvV1(debug=True, normalize_observation=False)
    check_env(env)

    assert env.observation_space.shape == (14, )
    obs = env.reset()
    stacked_obs_part = np.reshape(obs[:env.obs_column_number * env.observation_frame], (-1, env.stock_number))
    index_end = env.current_step + env.start_point
    index_start = env.current_step + env.start_point - env.observation_frame + 1
    ref_obs = env.df[env.observation_column_name_array].loc[index_start:index_end].values
    assert ((ref_obs == stacked_obs_part).all()).all()


def test_print(stock_trading_v1_make):
    env = StockTradingEnvV1(debug=False,
                            observation_frame=5,
                            observe_future_frame=5,
                            stack=20,
                            file_array=['data/daily_MSFT.csv', 'data/daily_IBM.csv', ],
                            observation_column=['close'],
                            normalize_observation=False,
                            )

    # env = stock_trading_v1_make
    # print(env.observation_space)
    # env.total_time_step = 500
    # env.render(label_action=True)
    for _ in range(300):
        env.step(env.action_space.sample())
    env.render(label_action=True)

    print('')
    # print(np.reshape(env.reset(), (-1, env.stock_number)))
    # print(env.df[env.observation_column_name_array].loc[env.start_point-4:env.start_point+5])
    # print(env.observation_space.shape)

    # obs = env.reset()
    # print(np.reshape(obs, (-1, env.stock_number)))
    #
    # action = [0]
    # obs, reward, done, info = env.step(action)
    # print(np.reshape(obs, (-1, env.stock_number)))

