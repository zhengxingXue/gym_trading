# modified from https://github.com/notadamking/Stock-Trading-Environment/blob/master/env/StockTradingEnv.py
import os
import random

import gym
import numpy as np
import pandas as pd
from gym import spaces

from gym_trading.config import PACKAGE_DIR
from gym_trading.envs.helper import normalize_df

# get rid of gym UserWarning: Box bound precision lowered...
gym.logger.set_level(40)


class StockTradingEnvV0(gym.Env):
    """
    Stock trading environment.

    **STATE:**
    Prices contains the OHCL values for the last five prices

    **ACTIONS:**
    Actions of the format Buy x%, Sell x%, Hold, etc.

    """

    metadata = {'render.modes': ['human']}

    MAX_ACCOUNT_BALANCE = 200000
    MAX_NUM_SHARES = 20000
    MAX_SHARE_PRICE = 800  # depend on data set
    INITIAL_ACCOUNT_BALANCE = 10000
    TIME_STEP = 300

    def __init__(self, df=None, file='data/AAPL.csv', absolute_path=False):
        super(StockTradingEnvV0, self).__init__()

        if df is None:
            if absolute_path:
                file_path = file
            else:
                file_path = os.path.join(PACKAGE_DIR, file)
            self.df = pd.read_csv(file_path)
            self.df.columns = self.df.columns.str.lower()
            self.df = self.df.sort_values('date')
        else:
            self.df = df

        self.normalized_df = normalize_df(self.df)
        self.reward_range = (0, self.MAX_ACCOUNT_BALANCE - self.INITIAL_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32), high=np.array([3, 1], dtype=np.float32), dtype=np.float32)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float32)

        self.reset()

    def step(self, action):

        init_net_worth = self.net_worth

        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        reward = self.net_worth - init_net_worth
        done = self.net_worth <= 0 or self.current_step >= self.TIME_STEP

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = self.INITIAL_ACCOUNT_BALANCE
        self.net_worth = self.INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = self.INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Random select start
        self.start_point = random.randint(6, len(self.df.index) - self.TIME_STEP - 1)

        # Set the current step to 0
        self.current_step = 0

        return self._next_observation()

    def render(self, mode='human'):
        # Render the environment to the screen
        profit = self.net_worth - self.INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')

    def _next_observation(self):
        # Get the stock data points for the last 6 days and scale to between 0-1
        index_start = self.current_step + self.start_point - 6
        index_end = self.current_step + self.start_point - 1
        frame = np.array([
            self.normalized_df.loc[index_start: index_end, 'open'].values,
            self.normalized_df.loc[index_start: index_end, 'high'].values,
            self.normalized_df.loc[index_start: index_end, 'low'].values,
            self.normalized_df.loc[index_start: index_end, 'close'].values,
            self.normalized_df.loc[index_start: index_end, 'volume'].values,
        ])

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / self.MAX_ACCOUNT_BALANCE,
            self.max_net_worth / self.MAX_ACCOUNT_BALANCE,
            self.shares_held / self.MAX_NUM_SHARES,
            self.cost_basis / self.MAX_SHARE_PRICE,
            self.total_shares_sold / self.MAX_NUM_SHARES,
            self.total_sales_value / (self.MAX_NUM_SHARES * self.MAX_SHARE_PRICE),
        ]], axis=0)

        return obs

    def _take_action(self, action):
        # Set the current price to avg of High and Low
        current_point = self.current_step + self.start_point
        current_price = (self.df.loc[current_point, "high"] + self.df.loc[current_point, "low"]) / 2

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.shares_held += shares_bought
            self.cost_basis = (prev_cost + additional_cost) / self.shares_held if self.shares_held != 0 else 0

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0
