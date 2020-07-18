import copy
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

from gym_trading.envs.helper import normalize_df, construct_df_array


class ActionInvalid(Exception):
    """Exception raised for invalid action"""

    def __init__(self, action, message="action must be 0 or 1 or 2"):
        self.action = action
        self.message = "action : " + str(action) + " is not valid \n" + message
        super().__init__(self.message)


class StockTradingEnvV1(gym.Env):
    """
    Stock trading environment for Reinforcement learning.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }

    INITIAL_BALANCE = 10000.  # initial balance
    MAX_BALANCE = INITIAL_BALANCE * 3  # used for normalization
    TOTAL_STEP = 300  # total number of timestep of the simulation

    def __init__(self, df=None, file_array=None, absolute_path=False, debug=False,
                 observation_frame=10, stack=10, observe_future_frame=0):
        super(StockTradingEnvV1, self).__init__()
        self.debug = debug

        # number of frames feed to observation
        self.observation_frame = observation_frame

        # number of future frames feed to observation
        # used for experiments to see if the RL agents know (or can predict) the future, can they imporve
        self.observe_future_frame = observe_future_frame

        # number of shares to operate with
        self.STACK = stack

        file_array = ['data/daily_IBM.csv'] if file_array is None else file_array

        self.df, self.max_share_value_array, self.stock_name_array = \
            construct_df_array(file_array, absolute_path) if df is None else df

        self.stock_number = len(self.max_share_value_array)
        self.normalized_df = normalize_df(self.df)
        self.column_number = self.normalized_df.shape[1]

        # action space : [0] - buy; [1] - sell; [2] - hold
        # for each stock
        self.action_space = spaces.MultiDiscrete([3] * self.stock_number)

        self.observation_space = \
            spaces.Box(low=0,
                       high=1,
                       shape=(self.column_number * (self.observation_frame + self.observe_future_frame)
                              + self.stock_number * 2 + 2,),
                       dtype=np.float32)

        self.net_worth = self.INITIAL_BALANCE
        self.net_worth_history = [self.net_worth]
        self.balance = self.INITIAL_BALANCE
        self.balance_history = [self.balance]

        # keep successfully executed actions
        # if sell or buy not valid, convert to hold action
        self.action_history = []

        # whether hold share of the stock
        # [0] - No ; [1] - Yes
        self.hold_share_array = [0] * self.stock_number

        # cost basis of each stock
        self.cost_basis_array = np.array([0] * self.stock_number)

        # index of start point
        self.start_point = 10 if self.debug \
            else random.randint(self.observation_frame+1,
                                len(self.df.index)-self.TOTAL_STEP-self.observe_future_frame-1)

        self.current_step = 0

    def reset(self):
        self.net_worth = self.INITIAL_BALANCE
        self.net_worth_history = [self.net_worth]
        self.balance = self.INITIAL_BALANCE
        self.balance_history = [self.balance]
        self.action_history = []
        self.hold_share_array = [0] * self.stock_number
        self.cost_basis_array = np.array([0] * self.stock_number)
        self.start_point = 10 if self.debug \
            else random.randint(self.observation_frame+1,
                                len(self.df.index)-self.TOTAL_STEP-self.observe_future_frame-1)
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        net_worth_before_step = self.net_worth
        self.action_history.append(copy.copy(action))
        # Execute one time step within the environment
        info = self._take_action(action)
        self.balance_history.append(self.balance)
        self.net_worth_history.append(self.net_worth)

        self.current_step += 1
        reward = self.net_worth - net_worth_before_step
        done = self.net_worth <= 0 or self.current_step >= self.TOTAL_STEP
        obs = self._get_obs()
        return obs, reward, done, info

    def render(self, mode='human', label_action=False):
        height = 3 * (self.stock_number + 1)
        fig = plt.figure(figsize=(9, height))
        padding = 10
        lower = - self.observation_frame - padding
        upper = self.TOTAL_STEP + 1 + padding + self.observe_future_frame

        ax_balance = plt.subplot(self.stock_number + 1, 1, 1)

        plt.axvline(x=self.current_step, label='current step', c='0.75')

        plt.plot(self.balance_history, label='balance')
        plt.plot(self.net_worth_history, label='net worth')

        ax_balance.set_xlim(lower, upper)

        plt.legend()
        # ax_balance.set_ylim(bottom=0)
        for i, stock_name in zip(range(self.stock_number), self.stock_name_array):
            ax = plt.subplot(self.stock_number + 1, 1, i + 2)
            plt.axvline(x=self.current_step, label='current step', c='0.7')
            ax.axvspan(self.current_step - self.observation_frame + 1, self.current_step + self.observe_future_frame,
                       color='0.9', label='observations')
            open_label = stock_name + '_' + 'open'
            close_label = stock_name + '_' + 'close'

            x = range(lower + padding, self.current_step + 1 + self.observe_future_frame)

            index_start = self.start_point - self.observation_frame
            index_end = self.start_point + self.current_step + self.observe_future_frame
            data = self.df[[open_label, close_label]].loc[index_start: self.start_point + self.TOTAL_STEP].values
            top = np.max(np.max(data)) + padding
            bottom = np.min(np.min(data)) - padding
            ax.set_ylim(bottom, top)

            x_tick_step = int(self.TOTAL_STEP / (len(ax_balance.get_xticklabels()) - 3))
            plt.xticks(range(0, self.TOTAL_STEP+1, x_tick_step),
                       self.df['date'].loc[self.start_point+0:self.start_point+self.TOTAL_STEP+1:x_tick_step])
            ax.set_xlim(lower, upper)

            line_open, = plt.plot(x, self.df[open_label].loc[index_start: index_end].values, label=open_label)
            line_close, = plt.plot(x, self.df[close_label].loc[index_start: index_end].values, label=close_label)

            plt.legend(handles=[line_open, line_close])
            if label_action:
                for j, actions in zip(range(self.current_step), self.action_history):
                    current_action = actions[i]
                    if current_action == 0:  # buy
                        y = self.df[close_label].loc[self.start_point+j+1]
                        circle = plt.Circle((j, y+10), .5, color='r', label='buy')
                        ax.add_artist(circle)
                    elif current_action == 1:  # sell
                        y = self.df[close_label].loc[self.start_point+j+1]
                        circle = plt.Circle((j, y-10), .5, color='g', label='sell')
                        ax.add_artist(circle)
                    else:  # hold (and other)
                        pass
            else:
                pass

            plt.legend()

        if mode == 'human':
            plt.show()
        elif mode == 'rgb_array':
            # TODO: convert plt to rgb array
            pass
        else:
            pass

    def _take_action(self, action):
        # TODO: Examine whether to update the self.cost_basis_array to 0 if no share hold
        # sell and buy price is based on next day's open price
        # action space : [0] - buy; [1] - sell; [2] - hold
        info = {}
        if self.debug:
            info['balance_before_step'] = self.balance
            info['net_worth_before_step'] = self.net_worth
        for i, action, stock_name in zip(range(self.stock_number), action, self.stock_name_array):
            info[stock_name] = {}
            if action == 0:  # buy
                info[stock_name]['action'] = 'buy'
                # whether hold share of the stock [0] - No ; [1] - Yes
                if self.hold_share_array[i]:
                    # already hold STACK number of stock, no action
                    info[stock_name]['price'] = 'fail'
                    # update action history to hold (as the effect is the same)
                    self.action_history[self.current_step][i] = 2
                else:
                    # buy STACK number of shares
                    # execute on next day's open price
                    share_value_column = stock_name + '_' + 'open'
                    share_value_row = self.current_step + self.start_point + 1
                    share_value = self.df[share_value_column].loc[share_value_row]

                    info[stock_name]['price'] = share_value

                    self.balance -= self.STACK * share_value
                    self.net_worth = self.net_worth  # net worth unchanged
                    # TODO: add commission
                    self.cost_basis_array[i] = share_value
                    self.hold_share_array[i] = 1  # hold share
            elif action == 1:  # sell
                info[stock_name]['action'] = 'sell'
                # whether hold share of the stock [0] - No ; [1] - Yes
                if self.hold_share_array[i]:
                    # already hold STACK number of stock, sell
                    share_value_column = stock_name + '_' + 'open'
                    share_value_row = self.current_step + self.start_point + 1
                    share_value = self.df[share_value_column].loc[share_value_row]

                    info[stock_name]['price'] = share_value

                    self.balance += self.STACK * share_value
                    # TODO: add commission
                    self.cost_basis_array[i] = share_value
                    self.hold_share_array[i] = 0  # no share hold
                else:
                    # hold no stock, cannot sell no action
                    info[stock_name]['price'] = 'fail'
                    # update action history to hold (as the effect is the same)
                    self.action_history[self.current_step][i] = 2
            elif action == 2:  # hold
                info[stock_name]['action'] = 'hold'
                info[stock_name]['price'] = 'success'
                pass
            else:
                raise ActionInvalid(action)

        self._update_net_worth()

        if self.debug:
            info['balance_after_step'] = self.balance
            info['net_worth_after_step'] = self.net_worth

        return info

    def _update_net_worth(self):
        # TODO: Net worth is calculated by the close price of today,
        #  balance is calculated by the open price of next day,
        #  this might cause problem for RL

        # net worth is based on the close price of current timestep
        stock_close_value_array = np.zeros(self.stock_number)

        for i, stock_name in zip(range(self.stock_number), self.stock_name_array):
            share_value_column = stock_name + '_' + 'close'
            share_value_row = self.current_step + self.start_point
            share_value = self.df[share_value_column].loc[share_value_row]
            stock_close_value_array[i] = share_value * self.STACK

        self.net_worth = self.balance + np.sum(stock_close_value_array * self.hold_share_array)

    def _get_obs(self):
        obs = np.array([])
        index_start = self.current_step + self.start_point - self.observation_frame + 1
        index_end = self.current_step + self.start_point + self.observe_future_frame
        obs = np.append(obs, self.normalized_df.loc[index_start:index_end].values.flatten())  # as date pop
        obs = np.append(obs, self.hold_share_array)
        obs = np.append(obs, self.cost_basis_array / self.max_share_value_array)
        obs = np.append(obs, np.array([self.balance, self.net_worth]) / self.MAX_BALANCE)
        return obs
