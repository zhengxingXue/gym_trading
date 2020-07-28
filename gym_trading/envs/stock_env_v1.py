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
        self.message = "action : " + str(action) + " is not valid. \n" + message
        super().__init__(self.message)


class StockTradingEnvV1(gym.Env):
    """
    Stock trading environment for Reinforcement learning.

    **STATE:**
    Stock price and balance information depend on the observation frame.
    Flattened and normalized.

    **ACTION:**
    Every stock has MultiDiscrete action space [0] - buy; [1] - sell; [2] - hold
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }

    def __init__(self, df=None, file_array=None, absolute_path=False, **kwargs):
        super(StockTradingEnvV1, self).__init__()

        attribute_dict = {
            'debug': False,  # used for testing and debugging
            'observation_frame': 10,  # number of frames feed to observation
            'stack': 10,  # number of shares to operate with
            'observe_future_frame': 0,  # number of future frames feed to observation
            'initial_balance': 10000.,  # initial balance
            'max_balance': 20000.,  # used for normalization
            'total_time_step': 300,  # total time step for the simulation
            'observation_column': ['close'],  # column names for observation
            'sell_buy_column': 'close',  # column name for sell and buy action's share value
            'net_worth_column': 'close',  # column name for net worth calculation
            'normalize_observation': True,  # whether to normalize the observation
        }

        attribute_dict.update((k, kwargs[k]) for k in attribute_dict.keys() & kwargs.keys())

        invalid_attr_key_in_kwargs = kwargs.keys() - attribute_dict.keys()
        if invalid_attr_key_in_kwargs != set():
            print("invalid attribute key " + str(invalid_attr_key_in_kwargs) + " for class initialization.")

        for key in attribute_dict:
            self.__setattr__(key, attribute_dict[key])

        file_array = ['data/daily_IBM.csv'] if file_array is None else file_array

        self.df, self.max_share_value_array, self.stock_name_array = \
            construct_df_array(file_array, absolute_path) if df is None else df

        self.stock_number = len(self.max_share_value_array)
        self.normalized_df = normalize_df(self.df)

        self.observation_column_name_array = []

        for stock_name in self.stock_name_array:
            for col in self.observation_column:
                self.observation_column_name_array.append(stock_name + '_' + col)

        self.obs_column_number = len(self.observation_column_name_array)

        # action space : [0] - buy; [1] - sell; [2] - hold
        # for each stock
        self.action_space = spaces.MultiDiscrete([3] * self.stock_number)

        obs_len = self.obs_column_number * \
                    (self.observation_frame + self.observe_future_frame) + self.stock_number * 2 + 2

        if self.normalize_observation:
            self.observation_space = \
                spaces.Box(low=0,
                           high=1,
                           shape=(obs_len,),
                           dtype=np.float32)
        else:
            # TODO: 'volume' column is not working in not normalized observation
            low = [0] * obs_len
            high = []
            for max_share_value in self.max_share_value_array:
                high += [max_share_value] * len(self.observation_column)
            high *= self.observation_frame + self.observe_future_frame
            # hold share array
            high += [1] * self.stock_number
            # cost basis array
            high += self.max_share_value_array.tolist()
            # net worth and balance
            high += [self.max_balance] * 2

            self.observation_space = \
                spaces.Box(low=np.array(low),
                           high=np.array(high),
                           dtype=np.float32)

        self.net_worth = self.initial_balance
        self.net_worth_history = []
        self.balance = self.initial_balance
        self.balance_history = []

        # keep successfully executed actions
        # if sell or buy not valid, convert to hold action
        self.action_history = []

        # whether hold share of the stock
        # [0] - No ; [1] - Yes
        self.hold_share_array = np.array([0] * self.stock_number)

        # cost basis of each stock
        self.cost_basis_array = np.array([0] * self.stock_number)

        # index of start point
        self.start_point = 10 if self.debug \
            else random.randint(self.observation_frame + 1,
                                len(self.df.index) - self.total_time_step - self.observe_future_frame - 1)

        self.current_step = 0

    def reset(self):
        self.net_worth = self.initial_balance
        self.net_worth_history = []
        self.balance = self.initial_balance
        self.balance_history = []
        self.action_history = []
        self.hold_share_array = np.array([0] * self.stock_number)
        self.cost_basis_array = np.array([0] * self.stock_number)
        self.start_point = 10 if self.debug \
            else random.randint(self.observation_frame + 1,
                                len(self.df.index) - self.total_time_step - self.observe_future_frame - 1)
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
        done = self.net_worth <= 0 or self.current_step >= self.total_time_step
        obs = self._get_obs()
        return obs, reward, done, info

    def render(self, mode='human', label_action=False):
        height = 3 * (self.stock_number + 1)
        fig = plt.figure(figsize=(9, height))
        padding = 10
        lower = - self.observation_frame - padding
        upper = self.total_time_step + 1 + padding + self.observe_future_frame

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

            column_label_array = []
            for column_name in self.observation_column:
                if column_name != 'volume':
                    column_label_array.append(stock_name + '_' + column_name)

            x = range(lower + padding, self.current_step + 1 + self.observe_future_frame)

            index_start = self.start_point - self.observation_frame
            index_end = self.start_point + self.current_step + self.observe_future_frame
            data = self.df[column_label_array].loc[index_start: self.start_point + self.total_time_step].values
            top = np.max(np.max(data)) + padding
            bottom = np.min(np.min(data)) - padding
            ax.set_ylim(bottom, top)

            x_tick_step = int(self.total_time_step / (len(ax_balance.get_xticklabels()) - 3))
            plt.xticks(range(0, self.total_time_step + 1, x_tick_step),
                       self.df['date'].loc[
                       self.start_point + 0:self.start_point + self.total_time_step + 1:x_tick_step])
            ax.set_xlim(lower, upper)

            line_handles = []
            for column_label in column_label_array:
                line, = plt.plot(x, self.df[column_label].loc[index_start: index_end].values, label=column_label)
                line_handles.append(line)

            plt.legend(handles=line_handles)

            if label_action:
                for j, actions in zip(range(self.current_step), self.action_history):
                    current_action = actions[i]
                    if current_action == 0:  # buy
                        y = self.df[stock_name + '_' + self.sell_buy_column].loc[self.start_point + j + 1]
                        circle = plt.Circle((j, y + 10), .5, color='r', label='buy')
                        ax.add_artist(circle)
                    elif current_action == 1:  # sell
                        y = self.df[stock_name + '_' + self.sell_buy_column].loc[self.start_point + j + 1]
                        circle = plt.Circle((j, y - 10), .5, color='g', label='sell')
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
        # sell and buy price is based on today's close price
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
                    # already hold stack number of stock, no action
                    info[stock_name]['price'] = 'fail'
                    # update action history to hold (as the effect is the same)
                    self.action_history[self.current_step][i] = 2
                else:
                    # buy stack number of shares
                    # execute on today's close price
                    share_value = self._get_current_share_value(stock_name, column_name=self.sell_buy_column)

                    info[stock_name]['price'] = share_value

                    self.balance -= self.stack * share_value
                    self.net_worth = self.net_worth  # net worth unchanged
                    # TODO: add commission
                    self.cost_basis_array[i] = share_value
                    self.hold_share_array[i] = 1  # hold share
            elif action == 1:  # sell
                info[stock_name]['action'] = 'sell'
                # whether hold share of the stock [0] - No ; [1] - Yes
                if self.hold_share_array[i]:
                    # already hold stack number of stock, sell
                    share_value = self._get_current_share_value(stock_name, column_name=self.sell_buy_column)

                    info[stock_name]['price'] = share_value

                    self.balance += self.stack * share_value
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

    def _get_current_share_value(self, stock_name, column_name='close', add_step=0):
        """
        Get share value of specified step. Default to close value of current time step.
        :param stock_name: name of the stock
        :param column_name: which column value to use
        :param add_step: 0 - for current time step. modify to get past or future price
        :return: share value of the specified timestep
        """
        share_value_column = stock_name + '_' + column_name
        share_value_row = self.current_step + self.start_point + add_step
        share_value = self.df[share_value_column].loc[share_value_row]
        return share_value

    def _update_net_worth(self):
        # net worth is based on the close price of current timestep
        stock_close_value_array = np.zeros(self.stock_number)

        for i, stock_name in zip(range(self.stock_number), self.stock_name_array):
            share_value = self._get_current_share_value(stock_name, column_name=self.net_worth_column)
            stock_close_value_array[i] = share_value * self.stack

        self.net_worth = self.balance + np.sum(stock_close_value_array * self.hold_share_array)

    def _get_obs(self):
        index_start = self.current_step + self.start_point - self.observation_frame + 1
        index_end = self.current_step + self.start_point + self.observe_future_frame
        # obs = np.array([])
        obs = []
        if self.normalize_observation:
            # list append is more efficient than numpy append
            # https://www.quora.com/Is-it-better-to-use-np-append-or-list-append
            obs += self.normalized_df[self.observation_column_name_array].loc[
                   index_start:index_end].values.flatten().tolist()
            obs += self.hold_share_array.tolist()
            obs += (self.cost_basis_array / self.max_share_value_array).tolist()
            obs += [self.balance/self.max_balance, self.net_worth/self.max_balance]
            obs = np.array(obs)
            # obs = np.append(obs, self.normalized_df[
            #                      self.observation_column_name_array].loc[index_start:index_end].values.flatten())
            # obs = np.append(obs, self.hold_share_array)
            # obs = np.append(obs, self.cost_basis_array / self.max_share_value_array)
            # obs = np.append(obs, np.array([self.balance, self.net_worth]) / self.max_balance)
        else:
            obs += self.df[self.observation_column_name_array].loc[index_start:index_end].values.flatten().tolist()
            obs += self.hold_share_array.tolist()
            obs += self.cost_basis_array.tolist()
            obs += [self.balance, self.net_worth]
            obs = np.array(obs)
            # obs = np.append(obs, self.df[
            #                         self.observation_column_name_array].loc[index_start:index_end].values.flatten())
            # obs = np.append(obs, self.hold_share_array)
            # obs = np.append(obs, self.cost_basis_array)
            # obs = np.append(obs, [self.balance, self.net_worth])
        return obs
