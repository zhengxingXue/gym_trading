import gym
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
    metadata = {'render.modes': ['human']}

    observation_frame = 3  # number of frames feed to observation
    STACK = 10  # number of shares to operate with
    INITIAL_BALANCE = 10000.  # initial balance
    MAX_BALANCE = INITIAL_BALANCE * 3  # used for normalization
    TOTAL_STEP = 300    # total number of timestep of the simulation

    def __init__(self, df=None, file_array=None, absolute_path=False, debug=False):
        super(StockTradingEnvV1, self).__init__()
        self.debug = debug

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
                       shape=(self.column_number * self.observation_frame + self.stock_number * 2 + 2,),
                       dtype=np.float32)

        self.net_worth = self.INITIAL_BALANCE
        self.balance = self.INITIAL_BALANCE

        # whether hold share of the stock
        # [0] - No ; [1] - Yes
        self.hold_share_array = [0] * self.stock_number

        # cost basis of each stock
        self.cost_basis_array = np.array([0] * self.stock_number)

        self.start_point = 10 if self.debug else 10  # index of start point
        self.current_step = 0

    def reset(self):
        self.net_worth = self.INITIAL_BALANCE
        self.balance = self.INITIAL_BALANCE
        self.hold_share_array = [0] * self.stock_number
        self.cost_basis_array = np.array([0] * self.stock_number)
        self.start_point = 10 if self.debug else 10  # index of start point
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        net_worth_before_step = self.net_worth
        # Execute one time step within the environment
        info = self._take_action(action)
        self.current_step += 1
        reward = self.net_worth - net_worth_before_step
        done = self.net_worth <= 0 or self.current_step >= self.TOTAL_STEP
        obs = self._get_obs()
        return obs, reward, done, info

    def render(self, mode='human'):
        if mode == 'human':
            profit = self.net_worth - self.INITIAL_BALANCE
            print(f'Step: {self.current_step}')
            print(f'Balance: {self.balance}')
            print(f'Net worth: {self.net_worth}')
            print(f'Profit: {profit}')
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
                    pass
                else:
                    # buy STACK number of shares
                    # execute on next day's open price
                    share_value_column = stock_name + '_' + 'open'
                    share_value_row = self.current_step + self.start_point + 1
                    share_value = self.df[share_value_column].loc[share_value_row]

                    info[stock_name]['price'] = share_value

                    self.balance -= self.STACK * share_value
                    self.net_worth = self.net_worth     # net worth unchanged
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
                    pass
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
        index_start = self.current_step + self.start_point - self.observation_frame
        index_end = self.current_step + self.start_point - 1
        obs = np.append(obs, self.normalized_df.loc[index_start:index_end].values.flatten())  # as date pop
        obs = np.append(obs, self.hold_share_array)
        obs = np.append(obs, self.cost_basis_array / self.max_share_value_array)
        obs = np.append(obs, np.array([self.balance, self.net_worth]) / self.MAX_BALANCE)
        return obs
