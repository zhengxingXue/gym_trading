import gym
import numpy as np
from gym import spaces
from gym_trading.envs.helper import normalize_df, construct_df_array


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

    def __init__(self, df=None, file_array=None, absolute_path=False):
        super(StockTradingEnvV1, self).__init__()

        # TODO: Process the pd to have same shape

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

        self.start_point = 10   # index of start point
        self.current_step = 0

    def reset(self):
        self.net_worth = self.INITIAL_BALANCE
        self.balance = self.INITIAL_BALANCE
        self.hold_share_array = [0] * self.stock_number
        self.cost_basis_array = np.array([0] * self.stock_number)
        self.start_point = 10   # index of start point
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        pass

    def render(self, mode='human'):
        pass

    def _take_action(self, action):
        pass

    def _get_obs(self):
        obs = np.array([])
        index_start = self.current_step + self.start_point - self.observation_frame
        index_end = self.current_step + self.start_point - 1

        obs = np.append(obs, self.normalized_df.loc[index_start:index_end].values.flatten())  # as date pop

        obs = np.append(obs, self.hold_share_array)
        obs = np.append(obs, self.cost_basis_array / self.max_share_value_array)
        obs = np.append(obs, np.array([self.balance, self.net_worth]) / self.MAX_BALANCE)
        return obs
