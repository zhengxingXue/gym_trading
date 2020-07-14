import pytest
from gym_trading.algorithms import ppo2_mlp_policy_train


def test_ppo2_mlp_policy_train():
    ppo2_mlp_policy_train(time_step=10 ** 4, verbose=0)
