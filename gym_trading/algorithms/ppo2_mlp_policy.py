from stable_baselines.common.policies import MlpPolicy
from gym_trading.algorithms.utils.stable_baseline_helper import ppo2_train


def ppo2_mlp_policy_train(env_id='StockTrading-IBM-MSFT-v1', env_n=8, time_step=10 ** 5, verbose=1):
    """
    :param env_id: environment id
    :param env_n: number of environment for ppo2
    :param time_step: training time steps
    :param verbose: 0 - no training info ; 1 - training info
    :return: trained model and save directory
    """
    policy = MlpPolicy
    policy_name = 'mlp_policy'
    model, save_dir = ppo2_train(policy, policy_name, env_id=env_id, env_n=env_n, time_step=time_step,
                                 enable_call_back=False, verbose=verbose)
    return model, save_dir


if __name__ == '__main__':
    ppo2_mlp_policy_train()
