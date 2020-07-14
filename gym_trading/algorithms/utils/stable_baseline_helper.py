import logging
import os
import time

import gym
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from gym_trading.config import PACKAGE_DIR

logging.getLogger('tensorflow').setLevel(logging.ERROR)


def create_custom_policy(net_arch):
    """
    function to create custom policy
    eg. net_arch = [64, 64]
    eg. net_arch = [64, dict(pi=[64],vf=[64])]
    """
    class CustomPolicy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomPolicy, self).__init__(*args, **kwargs, net_arch=net_arch, feature_extraction="mlp")
    return CustomPolicy


def create_n_env(env, log_dir='./temp', num_envs=8):
    """
    :param env: (str or gym.env) if env is string use gym.make() else directly use env
    :param log_dir: log directory
    :param num_envs: number of environment
    :return: DummyVecEnv for training
    """
    if isinstance(env, str):
        env = gym.make(env)
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env] * num_envs)
    return env


def create_eval_callback(env_id, save_dir='./logs', eval_freq=1000, n_eval_episodes=10):
    """
    :param env_id: environment id
    :param save_dir: the directory to save the best model
    :param eval_freq: the frequency of the evaluation callback
    :param n_eval_episodes: the number  of evaluation of each callback
    :return: EvalCallback for training
    """
    eval_env = gym.make(env_id)
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_dir,
                                 log_path=save_dir,
                                 eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
                                 deterministic=False, render=False)
    return eval_callback


def ppo2_train(policy, policy_name, env_id='StockTrading-IBM-MSFT-v1', env_n=8, time_step=10**5,
               enable_call_back=True, verbose=1):
    """
    :param policy: stable-baseline policy
    :param policy_name: string of policy name
    :param env_id: id of the environment
    :param env_n: number of env for the PPO2 model
    :param time_step: total time step for training
    :param enable_call_back: whether to use the eval call back
    :param verbose: whether to show training info
    :return: (PPO2, str)the trained ppo2 model, and the save directory
    """
    time_str = "{}".format(int(time.time()))
    log_dir = os.path.join(PACKAGE_DIR, 'training/tmp/'+policy_name+'-'+time_str)
    os.makedirs(log_dir, exist_ok=True)
    env = create_n_env(env_id, log_dir, env_n)

    save_dir = os.path.join(PACKAGE_DIR, 'training/logs/'+policy_name+'-'+time_str)
    os.makedirs(save_dir, exist_ok=True)

    model = PPO2(policy, env, verbose=verbose)
    if enable_call_back:
        eval_callback = create_eval_callback(env_id, save_dir=save_dir)
        model.learn(total_timesteps=time_step, callback=eval_callback)
    else:
        model.learn(total_timesteps=time_step)

    model_path = save_dir + '/' + policy_name
    model.save(model_path)
    print("model saved to " + model_path + '.zip')
    return model, save_dir
