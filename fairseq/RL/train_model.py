import os 

import numpy as np 
import torch
import fairseq
from typing import Callable

import gym 
from gym.wrappers import TimeLimit

import stable_baselines3d
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
 
def make_vec_envs(env_id: str, time_limit: int, n_envs: int):
    def make_envs(env_id, time_limit):
        return Monitor(TimeLimit(gym.make(env_id),time_limit))
    if n_envs==1:
        return DummyVecEnv([lambda:make_envs(env_id, time_limit)])
    elif n_envs>1:
        raise NotImplementedError("Can't handle multiple envs yet.")
    else:
        raise ValueError("Number of environments must be greater than 0")
    
def main(algo: str, policy: str ,eval_freq=1000, n_eval_eps=10, log_dir="./logs/"):
    env=make_vec_envs("EnFrSNmtEnv-v0", 50, 1)
    model=A2C(policy, env, verbose=1, tensorboard_log=log_dir, device='cuda') if algo=='A2C' else PPO(policy, env, verbose=0)
    
    # train model
    model.learn(total_timesteps=10000, eval_freq=500, n_eval_episodes=20, tb_log_name="A2C_EnFr_run1")

if __name__=="__main__":  
    main('A2C', 'MlpPolicy')