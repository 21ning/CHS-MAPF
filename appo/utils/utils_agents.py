import pathlib
from typing import Optional

import torch
from pogema.animation import AnimationConfig, AnimationMonitor
from pydantic import BaseModel

from pomapf_env.env import make_pomapf
from pomapf_env.pomapf_config import POMAPFConfig


class AlgoBase(BaseModel):
    name: str = None
    num_process: int = 3
    device: str = 'cuda'
    seed: Optional[int] = 0


def run_algorithm(algo, map_name='sc1-AcrosstheCape', max_episode_steps=512, seed=None, num_agents=64, animate=False):
    gc = POMAPFConfig(num_agents=16, size=32, density=0.3, seed = seed ,max_episode_steps=128, obs_radius = 3 )
    env = make_pomapf(grid_config=gc, with_animations=True)
    algo_name = type(algo).__name__
    if animate:
        anim_dir = str(pathlib.Path('renders') / algo_name)
        env = AnimationMonitor(env, AnimationConfig(directory=anim_dir))
    obs = env.reset()
    algo.after_reset()
    results_holder = ResultsHolder()
    dones = [False for _ in range(len(obs))]
    infos = [{'is_active': True} for _ in range(len(obs))]
    rew = [0 for _ in range(len(obs))]
    a=0
    with torch.no_grad():
        while True:
            obs, rew, dones, infos = env.step(algo.act(obs, rew, dones, infos))
            results_holder.after_step(infos)
            results_holder.after_step2(a)
            algo.after_step(dones)
            if all(dones):
                #print(a)
                break
            a+=1

    results = results_holder.get_final()
    results['algorithm'] = algo_name
    return results


class ResultsHolder:
    def __init__(self):
        self.results = dict()

    def after_step(self, infos):
        if 'metrics' in infos[0]:
            self.results.update(**infos[0]['metrics'])
    def after_step2(self, step):
        self.results.update({"Step": step})

    def get_final(self):
        return self.results
